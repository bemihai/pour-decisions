"""Run the M3 Phase 2 body-only versus contextual-search ablation.

This command rebuilds disposable Chroma and BM25 indexes from the accepted source
collection. Both variants retain the same IDs, clean documents, metadata, model,
and retrieval settings. Only the text used for dense indexing, sparse indexing,
and reranker pairs changes.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.chroma.bm25_builder import compute_chunk_ids_sha256, read_collection_documents
from src.eval.contextual_ablation import (
    SUPPORTED_SEARCH_REPRESENTATIONS,
    SearchRepresentation,
    SearchRepresentationReranker,
    build_ablation_documents,
    validate_aligned_ablation_documents,
)
from src.eval.dataset import load_golden_dataset
from src.eval.metrics import precision_at_k, reciprocal_rank
from src.eval.utils import run_rag_retrieval_only_sync
from src.retrieval import ChromaRetriever, HybridRetriever, build_reranker_from_config
from src.retrieval.keyword_search import BM25Index
from src.utils import compute_file_hash, get_config, get_embedder, initialize_chroma_client, logger
from src.utils.env import load_env


DEFAULT_COHORT_PATH = Path("src/eval/m3b_contextual_enrichment_cohort.json")
DEFAULT_OUTPUT_PATH = Path("eval-results") / f"m3b_contextual_ablation_{datetime.now(UTC):%Y%m%d}.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--variant",
        choices=["both", *SUPPORTED_SEARCH_REPRESENTATIONS],
        default="both",
        help="Run both representations or one diagnostic variant.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def load_frozen_cohort(path: Path) -> tuple[dict[str, Any], list[Any]]:
    """Load the cohort and reject dataset drift before expensive indexing."""
    cohort = json.loads(path.read_text(encoding="utf-8"))
    dataset_path = Path(str(cohort["dataset_path"]))
    current_hash = compute_file_hash(dataset_path)
    expected_hash = str(cohort["dataset_content_hash"])
    if current_hash != expected_hash:
        raise ValueError(
            "Frozen cohort dataset hash mismatch: "
            f"expected={expected_hash}, current={current_hash}"
        )

    samples_by_id = {sample.id: sample for sample in load_golden_dataset(dataset_path)}
    sample_ids = [str(entry["sample_id"]) for entry in cohort["samples"]]
    missing = sorted(set(sample_ids) - set(samples_by_id))
    if missing:
        raise ValueError(f"Frozen cohort contains missing sample IDs: {', '.join(missing)}")
    return cohort, [samples_by_id[sample_id] for sample_id in sample_ids]


def materialize_variant_collection(
    *,
    client: Any,
    collection_name: str,
    collection_metadata: dict[str, Any],
    documents: list[dict[str, Any]],
    embedder: Any,
    batch_size: int,
) -> Any:
    """Build one disposable Chroma collection from explicit search text."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    collection = client.create_collection(name=collection_name, metadata=collection_metadata)
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        embeddings = embedder.embed_documents([str(document["search_text"]) for document in batch])
        collection.add(
            ids=[str(document["id"]) for document in batch],
            embeddings=embeddings,
            documents=[str(document["document"]) for document in batch],
            metadatas=[dict(document["metadata"]) for document in batch],
        )
        logger.info(
            "Materialized ablation collection %s: %d/%d records",
            collection_name,
            min(start + len(batch), len(documents)),
            len(documents),
        )
    return collection


def run_variant(
    *,
    representation: SearchRepresentation,
    client: Any,
    collection_metadata: dict[str, Any],
    documents: list[dict[str, Any]],
    samples: list[Any],
    config: Any,
    batch_size: int,
) -> dict[str, Any]:
    """Build and evaluate one temporary representation variant."""
    temporary_collection_name = f"m3b-{representation.replace('_', '-')}-{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()
    try:
        embedder = get_embedder(model_name=str(config.chroma.settings.embedder))
        materialize_variant_collection(
            client=client,
            collection_name=temporary_collection_name,
            collection_metadata=collection_metadata,
            documents=documents,
            embedder=embedder,
            batch_size=batch_size,
        )

        bm25 = BM25Index()
        bm25.build_index(documents)
        retrieval_cfg = config.chroma.retrieval
        vector_retriever = ChromaRetriever(
            client=client,
            collection_name=temporary_collection_name,
            embedding_model=str(config.chroma.settings.embedder),
            n_results=int(retrieval_cfg.n_results),
            similarity_threshold=float(retrieval_cfg.similarity_threshold),
            enable_query_expansion=False,
            enable_cache=False,
        )
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_index=bm25,
            semantic_candidate_pool=int(retrieval_cfg.semantic_candidate_pool),
            bm25_candidate_pool=int(retrieval_cfg.bm25_candidate_pool),
            reranker_input_limit=int(retrieval_cfg.reranker_input_limit),
        )
        base_reranker = build_reranker_from_config(config)
        reranker = (
            SearchRepresentationReranker(base_reranker, representation)
            if base_reranker is not None
            else None
        )

        per_sample: list[dict[str, Any]] = []
        for sample in samples:
            sample_started = time.perf_counter()
            result = run_rag_retrieval_only_sync(sample, config, retriever, reranker)
            if result.retrieval_error:
                raise RuntimeError(f"Retrieval failed for {sample.id}: {result.retrieval_error}")
            retrieved_ids = [chunk.id for chunk in result.context_chunks]
            scores = {
                "mrr": reciprocal_rank(retrieved_ids, sample.ground_truth_chunk_ids),
                "precision_at_3": precision_at_k(retrieved_ids, sample.ground_truth_chunk_ids, 3),
                "precision_at_5": precision_at_k(retrieved_ids, sample.ground_truth_chunk_ids, 5),
            }
            per_sample.append(
                {
                    "sample_id": sample.id,
                    "retrieved_chunk_ids": retrieved_ids,
                    "contexts": [chunk.text for chunk in result.context_chunks],
                    "scores": scores,
                    "retrieval_confidence": result.retrieval_confidence,
                    "low_confidence": result.low_confidence,
                    "latency_ms": round((time.perf_counter() - sample_started) * 1000, 3),
                }
            )

        aggregate_metrics = {
            metric: sum(float(sample["scores"][metric]) for sample in per_sample) / len(per_sample)
            for metric in ("mrr", "precision_at_3", "precision_at_5")
        }
        return {
            "representation": representation,
            "record_count": len(documents),
            "aggregate_metrics": aggregate_metrics,
            "mean_retrieval_latency_ms": (
                sum(float(sample["latency_ms"]) for sample in per_sample) / len(per_sample)
            ),
            "total_runtime_seconds": round(time.perf_counter() - started, 3),
            "per_sample": per_sample,
        }
    finally:
        try:
            client.delete_collection(temporary_collection_name)
            logger.info("Deleted temporary ablation collection %s", temporary_collection_name)
        except Exception as exc:
            logger.warning(
                "Could not delete temporary ablation collection %s: %s",
                temporary_collection_name,
                exc,
            )


def build_comparison(variants: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Build contextual-minus-body deltas when both variants were run."""
    if set(variants) != set(SUPPORTED_SEARCH_REPRESENTATIONS):
        return None
    body = variants["body_only"]
    contextual = variants["contextual"]
    return {
        "metric_deltas": {
            metric: contextual["aggregate_metrics"][metric] - body["aggregate_metrics"][metric]
            for metric in ("mrr", "precision_at_3", "precision_at_5")
        },
        "mean_retrieval_latency_delta_ms": (
            contextual["mean_retrieval_latency_ms"] - body["mean_retrieval_latency_ms"]
        ),
    }


def build_configuration_snapshot(config: Any, batch_size: int) -> dict[str, Any]:
    """Record every setting that must remain fixed between variants."""
    retrieval = config.chroma.retrieval
    return {
        "embedding_model": str(config.chroma.settings.embedder),
        "reranking_enabled": bool(retrieval.enable_reranking),
        "reranker_model": str(retrieval.reranker_model),
        "rerank_threshold": (
            None if retrieval.rerank_threshold is None else float(retrieval.rerank_threshold)
        ),
        "rerank_top_k": int(retrieval.rerank_top_k),
        "semantic_candidate_pool": int(retrieval.semantic_candidate_pool),
        "bm25_candidate_pool": int(retrieval.bm25_candidate_pool),
        "reranker_input_limit": int(retrieval.reranker_input_limit),
        "similarity_threshold": float(retrieval.similarity_threshold),
        "metadata_boosting_enabled": bool(retrieval.enable_metadata_boost),
        "metadata_boost_factor": float(retrieval.metadata_boost_factor),
        "deduplication_enabled": bool(retrieval.use_deduplication),
        "deduplication_threshold": float(retrieval.deduplication_threshold),
        "batch_size": batch_size,
    }


def main() -> int:
    """Run the requested ablation variants and write one comparison artifact."""
    args = parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    load_env()
    config = get_config()
    cohort, samples = load_frozen_cohort(args.cohort)
    client = initialize_chroma_client(
        host=str(config.chroma.client.host),
        port=int(config.chroma.client.port),
    )
    source_collection_name = str(config.chroma.collections[0].name)
    source_collection = client.get_collection(source_collection_name)
    source_documents = read_collection_documents(
        source_collection,
        batch_size=int(config.chroma.settings.batch_size),
    )
    body_only = build_ablation_documents(source_documents, "body_only")
    contextual = build_ablation_documents(source_documents, "contextual")
    validate_aligned_ablation_documents(body_only, contextual)
    documents_by_variant = {"body_only": body_only, "contextual": contextual}

    requested_variants = (
        list(SUPPORTED_SEARCH_REPRESENTATIONS)
        if args.variant == "both"
        else [args.variant]
    )
    collection_metadata = OmegaConf.to_container(
        config.chroma.collections[0].metadata,
        resolve=True,
    )
    if not isinstance(collection_metadata, dict):
        raise ValueError("Configured Chroma collection metadata must be a mapping")
    batch_size = int(args.batch_size or config.chroma.settings.batch_size)
    variants = {
        representation: run_variant(
            representation=representation,
            client=client,
            collection_metadata=collection_metadata,
            documents=documents_by_variant[representation],
            samples=samples,
            config=config,
            batch_size=batch_size,
        )
        for representation in requested_variants
    }

    artifact = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_id": cohort["cohort_id"],
        "dataset_content_hash": cohort["dataset_content_hash"],
        "sample_ids": [sample.id for sample in samples],
        "source_collection": source_collection_name,
        "source_record_count": len(source_documents),
        "source_chunk_ids_sha256": compute_chunk_ids_sha256(
            document["id"] for document in source_documents
        ),
        "invariant": "Variants differ only in dense/BM25/reranker search text.",
        "configuration": build_configuration_snapshot(config, batch_size),
        "variants": variants,
        "comparison": build_comparison(variants),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Saved contextual-enrichment ablation to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
