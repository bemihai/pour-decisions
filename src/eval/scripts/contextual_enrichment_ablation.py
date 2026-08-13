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
from src.eval.models import GoldenSample, SampleResult
from src.eval.utils import run_rag_retrieval_only_sync
from src.retrieval import ChromaRetriever, HybridRetriever, build_reranker_from_config
from src.retrieval.keyword_search import BM25Index
from src.utils import compute_file_hash, get_config, get_embedder, initialize_chroma_client, logger
from src.utils.env import load_env


DEFAULT_COHORT_PATH = Path("src/eval/m3b_contextual_enrichment_cohort.json")
DEFAULT_OUTPUT_PATH = Path("eval-results") / f"m3b_contextual_enrichment_{datetime.now(UTC):%Y%m%d}.json"


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
    parser.add_argument(
        "--judge-context-metrics",
        action="store_true",
        help="Score context precision/recall on the frozen cohort with fixed reference answers.",
    )
    return parser.parse_args()


def load_frozen_cohort(path: Path) -> tuple[dict[str, Any], list[GoldenSample], list[GoldenSample]]:
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
    cohort_samples = [samples_by_id[sample_id] for sample_id in sample_ids]
    global_samples = [
        sample
        for sample in samples_by_id.values()
        if sample.category == "rag_only"
    ]
    return cohort, cohort_samples, global_samples


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
    samples: list[GoldenSample],
    cohort_sample_ids: set[str],
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
            retrieval_metrics_supported = bool(sample.ground_truth_chunk_ids)
            scores = (
                {
                    "mrr": reciprocal_rank(retrieved_ids, sample.ground_truth_chunk_ids),
                    "precision_at_3": precision_at_k(retrieved_ids, sample.ground_truth_chunk_ids, 3),
                    "precision_at_5": precision_at_k(retrieved_ids, sample.ground_truth_chunk_ids, 5),
                }
                if retrieval_metrics_supported
                else {}
            )
            per_sample.append(
                {
                    "sample_id": sample.id,
                    "question": sample.question,
                    "ground_truth": sample.ground_truth,
                    "is_cohort_sample": sample.id in cohort_sample_ids,
                    "retrieval_metrics_supported": retrieval_metrics_supported,
                    "retrieved_chunk_ids": retrieved_ids,
                    "contexts": [chunk.text for chunk in result.context_chunks],
                    "scores": scores,
                    "retrieval_confidence": result.retrieval_confidence,
                    "low_confidence": result.low_confidence,
                    "latency_ms": round((time.perf_counter() - sample_started) * 1000, 3),
                }
            )

        scorable = [sample for sample in per_sample if sample["retrieval_metrics_supported"]]
        cohort_scorable = [sample for sample in scorable if sample["is_cohort_sample"]]

        def mean_metrics(selected: list[dict[str, Any]]) -> dict[str, float]:
            """Average deterministic metrics over one explicit supported slice."""
            return {
                metric: sum(float(sample["scores"][metric]) for sample in selected) / len(selected)
                for metric in ("mrr", "precision_at_3", "precision_at_5")
            }

        return {
            "representation": representation,
            "record_count": len(documents),
            "metric_coverage": {
                "global_scorable": len(scorable),
                "global_unsupported": len(per_sample) - len(scorable),
                "cohort_scorable": len(cohort_scorable),
            },
            "aggregate_metrics": mean_metrics(scorable),
            "cohort_aggregate_metrics": mean_metrics(cohort_scorable),
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
        "global_metric_deltas": {
            metric: contextual["aggregate_metrics"][metric] - body["aggregate_metrics"][metric]
            for metric in ("mrr", "precision_at_3", "precision_at_5")
        },
        "cohort_metric_deltas": {
            metric: (
                contextual["cohort_aggregate_metrics"][metric]
                - body["cohort_aggregate_metrics"][metric]
            )
            for metric in ("mrr", "precision_at_3", "precision_at_5")
        },
        "mean_retrieval_latency_delta_ms": (
            contextual["mean_retrieval_latency_ms"] - body["mean_retrieval_latency_ms"]
        ),
    }


def score_context_metrics(
    variants: dict[str, dict[str, Any]],
    cohort_samples: list[GoldenSample],
) -> dict[str, Any]:
    """Score only context precision/recall with fixed reference answers."""
    from src.eval.ragas_scorer import RagasScorer

    cohort_by_id = {sample.id: sample for sample in cohort_samples}
    scores_by_variant: dict[str, Any] = {}
    for representation, variant in variants.items():
        results = [
            SampleResult(
                id=str(sample_result["sample_id"]),
                question=str(sample_result["question"]),
                answer=cohort_by_id[str(sample_result["sample_id"])].ground_truth,
                ground_truth=cohort_by_id[str(sample_result["sample_id"])].ground_truth,
                expected_facts=cohort_by_id[str(sample_result["sample_id"])].expected_facts,
                contexts=list(sample_result["contexts"]),
                retrieved_chunk_ids=list(sample_result["retrieved_chunk_ids"]),
                status="passed",
            )
            for sample_result in variant["per_sample"]
            if sample_result["is_cohort_sample"]
        ]
        scorer = RagasScorer()
        scorer.metric_names = ["context_precision", "context_recall"]
        scorer.score(results)

        metric_payload: dict[str, Any] = {}
        for metric_name in scorer.metric_names:
            values = [result.scores[metric_name] for result in results if metric_name in result.scores]
            metric_payload[metric_name] = {
                "mean": sum(values) / len(values) if values else None,
                "scored": len(values),
                "errored": sum(metric_name in result.metric_errors for result in results),
            }
        scores_by_variant[representation] = {
            "metrics": metric_payload,
            "per_sample": [
                {
                    "sample_id": result.id,
                    "scores": {
                        metric: value
                        for metric, value in result.scores.items()
                        if metric in scorer.metric_names
                    },
                    "metric_errors": result.metric_errors,
                }
                for result in results
            ],
        }

    comparison = None
    if set(scores_by_variant) == set(SUPPORTED_SEARCH_REPRESENTATIONS):
        comparison = {}
        for metric_name in ("context_precision", "context_recall"):
            scores_by_sample = {
                representation: {
                    sample["sample_id"]: sample["scores"][metric_name]
                    for sample in scores_by_variant[representation]["per_sample"]
                    if metric_name in sample["scores"]
                }
                for representation in SUPPORTED_SEARCH_REPRESENTATIONS
            }
            common_sample_ids = sorted(
                set(scores_by_sample["body_only"]) & set(scores_by_sample["contextual"])
            )
            body_mean = (
                sum(scores_by_sample["body_only"][sample_id] for sample_id in common_sample_ids)
                / len(common_sample_ids)
                if common_sample_ids
                else None
            )
            contextual_mean = (
                sum(scores_by_sample["contextual"][sample_id] for sample_id in common_sample_ids)
                / len(common_sample_ids)
                if common_sample_ids
                else None
            )
            comparison[metric_name] = {
                "common_sample_ids": common_sample_ids,
                "common_sample_count": len(common_sample_ids),
                "body_only_mean": body_mean,
                "contextual_mean": contextual_mean,
                "contextual_minus_body": (
                    contextual_mean - body_mean
                    if body_mean is not None and contextual_mean is not None
                    else None
                ),
            }
    return {"variants": scores_by_variant, "common_sample_comparison": comparison}


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Atomically preserve the latest completed experiment checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    logger.info("Saved contextual-enrichment ablation checkpoint to %s", path)


def build_acceptance_decision(artifact: dict[str, Any], tolerance: float = 0.02) -> dict[str, Any]:
    """Apply the reviewed Phase 2 keep/revise gate to completed metrics."""
    comparison = artifact.get("comparison")
    context_judge = artifact.get("context_judge")
    if comparison is None or context_judge is None:
        return {
            "decision": "not_evaluated",
            "reason": "Both retrieval variants and context judge metrics are required.",
        }

    global_deltas = comparison["global_metric_deltas"]
    judge_deltas = context_judge["common_sample_comparison"]
    checks = {
        "cohort_context_precision_improves": (
            judge_deltas["context_precision"]["contextual_minus_body"] > 0.0
        ),
        "global_mrr_within_tolerance": global_deltas["mrr"] >= -tolerance,
        "global_precision_at_3_within_tolerance": (
            global_deltas["precision_at_3"] >= -tolerance
        ),
        "global_precision_at_5_within_tolerance": (
            global_deltas["precision_at_5"] >= -tolerance
        ),
        "common_sample_context_recall_within_tolerance": (
            judge_deltas["context_recall"]["contextual_minus_body"] >= -tolerance
        ),
    }
    keep = all(checks.values())
    return {
        "decision": "keep" if keep else "revise",
        "reviewed_tolerance": tolerance,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
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
    cohort, cohort_samples, global_samples = load_frozen_cohort(args.cohort)
    cohort_sample_ids = {sample.id for sample in cohort_samples}
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
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_id": cohort["cohort_id"],
        "dataset_content_hash": cohort["dataset_content_hash"],
        "cohort_sample_ids": [sample.id for sample in cohort_samples],
        "global_sample_ids": [sample.id for sample in global_samples],
        "source_collection": source_collection_name,
        "source_record_count": len(source_documents),
        "source_chunk_ids_sha256": compute_chunk_ids_sha256(
            document["id"] for document in source_documents
        ),
        "invariant": "Variants differ only in dense/BM25/reranker search text.",
        "configuration": build_configuration_snapshot(config, batch_size),
        "variants": {},
        "comparison": None,
        "context_judge": None,
        "acceptance_decision": {"decision": "not_evaluated"},
    }
    variants: dict[str, dict[str, Any]] = artifact["variants"]
    for representation in requested_variants:
        variants[representation] = run_variant(
            representation=representation,
            client=client,
            collection_metadata=collection_metadata,
            documents=documents_by_variant[representation],
            samples=global_samples,
            cohort_sample_ids=cohort_sample_ids,
            config=config,
            batch_size=batch_size,
        )
        artifact["comparison"] = build_comparison(variants)
        write_artifact(args.output, artifact)

    if args.judge_context_metrics:
        artifact["context_judge"] = score_context_metrics(variants, cohort_samples)
        artifact["acceptance_decision"] = build_acceptance_decision(artifact)
        write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
