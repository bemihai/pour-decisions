"""Evaluate deterministic corrective retrieval against a frozen evidence cohort.

The command executes the shared production RAG path with generation disabled. It
records enough query-plan, candidate-channel, ranking, and corpus provenance to
reproduce the Gate 0 decision without making any LLM calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.chroma.bm25_builder import compute_chunk_ids_sha256, read_collection_ids
from src.eval.dataset import load_golden_dataset
from src.eval.metrics import precision_at_k, reciprocal_rank
from src.eval.models import GoldenSample
from src.retrieval import (
    build_correction_query,
    build_reranker_from_config,
    build_retriever_from_config,
    correction_trigger_reason,
    execute_production_rag,
)
from src.retrieval.rag_service import RAGChunkArtifact, RAGExecutionResult
from src.utils import get_config, initialize_chroma_client, logger
from src.utils.env import load_env


DEFAULT_COHORT_PATH = Path("src/eval/m12_gate0_cohort.json")
DEFAULT_DATASET_PATH = Path("src/eval/wine_qa_golden.jsonl")
DEFAULT_OUTPUT_PATH = Path("eval-results") / f"m12_gate0_corrective_retrieval_{datetime.now(UTC):%Y%m%d}.json"
RETRIEVAL_METRICS = ("mrr", "precision_at_3", "precision_at_5")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_cohort(path: Path) -> dict[str, Any]:
    """Load and validate the frozen three-target, three-control cohort."""
    cohort = json.loads(path.read_text(encoding="utf-8"))
    if len(cohort.get("targets", [])) != 3 or len(cohort.get("controls", [])) != 3:
        raise ValueError("Gate 0 requires exactly three targets and three controls")
    sample_ids = [entry["sample_id"] for group in ("targets", "controls") for entry in cohort[group]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Gate 0 cohort sample IDs must be unique")
    return cohort


def should_trigger(query_plan: dict[str, Any]) -> bool:
    """Select structured aging or classification questions with lost lexical detail."""
    from src.retrieval.query_analyzer import RetrievalQueryPlan

    return correction_trigger_reason(RetrievalQueryPlan.from_dict(query_plan)) is not None


def build_corrective_query(query_plan: dict[str, Any]) -> str:
    """Build a zero-LLM alternate query from the existing deterministic plan."""
    from src.retrieval.query_analyzer import RetrievalQueryPlan

    correction_query = build_correction_query(RetrievalQueryPlan.from_dict(query_plan))
    if correction_query is None:
        raise ValueError("No corrective query template for the supplied query plan")
    return correction_query.query


def score(retrieved_ids: list[str], ground_truth_ids: list[str]) -> dict[str, float]:
    """Score deterministic retrieval metrics for one result list."""
    return {
        "mrr": reciprocal_rank(retrieved_ids, ground_truth_ids),
        "precision_at_3": precision_at_k(retrieved_ids, ground_truth_ids, 3),
        "precision_at_5": precision_at_k(retrieved_ids, ground_truth_ids, 5),
    }


def chunk_diagnostic(chunk: RAGChunkArtifact) -> dict[str, Any]:
    """Serialize ranking and channel provenance without duplicating chunk text."""
    return {
        "id": chunk.id,
        "source": chunk.metadata.get("source") or chunk.metadata.get("filename"),
        "similarity": chunk.similarity,
        "rerank_score": chunk.rerank_score,
        "rrf_score": chunk.rrf_score,
        "dense_rank": chunk.dense_rank,
        "sparse_rank": chunk.sparse_rank,
        "dense_similarity": chunk.dense_similarity,
        "bm25_score": chunk.bm25_score,
        "metadata_matches": chunk.metadata_matches,
        "retrieval_channels": chunk.retrieval_channels,
    }


def result_diagnostic(result: RAGExecutionResult, sample: GoldenSample) -> dict[str, Any]:
    """Build a compact diagnostic snapshot for one production-path execution."""
    retrieved_ids = [chunk.id for chunk in result.context_chunks]
    pool_diagnostics = (
        result.raw_retrieved_chunks[0].retrieval_diagnostics if result.raw_retrieved_chunks else {}
    )
    return {
        "query_plan": result.retrieval_query_plan,
        "retrieval_confidence": result.retrieval_confidence,
        "low_confidence": result.low_confidence,
        "rerank_threshold": result.rerank_threshold,
        "feature_usage": result.feature_usage.to_dict(),
        "raw_candidate_count": len(result.raw_retrieved_chunks),
        "final_context_count": len(result.context_chunks),
        "candidate_pool_diagnostics": pool_diagnostics,
        "raw_candidates": [chunk_diagnostic(chunk) for chunk in result.raw_retrieved_chunks],
        "final_context": [chunk_diagnostic(chunk) for chunk in result.context_chunks],
        "retrieved_chunk_ids": retrieved_ids,
        "scores": score(retrieved_ids, sample.ground_truth_chunk_ids),
    }


def execute_sample(sample: GoldenSample, config: Any, retriever: Any, reranker: Any) -> RAGExecutionResult:
    """Execute retrieval and context assembly through the production RAG path."""
    result = execute_production_rag(
        prompt=sample.question,
        config=config,
        model=None,
        retriever=retriever,
        reranker=reranker,
        message_history=[],
        generation_enabled=False,
    )
    if result.retrieval_error:
        raise RuntimeError(result.retrieval_error)
    return result


def execute_query(query: str, config: Any, retriever: Any, reranker: Any) -> RAGExecutionResult:
    """Execute an alternate query through the same production RAG path."""
    result = execute_production_rag(
        prompt=query,
        config=config,
        model=None,
        retriever=retriever,
        reranker=reranker,
        message_history=[],
        generation_enabled=False,
    )
    if result.retrieval_error:
        raise RuntimeError(result.retrieval_error)
    return result


def evaluate_gate(
    cohort: dict[str, Any],
    samples: dict[str, GoldenSample],
    config: Any,
    retriever: Any,
    reranker: Any,
) -> dict[str, Any]:
    """Run the cohort and apply every Gate 0 pass criterion."""
    records: list[dict[str, Any]] = []
    target_ids = {entry["sample_id"] for entry in cohort["targets"]}
    cohort_entries = cohort["targets"] + cohort["controls"]
    for entry in cohort_entries:
        sample = samples[entry["sample_id"]]
        baseline = execute_sample(sample, config, retriever, reranker)
        triggered = should_trigger(baseline.retrieval_query_plan)
        corrective_query = build_corrective_query(baseline.retrieval_query_plan) if triggered else None
        candidate = execute_query(corrective_query, config, retriever, reranker) if corrective_query else baseline
        baseline_diag = result_diagnostic(baseline, sample)
        candidate_diag = result_diagnostic(candidate, sample)
        required = set(entry["required_evidence_chunk_ids"])
        baseline_evidence = required.intersection(baseline_diag["retrieved_chunk_ids"])
        candidate_evidence = required.intersection(candidate_diag["retrieved_chunk_ids"])
        records.append(
            {
                "sample_id": sample.id,
                "cohort_role": "target" if sample.id in target_ids else "control",
                "question": sample.question,
                "adjudication": entry["adjudication"],
                "required_evidence_chunk_ids": sorted(required),
                "triggered": triggered,
                "corrective_query": corrective_query,
                "baseline": baseline_diag,
                "candidate": candidate_diag,
                "new_required_evidence_recovered": sorted(candidate_evidence - baseline_evidence),
                "baseline_required_evidence_retained": baseline_evidence.issubset(candidate_evidence),
                "metric_deltas": {
                    metric: candidate_diag["scores"][metric] - baseline_diag["scores"][metric]
                    for metric in RETRIEVAL_METRICS
                },
            }
        )

    targets = [record for record in records if record["cohort_role"] == "target"]
    controls = [record for record in records if record["cohort_role"] == "control"]
    selected_targets = sum(record["triggered"] for record in targets)
    selected_controls = sum(record["triggered"] for record in controls)
    recovered_targets = sum(bool(record["new_required_evidence_recovered"]) for record in targets)
    checks = {
        "trigger_selects_at_least_two_targets": selected_targets >= 2,
        "trigger_selects_at_most_one_control": selected_controls <= 1,
        "strategy_recovers_at_least_two_targets": recovered_targets >= 2,
        "no_target_loses_required_evidence": all(record["baseline_required_evidence_retained"] for record in targets),
        "no_control_metric_regresses_over_two_points": all(
            delta >= -0.02 for record in controls for delta in record["metric_deltas"].values()
        ),
    }
    return {
        "decision": "pass" if all(checks.values()) else "defer",
        "checks": checks,
        "counts": {
            "targets": len(targets),
            "controls": len(controls),
            "selected_targets": selected_targets,
            "selected_controls": selected_controls,
            "recovered_targets": recovered_targets,
        },
        "per_sample": records,
    }


def main() -> int:
    """Run the Gate 0 experiment and write its reproducible evidence artifact."""
    args = parse_args()
    load_env()
    config = get_config()
    cohort = load_cohort(args.cohort)
    all_samples = {sample.id: sample for sample in load_golden_dataset(args.dataset)}
    required_sample_ids = {entry["sample_id"] for group in ("targets", "controls") for entry in cohort[group]}
    missing = sorted(required_sample_ids - all_samples.keys())
    if missing:
        raise ValueError(f"Cohort samples missing from golden dataset: {missing}")

    chroma_config = config.chroma
    collection_name = str(chroma_config.collections[0].name)
    client = initialize_chroma_client(str(chroma_config.client.host), int(chroma_config.client.port))
    collection = client.get_collection(collection_name)
    corpus_ids = read_collection_ids(collection)
    retriever = build_retriever_from_config(config, enable_cache=False, enable_query_expansion=False)
    reranker = build_reranker_from_config(config)
    gate = evaluate_gate(cohort, all_samples, config, retriever, reranker)

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dataset": {
            "path": str(args.dataset),
            "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        },
        "cohort": cohort,
        "corpus": {
            "collection_name": collection_name,
            "record_count": len(corpus_ids),
            "chunk_ids_sha256": compute_chunk_ids_sha256(corpus_ids),
        },
        "config_snapshot": OmegaConf.to_container(config.chroma.retrieval, resolve=True),
        "execution": {
            "generation_enabled": False,
            "llm_calls": 0,
            "production_rag_path": "src.retrieval.execute_production_rag",
        },
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Gate 0 decision=%s evidence=%s", gate["decision"], args.output)
    return 0 if gate["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
