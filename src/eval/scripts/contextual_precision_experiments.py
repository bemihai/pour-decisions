"""Run focused precision-recovery experiments for M3 contextual retrieval.

This command reuses the active contextual dense and synchronized BM25 indexes.
It changes only the final reranker representation and result count, comparing
body-only top-five and top-three variants with the recorded body-only control.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.chroma.bm25_builder import compute_chunk_ids_sha256, read_collection_documents
from src.eval.contextual_ablation import SearchRepresentationReranker
from src.eval.metrics import precision_at_k, reciprocal_rank
from src.eval.models import GoldenSample
from src.eval.scripts.contextual_enrichment_ablation import (
    load_frozen_cohort,
    score_context_metrics,
    write_artifact,
)
from src.eval.utils import run_rag_retrieval_only_sync
from src.retrieval import build_reranker_from_config, build_retriever_from_config
from src.utils import get_config, initialize_chroma_client, logger
from src.utils.env import load_env


DEFAULT_BASELINE_PATH = Path("eval-results/m3b_contextual_enrichment_20260813.json")
DEFAULT_OUTPUT_PATH = Path("eval-results") / f"m3b_precision_recovery_{datetime.now(UTC):%Y%m%d}.json"
VARIANT_TOP_K = {"body_rerank_top5": 5, "body_rerank_top3": 3}
RETRIEVAL_METRICS = ("mrr", "precision_at_3", "precision_at_5")
JUDGE_METRICS = ("context_precision", "context_recall")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--judge-context-metrics",
        action="store_true",
        help="Score context precision/recall for both focused variants.",
    )
    return parser.parse_args()


def config_with_top_k(config: DictConfig, top_k: int) -> DictConfig:
    """Copy config and override only the evaluation result count."""
    copied = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    copied.chroma.retrieval.rerank_top_k = top_k
    return copied


def evaluate_variant(
    *,
    name: str,
    top_k: int,
    config: DictConfig,
    retriever: Any,
    reranker: SearchRepresentationReranker,
    samples: list[GoldenSample],
    cohort_sample_ids: set[str],
) -> dict[str, Any]:
    """Evaluate one body-reranked result-count variant."""
    variant_config = config_with_top_k(config, top_k)
    started = time.perf_counter()
    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        sample_started = time.perf_counter()
        result = run_rag_retrieval_only_sync(sample, variant_config, retriever, reranker)
        retrieved_ids = [chunk.id for chunk in result.context_chunks]
        metrics_supported = bool(sample.ground_truth_chunk_ids)
        scores = (
            {
                "mrr": reciprocal_rank(retrieved_ids, sample.ground_truth_chunk_ids),
                "precision_at_3": precision_at_k(retrieved_ids, sample.ground_truth_chunk_ids, 3),
                "precision_at_5": precision_at_k(retrieved_ids, sample.ground_truth_chunk_ids, 5),
            }
            if metrics_supported
            else {}
        )
        per_sample.append(
            {
                "sample_id": sample.id,
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "is_cohort_sample": sample.id in cohort_sample_ids,
                "retrieval_metrics_supported": metrics_supported,
                "retrieved_chunk_ids": retrieved_ids,
                "rerank_scores": [chunk.rerank_score for chunk in result.context_chunks],
                "contexts": [chunk.text for chunk in result.context_chunks],
                "scores": scores,
                "retrieval_confidence": result.retrieval_confidence,
                "low_confidence": result.low_confidence,
                "latency_ms": round((time.perf_counter() - sample_started) * 1000, 3),
            }
        )

    scorable = [sample for sample in per_sample if sample["retrieval_metrics_supported"]]
    cohort_scorable = [sample for sample in scorable if sample["is_cohort_sample"]]

    def means(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            metric: sum(float(sample["scores"][metric]) for sample in selected) / len(selected)
            for metric in RETRIEVAL_METRICS
        }

    return {
        "name": name,
        "candidate_representation": "contextual",
        "reranker_representation": "body_only",
        "rerank_top_k": top_k,
        "metric_coverage": {
            "global_scorable": len(scorable),
            "global_unsupported": len(per_sample) - len(scorable),
            "cohort_scorable": len(cohort_scorable),
        },
        "aggregate_metrics": means(scorable),
        "cohort_aggregate_metrics": means(cohort_scorable),
        "mean_retrieval_latency_ms": (
            sum(float(sample["latency_ms"]) for sample in per_sample) / len(per_sample)
        ),
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
        "per_sample": per_sample,
    }


def common_judge_comparison(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare judge metrics only over samples scored for both variants."""
    comparison: dict[str, Any] = {}
    for metric in JUDGE_METRICS:
        control_scores = {
            sample["sample_id"]: sample["scores"][metric]
            for sample in control["per_sample"]
            if metric in sample["scores"]
        }
        candidate_scores = {
            sample["sample_id"]: sample["scores"][metric]
            for sample in candidate["per_sample"]
            if metric in sample["scores"]
        }
        common_ids = sorted(set(control_scores) & set(candidate_scores))
        control_mean = (
            sum(control_scores[sample_id] for sample_id in common_ids) / len(common_ids)
            if common_ids
            else None
        )
        candidate_mean = (
            sum(candidate_scores[sample_id] for sample_id in common_ids) / len(common_ids)
            if common_ids
            else None
        )
        comparison[metric] = {
            "common_sample_ids": common_ids,
            "common_sample_count": len(common_ids),
            "body_only_control_mean": control_mean,
            "candidate_mean": candidate_mean,
            "candidate_minus_control": (
                candidate_mean - control_mean
                if control_mean is not None and candidate_mean is not None
                else None
            ),
        }
    return comparison


def compare_with_control(
    control_variant: dict[str, Any],
    control_judge: dict[str, Any],
    candidate_variant: dict[str, Any],
    candidate_judge: dict[str, Any],
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Apply the Phase 2 gate to one focused candidate."""
    retrieval_deltas = {
        metric: candidate_variant["aggregate_metrics"][metric] - control_variant["aggregate_metrics"][metric]
        for metric in RETRIEVAL_METRICS
    }
    judge_comparison = common_judge_comparison(control_judge, candidate_judge)
    checks = {
        "context_precision_improves": (
            judge_comparison["context_precision"]["candidate_minus_control"] > 0.0
        ),
        "mrr_within_tolerance": retrieval_deltas["mrr"] >= -tolerance,
        "precision_at_3_within_tolerance": retrieval_deltas["precision_at_3"] >= -tolerance,
        "precision_at_5_within_tolerance": retrieval_deltas["precision_at_5"] >= -tolerance,
        "context_recall_within_tolerance": (
            judge_comparison["context_recall"]["candidate_minus_control"] >= -tolerance
        ),
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "reviewed_tolerance": tolerance,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "global_retrieval_deltas": retrieval_deltas,
        "judge_comparison": judge_comparison,
        "mean_retrieval_latency_delta_ms": (
            candidate_variant["mean_retrieval_latency_ms"]
            - control_variant["mean_retrieval_latency_ms"]
        ),
    }


def validate_baseline(
    baseline: dict[str, Any],
    cohort: dict[str, Any],
    source_documents: list[dict[str, Any]],
) -> None:
    """Reject stale baseline, dataset, or corpus comparisons."""
    if baseline["dataset_content_hash"] != cohort["dataset_content_hash"]:
        raise ValueError("Baseline and frozen cohort dataset hashes differ")
    if baseline["source_record_count"] != len(source_documents):
        raise ValueError("Baseline and active source collection counts differ")
    current_ids_hash = compute_chunk_ids_sha256(document["id"] for document in source_documents)
    if baseline["source_chunk_ids_sha256"] != current_ids_hash:
        raise ValueError("Baseline and active source collection ID hashes differ")


def main() -> int:
    """Run both focused variants and save a decision artifact."""
    args = parse_args()
    load_env()
    config = get_config()
    cohort, cohort_samples, global_samples = load_frozen_cohort(
        Path("src/eval/m3b_contextual_enrichment_cohort.json")
    )
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    client = initialize_chroma_client(
        host=str(config.chroma.client.host),
        port=int(config.chroma.client.port),
    )
    collection_name = str(config.chroma.collections[0].name)
    source_collection = client.get_collection(collection_name)
    source_documents = read_collection_documents(
        source_collection,
        batch_size=int(config.chroma.settings.batch_size),
    )
    validate_baseline(baseline, cohort, source_documents)

    retriever = build_retriever_from_config(
        config,
        collection_name=collection_name,
        enable_cache=False,
        enable_query_expansion=False,
    )
    base_reranker = build_reranker_from_config(config)
    if base_reranker is None:
        raise RuntimeError("The focused experiment requires the configured reranker")
    reranker = SearchRepresentationReranker(base_reranker, "body_only")
    cohort_sample_ids = {sample.id for sample in cohort_samples}

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_path": str(args.baseline),
        "dataset_content_hash": cohort["dataset_content_hash"],
        "source_collection": collection_name,
        "source_record_count": len(source_documents),
        "source_chunk_ids_sha256": baseline["source_chunk_ids_sha256"],
        "invariant": "Active contextual dense/BM25 candidates; only body rerank result count differs.",
        "variants": {},
        "context_judge": None,
        "comparisons_to_body_only_control": None,
    }
    variants: dict[str, dict[str, Any]] = artifact["variants"]
    for name, top_k in VARIANT_TOP_K.items():
        variants[name] = evaluate_variant(
            name=name,
            top_k=top_k,
            config=config,
            retriever=retriever,
            reranker=reranker,
            samples=global_samples,
            cohort_sample_ids=cohort_sample_ids,
        )
        write_artifact(args.output, artifact)

    if args.judge_context_metrics:
        judged_variants = {
            "body_only_control": baseline["variants"]["body_only"],
            "contextual_current": baseline["variants"]["contextual"],
            **variants,
        }
        judge_result = score_context_metrics(judged_variants, cohort_samples)
        artifact["context_judge"] = judge_result["variants"]
        control_variant = baseline["variants"]["body_only"]
        control_judge = judge_result["variants"]["body_only_control"]
        compared_variants = {
            "contextual_current": baseline["variants"]["contextual"],
            **variants,
        }
        artifact["comparisons_to_body_only_control"] = {
            name: compare_with_control(
                control_variant,
                control_judge,
                variant,
                judge_result["variants"][name],
            )
            for name, variant in compared_variants.items()
        }
        passed_candidates = [
            name
            for name, comparison in artifact["comparisons_to_body_only_control"].items()
            if comparison["decision"] == "pass"
        ]
        artifact["recommendation"] = {
            "production_choice": (
                passed_candidates[0] if passed_candidates else "body_only_control"
            ),
            "passed_candidates": passed_candidates,
            "discarded_candidates": [
                name
                for name in compared_variants
                if name not in passed_candidates
            ],
            "reason": (
                "No contextual candidate passed every reviewed precision/recall gate; "
                "retain or restore the simplest body-only control."
                if not passed_candidates
                else "At least one candidate passed every reviewed precision/recall gate."
            ),
            "production_change_applied": False,
        }
        write_artifact(args.output, artifact)

    logger.info("Saved focused precision-recovery experiment to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
