"""Run the frozen M12 correction-disabled/enabled retrieval evaluation.

This command exercises the shared production RAG path with generation and web
fallback disabled. It evaluates the local retrieval gates only; answer quality,
token usage, and cloud judges remain unavailable until separately authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.chroma.bm25_builder import compute_chunk_ids_sha256, read_collection_ids
from src.eval.dataset import load_golden_dataset
from src.eval.models import GoldenSample
from src.eval.scripts.corrective_retrieval_gate import (
    DEFAULT_COHORT_PATH,
    DEFAULT_DATASET_PATH,
    RETRIEVAL_METRICS,
    load_cohort,
    result_diagnostic,
)
from src.retrieval import build_reranker_from_config, build_retriever_from_config, execute_production_rag
from src.retrieval.rag_service import RAGExecutionResult
from src.utils import get_config, initialize_chroma_client, logger
from src.utils.env import load_env


DEFAULT_OUTPUT_PATH = Path("eval-results") / f"m12_phase4_corrective_pair_{datetime.now(UTC):%Y%m%d}.json"
FROZEN_SAMPLE_IDS = frozenset(
    {
        "rag_only_002",
        "rag_only_003",
        "rag_only_006",
        "rag_only_012",
        "rag_only_022",
        "rag_only_023",
    }
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def validate_frozen_cohort(cohort: dict[str, Any]) -> None:
    """Reject any cohort that differs from the six reviewed Gate 0 IDs."""
    sample_ids = {
        entry["sample_id"] for group in ("targets", "controls") for entry in cohort[group]
    }
    if sample_ids != FROZEN_SAMPLE_IDS:
        raise ValueError("Phase 4 requires the exact six frozen Gate 0 sample IDs")


def arm_config(config: DictConfig, *, correction_enabled: bool) -> DictConfig:
    """Copy config and set only the experiment's local safety switches."""
    copied = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    copied.chroma.retrieval.correction.enabled = correction_enabled
    copied.web_search.auto_fallback = False
    return copied


def retriever_cache_stats(retriever: Any) -> dict[str, Any]:
    """Return bounded cache counters for vector or hybrid retrieval."""
    cache_owner = getattr(retriever, "vector_retriever", retriever)
    getter = getattr(cache_owner, "get_cache_stats", None)
    if not callable(getter):
        return {"available": False}
    stats = getter()
    return {
        "available": True,
        "size": int(stats.get("size", 0)),
        "max_size": int(stats.get("max_size", 0)),
        "hits": int(stats.get("hits", 0)),
        "misses": int(stats.get("misses", 0)),
        "hit_rate": float(stats.get("hit_rate", 0.0)),
    }


def source_attribution_diagnostic(result: RAGExecutionResult) -> dict[str, Any]:
    """Describe whether every final chunk has a matching bounded source artifact."""
    context_ids = [chunk.id for chunk in result.context_chunks]
    source_ids = [source.chunk_id for source in result.sources]
    missing_source_metadata = [
        chunk.id
        for chunk in result.context_chunks
        if not (chunk.metadata.get("source") or chunk.metadata.get("filename"))
    ]
    missing_source_artifacts = sorted(set(context_ids) - set(source_ids))
    return {
        "valid": not missing_source_metadata and not missing_source_artifacts,
        "source_count": len(result.sources),
        "missing_source_metadata": missing_source_metadata,
        "missing_source_artifacts": missing_source_artifacts,
    }


def execute_arm(
    sample: GoldenSample,
    config: DictConfig,
    retriever: Any,
    reranker: Any,
) -> dict[str, Any]:
    """Execute one retrieval-only arm and retain failures as coverage data."""
    started = time.perf_counter()
    try:
        result = execute_production_rag(
            prompt=sample.question,
            config=config,
            model=None,
            retriever=retriever,
            reranker=reranker,
            message_history=[],
            generation_enabled=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if result.retrieval_error:
            return {
                "status": "error",
                "latency_ms": elapsed_ms,
                "error_type": "retrieval_error",
                "error": str(result.retrieval_error)[:500],
            }
        diagnostic = result_diagnostic(result, sample)
        diagnostic.update(
            {
                "status": "success",
                "latency_ms": elapsed_ms,
                "correction": result.correction.to_dict(),
                "source_attribution": source_attribution_diagnostic(result),
                "blank_answer": False,
            }
        )
        return diagnostic
    except Exception as exc:
        return {
            "status": "error",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def run_pairs(
    cohort: dict[str, Any],
    samples: dict[str, GoldenSample],
    disabled_config: DictConfig,
    enabled_config: DictConfig,
    disabled_retriever: Any,
    enabled_retriever: Any,
    reranker: Any,
) -> list[dict[str, Any]]:
    """Run both arms for each sample in reviewed cohort order."""
    target_ids = {entry["sample_id"] for entry in cohort["targets"]}
    records: list[dict[str, Any]] = []
    for entry in cohort["targets"] + cohort["controls"]:
        sample = samples[entry["sample_id"]]
        records.append(
            {
                "sample_id": sample.id,
                "cohort_role": "target" if sample.id in target_ids else "control",
                "question": sample.question,
                "required_evidence_chunk_ids": sorted(entry["required_evidence_chunk_ids"]),
                "disabled": execute_arm(sample, disabled_config, disabled_retriever, reranker),
                "enabled": execute_arm(sample, enabled_config, enabled_retriever, reranker),
            }
        )
    return records


def _successful(record: dict[str, Any], arm: str) -> bool:
    return record[arm].get("status") == "success"


def _required_evidence(record: dict[str, Any], arm: str) -> set[str]:
    if not _successful(record, arm):
        return set()
    return set(record["required_evidence_chunk_ids"]).intersection(record[arm]["retrieved_chunk_ids"])


def _p95(values: list[float]) -> float | None:
    """Return the nearest-rank p95 used by the frozen small cohort."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def evaluate_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply every retrieval-only Phase 4 hard gate without hiding failures."""
    targets = [record for record in records if record["cohort_role"] == "target"]
    controls = [record for record in records if record["cohort_role"] == "control"]
    common = [record for record in records if _successful(record, "disabled") and _successful(record, "enabled")]
    common_ids = {record["sample_id"] for record in common}

    for record in records:
        disabled_evidence = _required_evidence(record, "disabled")
        enabled_evidence = _required_evidence(record, "enabled")
        record["disabled_required_evidence"] = sorted(disabled_evidence)
        record["enabled_required_evidence"] = sorted(enabled_evidence)
        record["new_required_evidence_recovered"] = sorted(enabled_evidence - disabled_evidence)
        record["disabled_required_evidence_retained"] = disabled_evidence.issubset(enabled_evidence)
        record["metric_deltas"] = (
            {
                metric: record["enabled"]["scores"][metric] - record["disabled"]["scores"][metric]
                for metric in RETRIEVAL_METRICS
            }
            if record in common
            else None
        )

    selected_targets = sum(
        _successful(record, "enabled") and bool(record["enabled"]["correction"]["eligible"])
        for record in targets
    )
    selected_controls = sum(
        _successful(record, "enabled") and bool(record["enabled"]["correction"]["eligible"])
        for record in controls
    )
    recovered_targets = sum(bool(record["new_required_evidence_recovered"]) for record in targets)
    eligible_latencies = [
        float(record["enabled"]["correction"]["added_latency_ms"])
        for record in records
        if _successful(record, "enabled") and record["enabled"]["correction"]["eligible"]
    ]
    enabled_successes = [record for record in records if _successful(record, "enabled")]
    disabled_successes = [record for record in records if _successful(record, "disabled")]
    web_calls = sum(
        int(record[arm]["feature_usage"].get("web_fallback", False))
        for record in records
        for arm in ("disabled", "enabled")
        if _successful(record, arm)
    )
    model_attempts = sum(record["enabled"]["correction"]["model_attempts"] for record in enabled_successes)

    checks = {
        "trigger_selects_at_least_two_targets": selected_targets >= 2,
        "trigger_selects_at_most_one_control": selected_controls <= 1,
        "at_least_two_targets_recover_required_evidence": recovered_targets >= 2,
        "no_target_loses_required_evidence": all(
            record["sample_id"] in common_ids and record["disabled_required_evidence_retained"] for record in targets
        ),
        "no_control_metric_regresses_over_two_points": all(
            record["sample_id"] in common_ids
            and all(delta >= -0.02 for delta in record["metric_deltas"].values())
            for record in controls
        ),
        "no_new_execution_errors": len(enabled_successes) == len(records),
        "no_source_attribution_regressions": all(
            not record["disabled"]["source_attribution"]["valid"]
            or record["enabled"]["source_attribution"]["valid"]
            for record in common
        )
        and len(common) == len(records),
        "ineligible_queries_have_zero_attempts": all(
            record["enabled"]["correction"]["attempt_count"] == 0
            for record in enabled_successes
            if not record["enabled"]["correction"]["eligible"]
        ),
        "eligible_queries_have_at_most_one_attempt": all(
            record["enabled"]["correction"]["attempt_count"] <= 1
            for record in enabled_successes
            if record["enabled"]["correction"]["eligible"]
        ),
        "zero_correction_model_attempts": model_attempts == 0,
        "zero_web_calls": web_calls == 0,
        "eligible_p95_added_latency_at_most_2000_ms": bool(eligible_latencies)
        and float(_p95(eligible_latencies) or 0.0) <= 2000.0,
        "no_correction_exceeds_3000_ms": all(latency <= 3000.0 for latency in eligible_latencies),
    }
    retrieval_passed = all(checks.values())
    return {
        "retrieval_decision": "pass" if retrieval_passed else "fail",
        "promotion_decision": "pending_cloud_evaluation" if retrieval_passed else "keep_disabled",
        "checks": checks,
        "coverage": {
            "expected_ids": len(records),
            "disabled_success_ids": len(disabled_successes),
            "enabled_success_ids": len(enabled_successes),
            "common_success_ids": len(common),
            "common_sample_ids": [record["sample_id"] for record in common],
        },
        "counts": {
            "targets": len(targets),
            "controls": len(controls),
            "selected_targets": selected_targets,
            "selected_controls": selected_controls,
            "recovered_targets": recovered_targets,
        },
        "latency": {
            "eligible_added_latency_ms": eligible_latencies,
            "eligible_p95_added_latency_ms": _p95(eligible_latencies),
            "eligible_max_added_latency_ms": max(eligible_latencies, default=None),
        },
        "external_calls": {"correction_model_attempts": model_attempts, "web_calls": web_calls},
        "generation_and_judging": {
            "status": "not_run",
            "reason": "requires separate explicit cloud authorization",
            "answer_coverage": "not_run",
            "blank_answers": "not_run",
            "faithfulness": "not_run",
            "token_usage": "not_run",
            "judge_failures": "not_run",
        },
        "per_sample": records,
    }


def main() -> int:
    """Execute the paired local gate and write its reproducible artifact."""
    args = parse_args()
    load_env()
    config = get_config()
    cohort = load_cohort(args.cohort)
    validate_frozen_cohort(cohort)
    samples = {sample.id: sample for sample in load_golden_dataset(args.dataset)}
    missing = sorted(FROZEN_SAMPLE_IDS - samples.keys())
    if missing:
        raise ValueError(f"Frozen cohort samples missing from golden dataset: {missing}")

    disabled_config = arm_config(config, correction_enabled=False)
    enabled_config = arm_config(config, correction_enabled=True)
    disabled_retriever = build_retriever_from_config(disabled_config, enable_cache=True, enable_query_expansion=False)
    enabled_retriever = build_retriever_from_config(enabled_config, enable_cache=True, enable_query_expansion=False)
    reranker = build_reranker_from_config(config)

    chroma_config = config.chroma
    collection_name = str(chroma_config.collections[0].name)
    client = initialize_chroma_client(str(chroma_config.client.host), int(chroma_config.client.port))
    corpus_ids = read_collection_ids(client.get_collection(collection_name))
    records = run_pairs(
        cohort,
        samples,
        disabled_config,
        enabled_config,
        disabled_retriever,
        enabled_retriever,
        reranker,
    )
    gate = evaluate_pairs(records)
    gate["cache_behavior"] = {
        "disabled_arm": retriever_cache_stats(disabled_retriever),
        "enabled_arm": retriever_cache_stats(enabled_retriever),
        "arm_isolation": "separate retriever caches",
    }

    retrieval_snapshot = OmegaConf.to_container(config.chroma.retrieval, resolve=True)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "evaluation_scope": "retrieval_only",
        "dataset": {"path": str(args.dataset), "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest()},
        "cohort": {"path": str(args.cohort), "sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest()},
        "corpus": {
            "collection_name": collection_name,
            "record_count": len(corpus_ids),
            "chunk_ids_sha256": compute_chunk_ids_sha256(corpus_ids),
        },
        "config_snapshot": {
            "retrieval": retrieval_snapshot,
            "disabled_arm_correction_enabled": False,
            "enabled_arm_correction_enabled": True,
            "web_auto_fallback": False,
        },
        "execution": {
            "generation_enabled": False,
            "production_rag_path": "src.retrieval.execute_production_rag",
            "sample_ids_per_arm": len(FROZEN_SAMPLE_IDS),
            "retrieval_executions": len(FROZEN_SAMPLE_IDS) * 2,
        },
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "M12 Phase 4 retrieval decision=%s promotion=%s evidence=%s",
        gate["retrieval_decision"],
        gate["promotion_decision"],
        args.output,
    )
    return 0 if gate["retrieval_decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
