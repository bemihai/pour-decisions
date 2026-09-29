"""Tests for the M12 Phase 4 paired retrieval evaluator."""

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from src.eval.scripts.corrective_retrieval_gate import load_cohort
from src.eval.scripts.corrective_retrieval_pair import (
    arm_config,
    evaluate_pairs,
    retriever_cache_stats,
    validate_frozen_cohort,
)


def _arm(
    retrieved_ids: list[str],
    *,
    eligible: bool = False,
    attempt_count: int = 0,
    added_latency_ms: float = 0.0,
) -> dict:
    return {
        "status": "success",
        "retrieved_chunk_ids": retrieved_ids,
        "scores": {"mrr": 1.0, "precision_at_3": 1 / 3, "precision_at_5": 0.2},
        "feature_usage": {"web_fallback": False},
        "source_attribution": {"valid": True},
        "correction": {
            "eligible": eligible,
            "attempt_count": attempt_count,
            "added_latency_ms": added_latency_ms,
            "model_attempts": 0,
        },
    }


def _record(sample_id: str, role: str, *, selected: bool, recovered: bool) -> dict:
    evidence_id = f"{sample_id}-evidence"
    return {
        "sample_id": sample_id,
        "cohort_role": role,
        "required_evidence_chunk_ids": [evidence_id],
        "disabled": _arm([]),
        "enabled": _arm(
            [evidence_id] if recovered else [],
            eligible=selected,
            attempt_count=1 if selected else 0,
            added_latency_ms=250.0 if selected else 0.0,
        ),
    }


def test_phase4_rejects_any_cohort_other_than_frozen_gate0_ids() -> None:
    cohort = load_cohort(Path("src/eval/m12_gate0_cohort.json"))
    validate_frozen_cohort(cohort)

    cohort["controls"][0]["sample_id"] = "replacement"
    with pytest.raises(ValueError, match="exact six frozen"):
        validate_frozen_cohort(cohort)


def test_arm_config_isolatedly_sets_correction_and_disables_web() -> None:
    config = OmegaConf.create(
        {
            "chroma": {"retrieval": {"correction": {"enabled": False}}},
            "web_search": {"auto_fallback": True},
        }
    )

    enabled = arm_config(config, correction_enabled=True)

    assert enabled.chroma.retrieval.correction.enabled is True
    assert enabled.web_search.auto_fallback is False
    assert config.chroma.retrieval.correction.enabled is False
    assert config.web_search.auto_fallback is True


def test_retrieval_gate_passes_but_promotion_waits_for_cloud_evaluation() -> None:
    records = [
        _record("target-1", "target", selected=True, recovered=True),
        _record("target-2", "target", selected=True, recovered=True),
        _record("target-3", "target", selected=False, recovered=False),
        _record("control-1", "control", selected=False, recovered=False),
        _record("control-2", "control", selected=False, recovered=False),
        _record("control-3", "control", selected=False, recovered=False),
    ]

    result = evaluate_pairs(records)

    assert result["retrieval_decision"] == "pass"
    assert result["promotion_decision"] == "pending_cloud_evaluation"
    assert result["coverage"]["common_success_ids"] == 6
    assert result["generation_and_judging"]["faithfulness"] == "not_run"
    assert all(result["checks"].values())


def test_arm_failure_is_reported_as_coverage_and_keeps_correction_disabled() -> None:
    records = [
        _record("target-1", "target", selected=True, recovered=True),
        _record("target-2", "target", selected=True, recovered=True),
        _record("target-3", "target", selected=False, recovered=False),
        _record("control-1", "control", selected=False, recovered=False),
        _record("control-2", "control", selected=False, recovered=False),
        _record("control-3", "control", selected=False, recovered=False),
    ]
    records[-1]["enabled"] = {"status": "error", "error_type": "retrieval_error"}

    result = evaluate_pairs(records)

    assert result["retrieval_decision"] == "fail"
    assert result["promotion_decision"] == "keep_disabled"
    assert result["coverage"]["enabled_success_ids"] == 5
    assert result["coverage"]["common_success_ids"] == 5
    assert result["checks"]["no_new_execution_errors"] is False


def test_cache_stats_support_hybrid_retriever_wrapper() -> None:
    cache_owner = SimpleNamespace(
        get_cache_stats=lambda: {"size": 2, "max_size": 100, "hits": 3, "misses": 4, "hit_rate": 3 / 7}
    )

    stats = retriever_cache_stats(SimpleNamespace(vector_retriever=cache_owner))

    assert stats == {
        "available": True,
        "size": 2,
        "max_size": 100,
        "hits": 3,
        "misses": 4,
        "hit_rate": 3 / 7,
    }
