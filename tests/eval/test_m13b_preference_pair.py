"""Tests for the fail-closed M13B Phase 5 pair evaluator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.eval.scripts.m13b_preference_pair import (
    ANSWER_CONTROL_IDS,
    APPLICATION_IDS,
    BASELINE_COMMIT,
    MAX_PROVIDER_ATTEMPTS,
    TARGET_IDS,
    ProviderAttemptBudget,
    evaluate_phase5,
    load_cohort,
    validate_arm,
    write_json_exclusive,
)


def _arm(name: str) -> dict:
    executions = []
    for sample_id in APPLICATION_IDS:
        for repetition in range(1, 3):
            executions.append(
                {
                    "sample_id": sample_id,
                    "repetition": repetition,
                    "status": "passed",
                    "error": None,
                    "answer": "bounded answer",
                    "tool_calls": ["get_user_taste_profile"],
                    "tool_outputs": [],
                    "latency_ms": 10.0,
                    "model_attempts": 1,
                    "input_tokens": 100 if name == "baseline" else 110,
                    "output_tokens": 20,
                    "external_calls": [],
                    "preference_external_calls": [],
                    "visible_effect": sample_id in TARGET_IDS,
                    "hard_avoid_violations": 0,
                    "prohibited_or_fabricated_facts": 0,
                    "conflict_disclosed": sample_id == "m13b_pref_002",
                    "control_regression": False,
                    "control_signature": {"observed": sample_id},
                }
            )
    return {
        "schema_version": 1,
        "arm": name,
        "commit": BASELINE_COMMIT if name == "baseline" else "candidate-commit",
        "model_id": "gemma4:31b",
        "configuration_fingerprint": "config-hash",
        "prompt_fingerprint": "prompt-hash",
        "tool_snapshot": {"contract_hash": "tool-hash"},
        "provider_attempts": 14,
        "executions": executions,
    }


def _local_evidence() -> dict:
    return {
        "checks": {
            "all_ten_frozen_ids_executed": True,
            "zero_prohibited_facts": True,
            "ordinary_controls_are_clean": True,
        },
        "metrics": {
            "common_sample_coverage_rate": 1.0,
            "deterministic_expected_outcome_rate": 1.0,
            "declared_fact_precision": 1.0,
            "declared_fact_recall": 1.0,
            "lifecycle_success_rate": 1.0,
            "lifecycle_failures": 0,
            "preference_read_added_latency_p95_ms": 2.0,
            "ordinary_request_added_preference_reads": 0,
            "ordinary_request_model_attempt_delta": 0,
        },
    }


def test_approved_cohort_has_exact_phase5_ids() -> None:
    cohort = load_cohort()
    assert {sample["sample_id"] for sample in cohort["samples"]} == set(APPLICATION_IDS) | {
        "m13b_lifecycle_001",
        "m13b_lifecycle_002",
        "m13b_lifecycle_003",
    }


def test_complete_pair_is_eligible_for_rollout_review() -> None:
    report = evaluate_phase5(load_cohort(), _arm("baseline"), _arm("candidate"), _local_evidence())
    assert report["decision"] == "eligible_for_rollout_review"
    assert all(report["checks"].values())
    assert report["metrics"]["application_execution_count"] == 28
    assert report["metrics"]["visible_target_effect_count"] == 5
    assert report["metrics"]["preference_aware_input_token_increase_ratio"] == pytest.approx(0.1)


def test_missing_or_duplicate_execution_fails_closed() -> None:
    missing = _arm("candidate")
    missing["executions"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        validate_arm(missing, "candidate")

    duplicate = _arm("candidate")
    duplicate["executions"].append(deepcopy(duplicate["executions"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_arm(duplicate, "candidate")


def test_wrong_baseline_commit_is_rejected() -> None:
    arm = _arm("baseline")
    arm["commit"] = "wrong"
    with pytest.raises(ValueError, match="frozen released commit"):
        validate_arm(arm, "baseline")


def test_failed_execution_and_missing_tokens_fail_promotion() -> None:
    candidate = _arm("candidate")
    candidate["executions"][0]["status"] = "failed"
    candidate["executions"][0]["error"] = "synthetic failure"
    candidate["executions"][1]["input_tokens"] = None
    report = evaluate_phase5(load_cohort(), _arm("baseline"), candidate, _local_evidence())
    assert report["decision"] == "repair_and_rerun"
    assert not report["checks"]["all_application_executions_succeeded"]
    assert not report["checks"]["input_token_increase_passes"]


def test_missing_required_local_metric_fails_closed() -> None:
    local = _local_evidence()
    del local["metrics"]["ordinary_request_added_preference_reads"]
    report = evaluate_phase5(load_cohort(), _arm("baseline"), _arm("candidate"), local)
    assert report["decision"] == "repair_and_rerun"
    assert not report["checks"]["ordinary_preference_reads_pass"]


def test_control_signature_difference_triggers_rollback() -> None:
    candidate = _arm("candidate")
    control = next(
        execution for execution in candidate["executions"] if execution["sample_id"] == ANSWER_CONTROL_IDS[0]
    )
    control["control_signature"] = {"observed": "changed"}
    report = evaluate_phase5(load_cohort(), _arm("baseline"), candidate, _local_evidence())
    assert report["decision"] == "rollback_consumption"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hard_avoid_violations", 1),
        ("prohibited_or_fabricated_facts", 1),
        ("control_regression", True),
    ],
)
def test_immediate_rollback_triggers(field: str, value: object) -> None:
    candidate = _arm("candidate")
    if field == "control_regression":
        index = next(
            index
            for index, execution in enumerate(candidate["executions"])
            if execution["sample_id"] in ANSWER_CONTROL_IDS
        )
    else:
        index = 0
    candidate["executions"][index][field] = value
    report = evaluate_phase5(load_cohort(), _arm("baseline"), candidate, _local_evidence())
    assert report["decision"] == "rollback_consumption"


def test_fewer_than_four_visible_targets_triggers_rollback() -> None:
    candidate = _arm("candidate")
    for execution in candidate["executions"]:
        if execution["sample_id"] in TARGET_IDS[:2]:
            execution["visible_effect"] = False
    report = evaluate_phase5(load_cohort(), _arm("baseline"), candidate, _local_evidence())
    assert report["decision"] == "rollback_consumption"


def test_provider_budget_refuses_attempt_above_exact_cap() -> None:
    budget = ProviderAttemptBudget(maximum=MAX_PROVIDER_ATTEMPTS)
    for _ in range(MAX_PROVIDER_ATTEMPTS):
        budget.reserve()
    with pytest.raises(RuntimeError, match="ceiling reached"):
        budget.reserve()


def test_evidence_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    write_json_exclusive(output, {"decision": "repair_and_rerun"})
    with pytest.raises(FileExistsError):
        write_json_exclusive(output, {"decision": "eligible_for_rollout_review"})
