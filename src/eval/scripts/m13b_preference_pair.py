"""Run and evaluate the bounded M13B declared-preference pair.

The live run is split into explicit arm commands so the released baseline can
execute from a detached worktree while the candidate executes from the current
stack head. The final evaluator is deterministic and fails closed on malformed,
failed, duplicated, or missing executions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


BASELINE_COMMIT = "d53ac92a602eb236e7395d972a201fd09a07cac1"
TARGET_IDS = (
    "m13b_pref_001",
    "m13b_pref_002",
    "m13b_pref_003",
    "m13b_pref_004",
    "m13b_pref_006",
)
ANSWER_CONTROL_IDS = ("m13b_control_001", "m13b_control_002")
LIFECYCLE_IDS = ("m13b_lifecycle_001", "m13b_lifecycle_002", "m13b_lifecycle_003")
APPLICATION_IDS = TARGET_IDS + ANSWER_CONTROL_IDS
REPETITIONS = 2
APPLICATION_EXECUTIONS_PER_ARM = len(APPLICATION_IDS) * REPETITIONS
TOTAL_APPLICATION_EXECUTIONS = APPLICATION_EXECUTIONS_PER_ARM * 2
MAX_MODEL_ATTEMPTS_PER_EXECUTION = 5
MAX_PROVIDER_ATTEMPTS = TOTAL_APPLICATION_EXECUTIONS * MAX_MODEL_ATTEMPTS_PER_EXECUTION
DEFAULT_COHORT_PATH = Path("src/eval/m13b_gate0_cohort.json")
DEFAULT_OUTPUT_PATH = Path("eval-results/m13b_phase5_preference_pair.json")


@dataclass
class ProviderAttemptBudget:
    """Thread-safe provider-attempt ceiling shared by one evaluation arm."""

    maximum: int
    attempts: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def reserve(self) -> None:
        """Reserve one attempt before a provider call."""
        with self._lock:
            if self.attempts >= self.maximum:
                raise RuntimeError(f"provider attempt ceiling reached ({self.maximum})")
            self.attempts += 1

    def snapshot(self) -> dict[str, int]:
        """Return safe attempt counters."""
        with self._lock:
            return {
                "attempts": self.attempts,
                "maximum": self.maximum,
                "remaining": self.maximum - self.attempts,
            }


class ProviderUsageCallback(BaseCallbackHandler):
    """Count attempts, outcomes, and provider-reported token usage."""

    raise_error = True
    run_inline = True

    def __init__(self, budget: ProviderAttemptBudget) -> None:
        self.budget = budget
        self.completions = 0
        self.errors = 0
        self.token_reports = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = Lock()

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        """Reserve an attempt for a chat-model call."""
        self.budget.reserve()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Reserve an attempt for a non-chat model adapter."""
        self.budget.reserve()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record a completion and any reported tokens."""
        usage = extract_usage(response)
        with self._lock:
            self.completions += 1
            if usage is not None:
                self.token_reports += 1
                self.input_tokens += usage["input_tokens"]
                self.output_tokens += usage["output_tokens"]

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Record a failed provider attempt."""
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, int]:
        """Return cumulative usage counters."""
        with self._lock:
            return {
                "completions": self.completions,
                "errors": self.errors,
                "token_reports": self.token_reports,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }


def extract_usage(response: LLMResult) -> dict[str, int] | None:
    """Extract input and output tokens from supported LangChain result shapes."""
    input_tokens = 0
    output_tokens = 0
    found = False
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message is not None else None
            if not isinstance(usage, dict):
                continue
            input_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            output_tokens += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
            found = True
    if found:
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}

    raw_usage = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage")
    if not isinstance(raw_usage, dict):
        return None
    return {
        "input_tokens": int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)) or 0),
    }


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible evidence."""
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cohort(path: Path = DEFAULT_COHORT_PATH) -> dict[str, Any]:
    """Load the frozen approved M13B cohort without source-hash revalidation."""
    cohort = json.loads(path.read_text(encoding="utf-8"))
    samples = cohort.get("samples", [])
    ids = [sample.get("sample_id") for sample in samples]
    expected = set(APPLICATION_IDS + LIFECYCLE_IDS)
    if (
        cohort.get("schema_version") != 1
        or cohort.get("milestone") != "m13b"
        or cohort.get("status") != "frozen_and_approved"
        or cohort.get("approval", {}).get("status") != "approved"
        or set(ids) != expected
        or len(ids) != len(expected)
    ):
        raise ValueError("M13B Phase 5 requires the exact approved ten-ID cohort")
    return cohort


def validate_arm(arm: dict[str, Any], expected_name: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate one arm and index its exact execution keys."""
    if arm.get("schema_version") != 1 or arm.get("arm") != expected_name:
        raise ValueError(f"Malformed {expected_name} arm")
    if expected_name == "baseline" and arm.get("commit") != BASELINE_COMMIT:
        raise ValueError("Baseline arm commit differs from the frozen released commit")
    if not arm.get("commit") or not arm.get("model_id"):
        raise ValueError(f"{expected_name} arm lacks commit or model identity")
    for field_name in ("configuration_fingerprint", "prompt_fingerprint", "tool_snapshot"):
        if not arm.get(field_name):
            raise ValueError(f"{expected_name} arm lacks {field_name}")

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    executions = arm.get("executions", [])
    for execution in executions:
        key = (execution.get("sample_id"), execution.get("repetition"))
        if key in indexed:
            raise ValueError(f"Duplicate {expected_name} execution: {key}")
        indexed[key] = execution
    expected_keys = {(sample_id, repetition) for sample_id in APPLICATION_IDS for repetition in range(1, 3)}
    if set(indexed) != expected_keys or len(executions) != APPLICATION_EXECUTIONS_PER_ARM:
        raise ValueError(f"{expected_name} arm is incomplete")
    return indexed


def _successful(execution: dict[str, Any]) -> bool:
    return execution.get("status") == "passed" and not execution.get("error") and bool(execution.get("answer"))


def _input_tokens(executions: list[dict[str, Any]]) -> int | None:
    values = [execution.get("input_tokens") for execution in executions]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return None
    return sum(values)


def _number_at_most(metrics: dict[str, Any], name: str, maximum: float) -> bool:
    """Return whether one required numeric metric is present and bounded."""
    value = metrics.get(name)
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value <= maximum


def evaluate_phase5(
    cohort: dict[str, Any],
    baseline_arm: dict[str, Any],
    candidate_arm: dict[str, Any],
    local_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every approved promotion and rollback threshold."""
    baseline = validate_arm(baseline_arm, "baseline")
    candidate = validate_arm(candidate_arm, "candidate")
    thresholds = cohort["promotion_thresholds"]
    all_keys = sorted(baseline)
    successful_common = [key for key in all_keys if _successful(baseline[key]) and _successful(candidate[key])]

    target_visible: dict[str, bool] = {}
    for sample_id in TARGET_IDS:
        target_visible[sample_id] = all(
            bool(candidate[(sample_id, repetition)].get("visible_effect"))
            for repetition in range(1, REPETITIONS + 1)
        )
    visible_count = sum(target_visible.values())

    candidate_executions = list(candidate.values())
    hard_avoid_violations = sum(int(execution.get("hard_avoid_violations", 0)) for execution in candidate_executions)
    prohibited_facts = sum(int(execution.get("prohibited_or_fabricated_facts", 0)) for execution in candidate_executions)
    conflict_executions = [candidate[("m13b_pref_002", repetition)] for repetition in range(1, 3)]
    conflicts_disclosed = sum(bool(execution.get("conflict_disclosed")) for execution in conflict_executions)
    control_regressions = sum(
        bool(candidate[(sample_id, repetition)].get("control_regression"))
        or candidate[(sample_id, repetition)].get("control_signature")
        != baseline[(sample_id, repetition)].get("control_signature")
        for sample_id in ANSWER_CONTROL_IDS
        for repetition in range(1, 3)
    )

    preference_keys = [(sample_id, repetition) for sample_id in TARGET_IDS for repetition in range(1, 3)]
    baseline_input = _input_tokens([baseline[key] for key in preference_keys])
    candidate_input = _input_tokens([candidate[key] for key in preference_keys])
    input_token_increase = (
        (candidate_input - baseline_input) / baseline_input
        if baseline_input is not None and candidate_input is not None and baseline_input > 0
        else None
    )
    external_preference_calls = sum(
        len(execution.get("preference_external_calls", [])) for execution in candidate_executions
    )

    local_checks = local_evidence.get("checks", {})
    local_metrics = local_evidence.get("metrics", {})
    checks = {
        "all_28_application_executions_present": len(baseline) + len(candidate) == TOTAL_APPLICATION_EXECUTIONS,
        "all_application_executions_succeeded": len(successful_common) == APPLICATION_EXECUTIONS_PER_ARM,
        "common_sample_coverage_passes": local_metrics.get("common_sample_coverage_rate")
        == thresholds["common_sample_coverage_rate_min"],
        "deterministic_expected_outcome_passes": local_metrics.get("deterministic_expected_outcome_rate")
        == thresholds["deterministic_expected_outcome_rate_min"],
        "declared_fact_precision_passes": local_metrics.get("declared_fact_precision")
        == thresholds["declared_fact_precision_min"],
        "declared_fact_recall_passes": local_metrics.get("declared_fact_recall")
        == thresholds["declared_fact_recall_min"],
        "visible_target_effect_passes": visible_count >= thresholds["target_answer_visible_effect_count_min"],
        "hard_avoid_passes": hard_avoid_violations <= thresholds["hard_avoid_violations_max"],
        "prohibited_fact_passes": prohibited_facts <= thresholds["prohibited_or_fabricated_facts_max"],
        "conflict_disclosure_passes": conflicts_disclosed / len(conflict_executions)
        >= thresholds["conflict_disclosure_rate_min"],
        "lifecycle_passes": local_metrics.get("lifecycle_success_rate")
        == thresholds["lifecycle_success_rate_min"],
        "control_regression_passes": control_regressions <= thresholds["control_regressions_max"],
        "preference_read_latency_passes": _number_at_most(
            local_metrics,
            "preference_read_added_latency_p95_ms",
            thresholds["preference_read_added_latency_p95_ms_max"],
        ),
        "ordinary_preference_reads_pass": _number_at_most(
            local_metrics,
            "ordinary_request_added_preference_reads",
            thresholds["ordinary_request_added_preference_reads_max"],
        ),
        "ordinary_model_attempts_pass": _number_at_most(
            local_metrics,
            "ordinary_request_model_attempt_delta",
            thresholds["ordinary_request_model_attempt_delta_max"],
        ),
        "input_token_increase_passes": input_token_increase is not None
        and input_token_increase <= thresholds["preference_aware_input_token_increase_ratio_max"],
        "external_preference_calls_pass": external_preference_calls <= thresholds["external_preference_calls_max"],
        "local_gate_checks_pass": bool(local_checks) and all(local_checks.values()),
        "provider_attempts_within_cap": baseline_arm.get("provider_attempts", 0)
        + candidate_arm.get("provider_attempts", 0)
        <= MAX_PROVIDER_ATTEMPTS,
    }
    immediate_rollback = (
        hard_avoid_violations >= cohort["rollback_thresholds"]["hard_avoid_violations_trigger"]
        or prohibited_facts >= cohort["rollback_thresholds"]["prohibited_or_fabricated_facts_trigger"]
        or local_metrics.get("lifecycle_failures", 1)
        >= cohort["rollback_thresholds"]["lifecycle_failures_trigger"]
        or control_regressions >= cohort["rollback_thresholds"]["control_regressions_trigger"]
        or visible_count < cohort["rollback_thresholds"]["target_answer_visible_effect_count_below"]
        or external_preference_calls > 0
    )
    if immediate_rollback:
        decision = "rollback_consumption"
    elif all(checks.values()):
        decision = "eligible_for_rollout_review"
    else:
        decision = "repair_and_rerun"
    return {
        "schema_version": 1,
        "milestone": "m13b",
        "phase": 5,
        "decision": decision,
        "checks": checks,
        "metrics": {
            **local_metrics,
            "application_execution_count": len(baseline) + len(candidate),
            "successful_common_execution_count": len(successful_common),
            "visible_target_effect_count": visible_count,
            "visible_target_effects": target_visible,
            "hard_avoid_violations": hard_avoid_violations,
            "prohibited_or_fabricated_facts": prohibited_facts,
            "conflict_disclosure_rate": conflicts_disclosed / len(conflict_executions),
            "control_regressions": control_regressions,
            "baseline_preference_input_tokens": baseline_input,
            "candidate_preference_input_tokens": candidate_input,
            "preference_aware_input_token_increase_ratio": input_token_increase,
            "external_preference_calls": external_preference_calls,
            "provider_attempts": baseline_arm.get("provider_attempts", 0)
            + candidate_arm.get("provider_attempts", 0),
        },
        "failures": {
            "baseline": [key for key in all_keys if not _successful(baseline[key])],
            "candidate": [key for key in all_keys if not _successful(candidate[key])],
        },
        "provenance": {
            "baseline": {key: baseline_arm[key] for key in (
                "commit", "model_id", "configuration_fingerprint", "prompt_fingerprint", "tool_snapshot"
            )},
            "candidate": {key: candidate_arm[key] for key in (
                "commit", "model_id", "configuration_fingerprint", "prompt_fingerprint", "tool_snapshot"
            )},
        },
        "arms": {"baseline": baseline_arm, "candidate": candidate_arm},
        "local_evidence": local_evidence,
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Persist JSON evidence without overwriting a prior run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse deterministic evaluation mode arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--baseline-arm", type=Path)
    parser.add_argument("--candidate-arm", type=Path)
    parser.add_argument("--local-evidence", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    """Evaluate two already-captured arms plus deterministic local evidence."""
    args = parse_args()
    required = (args.baseline_arm, args.candidate_arm, args.local_evidence)
    if any(path is None for path in required):
        raise ValueError("--baseline-arm, --candidate-arm, and --local-evidence are required")
    cohort = load_cohort(args.cohort)
    baseline = json.loads(args.baseline_arm.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_arm.read_text(encoding="utf-8"))
    local = json.loads(args.local_evidence.read_text(encoding="utf-8"))
    report = evaluate_phase5(cohort, baseline, candidate, local)
    write_json_exclusive(args.output, report)
    print(json.dumps({"decision": report["decision"], "checks": report["checks"]}, indent=2, sort_keys=True))
    return 0 if report["decision"] == "eligible_for_rollout_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
