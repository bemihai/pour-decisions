"""Same-commit prompt comparison for the frozen M10 planning cohort."""

from __future__ import annotations

import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.agents.prompt_registry import PromptRegistry, get_prompt_registry, sha256_text
from src.eval.planning_baseline import PlanningCohortManifest

_PROMPT_NAME = "intelligent_agent_system"
_PROMPT_PATH = "src/agents/prompts/intelligent_agent_system_prompt.md.j2"
_MANIFEST_PATH = "src/agents/prompts/versions.yml"
_WEB_TOOL_NAMES = frozenset({"search_web_for_wine", "search_wine_price", "search_wine_reviews"})


def _git_output(*args: str) -> str:
    """Read one pinned local Git object without shell interpolation."""
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.PIPE).strip()


def load_prior_prompt_registry(ref: str) -> tuple[PromptRegistry, str]:
    """Inject an earlier registered prompt while keeping current agent code and tools."""
    if not ref or ref.startswith("-"):
        raise ValueError("A non-option baseline Git ref is required")
    commit = _git_output("rev-parse", "--verify", f"{ref}^{{commit}}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("Baseline prompt ref must be an ancestor of the current commit")

    source = subprocess.check_output(
        ["git", "show", f"{commit}:{_PROMPT_PATH}"],
        text=True,
        stderr=subprocess.PIPE,
    )
    manifest_text = _git_output("show", f"{commit}:{_MANIFEST_PATH}")
    manifest = OmegaConf.create(manifest_text)
    entry = manifest.prompts.intelligent_agent_system
    if entry.file != Path(_PROMPT_PATH).name or entry.renderer != "jinja2":
        raise ValueError("Baseline ref does not contain the registered intelligent prompt")

    current = get_prompt_registry()
    current_record = current.get(_PROMPT_NAME)
    prior_hash = sha256_text(source)
    if prior_hash == current_record.source_hash:
        raise ValueError("Baseline and treatment prompt sources are identical")
    records = {name: current.get(name) for name in current.get_source_version_map()}
    records[_PROMPT_NAME] = current_record.model_copy(
        update={"source": source, "source_hash": prior_hash, "label": str(entry.label or "")}
    )
    return PromptRegistry(records), commit


def assert_paired_artifacts(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    manifest: PlanningCohortManifest,
) -> None:
    """Reject any paired-run mismatch beyond the intentionally changed prompt."""
    if baseline.get("phase") != 2 or treatment.get("phase") != 2:
        raise ValueError("Both paired artifacts must be Phase 2 captures")
    if baseline.get("arm") != "baseline" or treatment.get("arm") != "treatment":
        raise ValueError("Paired artifacts must identify baseline and treatment arms")

    common_keys = (
        "manifest_content_hash",
        "dataset",
        "cellar_state_fingerprint",
        "app_config_content_hash",
        "repetitions",
        "max_concurrency",
        "git",
    )
    mismatches = [key for key in common_keys if baseline.get(key) != treatment.get(key)]
    if mismatches:
        raise ValueError(f"Paired artifact identity mismatch: {', '.join(mismatches)}")
    if baseline.get("git", {}).get("is_dirty") is not False:
        raise ValueError("Paired evaluation requires a clean current commit")
    if baseline.get("repetitions") != manifest.repetitions:
        raise ValueError("Paired repetition count differs from frozen manifest")

    baseline_provenance = baseline.get("model_provenance", {})
    treatment_provenance = treatment.get("model_provenance", {})
    for key in ("mode", "models", "tools", "agent_policy"):
        if baseline_provenance.get(key) != treatment_provenance.get(key):
            raise ValueError(f"Paired model provenance mismatch: {key}")
    baseline_prompt = baseline_provenance.get("prompts", {}).get(_PROMPT_NAME, {})
    treatment_prompt = treatment_provenance.get("prompts", {}).get(_PROMPT_NAME, {})
    if not baseline_prompt.get("source_hash") or not treatment_prompt.get("source_hash"):
        raise ValueError("Both paired arms require registered prompt provenance")
    if baseline_prompt["source_hash"] == treatment_prompt["source_hash"]:
        raise ValueError("Paired arms must use distinct prompt source identities")

    expected_keys = {
        (sample.id, repetition)
        for sample in manifest.samples
        for repetition in range(1, manifest.repetitions + 1)
    }
    for name, artifact in (("baseline", baseline), ("treatment", treatment)):
        executions = artifact.get("executions", [])
        actual_keys = [(item.get("sample_id"), item.get("repetition")) for item in executions]
        if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
            raise ValueError(f"{name} arm does not contain the complete unique frozen cohort")


def _mandatory_recall(executions: list[dict[str, Any]]) -> dict[str, int | float]:
    """Count mandatory evidence categories, not exact tool-name matches."""
    target_runs = [item for item in executions if item["role"] == "target"]
    satisfied = sum(len(item["evidence"]["satisfied"]) for item in target_runs)
    required = sum(
        len(item["evidence"]["satisfied"]) + len(item["evidence"]["missing"])
        for item in target_runs
    )
    return {"satisfied": satisfied, "required": required, "recall": satisfied / required if required else 0.0}


def _majority_correct_ids(executions: list[dict[str, Any]], role: str) -> set[str]:
    counts = Counter(
        item["sample_id"]
        for item in executions
        if item["role"] == role and item["outcome"] == "correct"
    )
    return {sample_id for sample_id, count in counts.items() if count >= 2}


def _arm_metrics(executions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize bounded outcome, cost, latency, and coverage signals."""
    target_runs = [item for item in executions if item["role"] == "target"]
    usages = [item["token_usage"] for item in executions if item.get("token_usage") is not None]
    guardrail_codes = Counter(
        event["code"]
        for item in executions
        for event in item.get("guardrail_events", [])
        if isinstance(event, dict) and isinstance(event.get("code"), str)
    )
    return {
        "outcomes": dict(sorted(Counter(str(item["outcome"]) for item in executions).items())),
        "mandatory_evidence": _mandatory_recall(executions),
        "majority_correct_targets": sorted(_majority_correct_ids(executions, "target")),
        "majority_correct_controls": sorted(_majority_correct_ids(executions, "control")),
        "blank_public_answers": sum(not item.get("answer", "").strip() for item in executions),
        "model_attempts": sum(item["llm_call_count"] for item in executions),
        "mean_target_attempts": statistics.mean(item["llm_call_count"] for item in target_runs),
        "mean_latency_ms": round(statistics.mean(item["latency_ms"] for item in executions), 3),
        "token_coverage": len(usages),
        "input_tokens_when_available": sum(item["input_tokens"] for item in usages),
        "output_tokens_when_available": sum(item["output_tokens"] for item in usages),
        "web_tool_invocations": sum(
            call["tool_name"] in _WEB_TOOL_NAMES
            for item in executions
            for call in item.get("tool_calls", [])
        ),
        "execution_errors": sum(item["outcome"] in {"timeout", "execution_error"} for item in executions),
        "judge_coverage": sum(item.get("judge", {}).get("coverage", 0) for item in executions),
        "judge_errors": sum(len(item.get("judge", {}).get("errors", [])) for item in executions),
        "guardrail_event_counts": dict(sorted(guardrail_codes.items())),
    }


def assess_gate2(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    manifest: PlanningCohortManifest,
) -> dict[str, Any]:
    """Assess the reviewed Phase 2 thresholds after explicit answer adjudication."""
    assert_paired_artifacts(baseline, treatment, manifest)
    for name, artifact in (("baseline", baseline), ("treatment", treatment)):
        for item in artifact["executions"]:
            if item.get("outcome") is None:
                raise ValueError(f"{name} arm has an unadjudicated answer")
            if item["outcome"] == "correct" and item["evidence"]["missing"]:
                raise ValueError(f"{name} arm marks an evidence-incomplete answer correct")

    baseline_metrics = _arm_metrics(baseline["executions"])
    treatment_metrics = _arm_metrics(treatment["executions"])
    new_targets = sorted(
        set(treatment_metrics["majority_correct_targets"])
        - set(baseline_metrics["majority_correct_targets"])
    )
    lost_controls = sorted(
        set(baseline_metrics["majority_correct_controls"])
        - set(treatment_metrics["majority_correct_controls"])
    )
    attempt_delta = round(
        treatment_metrics["mean_target_attempts"] - baseline_metrics["mean_target_attempts"], 4
    )
    checks = {
        "no_blank_public_answers": treatment_metrics["blank_public_answers"] == 0,
        "mandatory_evidence_improved": (
            treatment_metrics["mandatory_evidence"]["recall"]
            > baseline_metrics["mandatory_evidence"]["recall"]
        ),
        "two_new_majority_correct_targets": len(new_targets) >= 2,
        "no_control_majority_regression": not lost_controls,
        "target_attempt_delta_within_limit": attempt_delta <= 0.25,
        "no_execution_errors": treatment_metrics["execution_errors"] == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "new_majority_correct_targets": new_targets,
        "lost_majority_correct_controls": lost_controls,
        "mean_target_attempt_delta": attempt_delta,
        "baseline": baseline_metrics,
        "treatment": treatment_metrics,
        "external_call_note": (
            "Application-level Ollama attempts and web-tool invocations are counted. "
            "Provider HTTP calls, cache hits, and billable tokens are not directly measured."
        ),
    }
