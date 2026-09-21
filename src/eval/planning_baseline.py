"""Frozen-cohort measurement helpers for M10 planning decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig
from pydantic import BaseModel, Field, model_validator

from src.agents.guardrails import (
    CALL_BUDGET_EVENT_CODE,
    EMPTY_FINAL_ANSWER_EVENT_CODE,
    LOOP_DETECTED_EVENT_CODE,
    RELEVANCE_DEFLECTED_EVENT_CODE,
)
from src.agents.intelligent.agent import WineAgent
from src.eval.dataset import load_golden_dataset
from src.eval.models import AgentToolCall, GoldenSample
from src.eval.utils import (
    get_git_metadata,
    load_execution_model,
    resolve_execution_model_config,
    run_agent_sample_sync,
)
from src.utils import compute_file_hash, get_config, get_default_db_path, get_project_root

DEFAULT_MANIFEST_PATH = Path("src/eval/m10_planning_cohort.json")
DEFAULT_DATASET_PATH = Path("src/eval/wine_qa_golden.jsonl")
RELEVANT_CELLAR_TABLES = (
    "producers",
    "regions",
    "wines",
    "bottles",
    "tastings",
    "food_pairing_rules",
)

_DEFAULT_TOOLS_BY_EVIDENCE = {
    "cellar": frozenset({"get_cellar_wines", "get_wine_details", "get_cellar_statistics"}),
    "knowledge": frozenset(
        {
            "search_wine_knowledge",
            "search_wine_region_info",
            "search_grape_variety_info",
            "search_wine_term_definition",
            "search_wine_producer_info",
        }
    ),
    "pairing": frozenset(
        {"get_food_pairing_wines", "get_pairing_for_wine", "get_wine_and_cheese_pairings"}
    ),
}
_SAFE_TERMINAL_CODES = frozenset(
    {CALL_BUDGET_EVENT_CODE, LOOP_DETECTED_EVENT_CODE, RELEVANCE_DEFLECTED_EVENT_CODE}
)


class PlanningCohortSample(BaseModel):
    """One frozen sample and its argument-level evidence policy."""

    id: str
    role: Literal["target", "control"]
    mandatory_evidence: list[Literal["cellar", "knowledge", "pairing"]]
    accepted_tools_by_evidence: dict[str, list[str]] = Field(default_factory=dict)
    allowed_additional_tools: list[str] = Field(default_factory=list)
    adjudication_note: str | None = None


class PlanningCohortManifest(BaseModel):
    """Versioned contract for the bounded M10 Phase 0 cohort."""

    schema_version: Literal[1]
    milestone: Literal["m10"]
    phase: Literal[0]
    dataset_path: str
    dataset_content_hash: str
    repetitions: int = Field(ge=1)
    max_concurrency: Literal[1]
    samples: list[PlanningCohortSample]

    @model_validator(mode="after")
    def validate_frozen_shape(self) -> "PlanningCohortManifest":
        """Require 13 targets, three controls, and unique identifiers."""
        ids = [sample.id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Planning cohort sample IDs must be unique")
        roles = Counter(sample.role for sample in self.samples)
        if roles != {"target": 13, "control": 3}:
            raise ValueError("Planning cohort must contain 13 targets and 3 controls")
        return self


class EvidenceAssessment(BaseModel):
    """Deterministic argument-level assessment of one observed trajectory."""

    satisfied: list[str]
    missing: list[str]
    repeated_tool_names: dict[str, int]
    exact_duplicate_calls: list[AgentToolCall]
    distinct_repeated_calls: list[str]
    unexpected_tools: list[str]


def load_planning_cohort(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dataset_path: Path | None = None,
) -> tuple[PlanningCohortManifest, list[GoldenSample]]:
    """Load and cross-check the frozen manifest against the golden dataset."""
    manifest = PlanningCohortManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    resolved_dataset = dataset_path or Path(manifest.dataset_path)
    actual_hash = compute_file_hash(resolved_dataset)
    if actual_hash != manifest.dataset_content_hash:
        raise ValueError(
            "Frozen planning cohort dataset mismatch: "
            f"expected {manifest.dataset_content_hash}, got {actual_hash}"
        )

    dataset_by_id = {sample.id: sample for sample in load_golden_dataset(resolved_dataset)}
    missing_ids = [sample.id for sample in manifest.samples if sample.id not in dataset_by_id]
    if missing_ids:
        raise ValueError(f"Planning cohort IDs missing from golden dataset: {', '.join(missing_ids)}")
    return manifest, [dataset_by_id[sample.id] for sample in manifest.samples]


def assess_planning_evidence(
    policy: PlanningCohortSample,
    calls: list[AgentToolCall],
) -> EvidenceAssessment:
    """Assess required evidence and exact duplicates using canonical arguments."""
    names = [call.tool_name for call in calls]
    satisfied: list[str] = []
    missing: list[str] = []
    accepted_tools: set[str] = set(policy.allowed_additional_tools)
    for evidence in policy.mandatory_evidence:
        candidates = set(
            policy.accepted_tools_by_evidence.get(evidence, _DEFAULT_TOOLS_BY_EVIDENCE[evidence])
        )
        accepted_tools.update(candidates)
        (satisfied if candidates.intersection(names) else missing).append(evidence)

    name_counts = Counter(names)
    repeated_tool_names = {name: count for name, count in name_counts.items() if count > 1}
    fingerprints: Counter[tuple[str, str]] = Counter(
        (call.tool_name, call.canonical_arguments) for call in calls
    )
    exact_duplicate_calls = [
        call
        for call in calls
        if fingerprints[(call.tool_name, call.canonical_arguments)] > 1
    ]
    distinct_repeated_calls = sorted(
        name
        for name in repeated_tool_names
        if len({call.canonical_arguments for call in calls if call.tool_name == name}) > 1
    )
    return EvidenceAssessment(
        satisfied=satisfied,
        missing=missing,
        repeated_tool_names=repeated_tool_names,
        exact_duplicate_calls=exact_duplicate_calls,
        distinct_repeated_calls=distinct_repeated_calls,
        unexpected_tools=sorted(set(names) - accepted_tools),
    )


def compute_cellar_state_fingerprint(db_path: Path | None = None) -> str:
    """Hash stable semantic rows from all cellar tables used by the cohort."""
    resolved_path = db_path or get_default_db_path()
    digest = hashlib.sha256()
    with sqlite3.connect(resolved_path) as connection:
        for table in RELEVANT_CELLAR_TABLES:
            columns = [
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            if not columns:
                raise ValueError(f"Required cellar table is missing: {table}")
            quoted_columns = ", ".join(f'"{column}"' for column in columns)
            rows = connection.execute(
                f'SELECT {quoted_columns} FROM "{table}" ORDER BY {quoted_columns}'
            ).fetchall()
            payload = {"table": table, "columns": columns, "rows": rows}
            digest.update(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    return f"sha256:{digest.hexdigest()}"


def estimate_run_cost(manifest: PlanningCohortManifest, max_calls_per_sample: int) -> dict[str, Any]:
    """Return an explicit logical-call bound without inventing token billing."""
    executions = len(manifest.samples) * manifest.repetitions
    return {
        "execution_count": executions,
        "minimum_logical_model_calls": executions,
        "maximum_logical_model_calls": executions * max_calls_per_sample,
        "judge_enabled": False,
        "judge_logical_model_calls": 0,
        "provider_cost_usd": None,
        "provider_cost_note": (
            "No judge calls are enabled. Dollar cost requires provider token usage and current pricing; "
            "the artifact records actual logical calls instead of fabricating a billing estimate."
        ),
    }


def build_planning_baseline_agent(config: DictConfig | None = None) -> WineAgent:
    """Build the Phase 0 agent with the explicitly configured Ollama eval model."""
    resolved_config = config or get_config()
    provider, model_name, _ = resolve_execution_model_config(resolved_config)
    if provider.casefold() != "ollama":
        raise ValueError(
            "M10 Phase 0 evaluation requires eval.execution_provider=ollama; "
            f"configured provider is {provider or '<unset>'}"
        )
    if not model_name:
        raise ValueError("M10 Phase 0 evaluation requires eval.execution_model")
    return WineAgent(llm=load_execution_model(resolved_config), verbose=False)


def _terminal_outcome(
    answer: str,
    guardrail_events: list[dict[str, Any]],
    terminal_outcome: str | None = None,
) -> Literal["safe_terminal", "blank"] | None:
    if terminal_outcome == EMPTY_FINAL_ANSWER_EVENT_CODE:
        return "blank"
    event_codes = {str(event.get("code", "")) for event in guardrail_events}
    if event_codes.intersection(_SAFE_TERMINAL_CODES):
        return "safe_terminal"
    if not answer.strip():
        return "blank"
    return None


def capture_planning_baseline(
    output_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dataset_path: Path | None = None,
    repetitions: int | None = None,
    agent: WineAgent | None = None,
) -> dict[str, Any]:
    """Run the frozen cohort sequentially and write a private local artifact."""
    manifest, samples = load_planning_cohort(manifest_path, dataset_path)
    repetition_count = repetitions or manifest.repetitions
    if repetition_count < 1:
        raise ValueError("repetitions must be at least one")

    active_agent = agent or build_planning_baseline_agent()
    git_metadata = get_git_metadata()
    project_root = get_project_root()
    resolved_dataset = dataset_path or Path(manifest.dataset_path)
    policies = {sample.id: sample for sample in manifest.samples}
    executions: list[dict[str, Any]] = []

    for repetition in range(1, repetition_count + 1):
        for sample in samples:
            started = time.perf_counter()
            try:
                result = run_agent_sample_sync(active_agent, sample)
                latency_ms = (time.perf_counter() - started) * 1000
                evidence = assess_planning_evidence(
                    policies[sample.id],
                    result.tool_call_records,
                )
                executions.append(
                    {
                        "sample_id": sample.id,
                        "role": policies[sample.id].role,
                        "repetition": repetition,
                        "question": sample.question,
                        "answer": result.answer,
                        "outcome": _terminal_outcome(result.answer, result.guardrail_events, result.terminal_outcome),
                        "error": None,
                        "latency_ms": round(latency_ms, 3),
                        "llm_call_count": result.llm_call_count,
                        "tool_calls": [call.model_dump(mode="json") for call in result.tool_call_records],
                        "evidence": evidence.model_dump(mode="json"),
                        "guardrail_events": result.guardrail_events,
                        "judge": {"enabled": False, "coverage": 0, "errors": []},
                    }
                )
            except TimeoutError as exc:
                executions.append(
                    _failed_execution(sample, policies[sample.id], repetition, started, "timeout", exc)
                )
            except Exception as exc:
                executions.append(
                    _failed_execution(
                        sample,
                        policies[sample.id],
                        repetition,
                        started,
                        "execution_error",
                        exc,
                    )
                )

    artifact = {
        "schema_version": 1,
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(UTC).isoformat(),
        "milestone": "m10",
        "phase": 0,
        "manifest_content_hash": compute_file_hash(manifest_path),
        "dataset": {
            "path": str(resolved_dataset),
            "content_hash": compute_file_hash(resolved_dataset),
        },
        "cellar_state_fingerprint": compute_cellar_state_fingerprint(),
        "git": git_metadata,
        "app_config_content_hash": compute_file_hash(project_root / "app_config.yml"),
        "model_provenance": active_agent.execution_provenance.to_eval_dict(),
        "repetitions": repetition_count,
        "max_concurrency": manifest.max_concurrency,
        "cost_estimate": estimate_run_cost(
            manifest,
            active_agent.call_budget.max_llm_calls_per_query,
        ),
        "executions": executions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def _failed_execution(
    sample: GoldenSample,
    policy: PlanningCohortSample,
    repetition: int,
    started: float,
    outcome: Literal["timeout", "execution_error"],
    error: Exception,
) -> dict[str, Any]:
    """Build one terminal failure record without losing cohort identity."""
    return {
        "sample_id": sample.id,
        "role": policy.role,
        "repetition": repetition,
        "question": sample.question,
        "answer": "",
        "outcome": outcome,
        "error": str(error),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "llm_call_count": 0,
        "tool_calls": [],
        "evidence": assess_planning_evidence(policy, []).model_dump(mode="json"),
        "guardrail_events": [],
        "judge": {"enabled": False, "coverage": 0, "errors": []},
    }


def assert_comparable_artifacts(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Reject comparisons whose dataset, cellar, config, or model identity differs."""
    comparable_paths = (
        ("dataset", "content_hash"),
        ("cellar_state_fingerprint",),
        ("app_config_content_hash",),
        ("model_provenance",),
        ("max_concurrency",),
    )
    mismatches: list[str] = []
    for path in comparable_paths:
        left_value: Any = left
        right_value: Any = right
        for key in path:
            left_value = left_value.get(key) if isinstance(left_value, dict) else None
            right_value = right_value.get(key) if isinstance(right_value, dict) else None
        if left_value != right_value:
            mismatches.append(".".join(path))
    if mismatches:
        raise ValueError(f"Planning artifacts are not comparable: {', '.join(mismatches)}")


def apply_adjudications(artifact: dict[str, Any], adjudications: dict[str, str]) -> dict[str, Any]:
    """Apply explicit correct/incorrect decisions to non-terminal executions."""
    allowed = {"correct", "incorrect"}
    for execution in artifact.get("executions", []):
        if execution.get("outcome") is not None:
            continue
        key = f"{execution['sample_id']}#{execution['repetition']}"
        outcome = adjudications.get(key)
        if outcome not in allowed:
            raise ValueError(f"Missing correct/incorrect adjudication for {key}")
        execution["outcome"] = outcome
    return artifact


def validate_gate0_artifact(
    artifact: dict[str, Any],
    manifest: PlanningCohortManifest,
) -> dict[str, Any]:
    """Validate Gate 0 completeness and return a compact decision summary."""
    executions = artifact.get("executions", [])
    expected_keys = {
        (sample.id, repetition)
        for sample in manifest.samples
        for repetition in range(1, manifest.repetitions + 1)
    }
    actual_keys = {
        (execution.get("sample_id"), execution.get("repetition")) for execution in executions
    }
    if actual_keys != expected_keys:
        raise ValueError("Gate 0 artifact does not contain exactly three runs of all 16 samples")

    incomplete = [
        f"{execution.get('sample_id')}#{execution.get('repetition')}"
        for execution in executions
        if execution.get("outcome") is None or not isinstance(execution.get("llm_call_count"), int)
    ]
    if incomplete:
        raise ValueError(f"Gate 0 executions are incomplete: {', '.join(incomplete)}")

    outcome_counts = Counter(str(execution["outcome"]) for execution in executions)
    stable_failures = sorted(
        sample.id
        for sample in manifest.samples
        if {
            execution["outcome"]
            for execution in executions
            if execution["sample_id"] == sample.id
        }
        == {"incorrect"}
    )
    stochastic_failures = sorted(
        sample.id
        for sample in manifest.samples
        if len(
            {
                execution["outcome"]
                for execution in executions
                if execution["sample_id"] == sample.id
            }
        )
        > 1
    )
    return {
        "gate": "Gate 0",
        "passed": True,
        "execution_count": len(executions),
        "outcomes": dict(sorted(outcome_counts.items())),
        "stable_incorrect_samples": stable_failures,
        "stochastic_outcome_samples": stochastic_failures,
        "actual_llm_call_count": sum(execution["llm_call_count"] for execution in executions),
        "judge_coverage": 0,
        "judge_errors": 0,
    }
