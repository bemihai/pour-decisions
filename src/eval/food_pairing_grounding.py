"""Bounded prompt comparison for food-pairing source boundaries."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.intelligent.agent import WineAgent
from src.agents.prompt_registry import PromptRegistry, get_prompt_registry, sha256_text
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolCategory, ToolDefinition, ToolRegistry
from src.eval.agent_metrics import classify_tool_output
from src.eval.models import GoldenSample
from src.eval.planning_baseline import compute_cellar_state_fingerprint
from src.eval.utils import (
    get_git_metadata,
    load_execution_model,
    resolve_execution_model_config,
    run_agent_sample_sync,
)
from src.utils import compute_file_hash, get_config, get_project_root

DEFAULT_MANIFEST_PATH = Path("src/eval/food_pairing_grounding_cohort.json")
_PROMPT_NAME = "intelligent_agent_system"
_PAIRING_TOOLS = frozenset({"get_food_pairing_wines", "get_pairing_for_wine", "get_wine_and_cheese_pairings"})


class PromptCandidate(BaseModel):
    """One isolated prompt candidate appended only inside the evaluation registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    prompt_append: str


class PairingScenario(BaseModel):
    """One fixed food-pairing scenario and its observable evidence contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    availability: Literal["standard", "pairing_unavailable"]
    required_evidence: list[Literal["cellar", "pairing"]]
    accepted_tools: list[str]
    expected_facts: list[str]
    reuses_golden_sample: str | None = None


class PairingManifest(BaseModel):
    """Frozen eight-scenario comparison contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    study: Literal["food_pairing_grounding"]
    repetitions: Literal[2]
    max_concurrency: Literal[1]
    candidate: PromptCandidate
    scenarios: list[PairingScenario]

    @model_validator(mode="after")
    def validate_fixed_shape(self) -> "PairingManifest":
        """Require exactly the approved eight unique scenarios."""
        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) != 8 or len(set(ids)) != 8:
            raise ValueError("Food-pairing cohort must contain exactly eight unique scenarios")
        if Counter(scenario.availability for scenario in self.scenarios)["pairing_unavailable"] != 1:
            raise ValueError("Food-pairing cohort must contain one pairing-unavailable scenario")
        return self


class ManualDecision(BaseModel):
    """Human audit of one answer against the claim-level grounding rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm: Literal["baseline", "candidate"]
    scenario_id: str
    repetition: int = Field(ge=1)
    source_attribution_supported: bool
    cellar_facts_traceable: bool
    required_pairing_tool_used: bool
    general_knowledge_disclosed: bool
    no_fabricated_specifics: bool
    no_internal_tool_disclosure: bool
    expected_facts_met: list[str]
    note: str


class DecisionFile(BaseModel):
    """Complete manual audit file for both comparison arms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    decisions: list[ManualDecision]


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> PairingManifest:
    """Load the fixed food-pairing cohort."""
    return PairingManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_candidate_registry(candidate: PromptCandidate) -> PromptRegistry:
    """Build an in-memory prompt variant without editing the production prompt."""
    current = get_prompt_registry()
    records = {name: current.get(name) for name in current.get_source_version_map()}
    source = records[_PROMPT_NAME].source.rstrip() + "\n" + candidate.prompt_append.strip() + "\n"
    records[_PROMPT_NAME] = records[_PROMPT_NAME].model_copy(
        update={"source": source, "source_hash": sha256_text(source), "label": candidate.id}
    )
    return PromptRegistry(records)


def definitions_for_availability(availability: str) -> tuple[ToolDefinition, ...]:
    """Return the local-only tool catalogue for one study availability profile."""
    excluded_categories = {ToolCategory.RAG, ToolCategory.WEB_SEARCH}
    if availability == "pairing_unavailable":
        excluded_categories.add(ToolCategory.PAIRING)
    return tuple(
        definition for definition in TOOL_DEFINITIONS if definition.metadata.category not in excluded_categories
    )


def _build_agent(prompt_registry: PromptRegistry, availability: str) -> WineAgent:
    """Build one eval agent with a fixed prompt and availability profile."""
    config = get_config()
    provider, model_name, _ = resolve_execution_model_config(config)
    if provider.casefold() != "ollama" or not model_name:
        raise ValueError("Food-pairing evaluation requires a configured Ollama execution model")

    definitions = definitions_for_availability(availability)
    registry = ToolRegistry(definitions, config=config)
    return WineAgent(
        llm=load_execution_model(config),
        tool_registry=registry,
        prompt_registry=prompt_registry,
        verbose=False,
    )


def estimate_comparison(manifest: PairingManifest) -> dict[str, Any]:
    """Report the approved execution bound without making provider calls."""
    arm_executions = len(manifest.scenarios) * manifest.repetitions
    return {
        "arms": 2,
        "scenarios": len(manifest.scenarios),
        "repetitions": manifest.repetitions,
        "agent_runs": arm_executions * 2,
        "maximum_agent_runs": 48,
        "within_approved_run_cap": arm_executions * 2 <= 48,
        "judge_calls": 0,
    }


def capture_comparison(
    output_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Capture both prompt arms serially with answers, prompt identities, and tool traces."""
    manifest = load_manifest(manifest_path)
    estimate = estimate_comparison(manifest)
    if not estimate["within_approved_run_cap"]:
        raise ValueError("Food-pairing comparison exceeds the approved 48-run cap")
    if get_git_metadata()["is_dirty"] is not False:
        raise ValueError("Commit evaluation code before capturing the comparison")

    current = get_prompt_registry()
    registries = {
        "baseline": current,
        "candidate": build_candidate_registry(manifest.candidate),
    }
    agents = {
        (arm, availability): _build_agent(registry, availability)
        for arm, registry in registries.items()
        for availability in ("standard", "pairing_unavailable")
    }
    config = get_config()
    provider, model_name, model_kwargs = resolve_execution_model_config(config)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "study": manifest.study,
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(UTC).isoformat(),
        "manifest": {"path": str(manifest_path), "content_hash": compute_file_hash(manifest_path)},
        "git": get_git_metadata(),
        "app_config_content_hash": compute_file_hash(get_project_root() / "app_config.yml"),
        "cellar_state_fingerprint": compute_cellar_state_fingerprint(),
        "model": {"provider": provider, "name": model_name, "kwargs": model_kwargs},
        "repetitions": manifest.repetitions,
        "max_concurrency": manifest.max_concurrency,
        "estimate": estimate,
        "candidate": manifest.candidate.model_dump(mode="json"),
        "profiles": {
            f"{arm}:{availability}": agent.execution_provenance.to_eval_dict()
            for (arm, availability), agent in agents.items()
        },
        "executions": [],
    }

    for arm in ("baseline", "candidate"):
        for repetition in range(1, manifest.repetitions + 1):
            for scenario in manifest.scenarios:
                agent = agents[(arm, scenario.availability)]
                sample = GoldenSample(
                    id=f"pairing_{len(artifact['executions']) + 1:03d}",
                    question=scenario.question,
                    category="pairing",
                    difficulty="medium",
                    expected_facts=scenario.expected_facts,
                    expected_tool_calls=scenario.accepted_tools,
                    ground_truth="Manual claim-level audit required.",
                    tags=["pairing", "grounding"],
                )
                started = time.perf_counter()
                try:
                    result = run_agent_sample_sync(agent, sample)
                    calls = [call.model_dump(mode="json") for call in result.tool_call_records]
                    outputs = [output.model_dump(mode="json") for output in result.tool_outputs]
                    error = None
                except Exception as exc:
                    result = None
                    calls = []
                    outputs = []
                    error = f"{type(exc).__name__}: {exc}"

                prompt = agent.rendered_system_prompt
                artifact["executions"].append(
                    {
                        "arm": arm,
                        "scenario_id": scenario.id,
                        "repetition": repetition,
                        "question": scenario.question,
                        "availability": scenario.availability,
                        "expected_facts": scenario.expected_facts,
                        "required_evidence": scenario.required_evidence,
                        "accepted_tools": scenario.accepted_tools,
                        "prompt": {
                            "label": prompt.label,
                            "source_hash": prompt.source_hash,
                            "rendered_hash": prompt.rendered_hash,
                        },
                        "answer": "" if result is None else result.answer,
                        "tool_calls": calls,
                        "tool_outputs": outputs,
                        "observed_evidence": sorted(
                            {
                                classify_tool_output(call["tool_name"]).removesuffix("_result")
                                for call in calls
                                if classify_tool_output(call["tool_name"]) != "other_result"
                            }
                        ),
                        "llm_call_count": 0 if result is None else result.llm_call_count,
                        "token_usage": None if result is None else result.token_usage,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                        "error": error,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return artifact


def assess_comparison(
    artifact: dict[str, Any],
    decisions: DecisionFile,
    manifest: PairingManifest,
) -> dict[str, Any]:
    """Apply the hard policy gate and expected-fact non-regression check."""
    expected_keys = {
        (arm, scenario.id, repetition)
        for arm in ("baseline", "candidate")
        for repetition in range(1, manifest.repetitions + 1)
        for scenario in manifest.scenarios
    }
    execution_by_key = {
        (item["arm"], item["scenario_id"], item["repetition"]): item for item in artifact.get("executions", [])
    }
    decision_by_key = {(item.arm, item.scenario_id, item.repetition): item for item in decisions.decisions}
    if set(execution_by_key) != expected_keys:
        raise ValueError("Capture does not contain the complete unique comparison cohort")
    if set(decision_by_key) != expected_keys:
        raise ValueError("Manual decisions do not cover the complete unique comparison cohort")

    hard_fields = (
        "source_attribution_supported",
        "cellar_facts_traceable",
        "required_pairing_tool_used",
        "general_knowledge_disclosed",
        "no_fabricated_specifics",
        "no_internal_tool_disclosure",
    )
    arm_results: dict[str, Any] = {}
    for arm in ("baseline", "candidate"):
        arm_keys = sorted(key for key in expected_keys if key[0] == arm)
        arm_decisions = [decision_by_key[key] for key in arm_keys]
        executions = [execution_by_key[key] for key in arm_keys]
        expected_count = sum(len(item["expected_facts"]) for item in executions)
        met_count = sum(len(decision.expected_facts_met) for decision in arm_decisions)
        failures = [
            {
                "scenario_id": decision.scenario_id,
                "repetition": decision.repetition,
                "failed": [field for field in hard_fields if not getattr(decision, field)],
                "note": decision.note,
            }
            for decision in arm_decisions
            if any(not getattr(decision, field) for field in hard_fields)
        ]
        arm_results[arm] = {
            "runs": len(executions),
            "execution_errors": sum(bool(item["error"]) for item in executions),
            "hard_requirement_failures": failures,
            "all_hard_requirements_passed": not failures and not any(item["error"] for item in executions),
            "mandatory_evidence_satisfied": sum(
                bool(item["required_evidence"]) and set(item["required_evidence"]).issubset(item["observed_evidence"])
                for item in executions
            ),
            "mandatory_evidence_required": sum(bool(item["required_evidence"]) for item in executions),
            "expected_facts_met": met_count,
            "expected_facts_total": expected_count,
            "expected_fact_coverage": met_count / expected_count if expected_count else 1.0,
        }

    candidate = arm_results["candidate"]
    baseline = arm_results["baseline"]
    candidate_failed_fields = {
        field for failure in candidate["hard_requirement_failures"] for field in failure["failed"]
    }
    checks = {
        "candidate_all_hard_requirements": candidate["all_hard_requirements_passed"],
        "candidate_mandatory_evidence_complete": (
            candidate["mandatory_evidence_satisfied"] == candidate["mandatory_evidence_required"]
        ),
        "mandatory_evidence_not_regressed": (
            candidate["mandatory_evidence_satisfied"] >= baseline["mandatory_evidence_satisfied"]
        ),
        "expected_fact_coverage_not_regressed": (
            candidate["expected_fact_coverage"] >= baseline["expected_fact_coverage"]
        ),
    }
    passed = all(checks.values())
    return {
        "study": manifest.study,
        "passed": passed,
        "recommendation": (
            "recommend pairing-source-boundaries-v1 for a separate prompt-change task"
            if passed
            else "do not change the prompt"
        ),
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_prompt_append": manifest.candidate.prompt_append,
        "follow_up_scope": (
            "Apply only the recorded prompt append and add these eight scenarios as regression coverage."
            if passed
            else (
                "Prompt layer: replace internal tool-availability wording with a plain-language "
                "verified-evidence limitation, then rerun this fixed cohort before requesting "
                "prompt-change approval."
                if candidate_failed_fields == {"no_internal_tool_disclosure"}
                else "Address the recorded failures in the tool, data, or evaluation layer before another prompt proposal."
            )
        ),
    }


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value
