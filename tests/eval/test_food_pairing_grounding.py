"""Tests for the bounded food-pairing grounding study."""

from pathlib import Path

import pytest

from src.eval.food_pairing_grounding import (
    DecisionFile,
    assess_comparison,
    build_candidate_registry,
    estimate_comparison,
    load_manifest,
)


def _artifact() -> dict:
    """Build one complete deterministic comparison artifact."""
    manifest = load_manifest()
    return {
        "executions": [
            {
                "arm": arm,
                "scenario_id": scenario.id,
                "repetition": repetition,
                "expected_facts": scenario.expected_facts,
                "required_evidence": scenario.required_evidence,
                "observed_evidence": scenario.required_evidence,
                "error": None,
            }
            for arm in ("baseline", "candidate")
            for repetition in range(1, manifest.repetitions + 1)
            for scenario in manifest.scenarios
        ]
    }


def _decisions(*, candidate_failure: bool = False) -> DecisionFile:
    """Build complete manual decisions with an optional candidate failure."""
    manifest = load_manifest()
    rows = []
    for arm in ("baseline", "candidate"):
        for repetition in range(1, manifest.repetitions + 1):
            for scenario in manifest.scenarios:
                passed = not (candidate_failure and arm == "candidate" and scenario.id == "unsupported_food")
                rows.append(
                    {
                        "arm": arm,
                        "scenario_id": scenario.id,
                        "repetition": repetition,
                        "source_attribution_supported": passed,
                        "cellar_facts_traceable": True,
                        "required_pairing_tool_used": True,
                        "general_knowledge_disclosed": True,
                        "no_fabricated_specifics": True,
                        "expected_facts_met": scenario.expected_facts,
                        "note": "Audited.",
                    }
                )
    return DecisionFile.model_validate({"schema_version": 1, "decisions": rows})


def test_manifest_freezes_eight_scenarios_and_32_runs() -> None:
    """The cohort stays within the approved two-arm, two-repeat bound."""
    manifest = load_manifest()
    estimate = estimate_comparison(manifest)

    assert len(manifest.scenarios) == 8
    assert estimate["agent_runs"] == 32
    assert estimate["within_approved_run_cap"] is True
    assert {scenario.availability for scenario in manifest.scenarios} == {
        "standard",
        "pairing_unavailable",
    }


def test_candidate_registry_changes_only_intelligent_prompt_identity() -> None:
    """The candidate is isolated from the production registry and other prompts."""
    manifest = load_manifest()
    candidate = build_candidate_registry(manifest.candidate)
    from src.agents.prompt_registry import get_prompt_registry

    current = get_prompt_registry()
    assert candidate.get("intelligent_agent_system").source_hash != current.get("intelligent_agent_system").source_hash
    assert candidate.get("intelligent_agent_system").label == "pairing-source-boundaries-v1"
    assert current.get("intelligent_agent_system").label == "concise-direct-v1"
    for name in current.get_source_version_map():
        if name != "intelligent_agent_system":
            assert candidate.get(name) == current.get(name)


def test_assessment_recommends_candidate_only_when_every_gate_passes() -> None:
    """One claim-level failure must produce the explicit no-change conclusion."""
    manifest = load_manifest()
    passed = assess_comparison(_artifact(), _decisions(), manifest)
    failed = assess_comparison(_artifact(), _decisions(candidate_failure=True), manifest)

    assert passed["passed"] is True
    assert passed["recommendation"].startswith("recommend pairing-source-boundaries-v1")
    assert failed["passed"] is False
    assert failed["recommendation"] == "do not change the prompt"


def test_assessment_rejects_incomplete_manual_audit() -> None:
    """Every captured output requires an explicit manual source-boundary audit."""
    manifest = load_manifest()
    decisions = _decisions().model_dump(mode="json")
    decisions["decisions"].pop()

    with pytest.raises(ValueError, match="complete unique comparison cohort"):
        assess_comparison(_artifact(), DecisionFile.model_validate(decisions), manifest)
