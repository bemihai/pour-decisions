"""Tests for same-commit M10 Phase 2 comparison and gate assessment."""

from __future__ import annotations

import subprocess
from copy import deepcopy

import pytest

from src.agents.prompt_registry import get_prompt_registry, sha256_text
from src.eval.planning_baseline import load_planning_cohort
from src.eval.planning_pair import assess_gate2, assert_paired_artifacts, load_prior_prompt_registry


def _paired_artifacts() -> tuple[dict, dict]:
    """Create complete synthetic arms on one clean commit and frozen cohort."""
    manifest, _ = load_planning_cohort()
    executions = [
        {
            "sample_id": sample.id,
            "role": sample.role,
            "repetition": repetition,
            "answer": "A supported wine answer.",
            "outcome": "correct",
            "error": None,
            "evidence": {"satisfied": sample.mandatory_evidence, "missing": []},
            "llm_call_count": 2,
            "latency_ms": 100.0,
            "token_usage": None,
            "tool_calls": [],
            "judge": {"coverage": 0, "errors": []},
        }
        for repetition in range(1, 4)
        for sample in manifest.samples
    ]
    common = {
        "phase": 2,
        "manifest_content_hash": "manifest",
        "dataset": {"content_hash": "dataset"},
        "cellar_state_fingerprint": "cellar",
        "app_config_content_hash": "config",
        "repetitions": 3,
        "max_concurrency": 1,
        "git": {"sha": "same", "branch": "m10-phase-2", "is_dirty": False},
        "model_provenance": {
            "mode": "intelligent",
            "models": {"planning": "ollama", "generation": "ollama"},
            "tools": {"contract_hash": "same"},
            "agent_policy": {"hash": "same"},
            "prompts": {"intelligent_agent_system": {"source_hash": "baseline"}},
        },
        "executions": executions,
    }
    baseline = {**deepcopy(common), "arm": "baseline"}
    treatment = {**deepcopy(common), "arm": "treatment"}
    treatment["model_provenance"]["prompts"]["intelligent_agent_system"]["source_hash"] = "treatment"
    return baseline, treatment


def test_prior_prompt_registry_uses_pinned_git_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Baseline source and label come from the prior registered Git revision."""
    source = "Prior registered wine prompt.\n"
    monkeypatch.setattr(
        "src.eval.planning_pair._git_output",
        lambda *args: "a" * 40 if args[0] == "rev-parse" else (
            "prompts:\n  intelligent_agent_system:\n    file: intelligent_agent_system_prompt.md.j2\n"
            "    renderer: jinja2\n    label: prior\n"
        ),
    )
    monkeypatch.setattr(
        "src.eval.planning_pair.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0),
    )
    monkeypatch.setattr(
        "src.eval.planning_pair.subprocess.check_output",
        lambda *_args, **_kwargs: source,
    )

    registry, commit = load_prior_prompt_registry("phase-one")

    assert commit == "a" * 40
    assert registry.get("intelligent_agent_system").source_hash == sha256_text(source)
    assert registry.get("intelligent_agent_system").label == "prior"
    assert get_prompt_registry().get("intelligent_agent_system").source != source


def test_pair_rejects_state_and_tool_mismatches() -> None:
    """A different cellar or tool contract cannot masquerade as a prompt-only test."""
    manifest, _ = load_planning_cohort()
    baseline, treatment = _paired_artifacts()
    treatment["cellar_state_fingerprint"] = "other"
    with pytest.raises(ValueError, match="cellar_state_fingerprint"):
        assert_paired_artifacts(baseline, treatment, manifest)

    treatment = _paired_artifacts()[1]
    treatment["model_provenance"]["tools"] = {"contract_hash": "other"}
    with pytest.raises(ValueError, match="tools"):
        assert_paired_artifacts(baseline, treatment, manifest)


def test_pair_requires_distinct_prompt_and_complete_clean_cohort() -> None:
    """Prompt identity, clean Git state, and all 48 unique executions are required."""
    manifest, _ = load_planning_cohort()
    baseline, treatment = _paired_artifacts()
    treatment["model_provenance"]["prompts"]["intelligent_agent_system"]["source_hash"] = "baseline"
    with pytest.raises(ValueError, match="distinct prompt"):
        assert_paired_artifacts(baseline, treatment, manifest)

    baseline, treatment = _paired_artifacts()
    baseline["git"]["is_dirty"] = True
    treatment["git"]["is_dirty"] = True
    with pytest.raises(ValueError, match="clean current commit"):
        assert_paired_artifacts(baseline, treatment, manifest)

    baseline, treatment = _paired_artifacts()
    treatment["executions"].pop()
    with pytest.raises(ValueError, match="complete unique"):
        assert_paired_artifacts(baseline, treatment, manifest)


def test_gate2_counts_new_correct_targets_and_mandatory_recall() -> None:
    """Gate assessment uses adjudicated completion and evidence, not exact tool names."""
    manifest, _ = load_planning_cohort()
    baseline, treatment = _paired_artifacts()
    mandatory_by_id = {sample.id: sample.mandatory_evidence for sample in manifest.samples}
    for item in baseline["executions"]:
        if item["sample_id"] in {"multi_hop_001", "multi_hop_003"}:
            item["outcome"] = "incorrect"
            item["evidence"] = {"satisfied": [], "missing": mandatory_by_id[item["sample_id"]]}

    report = assess_gate2(baseline, treatment, manifest)

    assert report["passed"] is True
    assert report["new_majority_correct_targets"] == ["multi_hop_001", "multi_hop_003"]
    assert report["checks"]["mandatory_evidence_improved"] is True
    assert report["mean_target_attempt_delta"] == 0
    assert report["baseline"]["token_coverage"] == 0


def test_gate2_rejects_correct_label_without_required_evidence() -> None:
    """Manual factual adjudication cannot override missing mandatory sources."""
    manifest, _ = load_planning_cohort()
    baseline, treatment = _paired_artifacts()
    treatment["executions"][0]["evidence"]["missing"] = ["cellar"]
    with pytest.raises(ValueError, match="evidence-incomplete"):
        assess_gate2(baseline, treatment, manifest)
