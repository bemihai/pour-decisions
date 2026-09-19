"""Tests for the frozen M10 planning cohort and evidence gate."""

import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from src.eval.models import AgentToolCall
from src.eval.planning_baseline import (
    RELEVANT_CELLAR_TABLES,
    PlanningCohortManifest,
    PlanningCohortSample,
    apply_adjudications,
    assess_planning_evidence,
    assert_comparable_artifacts,
    build_planning_baseline_agent,
    compute_cellar_state_fingerprint,
    load_planning_cohort,
    validate_gate0_artifact,
)


def test_planning_baseline_agent_uses_configured_ollama_eval_model() -> None:
    """Phase 0 injects the eval model instead of loading the production Gemini default."""
    config = OmegaConf.create(
        {
            "eval": {
                "execution_provider": "ollama",
                "execution_model": "gemma4:cloud",
                "ollama": {"base_url": "http://localhost:11434"},
                "sample_timeout_seconds": 300,
            }
        }
    )
    model = Mock(name="ollama_eval_model")
    agent = Mock(name="planning_agent")

    with (
        patch("src.eval.planning_baseline.load_execution_model", return_value=model) as load_model,
        patch("src.eval.planning_baseline.WineAgent", return_value=agent) as wine_agent,
    ):
        result = build_planning_baseline_agent(config)

    assert result is agent
    load_model.assert_called_once_with(config)
    wine_agent.assert_called_once_with(llm=model, verbose=False)


def test_planning_baseline_agent_rejects_production_provider() -> None:
    """Phase 0 fails closed rather than sending evaluation traffic to Gemini."""
    config = OmegaConf.create(
        {
            "eval": {
                "execution_provider": "google",
                "execution_model": "gemini-2.5-flash",
            }
        }
    )

    with pytest.raises(ValueError, match="requires eval.execution_provider=ollama"):
        build_planning_baseline_agent(config)


def test_m10_manifest_freezes_expected_dataset_and_roles() -> None:
    """The version-controlled contract contains exactly 13 targets and 3 controls."""
    manifest, samples = load_planning_cohort()

    assert len(manifest.samples) == 16
    assert len(samples) == 16
    assert [sample.id for sample in manifest.samples[:3]] == [
        "multi_hop_001",
        "multi_hop_002",
        "multi_hop_003",
    ]
    assert {sample.id for sample in manifest.samples if sample.role == "control"} == {
        "rag_only_001",
        "cellar_002",
        "pairing_001",
    }
    assert manifest.repetitions == 3
    assert manifest.max_concurrency == 1


def test_specialized_equivalence_and_argument_duplicates_are_distinct() -> None:
    """Specialized evidence passes while exact and distinct repeats remain visible."""
    policy = PlanningCohortSample(
        id="multi_hop_005",
        role="target",
        mandatory_evidence=["cellar", "pairing"],
        accepted_tools_by_evidence={
            "pairing": ["get_food_pairing_wines", "get_pairing_for_wine"]
        },
    )
    calls = [
        AgentToolCall(tool_name="get_cellar_wines"),
        AgentToolCall(
            tool_name="get_pairing_for_wine",
            arguments={"wine": "A"},
            canonical_arguments='{"wine":"A"}',
        ),
        AgentToolCall(
            tool_name="get_pairing_for_wine",
            arguments={"wine": "B"},
            canonical_arguments='{"wine":"B"}',
        ),
        AgentToolCall(
            tool_name="get_pairing_for_wine",
            arguments={"wine": "B"},
            canonical_arguments='{"wine":"B"}',
        ),
    ]

    assessment = assess_planning_evidence(policy, calls)

    assert assessment.satisfied == ["cellar", "pairing"]
    assert assessment.missing == []
    assert assessment.repeated_tool_names == {"get_pairing_for_wine": 3}
    assert assessment.distinct_repeated_calls == ["get_pairing_for_wine"]
    assert len(assessment.exact_duplicate_calls) == 2


def test_cellar_fingerprint_is_semantic_and_changes_with_rows(tmp_path: Path) -> None:
    """The fingerprint ignores SQLite file layout but detects relevant row changes."""
    db_path = tmp_path / "cellar.db"
    with sqlite3.connect(db_path) as connection:
        for table in RELEVANT_CELLAR_TABLES:
            connection.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, value TEXT)')
        connection.execute("INSERT INTO wines (id, value) VALUES (1, 'Barolo')")

    original = compute_cellar_state_fingerprint(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("VACUUM")
    assert compute_cellar_state_fingerprint(db_path) == original

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE wines SET value = 'Bandol' WHERE id = 1")
    assert compute_cellar_state_fingerprint(db_path) != original


def test_comparison_rejects_dataset_and_cellar_mismatch() -> None:
    """Paired analysis fails closed when either identity input changes."""
    baseline = {
        "dataset": {"content_hash": "dataset-a"},
        "cellar_state_fingerprint": "cellar-a",
        "app_config_content_hash": "config-a",
        "model_provenance": {"model": "same"},
        "max_concurrency": 1,
    }
    changed = {**baseline, "dataset": {"content_hash": "dataset-b"}}
    with pytest.raises(ValueError, match="dataset.content_hash"):
        assert_comparable_artifacts(baseline, changed)

    changed = {**baseline, "cellar_state_fingerprint": "cellar-b"}
    with pytest.raises(ValueError, match="cellar_state_fingerprint"):
        assert_comparable_artifacts(baseline, changed)


def test_gate0_requires_all_adjudications_and_summarizes_outcomes() -> None:
    """Gate 0 needs a resolved outcome and actual call count for every execution."""
    samples = [
        PlanningCohortSample(
            id=f"sample_{index:03d}",
            role="target" if index <= 13 else "control",
            mandatory_evidence=["cellar"],
        )
        for index in range(1, 17)
    ]
    manifest = PlanningCohortManifest(
        schema_version=1,
        milestone="m10",
        phase=0,
        dataset_path="unused",
        dataset_content_hash="unused",
        repetitions=3,
        max_concurrency=1,
        samples=samples,
    )
    executions = [
        {
            "sample_id": sample.id,
            "repetition": repetition,
            "outcome": None,
            "llm_call_count": 2,
        }
        for sample in samples
        for repetition in range(1, 4)
    ]
    artifact = {"executions": executions}

    with pytest.raises(ValueError, match="incomplete"):
        validate_gate0_artifact(artifact, manifest)

    decisions = {
        f"{execution['sample_id']}#{execution['repetition']}": "correct"
        for execution in executions
    }
    decisions.update({f"sample_001#{repetition}": "incorrect" for repetition in range(1, 4)})
    adjudicated = apply_adjudications(artifact, decisions)
    summary = validate_gate0_artifact(adjudicated, manifest)

    assert summary["passed"] is True
    assert summary["execution_count"] == 48
    assert summary["stable_incorrect_samples"] == ["sample_001"]
    assert summary["actual_llm_call_count"] == 96
