"""Validation for the frozen M9A Gate 0 relevance cohort."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.guardrails import RELEVANCE_REDIRECT, RelevanceConfig, evaluate_relevance
from src.agents.intelligent.agent import WineAgent
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORT_PATH = PROJECT_ROOT / "src" / "eval" / "m09a_relevance_cohort.json"
EXPECTED_ROUTES = {
    "on_topic": "allow",
    "off_topic": "deflect",
    "ambiguous": "allow",
}


def _load_cohort() -> dict[str, Any]:
    """Load the checked-in relevance cohort."""
    return json.loads(COHORT_PATH.read_text(encoding="utf-8"))


def test_relevance_cohort_has_balanced_unique_samples() -> None:
    """Freeze a balanced set of unique on-topic, off-topic, and ambiguous queries."""
    cohort = _load_cohort()
    samples = cohort["samples"]
    sample_ids = [sample["id"] for sample in samples]
    queries = [sample["query"] for sample in samples]
    category_counts = {
        category: sum(sample["classification"] == category for sample in samples)
        for category in EXPECTED_ROUTES
    }

    assert cohort["cohort_id"] == "m09a_relevance_v1"
    assert cohort["reviewed_on"] == "2026-08-23"
    assert category_counts == {"on_topic": 4, "off_topic": 4, "ambiguous": 4}
    assert len(sample_ids) == len(set(sample_ids))
    assert len(queries) == len(set(queries))
    assert all(query.strip() for query in queries)


def test_relevance_cohort_declares_reviewed_routes_and_phrases() -> None:
    """Every sample should encode its reviewed route and deterministic phrase evidence."""
    samples = _load_cohort()["samples"]

    for sample in samples:
        classification = sample["classification"]
        matched_phrase = sample["matched_phrase"]
        assert sample["expected_route"] == EXPECTED_ROUTES[classification]
        if classification == "ambiguous":
            assert matched_phrase is None
        else:
            assert isinstance(matched_phrase, str)
            assert matched_phrase in sample["query"].casefold()


def test_relevance_matcher_matches_every_reviewed_cohort_route() -> None:
    """Report deterministic routes and false positives over the frozen cohort."""
    samples = _load_cohort()["samples"]
    observed_routes = {
        sample["id"]: evaluate_relevance(sample["query"], RelevanceConfig()).route
        for sample in samples
    }
    false_positives = [
        sample["id"]
        for sample in samples
        if sample["classification"] in {"on_topic", "ambiguous"}
        and observed_routes[sample["id"]] == "deflect"
    ]
    missed_off_topic = [
        sample["id"]
        for sample in samples
        if sample["classification"] == "off_topic"
        and observed_routes[sample["id"]] != "deflect"
    ]

    assert observed_routes == {
        sample["id"]: sample["expected_route"] for sample in samples
    }
    assert false_positives == []
    assert missed_off_topic == []


def test_agent_routes_frozen_cohort_with_expected_model_call_savings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear off-topic samples should deflect while reviewed allow samples use one call."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: MagicMock(content="Test system prompt."),
    )

    snapshot = ToolSelectionSnapshot(definitions=(), readiness=())
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = snapshot
    bound_model = MagicMock()
    bound_model.invoke.return_value = AIMessage(content="Baseline model answer.")
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(llm=llm, tool_registry=registry)
    samples = _load_cohort()["samples"]

    for sample in samples:
        calls_before = bound_model.invoke.call_count
        result = agent.invoke(sample["query"])
        calls_after = bound_model.invoke.call_count
        if sample["expected_route"] == "deflect":
            assert result["final_answer"] == RELEVANCE_REDIRECT
            assert result["llm_call_count"] == 0
            assert result["tools_used"] == []
            assert calls_after == calls_before
        else:
            assert result["final_answer"] == "Baseline model answer."
            assert result["llm_call_count"] == 1
            assert calls_after == calls_before + 1

    expected_allowed = sum(sample["expected_route"] == "allow" for sample in samples)
    assert bound_model.invoke.call_count == expected_allowed
