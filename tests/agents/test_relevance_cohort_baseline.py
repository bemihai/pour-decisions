"""Validation for the frozen M9A Gate 0 relevance cohort."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

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


def test_current_agent_routes_every_relevance_sample_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze the pre-guardrail behavior in which every cohort query reaches the model."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
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
        result = agent.invoke(sample["query"])
        assert result["final_answer"] == "Baseline model answer."

    assert bound_model.invoke.call_count == len(samples)
