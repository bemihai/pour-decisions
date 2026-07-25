"""Unit tests for deterministic agent trajectory evaluation."""

from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from src.agents.tools import ALL_TOOLS
from src.eval.agent_metrics import (
    extract_agent_tool_calls,
    extract_agent_tool_outputs,
    score_expected_tool_calls,
)
from src.eval.dataset import load_golden_dataset


def test_tool_metrics_score_required_calls_and_order() -> None:
    """Required calls may have extras while still preserving expected order."""
    scores = score_expected_tool_calls(
        expected=["get_cellar_wines", "search_wine_knowledge"],
        actual=["get_cellar_wines", "search_web_for_wine", "search_wine_knowledge"],
    )

    assert scores == {
        "tool_recall": 1.0,
        "tool_precision": 2 / 3,
        "tool_exact_match": 0.0,
        "tool_ordered_match": 1.0,
    }


def test_tool_metrics_penalize_missing_redundant_and_reordered_calls() -> None:
    """Multiset scoring penalizes redundant calls and ordering mismatches."""
    scores = score_expected_tool_calls(
        expected=["get_cellar_wines", "search_wine_knowledge"],
        actual=["search_wine_knowledge", "search_wine_knowledge"],
    )

    assert scores["tool_recall"] == 0.5
    assert scores["tool_precision"] == 0.5
    assert scores["tool_exact_match"] == 0.0
    assert scores["tool_ordered_match"] == 0.0


def test_tool_metrics_are_absent_without_explicit_requirements() -> None:
    """An empty expectation is unspecified, not a requirement to use no tools."""
    assert score_expected_tool_calls([], ["search_wine_knowledge"]) == {}


def test_extract_agent_trajectory_preserves_calls_and_typed_outputs() -> None:
    """Structured calls retain order and tool results retain semantic types."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_cellar_wines", "args": {}, "id": "call-1"},
                {"name": "search_wine_knowledge", "args": {}, "id": "call-2"},
            ],
        ),
        ToolMessage(
            name="get_cellar_wines",
            content='{"wines": ["Barolo"]}',
            tool_call_id="call-1",
        ),
        ToolMessage(
            name="search_wine_knowledge",
            content="Barolo is made from Nebbiolo.",
            tool_call_id="call-2",
        ),
        ToolMessage(
            name="search_web_for_wine",
            content="Current release information.",
            tool_call_id="call-3",
        ),
    ]

    assert extract_agent_tool_calls(messages) == [
        "get_cellar_wines",
        "search_wine_knowledge",
    ]
    outputs = extract_agent_tool_outputs(messages)
    assert [output.output_type for output in outputs] == [
        "cellar_result",
        "rag_context",
        "web_result",
    ]
    assert outputs[0].tool_name == "get_cellar_wines"
    assert outputs[1].content == "Barolo is made from Nebbiolo."


def test_extract_agent_calls_falls_back_to_completed_tool_messages() -> None:
    """Tool results preserve execution order when structured AI calls are absent."""
    messages = [
        ToolMessage(name="get_food_pairing_wines", content="result", tool_call_id="call-1"),
        ToolMessage(name="custom_tool", content=["one", "two"], tool_call_id="call-2"),
    ]

    assert extract_agent_tool_calls(messages) == [
        "get_food_pairing_wines",
        "custom_tool",
    ]
    outputs = extract_agent_tool_outputs(messages)
    assert outputs[0].output_type == "pairing_result"
    assert outputs[1].output_type == "other_result"
    assert outputs[1].content == "one two"


def test_golden_agent_expectations_reference_registered_production_tools() -> None:
    """Agent-oriented golden samples require only tools the agent can invoke."""
    samples = load_golden_dataset(Path("src/eval/wine_qa_golden.jsonl"))
    registered_tool_names = {tool.name for tool in ALL_TOOLS}
    agent_samples = [
        sample
        for sample in samples
        if sample.category in {"cellar", "pairing", "multi_hop"}
    ]

    assert agent_samples
    assert all(sample.expected_tool_calls for sample in agent_samples)
    assert {
        tool_name
        for sample in agent_samples
        for tool_name in sample.expected_tool_calls
        if tool_name not in registered_tool_names
    } == set()
