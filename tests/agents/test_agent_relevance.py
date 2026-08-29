"""Graph-level tests for deterministic pre-agent relevance routing."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.guardrails import (
    RELEVANCE_DEFLECTED_EVENT_CODE,
    RELEVANCE_REDIRECT,
    RelevanceConfig,
)
from src.agents.intelligent.agent import WineAgent
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot
from src.agents.tools.web_search_tools import TOOL_DEFINITIONS, search_web_for_wine


def _tool_snapshot(with_tools: bool) -> ToolSelectionSnapshot:
    """Build a deterministic optional web-tool snapshot."""
    if not with_tools:
        return ToolSelectionSnapshot(definitions=(), readiness=())
    definition = next(
        definition for definition in TOOL_DEFINITIONS if definition.tool is search_web_for_wine
    )
    return ToolSelectionSnapshot(definitions=(definition,), readiness=())


def _build_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hybrid: bool,
    with_tools: bool,
) -> tuple[WineAgent, MagicMock, MagicMock, MagicMock]:
    """Build an isolated agent and expose all possible work executors."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )
    engine = MagicMock()
    engine.search.return_value = []
    monkeypatch.setattr("src.agents.tools.web_search_tools._engine", engine)

    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = _tool_snapshot(with_tools)
    planner = MagicMock()
    planner.invoke.return_value = AIMessage(content="Allowed model answer.")

    if hybrid:
        tool_llm = MagicMock()
        tool_llm.bind_tools.return_value = planner
        generation_llm = MagicMock()
        generation_llm.invoke.return_value = AIMessage(content="Allowed hybrid answer.")
        agent = WineAgent(
            llm=generation_llm,
            tool_llm=tool_llm,
            tool_registry=registry,
        )
    else:
        generation_llm = MagicMock()
        generation_llm.bind_tools.return_value = planner
        agent = WineAgent(llm=generation_llm, tool_registry=registry)

    return agent, planner, generation_llm, engine


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
def test_clear_off_topic_query_performs_zero_model_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    hybrid: bool,
    with_tools: bool,
) -> None:
    """Clear off-topic routing should stop before every costly executor."""
    agent, planner, generation_llm, engine = _build_agent(
        monkeypatch,
        hybrid=hybrid,
        with_tools=with_tools,
    )

    result = agent.invoke("What is the weather in Bucharest tomorrow?")

    assert result["final_answer"] == RELEVANCE_REDIRECT
    assert result["llm_call_count"] == 0
    assert result["tools_used"] == []
    assert result["tool_call_history"] == []
    assert result["guardrail_events"] == [
        {"code": RELEVANCE_DEFLECTED_EVENT_CODE, "route": "deflect"}
    ]
    planner.invoke.assert_not_called()
    generation_llm.invoke.assert_not_called()
    engine.search.assert_not_called()


@pytest.mark.parametrize(
    "query",
    [
        "How does weather affect a vineyard during harvest?",
        "What should I open tonight?",
    ],
)
def test_on_topic_and_ambiguous_queries_reach_agent(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
) -> None:
    """Allowlisted and ambiguous inputs should preserve normal standard routing."""
    agent, planner, _generation_llm, engine = _build_agent(
        monkeypatch,
        hybrid=False,
        with_tools=True,
    )

    result = agent.invoke(query)

    assert result["final_answer"] == "Allowed model answer."
    assert result["llm_call_count"] == 1
    assert result["guardrail_events"] == []
    planner.invoke.assert_called_once()
    engine.search.assert_not_called()


def test_disabled_relevance_preserves_existing_agent_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling the behavioral guardrail should let clear fixtures reach the model."""
    agent, planner, _generation_llm, engine = _build_agent(
        monkeypatch,
        hybrid=False,
        with_tools=True,
    )
    agent.relevance = RelevanceConfig(enabled=False)

    result = agent.invoke("What is the weather in Bucharest tomorrow?")

    assert result["final_answer"] == "Allowed model answer."
    assert result["llm_call_count"] == 1
    assert result["guardrail_events"] == []
    planner.invoke.assert_called_once()
    engine.search.assert_not_called()
