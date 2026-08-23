"""M9A Gate 0 baselines for current intelligent-agent model-call counts."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.intelligent.agent import WineAgent
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot
from src.agents.tools.web_search_tools import TOOL_DEFINITIONS, search_web_for_wine


def _tool_snapshot() -> ToolSelectionSnapshot:
    """Create a deterministic snapshot containing one active catalogue tool."""
    definition = next(
        definition for definition in TOOL_DEFINITIONS if definition.tool is search_web_for_wine
    )
    return ToolSelectionSnapshot(definitions=(definition,), readiness=())


def _prepare_dependencies(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    """Patch external dependencies and return a deterministic tool registry."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )

    engine = MagicMock()
    engine.search.return_value = []
    monkeypatch.setattr("src.agents.tools.web_search_tools._engine", engine)

    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = _tool_snapshot()
    return registry


def _tool_call(call_id: str, query: str) -> AIMessage:
    """Create one deterministic model response requesting the baseline tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": search_web_for_wine.name,
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _build_standard_agent(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[AIMessage],
) -> tuple[WineAgent, MagicMock]:
    """Build a standard agent and expose its bound model mock for accounting."""
    bound_model = MagicMock()
    bound_model.invoke.side_effect = responses
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(llm=llm, tool_registry=_prepare_dependencies(monkeypatch))
    return agent, bound_model


def test_standard_no_tool_request_uses_one_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A direct standard answer currently needs one model invocation."""
    agent, bound_model = _build_standard_agent(
        monkeypatch,
        [AIMessage(content="A direct wine answer.")],
    )

    result = agent.invoke("What is tannin?")

    assert result["final_answer"] == "A direct wine answer."
    assert result["tools_used"] == []
    assert bound_model.invoke.call_count == 1


def test_standard_single_tool_request_uses_two_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One standard tool round trip currently needs planning and final-answer calls."""
    agent, bound_model = _build_standard_agent(
        monkeypatch,
        [
            _tool_call("single-tool-call", "Barolo current news"),
            AIMessage(content="A tool-grounded wine answer."),
        ],
    )

    result = agent.invoke("What is the latest Barolo news?")

    assert result["final_answer"] == "A tool-grounded wine answer."
    assert result["tools_used"] == [search_web_for_wine.name]
    assert bound_model.invoke.call_count == 2


def test_standard_multi_iteration_request_uses_three_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two standard tool iterations currently need three model invocations."""
    agent, bound_model = _build_standard_agent(
        monkeypatch,
        [
            _tool_call("first-tool-call", "Barolo current news"),
            _tool_call("second-tool-call", "Barbaresco current news"),
            AIMessage(content="A multi-step wine answer."),
        ],
    )

    result = agent.invoke("Compare current Barolo and Barbaresco news.")

    assert result["final_answer"] == "A multi-step wine answer."
    assert result["tools_used"] == [search_web_for_wine.name]
    assert bound_model.invoke.call_count == 3


def test_hybrid_tool_request_uses_two_total_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid tool request currently uses one planning and one generation call."""
    planner = MagicMock()
    planner.invoke.return_value = _tool_call("hybrid-tool-call", "Bordeaux current news")
    tool_llm = MagicMock()
    tool_llm.bind_tools.return_value = planner

    generation_llm = MagicMock()
    generation_llm.invoke.return_value = AIMessage(content="A hybrid wine answer.")
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=tool_llm,
        tool_registry=_prepare_dependencies(monkeypatch),
    )

    result = agent.invoke("What is the latest Bordeaux news?")

    assert result["final_answer"] == "A hybrid wine answer."
    assert result["tools_used"] == [search_web_for_wine.name]
    assert planner.invoke.call_count == 1
    assert generation_llm.invoke.call_count == 1
    assert planner.invoke.call_count + generation_llm.invoke.call_count == 2
