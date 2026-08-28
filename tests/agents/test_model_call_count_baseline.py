"""M9A Gate 0 baselines for current intelligent-agent model-call counts."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from src.agents.guardrails import (
    CALL_BUDGET_EVENT_CODE,
    LOOP_DETECTED_EVENT_CODE,
    REDACTION_TOKEN,
    CallBudgetConfig,
    LoopDetectionConfig,
)
from src.agents.intelligent.agent import WineAgent
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot
from src.agents.tools.web_search_tools import TOOL_DEFINITIONS, search_web_for_wine


def _tool_snapshot() -> ToolSelectionSnapshot:
    """Create a deterministic snapshot containing one active catalogue tool."""
    definition = next(
        definition for definition in TOOL_DEFINITIONS if definition.tool is search_web_for_wine
    )
    return ToolSelectionSnapshot(definitions=(definition,), readiness=())


def _prepare_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    engine: MagicMock | None = None,
) -> ToolRegistry:
    """Patch external dependencies and return a deterministic tool registry."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )

    engine = engine or MagicMock()
    engine.search.return_value = []
    monkeypatch.setattr("src.agents.tools.web_search_tools._engine", engine)

    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = _tool_snapshot()
    return registry


def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    """Return a deterministic registry with no available tools."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = ToolSelectionSnapshot(definitions=(), readiness=())
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
    assert result["llm_call_count"] == 1


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
    assert result["llm_call_count"] == 2


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
    assert result["llm_call_count"] == 3


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
    assert result["llm_call_count"] == 2


def test_zero_budget_performs_no_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero call budget should terminate before the first model attempt."""
    agent, bound_model = _build_standard_agent(
        monkeypatch,
        [AIMessage(content="This response must not be used.")],
    )
    agent.call_budget = CallBudgetConfig(max_llm_calls_per_query=0)

    result = agent.invoke("What is tannin?")

    bound_model.invoke.assert_not_called()
    assert result["llm_call_count"] == 0
    assert result["guardrail_events"][-1]["code"] == CALL_BUDGET_EVENT_CODE
    assert "narrower question" in result["final_answer"]


def test_hybrid_budget_counts_planning_and_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid planning and final generation should consume separate reservations."""
    planner = MagicMock()
    planner.invoke.return_value = AIMessage(content="No tools required.")
    tool_llm = MagicMock()
    tool_llm.bind_tools.return_value = planner

    generation_llm = MagicMock()
    generation_llm.invoke.return_value = AIMessage(content="A hybrid direct answer.")
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=tool_llm,
        tool_registry=_prepare_dependencies(monkeypatch),
    )

    result = agent.invoke("What is tannin?")

    assert result["llm_call_count"] == 2
    planner.invoke.assert_called_once()
    generation_llm.invoke.assert_called_once()


def test_graph_limit_is_independent_from_call_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A low graph limit should stop execution before the larger call budget."""
    agent, bound_model = _build_standard_agent(
        monkeypatch,
        [_tool_call("recursion-test", "Barolo current news")],
    )
    agent.call_budget = CallBudgetConfig(
        max_llm_calls_per_query=10,
        max_graph_steps_per_query=2,
    )

    with pytest.raises(GraphRecursionError):
        agent.invoke("What is the latest Barolo news?")

    assert bound_model.invoke.call_count == 1


def test_standard_zero_tool_path_remains_functional(monkeypatch: pytest.MonkeyPatch) -> None:
    """A standard agent with no tools should still return its direct answer."""
    bound_model = MagicMock()
    bound_model.invoke.return_value = AIMessage(content="A zero-tool answer.")
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(llm=llm, tool_registry=_empty_registry(monkeypatch))

    result = agent.invoke("What is tannin?")

    assert result["final_answer"] == "A zero-tool answer."
    assert result["llm_call_count"] == 1
    assert result["tool_call_history"] == []


def test_hybrid_zero_tool_path_remains_functional(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hybrid agent with no tools should retain planning and generation."""
    planner = MagicMock()
    planner.invoke.return_value = AIMessage(content="Plan complete.")
    tool_llm = MagicMock()
    tool_llm.bind_tools.return_value = planner
    generation_llm = MagicMock()
    generation_llm.invoke.return_value = AIMessage(content="A hybrid zero-tool answer.")
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=tool_llm,
        tool_registry=_empty_registry(monkeypatch),
    )

    result = agent.invoke("What is tannin?")

    assert result["final_answer"] == "A hybrid zero-tool answer."
    assert result["llm_call_count"] == 2
    planner.invoke.assert_called_once()
    generation_llm.invoke.assert_called_once()


def test_standard_history_duplicate_stops_before_second_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated call from prior history should stop before another side effect."""
    engine = MagicMock()
    engine.search.return_value = []
    bound_model = MagicMock()
    bound_model.invoke.side_effect = [
        _tool_call("first-call", "Barolo current news"),
        _tool_call("repeated-call", "Barolo current news"),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_prepare_dependencies(monkeypatch, engine),
    )

    result = agent.invoke("Keep searching for the same Barolo news.")

    assert engine.search.call_count == 1
    assert bound_model.invoke.call_count == 2
    assert result["llm_call_count"] == 2
    assert len(result["tool_call_history"]) == 1
    assert result["guardrail_events"][-1] == {
        "code": LOOP_DETECTED_EVENT_CODE,
        "tool_name": search_web_for_wine.name,
        "duplicate_scope": "history",
    }
    assert "narrower question" in result["final_answer"]


def test_standard_same_batch_duplicate_executes_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate pending batch should be rejected before any tool starts."""
    engine = MagicMock()
    engine.search.return_value = []
    duplicate_batch = AIMessage(
        content="",
        tool_calls=[
            {
                "name": search_web_for_wine.name,
                "args": {"query": "Barolo current news"},
                "id": "batch-call-1",
                "type": "tool_call",
            },
            {
                "name": search_web_for_wine.name,
                "args": {"query": "Barolo current news"},
                "id": "batch-call-2",
                "type": "tool_call",
            },
        ],
    )
    bound_model = MagicMock()
    bound_model.invoke.return_value = duplicate_batch
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_prepare_dependencies(monkeypatch, engine),
    )

    result = agent.invoke("Search twice for the same Barolo news.")

    engine.search.assert_not_called()
    assert result["llm_call_count"] == 1
    assert result["tool_call_history"] == []
    assert result["guardrail_events"][-1]["duplicate_scope"] == "pending_batch"


def test_different_arguments_in_one_batch_execute_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same tool with different arguments should not trigger exact detection."""
    engine = MagicMock()
    engine.search.return_value = []
    distinct_batch = AIMessage(
        content="",
        tool_calls=[
            {
                "name": search_web_for_wine.name,
                "args": {"query": "Barolo current news"},
                "id": "distinct-call-1",
                "type": "tool_call",
            },
            {
                "name": search_web_for_wine.name,
                "args": {"query": "Barbaresco current news"},
                "id": "distinct-call-2",
                "type": "tool_call",
            },
        ],
    )
    bound_model = MagicMock()
    bound_model.invoke.side_effect = [distinct_batch, AIMessage(content="Comparison complete.")]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_prepare_dependencies(monkeypatch, engine),
    )

    result = agent.invoke("Compare Barolo and Barbaresco news.")

    assert engine.search.call_count == 2
    assert result["final_answer"] == "Comparison complete."
    assert len(result["tool_call_history"]) == 2
    assert result["guardrail_events"] == []


def test_hybrid_duplicate_batch_skips_tools_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid violations should terminate before tools and the generation model."""
    engine = MagicMock()
    engine.search.return_value = []
    repeated = _tool_call("hybrid-1", "Bordeaux current news").tool_calls[0]
    duplicate = dict(repeated, id="hybrid-2")
    planner = MagicMock()
    planner.invoke.return_value = AIMessage(content="", tool_calls=[repeated, duplicate])
    tool_llm = MagicMock()
    tool_llm.bind_tools.return_value = planner
    generation_llm = MagicMock()
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=tool_llm,
        tool_registry=_prepare_dependencies(monkeypatch, engine),
    )

    result = agent.invoke("Search twice for the same Bordeaux news.")

    engine.search.assert_not_called()
    generation_llm.invoke.assert_not_called()
    assert result["llm_call_count"] == 1
    assert result["guardrail_events"][-1]["code"] == LOOP_DETECTED_EVENT_CODE


def test_fail_soft_answer_is_sanitized_after_budget_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserved fail-soft text should still pass through final sanitization."""
    planner = MagicMock()
    planner.invoke.return_value = AIMessage(content="Use GOOGLE_API_KEY=synthetic-secret-value")
    tool_llm = MagicMock()
    tool_llm.bind_tools.return_value = planner
    generation_llm = MagicMock()
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=tool_llm,
        tool_registry=_prepare_dependencies(monkeypatch),
    )
    agent.call_budget = CallBudgetConfig(max_llm_calls_per_query=1)

    result = agent.invoke("What is tannin?")

    generation_llm.invoke.assert_not_called()
    assert REDACTION_TOKEN in result["final_answer"]
    assert "GOOGLE_API_KEY" not in result["final_answer"]
    assert "synthetic-secret-value" not in result["final_answer"]


def test_disabled_loop_detection_allows_duplicate_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The behavioral rollout flag should bypass exact duplicate rejection."""
    engine = MagicMock()
    engine.search.return_value = []
    repeated = _tool_call("disabled-1", "Rioja current news").tool_calls[0]
    duplicate = dict(repeated, id="disabled-2")
    bound_model = MagicMock()
    bound_model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[repeated, duplicate]),
        AIMessage(content="Both searches completed."),
    ]
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_prepare_dependencies(monkeypatch, engine),
        loop_detection=LoopDetectionConfig(enabled=False),
    )

    result = agent.invoke("Search twice for Rioja news.")

    assert engine.search.call_count == 2
    assert result["final_answer"] == "Both searches completed."
    assert result["tool_call_history"] == []
    assert result["guardrail_events"] == []
