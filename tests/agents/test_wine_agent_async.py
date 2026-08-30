"""Deterministic async-runtime tests for WineAgent."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.intelligent.agent import WineAgent
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot


def _registry(monkeypatch: pytest.MonkeyPatch, *, with_tools: bool) -> ToolRegistry:
    """Return a deterministic registry with the requested construction snapshot."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )
    registry = MagicMock(spec=ToolRegistry)
    definitions = TOOL_DEFINITIONS[:1] if with_tools else ()
    registry.select.return_value = ToolSelectionSnapshot(definitions=definitions, readiness=())
    return registry


def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    """Return a deterministic registry with no selected tools."""
    return _registry(monkeypatch, with_tools=False)


def _assert_equivalent_results(sync_result: dict, async_result: dict) -> None:
    """Compare complete WineAgent results without relying on message identity."""
    assert async_result.keys() == sync_result.keys()
    for key in sync_result.keys() - {"messages"}:
        assert async_result[key] == sync_result[key]

    sync_messages = [
        (type(message).__name__, message.content, getattr(message, "tool_calls", None))
        for message in sync_result["messages"]
    ]
    async_messages = [
        (type(message).__name__, message.content, getattr(message, "tool_calls", None))
        for message in async_result["messages"]
    ]
    assert async_messages == sync_messages


@pytest.mark.asyncio
async def test_standard_graph_ainvoke_uses_async_model_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standard compiled graph must await its bound model."""
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Async standard answer."))
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(llm=llm, tool_registry=_empty_registry(monkeypatch))

    response = await agent.agent.ainvoke(
        agent._build_invoke_payload("What is tannin?", None),
        config=agent._build_runnable_config(None),
    )

    assert response["messages"][-1].content == "Async standard answer."
    bound_model.ainvoke.assert_awaited_once()
    bound_model.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_graph_ainvoke_uses_async_planning_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hybrid compiled graph must await both model nodes."""
    planner = MagicMock()
    planner.ainvoke = AsyncMock(return_value=AIMessage(content="No tools required."))
    planning_llm = MagicMock()
    planning_llm.bind_tools.return_value = planner

    generation_llm = MagicMock()
    generation_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Async hybrid answer."))
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=planning_llm,
        tool_registry=_empty_registry(monkeypatch),
    )

    response = await agent.agent.ainvoke(
        agent._build_invoke_payload("Recommend a wine.", None),
        config=agent._build_runnable_config(None),
    )

    assert response["messages"][-1].content == "Async hybrid answer."
    planner.ainvoke.assert_awaited_once()
    generation_llm.ainvoke.assert_awaited_once()
    planner.invoke.assert_not_called()
    generation_llm.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_standard_invoke_and_ainvoke_return_equivalent_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standard agents should preserve their complete result across modes."""
    bound_model = MagicMock()
    bound_model.invoke.return_value = AIMessage(content="Standard answer.")
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Standard answer."))
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch, with_tools=True),
        verbose=True,
    )
    history = [{"role": "human", "content": "Earlier question."}]
    trace_context = {"request_id": "phase-1-standard"}

    sync_result = agent.invoke("What is tannin?", history, trace_context)
    async_result = await agent.ainvoke("What is tannin?", history, trace_context)

    _assert_equivalent_results(sync_result, async_result)
    assert sync_result["llm_call_count"] == 1
    assert sync_result["intermediate_steps"] == []
    bound_model.invoke.assert_called_once()
    bound_model.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_hybrid_invoke_and_ainvoke_return_equivalent_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid planning and generation should preserve complete result parity."""
    planner = MagicMock()
    planner.invoke.return_value = AIMessage(content="No tools required.")
    planner.ainvoke = AsyncMock(return_value=AIMessage(content="No tools required."))
    planning_llm = MagicMock()
    planning_llm.bind_tools.return_value = planner

    generation_llm = MagicMock()
    generation_llm.invoke.return_value = AIMessage(content="Hybrid answer.")
    generation_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hybrid answer."))
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=planning_llm,
        tool_registry=_registry(monkeypatch, with_tools=True),
    )

    sync_result = agent.invoke("Recommend a wine.")
    async_result = await agent.ainvoke("Recommend a wine.")

    _assert_equivalent_results(sync_result, async_result)
    assert sync_result["llm_call_count"] == 2
    planner.invoke.assert_called_once()
    planner.ainvoke.assert_awaited_once()
    generation_llm.invoke.assert_called_once()
    generation_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_zero_tool_invoke_and_ainvoke_return_equivalent_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty construction snapshot should work identically in both modes."""
    bound_model = MagicMock()
    bound_model.invoke.return_value = AIMessage(content="Zero-tool answer.")
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Zero-tool answer."))
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(llm=llm, tool_registry=_empty_registry(monkeypatch))

    sync_result = agent.invoke("Explain acidity.")
    async_result = await agent.ainvoke("Explain acidity.")

    _assert_equivalent_results(sync_result, async_result)
    assert sync_result["tools_used"] == []
    assert sync_result["llm_call_count"] == 1
