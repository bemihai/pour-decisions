"""Deterministic async-runtime tests for WineAgent."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from src.agents.intelligent.agent import WineAgent
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import (
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolRegistry,
    ToolSelectionSnapshot,
    ToolTier,
)


def _registry_from_definitions(
    monkeypatch: pytest.MonkeyPatch,
    definitions: tuple[ToolDefinition, ...],
) -> ToolRegistry:
    """Return a deterministic registry with an explicit construction snapshot."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = ToolSelectionSnapshot(definitions=definitions, readiness=())
    return registry


def _registry(monkeypatch: pytest.MonkeyPatch, *, with_tools: bool) -> ToolRegistry:
    """Return a deterministic registry with the requested construction snapshot."""
    definitions = TOOL_DEFINITIONS[:1] if with_tools else ()
    return _registry_from_definitions(monkeypatch, definitions)


def _tool_definition(tool_instance: BaseTool, category: ToolCategory) -> ToolDefinition:
    """Build deterministic metadata for a synthetic integration-test tool."""
    return ToolDefinition(
        tool=tool_instance,
        metadata=ToolMetadata(
            name=tool_instance.name,
            category=category,
            tier=ToolTier.CORE,
            capability=f"Exercise {tool_instance.name} through the compiled graph.",
        ),
    )


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
async def test_compiled_ainvoke_dispatches_sync_and_async_tools_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiled graph must await async tools and isolate blocking sync tools."""
    event_loop_thread = threading.get_ident()
    sync_tool_threads: list[int] = []
    async_tool_threads: list[int] = []

    @tool
    def blocking_sync_tool(value: str) -> str:
        """Return a marker after representative blocking synchronous work."""
        sync_tool_threads.append(threading.get_ident())
        time.sleep(0.01)
        return f"sync:{value}"

    @tool
    async def coroutine_tool(value: str) -> str:
        """Return a marker from a natively asynchronous tool."""
        async_tool_threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return f"async:{value}"

    definitions = (
        _tool_definition(blocking_sync_tool, ToolCategory.CELLAR),
        _tool_definition(coroutine_tool, ToolCategory.RAG),
    )
    planned_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": blocking_sync_tool.name, "args": {"value": "one"}, "id": "sync-call"},
            {"name": coroutine_tool.name, "args": {"value": "two"}, "id": "async-call"},
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(
        side_effect=[planned_calls, AIMessage(content="Both tools completed.")],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry_from_definitions(monkeypatch, definitions),
    )

    result = await agent.ainvoke("Run both test tools.")

    tool_messages = {
        message.name: message for message in result["messages"] if isinstance(message, ToolMessage)
    }
    assert set(tool_messages) == {blocking_sync_tool.name, coroutine_tool.name}
    assert tool_messages[blocking_sync_tool.name].content == "sync:one"
    assert tool_messages[blocking_sync_tool.name].tool_call_id == "sync-call"
    assert tool_messages[coroutine_tool.name].content == "async:two"
    assert tool_messages[coroutine_tool.name].tool_call_id == "async-call"
    assert set(result["tools_used"]) == {blocking_sync_tool.name, coroutine_tool.name}
    assert result["llm_call_count"] == 2
    assert sync_tool_threads and sync_tool_threads[0] != event_loop_thread
    assert async_tool_threads == [event_loop_thread]
    assert agent.tool_selection_snapshot.definitions == definitions
    assert bound_model.ainvoke.await_count == 2
    bound_model.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_wine_agent_ainvoke_propagates_cancellation_from_async_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling an agent request must cancel rather than safe-wrap its tool task."""
    tool_started = asyncio.Event()
    tool_cancelled = asyncio.Event()

    @tool
    async def cancellable_tool(value: str) -> str:
        """Wait until the enclosing agent request is cancelled."""
        tool_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise
        return value

    definition = _tool_definition(cancellable_tool, ToolCategory.CELLAR)
    planned_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": cancellable_tool.name,
                "args": {"value": "pending"},
                "id": "cancelled-tool-call",
            }
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=planned_call)
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
    )

    request_task = asyncio.create_task(agent.ainvoke("Wait for cancellation."))
    await asyncio.wait_for(tool_started.wait(), timeout=1)
    request_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert tool_cancelled.is_set()
    assert bound_model.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_compiled_safe_error_parity_freezes_m9b_async_wrapper_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiled sync and async paths must expose the same safe tool failure."""
    raw_failure = "M06A_COMPILED_PRIVATE_TOOL_FAILURE"

    @tool
    def failing_wine_knowledge_tool(value: str) -> str:
        """Raise one private failure for safe-error boundary validation."""
        raise RuntimeError(f"{raw_failure}:{value}")

    definition = _tool_definition(failing_wine_knowledge_tool, ToolCategory.RAG)
    sync_plan = AIMessage(
        content="",
        tool_calls=[
            {
                "name": failing_wine_knowledge_tool.name,
                "args": {"value": "same"},
                "id": "safe-call",
            }
        ],
    )
    async_plan = AIMessage(
        content="",
        tool_calls=[
            {
                "name": failing_wine_knowledge_tool.name,
                "args": {"value": "same"},
                "id": "safe-call",
            }
        ],
    )
    bound_model = MagicMock()
    bound_model.invoke.side_effect = [sync_plan, AIMessage(content="Safe final answer.")]
    bound_model.ainvoke = AsyncMock(
        side_effect=[async_plan, AIMessage(content="Safe final answer.")],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
    )

    sync_result = agent.invoke("Exercise the safe tool boundary.")
    async_result = await agent.ainvoke("Exercise the safe tool boundary.")

    sync_error = next(
        message for message in sync_result["messages"] if isinstance(message, ToolMessage)
    )
    async_error = next(
        message for message in async_result["messages"] if isinstance(message, ToolMessage)
    )
    expected_content = (
        "[wine_knowledge_tool_failed] "
        "Wine knowledge search is temporarily unavailable. Continue without it."
    )
    assert sync_error.content == async_error.content == expected_content
    assert sync_error.name == async_error.name == failing_wine_knowledge_tool.name
    assert sync_error.tool_call_id == async_error.tool_call_id == "safe-call"
    assert sync_error.status == async_error.status == "error"
    assert raw_failure not in str(sync_error.content)
    assert raw_failure not in str(async_error.content)
    _assert_equivalent_results(sync_result, async_result)
    assert sync_result["llm_call_count"] == async_result["llm_call_count"] == 2
    assert sync_result["guardrail_events"] == async_result["guardrail_events"] == []
    assert sync_result["tools_used"] == async_result["tools_used"]
    assert len(sync_result["tool_call_history"]) == len(async_result["tool_call_history"]) == 1
    assert bound_model.invoke.call_count == 2
    assert bound_model.ainvoke.await_count == 2


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
async def test_standard_invoke_and_ainvoke_sanitize_final_answers_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both public invocation modes must apply the same final-output redaction."""
    sensitive_answer = "Set M06A_SYNTHETIC_PROVIDER_TOKEN before retrying."
    expected_answer = "Set [internal configuration redacted] before retrying."
    bound_model = MagicMock()
    bound_model.invoke.return_value = AIMessage(content=sensitive_answer)
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content=sensitive_answer))
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(llm=llm, tool_registry=_empty_registry(monkeypatch))

    sync_result = agent.invoke("Explain the provider failure.")
    async_result = await agent.ainvoke("Explain the provider failure.")

    _assert_equivalent_results(sync_result, async_result)
    assert sync_result["final_answer"] == async_result["final_answer"] == expected_answer
    assert "M06A_SYNTHETIC_PROVIDER_TOKEN" not in sync_result["final_answer"]


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
