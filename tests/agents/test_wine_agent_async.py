"""Deterministic async-runtime tests for WineAgent."""

import asyncio
import sqlite3
import threading
import time
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from src.agents.guardrails import (
    ToolExecutionConfig,
    ToolExecutionController,
    ToolRetryConfig,
    ToolTimeoutConfig,
)
from src.agents.intelligent.agent import WineAgent
from src.agents.prompt_registry import RenderedPrompt
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import (
    CostClass,
    LatencyClass,
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
        lambda _snapshot: RenderedPrompt(
            name="intelligent_agent_system",
            content="Test system prompt.",
            source_hash="sha256:test-source",
            rendered_hash="sha256:test-rendered",
            label="",
        ),
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
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
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
async def test_sync_tool_timeout_returns_while_worker_continues_and_records_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sync timeout must stop waiting without claiming the worker was terminated."""
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    @tool
    def blocking_sync_tool() -> str:
        """Block deterministically until the test releases the worker thread."""
        worker_started.set()
        try:
            release_worker.wait(timeout=2)
            return "released"
        finally:
            worker_finished.set()

    definition = _tool_definition(blocking_sync_tool, ToolCategory.CELLAR)
    planned_call = AIMessage(
        content="",
        tool_calls=[
            {"name": blocking_sync_tool.name, "args": {}, "id": "sync-timeout-call"}
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(
        side_effect=[planned_call, AIMessage(content="Continued safely.")]
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
        tool_execution=ToolExecutionConfig(
            max_concurrent_calls=1,
            timeout_seconds=ToolTimeoutConfig(fast=0.03, slow=0.06),
        ),
    )

    started_at = time.monotonic()
    try:
        result = await agent.ainvoke("Run the blocking tool.")
        elapsed = time.monotonic() - started_at

        assert worker_started.is_set()
        assert not worker_finished.is_set()
        assert elapsed < 0.5
        timeout_message = next(
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage) and message.name == blocking_sync_tool.name
        )
        assert timeout_message.status == "error"
        assert timeout_message.tool_call_id == "sync-timeout-call"
        assert [event["code"] for event in result["guardrail_events"]] == [
            "tool_deadline_exceeded",
            "tool_sync_timeout",
        ]
        assert all(event["sync_bridge"] is True for event in result["guardrail_events"])
        assert bound_model.ainvoke.await_count == 2
    finally:
        release_worker.set()
        assert await asyncio.to_thread(worker_finished.wait, 1)


@pytest.mark.asyncio
async def test_coroutine_timeout_records_no_sync_worker_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native coroutine deadline must not imply continuing thread work."""
    cancelled = asyncio.Event()

    @tool
    async def blocking_coroutine_tool() -> str:
        """Wait cooperatively until the execution deadline cancels this call."""
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    definition = _tool_definition(blocking_coroutine_tool, ToolCategory.CELLAR)
    planned_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": blocking_coroutine_tool.name,
                "args": {},
                "id": "coroutine-timeout-call",
            }
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(
        side_effect=[planned_call, AIMessage(content="Continued safely.")]
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
        tool_execution=ToolExecutionConfig(
            max_concurrent_calls=1,
            timeout_seconds=ToolTimeoutConfig(fast=0.02, slow=0.04),
        ),
    )

    result = await agent.ainvoke("Run the coroutine tool.")

    assert cancelled.is_set()
    assert [event["code"] for event in result["guardrail_events"]] == [
        "tool_deadline_exceeded"
    ]
    assert result["guardrail_events"][0]["sync_bridge"] is False
    assert bound_model.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_shared_controller_bounds_concurrent_agent_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two agents sharing one controller must share the same admission capacity."""
    controller = ToolExecutionController(1)
    first_started = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    @tool
    async def admitted_tool(value: str) -> str:
        """Record active calls while waiting for deterministic test release."""
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        first_started.set()
        try:
            await release.wait()
            return value
        finally:
            active -= 1

    definition = _tool_definition(admitted_tool, ToolCategory.CELLAR)
    policy = ToolExecutionConfig(
        max_concurrent_calls=1,
        timeout_seconds=ToolTimeoutConfig(fast=0.5, slow=0.5),
    )

    def build_agent(call_id: str, value: str) -> WineAgent:
        planned_call = AIMessage(
            content="",
            tool_calls=[
                {"name": admitted_tool.name, "args": {"value": value}, "id": call_id}
            ],
        )
        bound_model = MagicMock()
        bound_model.ainvoke = AsyncMock(
            side_effect=[planned_call, AIMessage(content=f"Finished {value}.")]
        )
        llm = MagicMock()
        llm.bind_tools.return_value = bound_model
        return WineAgent(
            llm=llm,
            tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
            tool_execution=policy,
            tool_execution_controller=controller,
        )

    first_agent = build_agent("shared-call-1", "one")
    second_agent = build_agent("shared-call-2", "two")
    first_task = asyncio.create_task(first_agent.ainvoke("Run first."))
    await asyncio.wait_for(first_started.wait(), timeout=0.2)
    second_task = asyncio.create_task(second_agent.ainvoke("Run second."))
    await asyncio.sleep(0.02)

    assert active == 1
    assert maximum_active == 1

    release.set()
    first_result, second_result = await asyncio.gather(first_task, second_task)
    assert maximum_active == 1
    assert first_result["guardrail_events"] == []
    assert second_result["guardrail_events"] == []


@pytest.mark.asyncio
async def test_hybrid_mixed_batch_preserves_siblings_commands_order_and_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One deadline must not disturb successful or command-producing siblings."""
    timeout_attempts = 0

    @tool
    async def successful_tool(value: str) -> str:
        """Return one successful sibling result."""
        return f"success:{value}"

    @tool
    async def timed_out_tool() -> str:
        """Wait cooperatively beyond the configured response deadline."""
        nonlocal timeout_attempts
        timeout_attempts += 1
        await asyncio.Event().wait()
        return "unreachable"

    @tool
    async def command_tool(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Return a state update carrying the required matching ToolMessage."""
        return Command(
            update={
                "messages": [
                    ToolMessage(content="command:ok", tool_call_id=tool_call_id)
                ]
            }
        )

    definitions = (
        _tool_definition(successful_tool, ToolCategory.CELLAR),
        _tool_definition(timed_out_tool, ToolCategory.RAG),
        _tool_definition(command_tool, ToolCategory.PAIRING),
    )
    planned_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": successful_tool.name, "args": {"value": "one"}, "id": "success-call"},
            {"name": timed_out_tool.name, "args": {}, "id": "timeout-call"},
            {"name": command_tool.name, "args": {}, "id": "command-call"},
        ],
    )
    planner = MagicMock()
    planner.ainvoke = AsyncMock(return_value=planned_calls)
    planning_llm = MagicMock()
    planning_llm.bind_tools.return_value = planner
    generation_llm = MagicMock()
    generation_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Hybrid final answer."))
    agent = WineAgent(
        llm=generation_llm,
        tool_llm=planning_llm,
        tool_registry=_registry_from_definitions(monkeypatch, definitions),
        tool_execution=ToolExecutionConfig(
            max_concurrent_calls=3,
            timeout_seconds=ToolTimeoutConfig(fast=0.02, slow=0.04),
        ),
    )

    result = await agent.ainvoke("Run the mixed batch.")

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert [message.tool_call_id for message in tool_messages] == [
        "success-call",
        "timeout-call",
        "command-call",
    ]
    assert tool_messages[0].content == "success:one"
    assert tool_messages[1].status == "error"
    assert tool_messages[2].content == "command:ok"
    assert result["guardrail_events"][0]["tool_name"] == timed_out_tool.name
    assert result["guardrail_events"][0]["sync_bridge"] is False
    assert timeout_attempts == 1
    assert planner.ainvoke.await_count == 1
    assert generation_llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_unknown_tool_retains_framework_invalid_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing snapshot metadata must leave invalid-call handling to ToolNode."""

    @tool
    async def known_tool() -> str:
        """Provide one registered name for the framework's validation message."""
        return "known"

    definition = _tool_definition(known_tool, ToolCategory.CELLAR)
    planned_call = AIMessage(
        content="",
        tool_calls=[{"name": "unknown_tool", "args": {}, "id": "unknown-call"}],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(
        side_effect=[planned_call, AIMessage(content="Handled invalid call.")]
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
        tool_execution=ToolExecutionConfig(
            max_concurrent_calls=1,
            timeout_seconds=ToolTimeoutConfig(fast=0.02, slow=0.04),
        ),
    )

    result = await agent.ainvoke("Call an unknown tool.")

    invalid_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "unknown-call"
    )
    assert invalid_message.name == "unknown_tool"
    assert invalid_message.status == "error"
    assert "unknown_tool" in str(invalid_message.content)
    assert known_tool.name in str(invalid_message.content)
    assert result["guardrail_events"] == []
    assert bound_model.ainvoke.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("hybrid", (False, True), ids=("standard", "hybrid"))
async def test_agent_recovers_sqlite_contention_without_extra_model_calls(
    monkeypatch: pytest.MonkeyPatch,
    hybrid: bool,
) -> None:
    """Approved contention should recover once without changing model-call behavior."""
    attempts = 0

    @tool
    async def contention_tool(value: str) -> str:
        """Fail once with a structured busy code, then return normally."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            exception = sqlite3.OperationalError("synthetic private contention")
            exception.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exception
        return f"recovered:{value}"

    definition = _tool_definition(contention_tool, ToolCategory.CELLAR)
    planned_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": contention_tool.name,
                "args": {"value": "wine"},
                "id": "retry-agent-call",
            }
        ],
    )
    execution_policy = ToolExecutionConfig(
        max_concurrent_calls=1,
        timeout_seconds=ToolTimeoutConfig(fast=0.2, slow=0.2),
        retry=ToolRetryConfig(delay_seconds=0.0, min_remaining_seconds=0.01),
    )

    if hybrid:
        planner = MagicMock()
        planner.ainvoke = AsyncMock(return_value=planned_call)
        planning_llm = MagicMock()
        planning_llm.bind_tools.return_value = planner
        generation_llm = MagicMock()
        generation_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Recovered."))
        agent = WineAgent(
            llm=generation_llm,
            tool_llm=planning_llm,
            tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
            tool_execution=execution_policy,
        )
    else:
        bound_model = MagicMock()
        bound_model.ainvoke = AsyncMock(
            side_effect=[planned_call, AIMessage(content="Recovered.")]
        )
        llm = MagicMock()
        llm.bind_tools.return_value = bound_model
        agent = WineAgent(
            llm=llm,
            tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
            tool_execution=execution_policy,
        )

    result = await agent.ainvoke("Recover from contention.")

    recovered_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == contention_tool.name
    )
    assert recovered_message.content == "recovered:wine"
    assert recovered_message.tool_call_id == "retry-agent-call"
    assert attempts == 2
    assert [event["code"] for event in result["guardrail_events"]] == [
        "tool_retry_started",
        "tool_retry_succeeded",
    ]
    assert all(event["sync_bridge"] is False for event in result["guardrail_events"])
    assert result["llm_call_count"] == 2
    if hybrid:
        assert planner.ainvoke.await_count == 1
        assert generation_llm.ainvoke.await_count == 1
    else:
        assert bound_model.ainvoke.await_count == 2


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
    _assert_equivalent_results(
        sync_result,
        {**async_result, "guardrail_events": []},
    )
    assert sync_result["llm_call_count"] == async_result["llm_call_count"] == 2
    assert sync_result["guardrail_events"] == []
    assert async_result["guardrail_events"] == [
        {
            "code": "tool_terminal_failure",
            "tool_name": failing_wine_knowledge_tool.name,
            "latency_class": "fast",
            "cost_class": "free",
            "attempt_number": 1,
            "sync_bridge": True,
        }
    ]
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
