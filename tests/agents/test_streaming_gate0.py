"""Gate 0 probes for observing the existing async agent without changing it."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from src.agents.guardrails import (
    EMPTY_FINAL_ANSWER_RETRY,
    ToolExecutionConfig,
    ToolRetryConfig,
    ToolTimeoutConfig,
)
from src.agents.guardrails.tool_execution import build_async_tool_execution_wrapper
from src.agents.intelligent.agent import WineAgent
from src.agents.memory import ConversationMemoryManager, SessionMemoryConfig
from src.agents.prompt_registry import RenderedPrompt
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
    """Create an isolated selected-tool snapshot for each local probe."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: RenderedPrompt(
            name="intelligent_agent_system",
            content="Gate 0 test prompt.",
            source_hash="sha256:gate0-source",
            rendered_hash="sha256:gate0-rendered",
            label="",
        ),
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = ToolSelectionSnapshot(definitions=definitions, readiness=())
    return registry


def _tool_definition(tool_instance: BaseTool, category: ToolCategory) -> ToolDefinition:
    """Give a fake tool the metadata required by the production wrapper."""
    return ToolDefinition(
        tool=tool_instance,
        metadata=ToolMetadata(
            name=tool_instance.name,
            category=category,
            tier=ToolTier.CORE,
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability=f"Probe {tool_instance.name} progress.",
        ),
    )


class BoundedProgressSink:
    """Probe a finite, nonblocking progress channel with one reserved terminal slot."""

    def __init__(self, capacity: int = 16) -> None:
        self.progress: asyncio.Queue[tuple[str, int]] = asyncio.Queue(maxsize=capacity)
        self.terminal: dict | None = None
        self.dropped = 0

    def offer(self, status: str, invocation_id: int) -> None:
        """Discard excess progress rather than blocking agent execution."""
        try:
            self.progress.put_nowait((status, invocation_id))
        except asyncio.QueueFull:
            self.dropped += 1

    def complete(self, response: dict) -> None:
        """Keep a single authoritative result outside the progress queue."""
        self.terminal = response


class ToolEvents(AsyncCallbackHandler):
    """Keep only tool identity and lifecycle, never callback payloads."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []
        self.end_statuses: list[tuple[str, str | None, str | None]] = []
        self.started = asyncio.Event()

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.items.append(("start", str(run_id), serialized.get("name", "")))
        self.started.set()

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self.items.append(("end", str(run_id), getattr(output, "name", "")))
        self.end_statuses.append(
            (str(run_id), getattr(output, "status", None), getattr(output, "tool_call_id", None))
        )

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self.items.append(("error", str(run_id), ""))


@pytest.fixture
async def memory_manager(tmp_path: Path) -> AsyncIterator[ConversationMemoryManager]:
    """Open isolated checkpoint storage for lifecycle probes."""
    manager = await ConversationMemoryManager.open(
        SessionMemoryConfig(enabled=True, db_path=str(tmp_path / "stream-gate0.db"))
    )
    assert manager is not None
    try:
        yield manager
    finally:
        await manager.close()


def _attach_callback(agent: WineAgent, events: ToolEvents) -> None:
    """Inject a callback at the existing WineAgent request-config seam."""
    original = agent._build_runnable_config

    def with_callback(trace_context: dict[str, str] | None, tool_execution_report: Any = None) -> Any:
        config = original(trace_context, tool_execution_report)
        config["callbacks"] = [events]
        return config

    agent._build_runnable_config = with_callback


def _probe_logical_tool_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporarily compose observation outside the M9B wrapper under test."""

    def compose(snapshot: Any, policy: Any, controller: Any) -> Any:
        existing = build_async_tool_execution_wrapper(snapshot, policy, controller)
        selected = {definition.metadata.name for definition in snapshot.definitions}

        async def observe(request: Any, execute: Any) -> Any:
            config = request.runtime.config
            sink = config.get("configurable", {}).get("_gate0_progress_sink")
            if sink is None:
                return await existing(request, execute)
            call = request.tool_call
            name = call["name"] if call["name"] in selected else "unknown"
            call_id = call["id"]
            sink.append(("start", call_id, name))
            try:
                result = await existing(request, execute)
            except BaseException:
                sink.append(("cancel_or_raise", call_id, name))
                raise
            sink.append((getattr(result, "status", "unknown"), call_id, name))
            return result

        return observe

    monkeypatch.setattr("src.agents.intelligent.agent.build_async_tool_execution_wrapper", compose)


def _attach_sink(agent: WineAgent, sink: list[tuple[str, str, str]]) -> None:
    """Attach a request-local observer without replacing M9B report configuration."""
    original = agent._build_runnable_config

    def with_sink(trace_context: dict[str, str] | None, tool_execution_report: Any = None) -> Any:
        config = original(trace_context, tool_execution_report)
        config.setdefault("configurable", {})["_gate0_progress_sink"] = sink
        return config

    agent._build_runnable_config = with_sink


def test_gate0_bounded_progress_does_not_block_terminal_delivery() -> None:
    """A stopped reader cannot retain unbounded progress or hide the final result."""
    sink = BoundedProgressSink()
    for index in range(100):
        sink.offer("started", index)
    sink.complete({"answer": "Safe final"})

    assert sink.progress.qsize() == 16
    assert sink.dropped == 84
    assert sink.terminal == {"answer": "Safe final"}


@pytest.mark.asyncio
async def test_callbacks_observe_parallel_and_repeated_tools_without_second_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callbacks identify each invocation and emit start before tool completion."""
    release = asyncio.Event()
    started = asyncio.Event()
    executions: list[str] = []

    @tool
    async def sample_tool(value: str) -> str:
        """Wait until the probe releases all concurrent tool calls."""
        executions.append(value)
        if len(executions) == 2:
            started.set()
        await release.wait()
        return f"secret-result-{value}"

    definition = _tool_definition(sample_tool, ToolCategory.CELLAR)
    planned = AIMessage(
        content="",
        tool_calls=[
            {"name": sample_tool.name, "args": {"value": "secret-input-one"}, "id": "call-one"},
            {"name": sample_tool.name, "args": {"value": "secret-input-two"}, "id": "call-two"},
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[planned, AIMessage(content="Safe final answer")])
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    _probe_logical_tool_boundary(monkeypatch)
    agent = WineAgent(
        llm=model,
        tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
    )
    events = ToolEvents()
    _attach_callback(agent, events)
    progress: list[tuple[str, str, str]] = []
    _attach_sink(agent, progress)

    task = asyncio.create_task(agent.ainvoke("Use the tool twice"))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.wait_for(events.started.wait(), timeout=2)
        assert [kind for kind, _, _ in events.items] == ["start", "start"]
        assert {entry[:2] for entry in progress} == {
            ("start", "call-one"),
            ("start", "call-two"),
        }
    finally:
        release.set()
    result = await asyncio.wait_for(task, timeout=2)

    starts = {run_id for kind, run_id, _ in events.items if kind == "start"}
    ends = {run_id for kind, run_id, _ in events.items if kind == "end"}
    assert len(starts) == 2
    assert starts == ends
    assert bound_model.ainvoke.await_count == 2
    assert result["final_answer"] == "Safe final answer"
    assert set(executions) == {"secret-input-one", "secret-input-two"}
    assert {entry[:2] for entry in progress if entry[0] == "success"} == {
        ("success", "call-one"),
        ("success", "call-two"),
    }
    assert "secret-input" not in repr(events.items)
    assert "secret-result" not in repr(events.items)
    assert "secret-input" not in repr(progress)
    assert "secret-result" not in repr(progress)


@pytest.mark.asyncio
async def test_callbacks_preserve_zero_tool_and_empty_final_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final-only path still uses existing WineAgent finalization."""
    registry = _registry_from_definitions(monkeypatch, ())
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[AIMessage(content="Direct"), AIMessage(content="")])
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    agent = WineAgent(llm=model, tool_registry=registry)
    events = ToolEvents()
    _attach_callback(agent, events)

    first = await agent.ainvoke("What is tannin?")
    second = await agent.ainvoke("What is acidity?")

    assert first["final_answer"] == "Direct"
    assert second["final_answer"] == EMPTY_FINAL_ANSWER_RETRY
    assert second["terminal_outcome"] == "empty_final_answer"
    assert events.items == []
    assert bound_model.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_callback_probe_records_safe_failure_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback completion must be classified using the safe tool result."""

    @tool
    async def failing_tool(value: str) -> str:
        """Raise a synthetic private failure."""
        raise RuntimeError(f"secret-failure-{value}")

    @tool
    async def slow_tool(value: str) -> str:
        """Wait until the reviewed M9B deadline expires."""
        await asyncio.Event().wait()
        return value

    definitions = (
        _tool_definition(failing_tool, ToolCategory.CELLAR),
        _tool_definition(slow_tool, ToolCategory.CELLAR),
    )
    planned = AIMessage(
        content="",
        tool_calls=[
            {"name": failing_tool.name, "args": {"value": "private"}, "id": "failed-call"},
            {"name": slow_tool.name, "args": {"value": "private"}, "id": "timeout-call"},
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[planned, AIMessage(content="Safe final")])
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    _probe_logical_tool_boundary(monkeypatch)
    agent = WineAgent(
        llm=model,
        tool_registry=_registry_from_definitions(monkeypatch, definitions),
        tool_execution=ToolExecutionConfig(
            timeout_seconds=ToolTimeoutConfig(fast=0.03, slow=0.03),
        ),
    )
    events = ToolEvents()
    _attach_callback(agent, events)
    progress: list[tuple[str, str, str]] = []
    _attach_sink(agent, progress)

    result = await agent.ainvoke("Exercise failures")

    assert result["final_answer"] == "Safe final"
    assert bound_model.ainvoke.await_count == 2
    assert [kind for kind, _, _ in events.items].count("start") == 2
    # A deadline can cancel the inner tool without delivering its callback error.
    assert [kind for kind, _, _ in events.items].count("error") == 1
    assert events.end_statuses == []
    safe_results = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert {message.tool_call_id for message in safe_results} == {"failed-call", "timeout-call"}
    assert {message.status for message in safe_results} == {"error"}
    assert {event["code"] for event in result["guardrail_events"]} == {
        "tool_terminal_failure",
        "tool_deadline_exceeded",
    }
    assert {item[:2] for item in progress if item[0] == "error"} == {
        ("error", "failed-call"),
        ("error", "timeout-call"),
    }
    assert "secret-failure" not in repr(events.items)
    assert "private" not in repr(events.items)
    assert "private" not in repr(progress)


@pytest.mark.asyncio
async def test_callback_probe_records_retry_attempts_under_one_logical_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry must not become a second UI tool invocation."""
    attempts = 0

    @tool
    async def busy_tool(value: str) -> str:
        """Fail once with structured SQLite contention, then succeed."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = sqlite3.OperationalError("private contention")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        return value

    planned = AIMessage(
        content="",
        tool_calls=[{"name": busy_tool.name, "args": {"value": "private"}, "id": "logical-call"}],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[planned, AIMessage(content="Recovered")])
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    _probe_logical_tool_boundary(monkeypatch)
    agent = WineAgent(
        llm=model,
        tool_registry=_registry_from_definitions(
            monkeypatch, (_tool_definition(busy_tool, ToolCategory.CELLAR),)
        ),
        tool_execution=ToolExecutionConfig(
            timeout_seconds=ToolTimeoutConfig(fast=0.2, slow=0.2),
            retry=ToolRetryConfig(delay_seconds=0, min_remaining_seconds=0.01),
        ),
    )
    events = ToolEvents()
    _attach_callback(agent, events)
    progress: list[tuple[str, str, str]] = []
    _attach_sink(agent, progress)

    result = await agent.ainvoke("Retry one tool")

    assert attempts == 2
    assert result["final_answer"] == "Recovered"
    assert [event["code"] for event in result["guardrail_events"]] == [
        "tool_retry_started",
        "tool_retry_succeeded",
    ]
    assert len([item for item in events.items if item[0] == "start"]) == 2
    assert events.end_statuses[-1][1:] == ("success", "logical-call")
    assert progress == [
        ("start", "logical-call", "busy_tool"),
        ("success", "logical-call", "busy_tool"),
    ]
    assert bound_model.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_callback_path_preserves_thread_append_replace_and_failed_turn(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """Progress observation must not change M8 turn commits."""
    registry = _registry_from_definitions(monkeypatch, ())
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(
        side_effect=[AIMessage(content="First"), AIMessage(content="Replacement"), AIMessage(content="")]
    )
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    agent = WineAgent(llm=model, tool_registry=registry, memory_manager=memory_manager)
    events = ToolEvents()
    _attach_callback(agent, events)
    thread_id = "00000000-0000-4000-8000-000000000070"

    first = await agent.ainvoke("First", thread_id=thread_id)
    after_first = await memory_manager.get_thread(thread_id)
    replaced = await agent.ainvoke("Replace", thread_id=thread_id, thread_action="replace_last")
    after_replace = await memory_manager.get_thread(thread_id)
    failed = await agent.ainvoke("Blank", thread_id=thread_id)
    after_failed = await memory_manager.get_thread(thread_id)

    assert first["final_answer"] == "First"
    assert replaced["final_answer"] == "Replacement"
    assert failed["final_answer"] == EMPTY_FINAL_ANSWER_RETRY
    assert after_first.completed_turns == after_replace.completed_turns == after_failed.completed_turns == 1
    assert after_first.active_checkpoint_id != after_replace.active_checkpoint_id
    assert after_replace.active_checkpoint_id == after_failed.active_checkpoint_id
    assert events.items == []


@pytest.mark.asyncio
async def test_logical_observer_preserves_precommit_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """Cancelling a running tool leaves the last committed thread pointer intact."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    @tool
    async def waiting_tool(value: str) -> str:
        """Hold one cooperative tool call until the request is cancelled."""
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return value

    planned = AIMessage(
        content="",
        tool_calls=[{"name": waiting_tool.name, "args": {"value": "private"}, "id": "pending"}],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[AIMessage(content="Committed"), planned])
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    _probe_logical_tool_boundary(monkeypatch)
    agent = WineAgent(
        llm=model,
        tool_registry=_registry_from_definitions(
            monkeypatch, (_tool_definition(waiting_tool, ToolCategory.CELLAR),)
        ),
        memory_manager=memory_manager,
    )
    progress: list[tuple[str, str, str]] = []
    _attach_sink(agent, progress)
    thread_id = "00000000-0000-4000-8000-000000000071"
    await agent.ainvoke("First", thread_id=thread_id)
    before = await memory_manager.get_thread(thread_id)

    task = asyncio.create_task(agent.ainvoke("Run tool", thread_id=thread_id))
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert ("start", "pending", "waiting_tool") in progress
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    after = await memory_manager.get_thread(thread_id)

    assert cancelled.is_set()
    assert after.active_checkpoint_id == before.active_checkpoint_id
    assert after.completed_turns == before.completed_turns == 1
    assert progress[-1] == ("cancel_or_raise", "pending", "waiting_tool")
    assert "private" not in repr(progress)


@pytest.mark.asyncio
async def test_delivery_cancellation_after_commit_does_not_undo_turn(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """A response already committed remains committed if its sender is cancelled."""
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(content="Completed"))
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=model,
        tool_registry=_registry_from_definitions(monkeypatch, ()),
        memory_manager=memory_manager,
    )
    thread_id = "00000000-0000-4000-8000-000000000072"
    result = await agent.ainvoke("Commit this", thread_id=thread_id)
    committed = await memory_manager.get_thread(thread_id)

    sender = asyncio.create_task(asyncio.Event().wait())
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender
    after_disconnect = await memory_manager.get_thread(thread_id)

    assert result["final_answer"] == "Completed"
    assert after_disconnect.active_checkpoint_id == committed.active_checkpoint_id
    assert after_disconnect.completed_turns == 1
