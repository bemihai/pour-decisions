"""Parity and lifecycle tests for request-local WineAgent tool progress."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool, tool
import pytest

from src.agents.guardrails import (
    TOOL_EXECUTION_REPORT_CONFIG_KEY,
    TOOL_PROGRESS_REPORTER_CONFIG_KEY,
    ToolExecutionReport,
    ToolProgressEvent,
    ToolProgressReporter,
    ToolProgressStatus,
)
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
    """Return an isolated immutable selected-tool snapshot."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: RenderedPrompt(
            name="intelligent_agent_system",
            content="Progress test prompt.",
            source_hash="sha256:progress-source",
            rendered_hash="sha256:progress-rendered",
            label="",
        ),
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = ToolSelectionSnapshot(definitions=definitions, readiness=())
    return registry


def _tool_definition(tool_instance: BaseTool) -> ToolDefinition:
    """Attach the metadata required by the production execution wrapper."""
    return ToolDefinition(
        tool=tool_instance,
        metadata=ToolMetadata(
            name=tool_instance.name,
            category=ToolCategory.CELLAR,
            tier=ToolTier.CORE,
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability=f"Exercise {tool_instance.name} progress.",
        ),
    )


def _normalized_result(result: dict) -> dict:
    """Detach generated message identities while retaining observable semantics."""
    normalized = dict(result)
    normalized["messages"] = [
        (
            type(message).__name__,
            message.content,
            getattr(message, "tool_calls", None),
            getattr(message, "status", None),
        )
        for message in result["messages"]
    ]
    return normalized


def _drain(reporter: ToolProgressReporter) -> list[ToolProgressEvent]:
    """Remove all progress events currently available to the consumer."""
    events: list[ToolProgressEvent] = []
    while reporter.pending_count:
        events.append(reporter.get_nowait())
    return events


@pytest.fixture
async def memory_manager(tmp_path: Path) -> AsyncIterator[ConversationMemoryManager]:
    """Open isolated durable checkpoint storage for cancellation tests."""
    manager = await ConversationMemoryManager.open(
        SessionMemoryConfig(enabled=True, db_path=str(tmp_path / "progress-memory.db"))
    )
    assert manager is not None
    try:
        yield manager
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_progress_reporter_preserves_complete_stateless_result_and_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attaching progress must not change final state, model attempts, or tool outcomes."""

    @tool
    async def cellar_probe(value: str) -> str:
        """Return a deterministic private value through the real ToolNode."""
        return f"private-result-{value}"

    definition = _tool_definition(cellar_probe)

    def build_agent() -> tuple[WineAgent, AsyncMock]:
        planned = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": cellar_probe.name,
                    "args": {"value": "private-input"},
                    "id": "private-call-id",
                }
            ],
        )
        bound_model = MagicMock()
        bound_model.ainvoke = AsyncMock(
            side_effect=[planned, AIMessage(content="Stable final answer")]
        )
        model = MagicMock()
        model.bind_tools.return_value = bound_model
        return (
            WineAgent(
                llm=model,
                tool_registry=_registry_from_definitions(monkeypatch, (definition,)),
            ),
            bound_model.ainvoke,
        )

    baseline_agent, baseline_model = build_agent()
    observed_agent, observed_model = build_agent()
    reporter = ToolProgressReporter()

    baseline = await baseline_agent.ainvoke("Run the cellar probe")
    observed = await observed_agent.ainvoke("Run the cellar probe", progress_reporter=reporter)

    assert _normalized_result(observed) == _normalized_result(baseline)
    assert baseline_model.await_count == observed_model.await_count == 2
    assert _drain(reporter) == [
        ToolProgressEvent(1, cellar_probe.name, ToolProgressStatus.STARTED),
        ToolProgressEvent(1, cellar_probe.name, ToolProgressStatus.COMPLETED),
    ]
    assert "private-input" not in repr(reporter)
    assert "private-result" not in repr(reporter)
    assert "private-call-id" not in repr(reporter)


def test_progress_config_preserves_provenance_report_and_checkpoint_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reporter must compose with trace metadata, M9B evidence, and memory config."""
    bound_model = MagicMock()
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    agent = WineAgent(llm=model, tool_registry=_registry_from_definitions(monkeypatch, ()))
    report = ToolExecutionReport()
    reporter = ToolProgressReporter()

    baseline = agent._build_runnable_config({"request_id": "request-a"}, report)
    observed = agent._build_runnable_config({"request_id": "request-a"}, report, reporter)
    merged = agent._merge_runnable_configs(
        {
            "configurable": {
                "thread_id": "thread-a",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-a",
            }
        },
        observed,
    )

    assert observed["recursion_limit"] == baseline["recursion_limit"]
    assert observed["metadata"] == baseline["metadata"]
    assert observed["configurable"][TOOL_EXECUTION_REPORT_CONFIG_KEY] is report
    assert observed["configurable"][TOOL_PROGRESS_REPORTER_CONFIG_KEY] is reporter
    assert merged["configurable"] == {
        "thread_id": "thread-a",
        "checkpoint_ns": "",
        "checkpoint_id": "checkpoint-a",
        TOOL_EXECUTION_REPORT_CONFIG_KEY: report,
        TOOL_PROGRESS_REPORTER_CONFIG_KEY: reporter,
    }


@pytest.mark.asyncio
async def test_progress_cancellation_releases_tool_and_preserves_committed_thread(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """A cancelled observed turn must stop cooperative work and leave memory unchanged."""
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    @tool
    async def waiting_tool(value: str) -> str:
        """Wait until the owning request cancels this cooperative call."""
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return value

    planned = AIMessage(
        content="",
        tool_calls=[
            {
                "name": waiting_tool.name,
                "args": {"value": "private-input"},
                "id": "private-call-id",
            }
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[AIMessage(content="Committed"), planned])
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=model,
        tool_registry=_registry_from_definitions(monkeypatch, (_tool_definition(waiting_tool),)),
        memory_manager=memory_manager,
    )
    thread_id = "00000000-0000-4000-8000-000000000107"
    await agent.ainvoke("Commit first", thread_id=thread_id)
    before = await memory_manager.get_thread(thread_id)
    assert before is not None
    reporter = ToolProgressReporter()

    request = asyncio.create_task(
        agent.ainvoke("Wait", thread_id=thread_id, progress_reporter=reporter)
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    started = await asyncio.wait_for(reporter.get(), timeout=2)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    after = await memory_manager.get_thread(thread_id)

    assert cancelled.is_set()
    assert after is not None
    assert after.active_checkpoint_id == before.active_checkpoint_id
    assert after.completed_turns == before.completed_turns == 1
    assert started == ToolProgressEvent(1, waiting_tool.name, ToolProgressStatus.STARTED)
    assert reporter.pending_count == 0
    assert "private" not in repr(reporter)
