"""Per-turn local clock grounding for the intelligent wine agent."""

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from omegaconf import OmegaConf

from src.agents.intelligent.agent import (
    AgentClockConfig,
    WineAgent,
    load_agent_clock_config,
)
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
from src.utils import get_config


def _registry(
    monkeypatch: pytest.MonkeyPatch,
    definitions: tuple[ToolDefinition, ...] = (),
) -> ToolRegistry:
    """Return an isolated tool snapshot with stable prompt provenance."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: RenderedPrompt(
            name="intelligent_agent_system",
            content="Stable registered system prompt.",
            source_hash="sha256:stable-source",
            rendered_hash="sha256:stable-rendered",
            label="test",
        ),
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = ToolSelectionSnapshot(definitions=definitions, readiness=())
    return registry


def _definition(tool_instance: BaseTool) -> ToolDefinition:
    """Build deterministic metadata for a local clock-grounding tool."""
    return ToolDefinition(
        tool=tool_instance,
        metadata=ToolMetadata(
            name=tool_instance.name,
            category=ToolCategory.CELLAR,
            tier=ToolTier.CORE,
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Return deterministic local evidence.",
        ),
    )


def _fixed_clock(*values: datetime) -> tuple[Callable[[ZoneInfo], datetime], list[str]]:
    """Return a clock that records requested zones and yields fixed instants."""
    remaining = iter(values)
    requested_zones: list[str] = []

    def read(timezone_info: ZoneInfo) -> datetime:
        requested_zones.append(timezone_info.key)
        return next(remaining)

    return read, requested_zones


def _system_contents(calls: list[list[BaseMessage]]) -> list[str]:
    """Extract the model-only system content from captured calls."""
    assert all(isinstance(messages[0], SystemMessage) for messages in calls)
    return [str(messages[0].content) for messages in calls]


def test_clock_config_defaults_to_reviewed_iana_timezone_and_rejects_unknown_zone() -> None:
    """Clock settings should resolve explicitly and fail with their config path."""
    assert load_agent_clock_config(OmegaConf.create({})).timezone.key == "Europe/Bucharest"
    assert load_agent_clock_config(get_config()).timezone.key == "Europe/Bucharest"

    invalid = OmegaConf.create({"agents": {"clock": {"timezone": "Mars/Olympus_Mons"}}})
    with pytest.raises(
        ValueError,
        match=r"agents\.clock\.timezone is not a recognized IANA timezone: 'Mars/Olympus_Mons'",
    ):
        load_agent_clock_config(invalid)


def test_clock_context_formats_weekday_minute_timezone_and_daylight_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen winter and summer instants should derive their offsets from zoneinfo."""
    clock, requested_zones = _fixed_clock(
        datetime(2026, 1, 15, 10, 5, 59, tzinfo=timezone.utc),
        datetime(2026, 7, 15, 10, 5, 59, tzinfo=timezone.utc),
    )
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch),
        clock_now=clock,
    )

    winter = agent._capture_turn_clock_context().to_model_context()
    summer = agent._capture_turn_clock_context().to_model_context()

    assert winter == (
        "Current local date and time for this turn:\n"
        "- Date: Thursday, 2026-01-15\n"
        "- Time: 12:05\n"
        "- Timezone: Europe/Bucharest\n"
        "- UTC offset: +02:00\n"
        "Interpret relative date expressions such as today, tomorrow, and this year against this snapshot."
    )
    assert "- Time: 13:05" in summer
    assert "- UTC offset: +03:00" in summer
    assert requested_zones == ["Europe/Bucharest", "Europe/Bucharest"]


def test_sync_tool_turn_reuses_one_snapshot_and_next_turn_refreshes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every sync model call in one turn should share one newly captured snapshot."""

    @tool
    def local_lookup(value: str) -> str:
        """Return a deterministic local result."""
        return f"local:{value}"

    first_plan = AIMessage(
        content="",
        tool_calls=[
            {
                "name": local_lookup.name,
                "args": {"value": "one"},
                "id": "clock-call",
                "type": "tool_call",
            }
        ],
    )
    responses = iter(
        [
            first_plan,
            AIMessage(content="First grounded answer."),
            AIMessage(content="Second grounded answer."),
        ]
    )
    calls: list[list[BaseMessage]] = []

    def respond(messages: list[BaseMessage]) -> AIMessage:
        calls.append(list(messages))
        return next(responses)

    bound_model = MagicMock()
    bound_model.invoke.side_effect = respond
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    clock, requested_zones = _fixed_clock(
        datetime(2026, 1, 15, 10, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 16, 10, 6, tzinfo=timezone.utc),
    )
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch, (_definition(local_lookup),)),
        clock_now=clock,
    )
    prompt_identity = agent.execution_provenance.prompts[0]
    tool_contract_hash = agent.execution_provenance.tools.contract_hash

    first = agent.invoke("Use local evidence.")
    second = agent.invoke("Answer directly.")

    system_contents = _system_contents(calls)
    assert system_contents[0] == system_contents[1]
    assert "Thursday, 2026-01-15" in system_contents[0]
    assert "Friday, 2026-01-16" in system_contents[2]
    assert system_contents[2] != system_contents[0]
    assert requested_zones == ["Europe/Bucharest", "Europe/Bucharest"]
    assert first["llm_call_count"] == 2
    assert second["llm_call_count"] == 1
    assert bound_model.invoke.call_count == 3
    assert "Current local date and time" not in repr(first["messages"])
    assert agent.system_prompt == "Stable registered system prompt."
    assert agent.execution_provenance.prompts[0] == prompt_identity
    assert agent.execution_provenance.tools.contract_hash == tool_contract_hash


@pytest.mark.asyncio
async def test_async_turns_receive_fresh_ephemeral_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native async path should refresh the clock without adding model calls."""
    calls: list[list[BaseMessage]] = []

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        calls.append(list(messages))
        return AIMessage(content="Async grounded answer.")

    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=respond)
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    clock, _requested_zones = _fixed_clock(
        datetime(2026, 4, 1, 8, 30, tzinfo=timezone.utc),
        datetime(2026, 4, 1, 8, 31, tzinfo=timezone.utc),
    )
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch),
        clock_now=clock,
    )

    first = await agent.ainvoke("First async turn.")
    second = await agent.ainvoke("Second async turn.")

    system_contents = _system_contents(calls)
    assert "- Time: 11:30" in system_contents[0]
    assert "- Time: 11:31" in system_contents[1]
    assert first["llm_call_count"] == second["llm_call_count"] == 1
    assert bound_model.ainvoke.await_count == 2
    bound_model.invoke.assert_not_called()


def test_streaming_receives_one_turn_clock_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synchronous streaming graph should receive the same runtime context."""
    calls: list[list[BaseMessage]] = []

    def respond(messages: list[BaseMessage]) -> AIMessage:
        calls.append(list(messages))
        return AIMessage(content="Streamed grounded answer.")

    bound_model = MagicMock()
    bound_model.invoke.side_effect = respond
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    clock, requested_zones = _fixed_clock(datetime(2026, 10, 14, 13, 34, tzinfo=timezone.utc))
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch),
        clock_now=clock,
    )

    chunks = list(agent.stream("Stream this turn."))

    assert chunks[-1]["llm_call_count"] == 1
    assert "Wednesday, 2026-10-14" in _system_contents(calls)[0]
    assert "- Time: 16:34" in _system_contents(calls)[0]
    assert requested_zones == ["Europe/Bucharest"]
    assert bound_model.invoke.call_count == 1


@pytest.mark.asyncio
async def test_threaded_checkpoint_excludes_clock_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Runtime clock instructions must not enter durable graph checkpoints."""
    manager = await ConversationMemoryManager.open(
        SessionMemoryConfig(enabled=True, db_path=str(tmp_path / "clock-memory.db"))
    )
    assert manager is not None
    seen_model_messages: list[list[BaseMessage]] = []

    async def respond(messages: list[BaseMessage]) -> AIMessage:
        seen_model_messages.append(list(messages))
        return AIMessage(content="Threaded grounded answer.")

    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=respond)
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    clock, _requested_zones = _fixed_clock(datetime(2026, 6, 20, 9, 45, tzinfo=timezone.utc))
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch),
        memory_manager=manager,
        clock_now=clock,
    )

    try:
        await agent.ainvoke("Remember this wine turn.", thread_id="clock-thread")
        thread = await manager.get_thread("clock-thread")
        assert thread is not None
        assert thread.active_checkpoint_id is not None
        checkpoint_config = {
            "configurable": {
                "thread_id": "clock-thread",
                "checkpoint_ns": "",
                "checkpoint_id": thread.active_checkpoint_id,
            }
        }
        snapshot = await agent.threaded_agent.aget_state(checkpoint_config)
        checkpoint_tuple = await manager.saver.aget_tuple(checkpoint_config)

        assert "Current local date and time" in _system_contents(seen_model_messages)[0]
        assert all(not isinstance(message, SystemMessage) for message in snapshot.values["messages"])
        assert "Current local date and time" not in repr(snapshot.values)
        assert "Current local date and time" not in repr(checkpoint_tuple)
    finally:
        await manager.close()


def test_clock_rejects_naive_injected_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid injected clock should fail before any model work begins."""
    bound_model = MagicMock()
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch),
        clock=AgentClockConfig(timezone=ZoneInfo("Europe/Bucharest")),
        clock_now=lambda _timezone: datetime(2026, 1, 1, 12, 0),
    )

    with pytest.raises(ValueError, match="Agent clock must return a timezone-aware datetime"):
        agent.invoke("What day is it?")
    bound_model.invoke.assert_not_called()
