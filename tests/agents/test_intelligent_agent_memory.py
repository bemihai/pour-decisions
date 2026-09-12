"""Threaded WineAgent state, pruning, isolation, and safety tests."""

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import NodeCancelledError

from src.agents.guardrails import RELEVANCE_DEFLECTED_EVENT_CODE, TOOL_EXECUTION_REPORT_CONFIG_KEY
from src.agents.intelligent.agent import WineAgent, _complete_turn_removals, create_wine_agent
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


@pytest.fixture
async def memory_manager(tmp_path: Path) -> AsyncIterator[ConversationMemoryManager]:
    """Provide isolated async checkpoint storage with a small pruning window."""
    manager = await ConversationMemoryManager.open(
        SessionMemoryConfig(
            enabled=True,
            db_path=str(tmp_path / "agent-memory.db"),
            max_prior_turns=2,
        )
    )
    assert manager is not None
    try:
        yield manager
    finally:
        await manager.close()


def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    """Return a deterministic empty tool registry and prompt."""
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
    registry.select.return_value = ToolSelectionSnapshot(definitions=(), readiness=())
    return registry


def _agent(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
    response_factory: Callable[[list[BaseMessage]], AIMessage],
) -> tuple[WineAgent, AsyncMock]:
    """Build a threaded agent whose async model response is controlled by a test."""
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=response_factory)
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    return (
        WineAgent(
            llm=llm,
            tool_registry=_empty_registry(monkeypatch),
            memory_manager=memory_manager,
        ),
        bound_model.ainvoke,
    )


async def _thread_messages(agent: WineAgent, thread_id: str) -> list[BaseMessage]:
    """Read the committed graph messages through the graph state API."""
    thread = await agent.memory_manager.get_thread(thread_id)
    assert thread is not None
    assert thread.active_checkpoint_id is not None
    snapshot = await agent.threaded_agent.aget_state(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": thread.active_checkpoint_id,
            }
        }
    )
    return snapshot.values["messages"]


@pytest.mark.asyncio
async def test_threaded_turns_accumulate_and_threads_stay_isolated(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """A thread should supply only its own committed history to later model calls."""
    seen_humans: list[list[str]] = []

    def respond(messages: list[BaseMessage]) -> AIMessage:
        humans = [message.content for message in messages if isinstance(message, HumanMessage)]
        seen_humans.append(humans)
        return AIMessage(content=f"Seen: {', '.join(humans)}")

    agent, _model = _agent(monkeypatch, memory_manager, respond)

    await agent.ainvoke("first", thread_id="thread-a")
    second = await agent.ainvoke(
        "second",
        message_history=[{"role": "human", "content": "ignored-client-history"}],
        thread_id="thread-a",
    )
    await agent.ainvoke("other", thread_id="thread-b")

    assert seen_humans == [["first"], ["first", "second"], ["other"]]
    assert [message.content for message in second["messages"]] == ["second", "Seen: first, second"]
    assert second["llm_call_count"] == 1


@pytest.mark.asyncio
async def test_threaded_graph_adds_only_the_memory_entry_to_shared_topology(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """Stateless topology should stay intact while threaded setup adds one entry node."""
    agent, _model = _agent(
        monkeypatch,
        memory_manager,
        lambda _messages: AIMessage(content="answer"),
    )
    stateless = agent.agent.get_graph()
    threaded = agent.threaded_agent.get_graph()
    stateless_edges = {(edge.source, edge.target, edge.conditional) for edge in stateless.edges}
    threaded_edges = {(edge.source, edge.target, edge.conditional) for edge in threaded.edges}

    assert set(threaded.nodes) == set(stateless.nodes) | {"prepare_threaded_turn"}
    assert threaded_edges == (
        stateless_edges - {("__start__", "check_relevance", False)}
    ) | {
        ("__start__", "prepare_threaded_turn", False),
        ("prepare_threaded_turn", "check_relevance", False),
    }


@pytest.mark.asyncio
async def test_factory_passes_memory_manager_to_threaded_agent(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """The shared factory should opt into threaded compilation only when requested."""
    bound_model = MagicMock()
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model

    agent = create_wine_agent(
        llm=llm,
        tool_registry=_empty_registry(monkeypatch),
        memory_manager=memory_manager,
    )

    assert agent.memory_manager is memory_manager
    assert agent.threaded_agent is not None


@pytest.mark.asyncio
async def test_pruning_keeps_prior_complete_turns_plus_current(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """The checkpoint should retain two complete prior turns and the current turn."""
    def respond(messages: list[BaseMessage]) -> AIMessage:
        current_human = next(message for message in reversed(messages) if isinstance(message, HumanMessage))
        return AIMessage(content=f"answer-{current_human.content}")

    agent, _model = _agent(monkeypatch, memory_manager, respond)

    for query in ("one", "two", "three", "four"):
        await agent.ainvoke(query, thread_id="bounded-thread")

    messages = await _thread_messages(agent, "bounded-thread")
    assert [message.content for message in messages] == [
        "two",
        "answer-two",
        "three",
        "answer-three",
        "four",
        "answer-four",
    ]


def test_complete_turn_pruning_preserves_tool_call_and_result_pairs() -> None:
    """Reducer removals should target every message in an expired tool-backed turn."""
    messages = [
        HumanMessage(content="old", id="human-old"),
        AIMessage(
            content="",
            id="ai-tool-old",
            tool_calls=[{"name": "lookup", "args": {}, "id": "call-old", "type": "tool_call"}],
        ),
        ToolMessage(content="result", tool_call_id="call-old", id="tool-old"),
        AIMessage(content="answer", id="ai-answer-old"),
        HumanMessage(content="recent", id="human-recent"),
        AIMessage(content="recent answer", id="ai-recent"),
        HumanMessage(content="current", id="human-current"),
    ]

    removals = _complete_turn_removals(messages, max_prior_turns=1)

    assert [removal.id for removal in removals] == [
        "human-old",
        "ai-tool-old",
        "tool-old",
        "ai-answer-old",
    ]


@pytest.mark.asyncio
async def test_failed_threaded_turn_is_not_visible_to_next_success(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """A failed branch must not become conversation history for a later append."""
    seen_humans: list[list[str]] = []

    def respond(messages: list[BaseMessage]) -> AIMessage:
        humans = [message.content for message in messages if isinstance(message, HumanMessage)]
        seen_humans.append(humans)
        if humans[-1] == "fail":
            raise RuntimeError("synthetic model failure")
        return AIMessage(content="ok")

    agent, _model = _agent(monkeypatch, memory_manager, respond)
    await agent.ainvoke("committed", thread_id="failure-thread")

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        await agent.ainvoke("fail", thread_id="failure-thread")

    await agent.ainvoke("recovery", thread_id="failure-thread")

    assert seen_humans == [
        ["committed"],
        ["committed", "fail"],
        ["committed", "recovery"],
    ]


@pytest.mark.asyncio
async def test_replace_last_rebuilds_only_the_latest_turn(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """Regeneration should branch from the previous committed checkpoint."""
    seen_humans: list[list[str]] = []

    def respond(messages: list[BaseMessage]) -> AIMessage:
        humans = [message.content for message in messages if isinstance(message, HumanMessage)]
        seen_humans.append(humans)
        return AIMessage(content=f"answer:{humans[-1]}")

    agent, _model = _agent(monkeypatch, memory_manager, respond)
    await agent.ainvoke("original", thread_id="replace-thread")
    replacement = await agent.ainvoke(
        "replacement",
        thread_id="replace-thread",
        thread_action="replace_last",
    )

    thread = await memory_manager.get_thread("replace-thread")
    stored_messages = await _thread_messages(agent, "replace-thread")
    assert seen_humans == [["original"], ["replacement"]]
    assert replacement["final_answer"] == "answer:replacement"
    assert [message.content for message in stored_messages] == ["replacement", "answer:replacement"]
    assert thread is not None
    assert thread.completed_turns == 1


@pytest.mark.asyncio
async def test_cancelled_threaded_turn_is_not_visible_to_next_success(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """A cancelled branch must not become conversation history for a later append."""
    seen_humans: list[list[str]] = []

    def respond(messages: list[BaseMessage]) -> AIMessage:
        humans = [message.content for message in messages if isinstance(message, HumanMessage)]
        seen_humans.append(humans)
        if humans[-1] == "cancel":
            raise asyncio.CancelledError
        return AIMessage(content="ok")

    agent, _model = _agent(monkeypatch, memory_manager, respond)
    await agent.ainvoke("committed", thread_id="cancel-thread")

    with pytest.raises(NodeCancelledError):
        await agent.ainvoke("cancel", thread_id="cancel-thread")

    await agent.ainvoke("recovery", thread_id="cancel-thread")

    assert seen_humans == [
        ["committed"],
        ["committed", "cancel"],
        ["committed", "recovery"],
    ]


@pytest.mark.asyncio
async def test_threaded_request_state_starts_clean_each_turn(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """Call counts and guardrail events from one turn must not carry into the next."""
    agent, model = _agent(
        monkeypatch,
        memory_manager,
        lambda _messages: AIMessage(content="wine answer"),
    )

    off_topic = await agent.ainvoke(
        "What is the weather in Bucharest tomorrow?",
        thread_id="reset-thread",
    )
    on_topic = await agent.ainvoke("What is tannin?", thread_id="reset-thread")

    assert off_topic["llm_call_count"] == 0
    assert off_topic["guardrail_events"] == [
        {"code": RELEVANCE_DEFLECTED_EVENT_CODE, "route": "deflect"}
    ]
    assert on_topic["llm_call_count"] == 1
    assert on_topic["tool_call_history"] == []
    assert on_topic["guardrail_events"] == []
    model.assert_awaited_once()


@pytest.mark.asyncio
async def test_threaded_tool_loop_history_starts_clean_each_turn(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """The same legitimate tool call on later turns must not look like a loop."""
    calls: list[str] = []

    @tool
    async def repeatable_lookup(value: str) -> str:
        """Return a deterministic lookup result."""
        calls.append(value)
        return f"result:{value}"

    definition = ToolDefinition(
        tool=repeatable_lookup,
        metadata=ToolMetadata(
            name=repeatable_lookup.name,
            category=ToolCategory.CELLAR,
            tier=ToolTier.CORE,
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Exercise repeatable calls across threaded turns.",
        ),
    )
    registry = _empty_registry(monkeypatch)
    registry.select.return_value = ToolSelectionSnapshot(definitions=(definition,), readiness=())
    next_call_id = 0

    def respond(messages: list[BaseMessage]) -> AIMessage:
        nonlocal next_call_id
        if isinstance(messages[-1], HumanMessage):
            next_call_id += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": repeatable_lookup.name,
                        "args": {"value": "same"},
                        "id": f"call-{next_call_id}",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="complete")

    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=respond)
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=llm,
        tool_registry=registry,
        memory_manager=memory_manager,
    )

    first = await agent.ainvoke("lookup once", thread_id="loop-reset-thread")
    second = await agent.ainvoke("lookup again", thread_id="loop-reset-thread")

    assert calls == ["same", "same"]
    assert first["llm_call_count"] == second["llm_call_count"] == 2
    assert len(first["tool_call_history"]) == len(second["tool_call_history"]) == 1
    assert first["guardrail_events"] == second["guardrail_events"] == []


def test_checkpoint_and_request_configurable_values_are_merged() -> None:
    """The checkpointer selector and request-local M9B report must coexist."""
    report = object()

    merged = WineAgent._merge_runnable_configs(
        {
            "configurable": {
                "thread_id": "thread-a",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-a",
            }
        },
        {
            "recursion_limit": 17,
            "metadata": {"request_id": "request-a"},
            "configurable": {TOOL_EXECUTION_REPORT_CONFIG_KEY: report},
        },
    )

    assert merged["recursion_limit"] == 17
    assert merged["metadata"] == {"request_id": "request-a"}
    assert merged["configurable"] == {
        "thread_id": "thread-a",
        "checkpoint_ns": "",
        "checkpoint_id": "checkpoint-a",
        TOOL_EXECUTION_REPORT_CONFIG_KEY: report,
    }


@pytest.mark.asyncio
async def test_threaded_ai_text_is_sanitized_before_checkpoint_write(
    monkeypatch: pytest.MonkeyPatch,
    memory_manager: ConversationMemoryManager,
) -> None:
    """Sensitive AI text must be redacted in both the result and durable state."""
    secret = "M06A_SYNTHETIC_PROVIDER_TOKEN"
    agent, _model = _agent(
        monkeypatch,
        memory_manager,
        lambda _messages: AIMessage(content=f"Set {secret} before retrying."),
    )

    result = await agent.ainvoke("show setup", thread_id="sanitized-thread")
    stored_messages = await _thread_messages(agent, "sanitized-thread")

    assert secret not in result["final_answer"]
    assert all(secret not in str(message.content) for message in stored_messages)


@pytest.mark.asyncio
async def test_thread_id_requires_a_configured_memory_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Threaded invocation should fail explicitly when no threaded graph exists."""
    bound_model = MagicMock()
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    agent = WineAgent(llm=llm, tool_registry=_empty_registry(monkeypatch))

    with pytest.raises(RuntimeError, match="enabled conversation memory manager"):
        await agent.ainvoke("hello", thread_id="missing-manager")

    bound_model.ainvoke.assert_not_called()
