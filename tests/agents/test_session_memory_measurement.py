"""Deterministic measurements for the M8 session-memory rollout gate."""

import json
import math
import statistics
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.agents.guardrails import (
    CallBudgetConfig,
    LoopDetectionConfig,
    RelevanceConfig,
    ToolExecutionConfig,
)
from src.agents.intelligent.agent import WineAgent
from src.agents.memory import ConversationMemoryManager, SessionMemoryConfig
from src.agents.prompt_registry import RenderedPrompt
from src.agents.provenance import build_agent_policy_provenance
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot
from src.utils import logger


def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> ToolRegistry:
    """Return a deterministic empty tool registry and fixed prompt."""
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


def _deterministic_input_tokens(messages: list[BaseMessage]) -> int:
    """Count whitespace-delimited tokens for the controlled measurement model."""
    return sum(len(str(message.content).split()) for message in messages)


def _measured_agent(
    monkeypatch: pytest.MonkeyPatch,
    registry: ToolRegistry,
    inputs: list[dict[str, int]],
    memory_manager: ConversationMemoryManager | None = None,
) -> WineAgent:
    """Build an agent whose model reports deterministic input-token usage."""

    def respond(messages: list[BaseMessage]) -> AIMessage:
        measurement = {
            "input_tokens": _deterministic_input_tokens(messages),
            "human_messages": sum(isinstance(message, HumanMessage) for message in messages),
        }
        inputs.append(measurement)
        return AIMessage(
            content="wine answer",
            usage_metadata={
                "input_tokens": measurement["input_tokens"],
                "output_tokens": 2,
                "total_tokens": measurement["input_tokens"] + 2,
            },
        )

    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=respond)
    llm = MagicMock()
    llm.bind_tools.return_value = bound_model
    return WineAgent(
        llm=llm,
        tool_registry=registry,
        memory_manager=memory_manager,
        session_memory=(memory_manager.config if memory_manager is not None else SessionMemoryConfig()),
    )


def _bounded_history(completed_turns: int, max_prior_turns: int) -> list[dict[str, str]]:
    """Return the client history equivalent to the threaded complete-turn window."""
    prior_turns = min(completed_turns - 1, max_prior_turns)
    history: list[dict[str, str]] = []
    for _ in range(prior_turns):
        history.extend(
            [
                {"role": "human", "content": "wine question"},
                {"role": "ai", "content": "wine answer"},
            ]
        )
    return history


async def _checkpoint_count(manager: ConversationMemoryManager, thread_id: str) -> int:
    """Count checkpoints through the saver's public async listing API."""
    count = 0
    async for _ in manager.saver.alist(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    ):
        count += 1
    return count


def test_stateless_and_threaded_provenance_differs_only_by_memory_policy() -> None:
    """Record the M5 policy distinction without making a model call."""
    shared = {
        "call_budget": CallBudgetConfig(),
        "loop_detection": LoopDetectionConfig(),
        "relevance": RelevanceConfig(),
        "tool_execution": ToolExecutionConfig(),
    }
    stateless = build_agent_policy_provenance(
        **shared,
        session_memory=SessionMemoryConfig(enabled=False),
    )
    threaded = build_agent_policy_provenance(
        **shared,
        session_memory=SessionMemoryConfig(enabled=True),
    )
    stateless_config = stateless.config.model_dump(mode="json")
    threaded_config = threaded.config.model_dump(mode="json")

    assert stateless.hash != threaded.hash
    assert stateless_config["session_memory"]["enabled"] is False
    assert threaded_config["session_memory"]["enabled"] is True
    threaded_config["session_memory"]["enabled"] = False
    assert threaded_config == stateless_config
    logger.info(
        "M8 provenance comparison: stateless_hash=%s threaded_hash=%s changed_field=session_memory.enabled",
        stateless.hash,
        threaded.hash,
    )


@pytest.mark.parametrize("completed_turns", [10, 50])
async def test_threaded_measurement_stays_bounded_without_extra_model_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    completed_turns: int,
) -> None:
    """Record bounded input, latency, checkpoint, and SQLite storage evidence."""
    max_prior_turns = 10
    db_path = tmp_path / f"memory-{completed_turns}.db"
    manager = await ConversationMemoryManager.open(
        SessionMemoryConfig(
            enabled=True,
            db_path=str(db_path),
            max_prior_turns=max_prior_turns,
        )
    )
    assert manager is not None
    threaded_inputs: list[dict[str, int]] = []
    stateless_inputs: list[dict[str, int]] = []
    registry = _empty_registry(monkeypatch)
    threaded_agent = _measured_agent(monkeypatch, registry, threaded_inputs, manager)
    thread_id = f"measurement-{completed_turns}"
    turn_latencies_ms: list[float] = []

    try:
        for _ in range(completed_turns):
            started = time.perf_counter()
            result = await threaded_agent.ainvoke("wine question", thread_id=thread_id)
            turn_latencies_ms.append((time.perf_counter() - started) * 1_000)
            assert result["llm_call_count"] == 1

        stateless_agent = _measured_agent(monkeypatch, registry, stateless_inputs)
        stateless_started = time.perf_counter()
        stateless_result = await stateless_agent.ainvoke(
            "wine question",
            message_history=_bounded_history(completed_turns, max_prior_turns),
        )
        stateless_latency_ms = (time.perf_counter() - stateless_started) * 1_000
        thread = await manager.get_thread(thread_id)
        checkpoints = await _checkpoint_count(manager, thread_id)
        await manager.close()
        sqlite_bytes = db_path.stat().st_size

        assert thread is not None
        assert thread.completed_turns == completed_turns
        assert len(threaded_inputs) == completed_turns
        assert stateless_result["llm_call_count"] == 1
        assert len(stateless_inputs) == 1
        assert threaded_inputs[-1]["input_tokens"] == stateless_inputs[-1]["input_tokens"]
        assert threaded_inputs[-1]["human_messages"] <= max_prior_turns + 1
        assert checkpoints > completed_turns
        assert sqlite_bytes > 0
        assert all(math.isfinite(latency) and latency >= 0 for latency in turn_latencies_ms)

        measurement = {
            "completed_turns": completed_turns,
            "configured_max_prior_turns": max_prior_turns,
            "final_model_input_tokens": threaded_inputs[-1]["input_tokens"],
            "final_model_human_messages": threaded_inputs[-1]["human_messages"],
            "threaded_llm_calls_per_turn": 1,
            "stateless_llm_calls_for_equivalent_turn": stateless_result["llm_call_count"],
            "mean_turn_latency_ms": round(statistics.fmean(turn_latencies_ms), 3),
            "max_turn_latency_ms": round(max(turn_latencies_ms), 3),
            "equivalent_stateless_latency_ms": round(stateless_latency_ms, 3),
            "sqlite_bytes": sqlite_bytes,
            "checkpoint_count": checkpoints,
        }
        logger.info("M8 deterministic measurement: %s", json.dumps(measurement, sort_keys=True))
    finally:
        await manager.close()
