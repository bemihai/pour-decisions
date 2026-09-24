"""Measure M07 blocking and streaming delivery with deterministic local agents.

The default command runs the Gate 0-approved 30 paired requests without an
external model, provider, database, or retrieval service::

    PYTHONPATH=. uv run python -m src.eval.scripts.streaming_closeout

It logs one ``M07_STREAMING_CLOSEOUT_MEASUREMENT`` JSON object and exits with a
non-zero status when the approved rollout evidence gate fails.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import socket
import statistics
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid5

from fastapi import FastAPI
import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from omegaconf import OmegaConf
import uvicorn

from src.agents.guardrails import ToolProgressEvent, ToolProgressReporter
from src.agents.intelligent.agent import WineAgent
from src.agents.memory import ConversationMemoryManager, SessionMemoryConfig
from src.agents.tools.registry import (
    CostClass,
    LatencyClass,
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolRegistry,
    ToolTier,
)
from src.api.routes import chat as chat_routes
from src.utils import logger


Scenario = Literal["zero_tool", "one_tool", "multi_tool"]
Delivery = Literal["blocking", "streaming"]
_SCENARIOS: tuple[Scenario, ...] = ("zero_tool", "one_tool", "multi_tool")
_THREAD_NAMESPACE = UUID("00000000-0000-4000-8000-000000000704")
_PROGRESS_P95_LIMIT_MS = 250.0
_OVERHEAD_P95_LIMIT_MS = 100.0


@dataclass(frozen=True)
class TokenUsage:
    """Available deterministic token accounting emitted by the fake model."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class SampleMeasurement:
    """One paired blocking/streaming request measurement."""

    scenario: Scenario
    pair_index: int
    blocking_total_ms: float
    streaming_final_ms: float
    streaming_total_ms: float
    added_final_delivery_overhead_ms: float
    time_to_first_useful_progress_ms: float | None
    progress_observation_delay_ms: float | None
    progress_before_tool_completion: bool | None
    event_count: int
    event_bytes: int
    peak_progress_buffer: int
    dropped_progress_events: int
    final_response_equal: bool
    tool_trajectory_equal: bool
    model_attempts_equal: bool
    tool_attempts_equal: bool
    committed_thread_state_equal: bool
    blocking_model_attempts: int
    streaming_model_attempts: int
    blocking_tool_attempts: int
    streaming_tool_attempts: int
    blocking_token_usage: TokenUsage
    streaming_token_usage: TokenUsage


@dataclass(frozen=True)
class Distribution:
    """Compact distribution for one measured duration."""

    minimum_ms: float
    median_ms: float
    p95_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class RolloutGate:
    """Approved deterministic evidence gates for the separate rollout decision."""

    complete_cohort: bool
    final_response_parity: bool
    tool_trajectory_parity: bool
    model_attempt_parity: bool
    tool_attempt_parity: bool
    committed_thread_state_parity: bool
    progress_precedes_tool_completion: bool
    progress_p95_within_limit: bool
    overhead_p95_within_limit: bool
    bounded_progress_buffer: bool
    no_dropped_progress: bool
    request_tasks_cleaned_up: bool
    passed: bool


@dataclass(frozen=True)
class StreamingCloseoutMeasurement:
    """Machine-readable M07 Phase 4 result."""

    cohort_size: int
    scenario_counts: dict[str, int]
    configured_repetitions: int
    configured_tool_delay_ms: float
    progress_p95_limit_ms: float
    overhead_p95_limit_ms: float
    blocking_total: Distribution
    streaming_final: Distribution
    streaming_total: Distribution
    added_final_delivery_overhead: Distribution
    time_to_first_useful_progress: Distribution
    progress_observation_delay: Distribution
    progress_applicable_samples: int
    progress_before_completion_samples: int
    total_event_count: int
    total_event_bytes: int
    peak_progress_buffer: int
    dropped_progress_events: int
    blocking_model_attempts: int
    streaming_model_attempts: int
    blocking_tool_attempts: int
    streaming_tool_attempts: int
    blocking_token_usage: TokenUsage
    streaming_token_usage: TokenUsage
    request_owned_task_leaks: int
    gate: RolloutGate
    recommendation: str
    samples: tuple[SampleMeasurement, ...]


@dataclass
class _ScenarioRuntime:
    """Request-local evidence captured outside public progress payloads."""

    delivery: Delivery
    tool_delay_seconds: float
    model_attempts: Counter[str] = field(default_factory=Counter)
    tool_attempts: Counter[str] = field(default_factory=Counter)
    tool_trajectory: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    tool_started_at: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    tool_completed_at: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    token_usage: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    internal_results: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    def record_usage(self, prompt: str, input_tokens: int, output_tokens: int) -> None:
        """Record the fake model's explicit usage metadata."""
        usage = self.token_usage[prompt]
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["total_tokens"] += input_tokens + output_tokens

    def usage_for(self, prompt: str) -> TokenUsage:
        """Return detached usage for one request."""
        usage = self.token_usage[prompt]
        return TokenUsage(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
        )


class _MeteredToolProgressReporter(ToolProgressReporter):
    """Measurement-only reporter retaining production queue behavior."""

    instances: list["_MeteredToolProgressReporter"] = []

    def __init__(self, capacity: int = 16) -> None:
        super().__init__(capacity=capacity)
        self.peak_pending_count = 0
        self.instances.append(self)

    def _offer(self, event: ToolProgressEvent) -> None:
        super()._offer(event)
        self.peak_pending_count = max(self.peak_pending_count, self.pending_count)


class _DeliveryRoutingAgent:
    """Select equivalent agents using the presence of the streaming observer."""

    def __init__(
        self,
        blocking_agent: WineAgent,
        streaming_agent: WineAgent,
        blocking_runtime: _ScenarioRuntime,
        streaming_runtime: _ScenarioRuntime,
    ) -> None:
        self._blocking_agent = blocking_agent
        self._streaming_agent = streaming_agent
        self._blocking_runtime = blocking_runtime
        self._streaming_runtime = streaming_runtime
        self.execution_provenance = blocking_agent.execution_provenance

    async def ainvoke(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Delegate once and retain bounded internal parity evidence."""
        is_streaming = kwargs.get("progress_reporter") is not None
        agent = self._streaming_agent if is_streaming else self._blocking_agent
        runtime = self._streaming_runtime if is_streaming else self._blocking_runtime
        result = await agent.ainvoke(query, **kwargs)
        runtime.internal_results[query] = _normalize_internal_result(result)
        return result


def _normalize_internal_result(result: dict[str, Any]) -> tuple[Any, ...]:
    """Retain deterministic execution semantics while omitting message IDs."""
    history = tuple(
        sorted(
            tuple(sorted((str(key), str(value)) for key, value in item.items()))
            for item in result["tool_call_history"]
        )
    )
    return (
        result["final_answer"],
        tuple(sorted(result["tools_used"])),
        result["llm_call_count"],
        history,
    )


def _prompt_from_messages(messages: list[Any]) -> str:
    """Return the current request prompt from one model input."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise RuntimeError("Deterministic model input has no human message")


def _build_agent(
    runtime: _ScenarioRuntime,
    memory_manager: ConversationMemoryManager,
) -> WineAgent:
    """Build one real WineAgent around a deterministic fake model and tools."""

    @tool
    async def held_wine_tool(value: str) -> str:
        """Return deterministic wine evidence after the configured local delay."""
        runtime.tool_attempts[value] += 1
        runtime.tool_trajectory[value].append("held_wine_tool")
        runtime.tool_started_at[value].append(time.perf_counter())
        await asyncio.sleep(runtime.tool_delay_seconds)
        runtime.tool_completed_at[value].append(time.perf_counter())
        return f"held-result-{value}"

    @tool
    async def companion_wine_tool(value: str) -> str:
        """Return a second deterministic wine result for multi-tool requests."""
        runtime.tool_attempts[value] += 1
        runtime.tool_trajectory[value].append("companion_wine_tool")
        runtime.tool_started_at[value].append(time.perf_counter())
        await asyncio.sleep(runtime.tool_delay_seconds)
        runtime.tool_completed_at[value].append(time.perf_counter())
        return f"companion-result-{value}"

    definitions = tuple(
        ToolDefinition(
            tool=tool_instance,
            metadata=ToolMetadata(
                name=tool_instance.name,
                category=ToolCategory.CELLAR,
                tier=ToolTier.CORE,
                cost_class=CostClass.FREE,
                latency_class=LatencyClass.FAST,
                idempotent=True,
                capability="Provide deterministic local M07 measurement evidence.",
            ),
        )
        for tool_instance in (held_wine_tool, companion_wine_tool)
    )
    registry = ToolRegistry(definitions)

    async def _model_invoke(messages: list[Any], **_kwargs: Any) -> AIMessage:
        prompt = _prompt_from_messages(messages)
        runtime.model_attempts[prompt] += 1
        has_tool_result = any(isinstance(message, ToolMessage) for message in messages)
        if has_tool_result:
            runtime.record_usage(prompt, 24, 8)
            return AIMessage(
                content=f"Deterministic final answer for {prompt}",
                usage_metadata={"input_tokens": 24, "output_tokens": 8, "total_tokens": 32},
            )
        if "zero-tool" in prompt:
            runtime.record_usage(prompt, 18, 8)
            return AIMessage(
                content=f"Deterministic final answer for {prompt}",
                usage_metadata={"input_tokens": 18, "output_tokens": 8, "total_tokens": 26},
            )

        calls = [
            {
                "name": held_wine_tool.name,
                "args": {"value": prompt},
                "id": f"held-{prompt}",
            }
        ]
        if "multi-tool" in prompt:
            calls.append(
                {
                    "name": companion_wine_tool.name,
                    "args": {"value": prompt},
                    "id": f"companion-{prompt}",
                }
            )
        runtime.record_usage(prompt, 16, 6)
        return AIMessage(
            content="",
            tool_calls=calls,
            usage_metadata={"input_tokens": 16, "output_tokens": 6, "total_tokens": 22},
        )

    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=_model_invoke)
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    return WineAgent(
        llm=model,
        tool_registry=registry,
        memory_manager=memory_manager,
        session_memory=memory_manager.config,
    )


def _build_app(agent: _DeliveryRoutingAgent, memory_manager: ConversationMemoryManager) -> FastAPI:
    """Build the same FastAPI route surface used by local production."""
    app = FastAPI()
    app.include_router(chat_routes.router)
    model = SimpleNamespace()
    config = OmegaConf.create({"streaming": {"enabled": True}})
    app.state.config = config
    app.state.async_rag_runtime = SimpleNamespace(config=config, retriever=None, reranker=None)
    app.state.local_model = None
    app.state.cloud_model = model
    app.state.model = model
    app.state.local_intelligent_agent = None
    app.state.cloud_intelligent_agent = agent
    app.state.intelligent_agent = agent
    app.state.conversation_memory_manager = memory_manager
    return app


@asynccontextmanager
async def _serve(app: FastAPI) -> AsyncIterator[str]:
    """Serve the measurement app on one real ephemeral TCP socket."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            lifespan="off",
            log_level="warning",
            access_log=False,
        )
    )
    server_task = asyncio.create_task(
        server.serve(sockets=[listener]),
        name="streaming-closeout-server",
    )
    try:
        for _attempt in range(200):
            if server.started:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.005)
        else:
            raise RuntimeError("Streaming closeout server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()


async def _blocking_request(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    request_id: str,
) -> tuple[dict[str, Any], float]:
    """Measure one complete existing JSON delivery."""
    started = time.perf_counter()
    response = await client.post(
        f"{base_url}/api/chat/",
        json=payload,
        headers={"X-Request-Id": request_id},
    )
    total_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), total_ms


async def _streaming_request(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    request_id: str,
) -> tuple[dict[str, Any], float, float, float | None, float | None, int, int]:
    """Measure useful progress, final delivery, EOF, and SSE payload size."""
    started = time.perf_counter()
    first_progress_at: float | None = None
    final_at: float | None = None
    final_response: dict[str, Any] | None = None
    event_name: str | None = None
    event_data: str | None = None
    event_count = 0
    event_bytes = 0

    async with client.stream(
        "POST",
        f"{base_url}/api/chat/stream",
        json=payload,
        headers={"X-Request-Id": request_id},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            event_bytes += len((line + "\n").encode("utf-8"))
            if line.startswith(":"):
                continue
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                event_data = line.removeprefix("data: ")
            elif line == "" and event_name is not None and event_data is not None:
                received_at = time.perf_counter()
                data = json.loads(event_data)
                event_count += 1
                if event_name == "tool_progress" and first_progress_at is None:
                    first_progress_at = received_at
                elif event_name == "agent_done":
                    final_at = received_at
                    final_response = data["response"]
                elif event_name == "stream_error":
                    raise RuntimeError(f"Unexpected stream error: {data!r}")
                event_name = None
                event_data = None

    completed_at = time.perf_counter()
    if final_response is None or final_at is None:
        raise RuntimeError("Streaming measurement ended without agent_done")
    return (
        final_response,
        (final_at - started) * 1000,
        (completed_at - started) * 1000,
        (first_progress_at - started) * 1000 if first_progress_at is not None else None,
        first_progress_at,
        event_count,
        event_bytes,
    )


def _distribution(values: list[float]) -> Distribution:
    """Summarize one non-empty timing population."""
    if not values:
        return Distribution(0.0, 0.0, 0.0, 0.0)
    return Distribution(
        minimum_ms=round(min(values), 3),
        median_ms=round(statistics.median(values), 3),
        p95_ms=round(_percentile_nearest_rank(values, 0.95), 3),
        maximum_ms=round(max(values), 3),
    )


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Return a dependency-free nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int((percentile * len(ordered)) + 0.999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


def _sum_usage(samples: list[SampleMeasurement], delivery: Delivery) -> TokenUsage:
    """Sum available fake-model usage for one delivery mode."""
    values = [
        sample.blocking_token_usage if delivery == "blocking" else sample.streaming_token_usage
        for sample in samples
    ]
    return TokenUsage(
        input_tokens=sum(value.input_tokens for value in values),
        output_tokens=sum(value.output_tokens for value in values),
        total_tokens=sum(value.total_tokens for value in values),
    )


def summarize_measurement(
    samples: list[SampleMeasurement],
    *,
    repetitions: int,
    tool_delay_seconds: float,
    request_owned_task_leaks: int,
) -> StreamingCloseoutMeasurement:
    """Aggregate paired evidence and evaluate every approved gate."""
    progress_samples = [
        sample for sample in samples if sample.progress_observation_delay_ms is not None
    ]
    progress_delays = [
        sample.progress_observation_delay_ms
        for sample in progress_samples
        if sample.progress_observation_delay_ms is not None
    ]
    first_progress = [
        sample.time_to_first_useful_progress_ms
        for sample in progress_samples
        if sample.time_to_first_useful_progress_ms is not None
    ]
    overhead = [sample.added_final_delivery_overhead_ms for sample in samples]
    expected_size = repetitions * len(_SCENARIOS)
    expected_progress = repetitions * 2
    gate_values = {
        "complete_cohort": len(samples) == expected_size,
        "final_response_parity": all(sample.final_response_equal for sample in samples),
        "tool_trajectory_parity": all(sample.tool_trajectory_equal for sample in samples),
        "model_attempt_parity": all(sample.model_attempts_equal for sample in samples),
        "tool_attempt_parity": all(sample.tool_attempts_equal for sample in samples),
        "committed_thread_state_parity": all(
            sample.committed_thread_state_equal for sample in samples
        ),
        "progress_precedes_tool_completion": (
            len(progress_samples) == expected_progress
            and all(sample.progress_before_tool_completion is True for sample in progress_samples)
        ),
        "progress_p95_within_limit": (
            len(progress_delays) == expected_progress
            and _percentile_nearest_rank(progress_delays, 0.95) <= _PROGRESS_P95_LIMIT_MS
        ),
        "overhead_p95_within_limit": (
            len(overhead) == expected_size
            and _percentile_nearest_rank(overhead, 0.95) <= _OVERHEAD_P95_LIMIT_MS
        ),
        "bounded_progress_buffer": all(sample.peak_progress_buffer <= 16 for sample in samples),
        "no_dropped_progress": all(sample.dropped_progress_events == 0 for sample in samples),
        "request_tasks_cleaned_up": request_owned_task_leaks == 0,
    }
    gate = RolloutGate(**gate_values, passed=all(gate_values.values()))
    return StreamingCloseoutMeasurement(
        cohort_size=len(samples),
        scenario_counts=dict(Counter(sample.scenario for sample in samples)),
        configured_repetitions=repetitions,
        configured_tool_delay_ms=round(tool_delay_seconds * 1000, 3),
        progress_p95_limit_ms=_PROGRESS_P95_LIMIT_MS,
        overhead_p95_limit_ms=_OVERHEAD_P95_LIMIT_MS,
        blocking_total=_distribution([sample.blocking_total_ms for sample in samples]),
        streaming_final=_distribution([sample.streaming_final_ms for sample in samples]),
        streaming_total=_distribution([sample.streaming_total_ms for sample in samples]),
        added_final_delivery_overhead=_distribution(overhead),
        time_to_first_useful_progress=_distribution(first_progress),
        progress_observation_delay=_distribution(progress_delays),
        progress_applicable_samples=len(progress_samples),
        progress_before_completion_samples=sum(
            sample.progress_before_tool_completion is True for sample in progress_samples
        ),
        total_event_count=sum(sample.event_count for sample in samples),
        total_event_bytes=sum(sample.event_bytes for sample in samples),
        peak_progress_buffer=max((sample.peak_progress_buffer for sample in samples), default=0),
        dropped_progress_events=sum(sample.dropped_progress_events for sample in samples),
        blocking_model_attempts=sum(sample.blocking_model_attempts for sample in samples),
        streaming_model_attempts=sum(sample.streaming_model_attempts for sample in samples),
        blocking_tool_attempts=sum(sample.blocking_tool_attempts for sample in samples),
        streaming_tool_attempts=sum(sample.streaming_tool_attempts for sample in samples),
        blocking_token_usage=_sum_usage(samples, "blocking"),
        streaming_token_usage=_sum_usage(samples, "streaming"),
        request_owned_task_leaks=request_owned_task_leaks,
        gate=gate,
        recommendation=(
            "eligible_for_separate_enablement_decision"
            if gate.passed
            else "keep_streaming_disabled"
        ),
        samples=tuple(samples),
    )


async def _count_request_task_leaks() -> int:
    """Wait briefly for request-owned tasks, then return the remaining count."""
    remaining: list[asyncio.Task[Any]] = []
    for _attempt in range(200):
        remaining = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and task.get_name().startswith("chat-stream-")
        ]
        if not remaining:
            return 0
        await asyncio.sleep(0.005)
    return len(remaining)


async def measure_streaming_closeout(
    *,
    repetitions: int = 10,
    tool_delay_seconds: float = 0.5,
    work_directory: Path | None = None,
) -> StreamingCloseoutMeasurement:
    """Run paired socket-backed delivery with identical deterministic agents."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not isinstance(tool_delay_seconds, (int, float)) or tool_delay_seconds <= 0:
        raise ValueError("tool_delay_seconds must be positive")

    owned_temp_directory: TemporaryDirectory[str] | None = None
    if work_directory is None:
        owned_temp_directory = TemporaryDirectory(prefix="m07-streaming-closeout-")
        work_directory = Path(owned_temp_directory.name)
    work_directory.mkdir(parents=True, exist_ok=True)
    blocking_memory = await ConversationMemoryManager.open(
        SessionMemoryConfig(
            enabled=True,
            db_path=str(work_directory / "blocking-memory.db"),
        )
    )
    streaming_memory = await ConversationMemoryManager.open(
        SessionMemoryConfig(
            enabled=True,
            db_path=str(work_directory / "streaming-memory.db"),
        )
    )
    assert blocking_memory is not None and streaming_memory is not None

    blocking_runtime = _ScenarioRuntime("blocking", float(tool_delay_seconds))
    streaming_runtime = _ScenarioRuntime("streaming", float(tool_delay_seconds))
    routing_agent = _DeliveryRoutingAgent(
        _build_agent(blocking_runtime, blocking_memory),
        _build_agent(streaming_runtime, streaming_memory),
        blocking_runtime,
        streaming_runtime,
    )
    app = _build_app(routing_agent, blocking_memory)
    samples: list[SampleMeasurement] = []
    original_reporter = chat_routes.ToolProgressReporter
    _MeteredToolProgressReporter.instances.clear()
    chat_routes.ToolProgressReporter = _MeteredToolProgressReporter
    task_leaks = -1

    try:
        async with _serve(app) as base_url:
            async with httpx.AsyncClient(timeout=max(5.0, tool_delay_seconds * 5)) as client:
                for scenario in _SCENARIOS:
                    for pair_index in range(repetitions):
                        prompt = f"M07 {scenario.replace('_', '-')} wine request {pair_index}"
                        thread_id = str(uuid5(_THREAD_NAMESPACE, prompt))
                        request_id = f"m07-{scenario}-{pair_index}"
                        payload = {
                            "message": prompt,
                            "agent_mode": "intelligent",
                            "model_provider": "cloud",
                            "thread_id": thread_id,
                            "thread_action": "append",
                        }

                        if pair_index % 2 == 0:
                            blocking_response, blocking_total_ms = await _blocking_request(
                                client, base_url, payload, request_id
                            )
                            stream_values = await _streaming_request(
                                client, base_url, payload, request_id
                            )
                        else:
                            stream_values = await _streaming_request(
                                client, base_url, payload, request_id
                            )
                            blocking_response, blocking_total_ms = await _blocking_request(
                                client, base_url, payload, request_id
                            )

                        (
                            streaming_response,
                            streaming_final_ms,
                            streaming_total_ms,
                            first_progress_ms,
                            first_progress_at,
                            event_count,
                            event_bytes,
                        ) = stream_values
                        reporter = _MeteredToolProgressReporter.instances[-1]
                        blocking_thread = await blocking_memory.get_thread(thread_id)
                        streaming_thread = await streaming_memory.get_thread(thread_id)
                        starts = streaming_runtime.tool_started_at[prompt]
                        completions = streaming_runtime.tool_completed_at[prompt]
                        observation_delay_ms = (
                            max(0.0, (first_progress_at - min(starts)) * 1000)
                            if first_progress_at is not None and starts
                            else None
                        )
                        progress_before_completion = (
                            first_progress_at < min(completions)
                            if first_progress_at is not None and completions
                            else None
                        )
                        committed_state_equal = (
                            blocking_thread is not None
                            and streaming_thread is not None
                            and blocking_thread.completed_turns == streaming_thread.completed_turns == 1
                            and blocking_thread.active_checkpoint_id != blocking_thread.base_checkpoint_id
                            and streaming_thread.active_checkpoint_id != streaming_thread.base_checkpoint_id
                            and (blocking_thread.previous_checkpoint_id == blocking_thread.base_checkpoint_id)
                            and (streaming_thread.previous_checkpoint_id == streaming_thread.base_checkpoint_id)
                        )
                        samples.append(
                            SampleMeasurement(
                                scenario=scenario,
                                pair_index=pair_index,
                                blocking_total_ms=round(blocking_total_ms, 3),
                                streaming_final_ms=round(streaming_final_ms, 3),
                                streaming_total_ms=round(streaming_total_ms, 3),
                                added_final_delivery_overhead_ms=round(
                                    streaming_final_ms - blocking_total_ms, 3
                                ),
                                time_to_first_useful_progress_ms=(
                                    round(first_progress_ms, 3)
                                    if first_progress_ms is not None
                                    else None
                                ),
                                progress_observation_delay_ms=(
                                    round(observation_delay_ms, 3)
                                    if observation_delay_ms is not None
                                    else None
                                ),
                                progress_before_tool_completion=progress_before_completion,
                                event_count=event_count,
                                event_bytes=event_bytes,
                                peak_progress_buffer=reporter.peak_pending_count,
                                dropped_progress_events=reporter.dropped_count,
                                final_response_equal=blocking_response == streaming_response,
                                tool_trajectory_equal=(
                                    sorted(blocking_runtime.tool_trajectory[prompt])
                                    == sorted(streaming_runtime.tool_trajectory[prompt])
                                    and blocking_runtime.internal_results[prompt]
                                    == streaming_runtime.internal_results[prompt]
                                ),
                                model_attempts_equal=(
                                    blocking_runtime.model_attempts[prompt]
                                    == streaming_runtime.model_attempts[prompt]
                                ),
                                tool_attempts_equal=(
                                    blocking_runtime.tool_attempts[prompt]
                                    == streaming_runtime.tool_attempts[prompt]
                                ),
                                committed_thread_state_equal=committed_state_equal,
                                blocking_model_attempts=blocking_runtime.model_attempts[prompt],
                                streaming_model_attempts=streaming_runtime.model_attempts[prompt],
                                blocking_tool_attempts=blocking_runtime.tool_attempts[prompt],
                                streaming_tool_attempts=streaming_runtime.tool_attempts[prompt],
                                blocking_token_usage=blocking_runtime.usage_for(prompt),
                                streaming_token_usage=streaming_runtime.usage_for(prompt),
                            )
                        )
            task_leaks = await _count_request_task_leaks()
    finally:
        chat_routes.ToolProgressReporter = original_reporter
        await blocking_memory.close()
        await streaming_memory.close()
        if owned_temp_directory is not None:
            owned_temp_directory.cleanup()

    return summarize_measurement(
        samples,
        repetitions=repetitions,
        tool_delay_seconds=float(tool_delay_seconds),
        request_owned_task_leaks=task_leaks,
    )


def main() -> int:
    """Run the deterministic closeout and log one JSON result."""
    try:
        measurement = asyncio.run(measure_streaming_closeout())
    except Exception as exc:
        logger.error("M07 streaming closeout measurement failed: %s", exc, exc_info=True)
        return 1
    logger.info(
        "M07_STREAMING_CLOSEOUT_MEASUREMENT=%s",
        json.dumps(asdict(measurement), ensure_ascii=True, sort_keys=True),
    )
    return 0 if measurement.gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
