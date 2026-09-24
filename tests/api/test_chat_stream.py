"""Contract and lifecycle tests for the intelligent chat SSE endpoint."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import TypeAdapter, ValidationError
import pytest

from src.api.schemas.chat import (
    AgentDoneStreamEvent,
    ChatResponse,
    ChatStreamEvent,
    STREAM_ERROR_MESSAGE,
    StreamErrorEvent,
    ToolProgressStreamEvent,
    ChatRequest,
)


THREAD_ID = "123e4567-e89b-12d3-a456-426614174000"


def _populate_state(*, enabled: bool, agent: object | None = None, memory_manager: object | None = None) -> None:
    """Populate the shared API app with deterministic streaming resources."""
    from src.api.main import app

    model = MagicMock()
    app.state.config = OmegaConf.create({"streaming": {"enabled": enabled}})
    app.state.local_model = None
    app.state.cloud_model = model
    app.state.model = model
    app.state.local_intelligent_agent = None
    app.state.cloud_intelligent_agent = agent
    app.state.intelligent_agent = agent
    app.state.conversation_memory_manager = memory_manager
    app.state.async_rag_runtime = SimpleNamespace(config=app.state.config, retriever=None, reranker=None)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse complete test frames without acting as the Phase 3 incremental client."""
    events: list[tuple[str, dict]] = []
    for frame in body.split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        lines = frame.splitlines()
        event_name = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        events.append((event_name, json.loads(data)))
    return events


def test_stream_event_contract_is_discriminated_and_bounded() -> None:
    """Public frames accept only the reviewed event shapes and lifecycle values."""
    adapter = TypeAdapter(ChatStreamEvent)
    progress = adapter.validate_python(
        {
            "type": "tool_progress",
            "invocation_id": 1,
            "tool_key": "query_cellar",
            "status": "started",
        }
    )
    response = ChatResponse(answer="done", agent_mode="intelligent", model_provider="cloud")

    assert progress == ToolProgressStreamEvent(
        invocation_id=1,
        tool_key="query_cellar",
        status="started",
    )
    assert adapter.validate_python(
        {"type": "agent_done", "response": response.model_dump(mode="json")}
    ) == AgentDoneStreamEvent(response=response)
    assert adapter.validate_python(
        {"type": "stream_error", "message": STREAM_ERROR_MESSAGE, "outcome": "uncertain"}
    ) == StreamErrorEvent()

    for invalid in (
        {"type": "tool_progress", "invocation_id": 0, "tool_key": "query_cellar", "status": "started"},
        {"type": "tool_progress", "invocation_id": 1, "tool_key": "query_cellar", "status": "retrying"},
        {
            "type": "tool_progress",
            "invocation_id": 1,
            "tool_key": "query_cellar",
            "status": "started",
            "arguments": {"secret": "private"},
        },
        {"type": "stream_error", "message": "private exception", "outcome": "uncertain"},
    ):
        with pytest.raises(ValidationError):
            adapter.validate_python(invalid)


@pytest.mark.asyncio
async def test_streaming_disabled_returns_only_the_stable_fallback_code() -> None:
    """The default flag must reject before agent execution with the reviewed code."""
    agent = MagicMock()
    agent.ainvoke = AsyncMock()
    _populate_state(enabled=False, agent=agent)
    from src.api.main import app
    from src.api.routes.chat import _validate_stream_preconditions

    with pytest.raises(HTTPException) as caught:
        await _validate_stream_preconditions(ChatRequest(message="Hello"), app.state, None, agent)

    assert caught.value.status_code == 404
    assert caught.value.detail["code"] == "streaming_disabled"
    agent.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_rag_only_streaming_returns_the_stable_unsupported_code() -> None:
    """RAG-only remains on the blocking endpoint and cannot start SSE execution."""
    _populate_state(enabled=True, agent=MagicMock())
    from src.api.main import app
    from src.api.routes.chat import _validate_stream_preconditions

    with pytest.raises(HTTPException) as caught:
        await _validate_stream_preconditions(
            ChatRequest(message="Hello", agent_mode="rag_only"),
            app.state,
            None,
            MagicMock(),
        )

    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "stream_mode_unsupported"


@pytest.mark.asyncio
async def test_stream_preflight_rejects_missing_agent_and_known_replace_conflict() -> None:
    """Known resource and thread failures must retain HTTP status before headers."""
    _populate_state(enabled=True, agent=None)
    from src.api.main import app
    from src.api.routes.chat import _validate_stream_preconditions

    with pytest.raises(HTTPException) as missing:
        await _validate_stream_preconditions(ChatRequest(message="Hello"), app.state, None, None)

    manager = MagicMock()
    manager.get_thread = AsyncMock(return_value=None)
    agent = MagicMock()
    agent.ainvoke = AsyncMock()
    _populate_state(enabled=True, agent=agent, memory_manager=manager)
    with pytest.raises(HTTPException) as conflict:
        await _validate_stream_preconditions(
            ChatRequest(
                message="Replace",
                thread_id=THREAD_ID,
                thread_action="replace_last",
            ),
            app.state,
            manager,
            agent,
        )

    assert missing.value.status_code == 503
    assert conflict.value.status_code == 409
    agent.ainvoke.assert_not_called()


def test_project_streaming_flag_defaults_to_false() -> None:
    """The checked-in rollout setting must remain disabled until Phase 4 approval."""
    from pathlib import Path

    config = OmegaConf.load(Path(__file__).parents[2] / "app_config.yml")

    assert config.streaming.enabled is False


def test_stream_route_emits_safe_progress_then_one_authoritative_response() -> None:
    """A connected request should expose only reviewed progress before its final response."""
    from src.agents.guardrails import ToolProgressStatus
    from src.api.main import app

    async def invoke(_prompt: str, **kwargs: object) -> dict:
        reporter = kwargs["progress_reporter"]
        invocation_id = reporter.start("query_cellar")
        reporter.finish(invocation_id, "query_cellar", ToolProgressStatus.COMPLETED)
        return {"final_answer": "Complete answer", "messages": []}

    agent = MagicMock()
    agent.ainvoke = AsyncMock(side_effect=invoke)
    _populate_state(enabled=True, agent=agent)

    response = TestClient(app).post(
        "/api/chat/stream",
        headers={"X-Request-Id": "stream-request"},
        json={"message": "Inspect my cellar"},
    )
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [name for name, _payload in events] == [
        "tool_progress",
        "tool_progress",
        "agent_done",
    ]
    assert [payload.get("status") for _name, payload in events[:-1]] == [
        "started",
        "completed",
    ]
    final = events[-1][1]["response"]
    assert final["answer"] == "Complete answer"
    assert final["model_provider"] == "cloud"
    assert "private" not in response.text
    assert sum(name in {"agent_done", "stream_error"} for name, _payload in events) == 1


def test_post_header_failure_emits_one_fixed_uncertain_terminal_event() -> None:
    """Execution failures after response start must not leak or attempt a status rewrite."""
    from src.api.main import app

    agent = MagicMock()
    agent.ainvoke = AsyncMock(side_effect=RuntimeError("private provider exception"))
    _populate_state(enabled=True, agent=agent)

    response = TestClient(app).post("/api/chat/stream", json={"message": "Fail safely"})
    events = _parse_sse(response.text)

    assert response.status_code == 200
    assert events == [
        (
            "stream_error",
            {
                "type": "stream_error",
                "message": STREAM_ERROR_MESSAGE,
                "outcome": "uncertain",
            },
        )
    ]
    assert "private provider exception" not in response.text


def test_zero_tool_stream_emits_only_one_agent_done_terminal() -> None:
    """A valid zero-tool answer should complete without fabricated progress."""
    from src.api.main import app

    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"final_answer": "Direct answer", "messages": []})
    _populate_state(enabled=True, agent=agent)

    response = TestClient(app).post("/api/chat/stream", json={"message": "Define tannin"})
    events = _parse_sse(response.text)

    assert events == [
        (
            "agent_done",
            {
                "type": "agent_done",
                "response": {
                    "answer": "Direct answer",
                    "sources": [],
                    "web_sources": [],
                    "agent_mode": "intelligent",
                    "model_provider": "cloud",
                    "error": None,
                    "trace_id": None,
                    "thread_id": None,
                },
            },
        )
    ]


@pytest.mark.asyncio
async def test_slow_consumer_keeps_progress_bounded_and_final_result_separate() -> None:
    """A paused reader must not block execution or displace the terminal response."""
    from src.api.routes.chat import _stream_agent_events

    captured: dict[str, object] = {}
    finished = asyncio.Event()

    async def invoke(_prompt: str, **kwargs: object) -> dict:
        reporter = kwargs["progress_reporter"]
        captured["reporter"] = reporter
        for _index in range(100):
            reporter.start("query_cellar")
        finished.set()
        return {"final_answer": "Complete despite slow reader", "messages": []}

    agent = MagicMock()
    agent.ainvoke = AsyncMock(side_effect=invoke)
    stream = _stream_agent_events(
        request=ChatRequest(message="Produce progress"),
        intelligent_agent=agent,
        actual_provider="cloud",
        request_id="bounded-request",
        trace_context={"request_id": "bounded-request"},
        message_history=[],
        active_thread_id=None,
    )

    first = await anext(stream)
    await asyncio.wait_for(finished.wait(), timeout=1)
    reporter = captured["reporter"]
    assert reporter.pending_count <= 16
    assert reporter.dropped_count > 0

    remaining = [frame async for frame in stream]
    events = _parse_sse(first + "".join(remaining))
    assert sum(name == "tool_progress" for name, _payload in events) <= 16
    assert events[-1][0] == "agent_done"
    assert events[-1][1]["response"]["answer"] == "Complete despite slow reader"


@pytest.mark.asyncio
async def test_generator_close_cancels_and_awaits_owned_execution() -> None:
    """Disconnect-equivalent generator closure must clean up the cooperative agent task."""
    from src.api.routes.chat import _stream_agent_events

    cancelled = asyncio.Event()

    async def invoke(_prompt: str, **kwargs: object) -> dict:
        reporter = kwargs["progress_reporter"]
        reporter.start("query_cellar")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    agent = MagicMock()
    agent.ainvoke = AsyncMock(side_effect=invoke)
    stream = _stream_agent_events(
        request=ChatRequest(message="Wait"),
        intelligent_agent=agent,
        actual_provider="cloud",
        request_id="cancel-request",
        trace_context={"request_id": "cancel-request"},
        message_history=[],
        active_thread_id=None,
    )

    assert "event: tool_progress" in await anext(stream)
    await stream.aclose()

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_idle_stream_emits_semantics_free_heartbeat_comment() -> None:
    """Transport liveness must remain a comment and must not terminate execution."""
    from src.api.routes.chat import _stream_agent_events

    cancelled = asyncio.Event()

    async def invoke(_prompt: str, **_kwargs: object) -> dict:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    agent = MagicMock()
    agent.ainvoke = AsyncMock(side_effect=invoke)
    stream = _stream_agent_events(
        request=ChatRequest(message="Wait quietly"),
        intelligent_agent=agent,
        actual_provider="cloud",
        request_id="heartbeat-request",
        trace_context={"request_id": "heartbeat-request"},
        message_history=[],
        active_thread_id=None,
        heartbeat_seconds=0.01,
    )

    assert await anext(stream) == ": heartbeat\n\n"
    await stream.aclose()

    assert cancelled.is_set()


def test_stream_route_preflight_errors_are_http_responses_before_sse_headers() -> None:
    """Disabled, unsupported, missing, and known conflict cases must stay pre-header."""
    from src.api.main import app

    agent = MagicMock()
    agent.ainvoke = AsyncMock()
    _populate_state(enabled=False, agent=agent)
    disabled = TestClient(app).post("/api/chat/stream", json={"message": "Hello"})

    _populate_state(enabled=True, agent=agent)
    unsupported = TestClient(app).post(
        "/api/chat/stream",
        json={"message": "Hello", "agent_mode": "rag_only"},
    )

    _populate_state(enabled=True, agent=None)
    unavailable = TestClient(app).post("/api/chat/stream", json={"message": "Hello"})

    manager = MagicMock()
    manager.get_thread = AsyncMock(return_value=None)
    _populate_state(enabled=True, agent=agent, memory_manager=manager)
    conflict = TestClient(app).post(
        "/api/chat/stream",
        json={
            "message": "Replace",
            "thread_id": THREAD_ID,
            "thread_action": "replace_last",
        },
    )

    assert (disabled.status_code, disabled.json()["detail"]["code"]) == (
        404,
        "streaming_disabled",
    )
    assert (unsupported.status_code, unsupported.json()["detail"]["code"]) == (
        400,
        "stream_mode_unsupported",
    )
    assert unavailable.status_code == 503
    assert conflict.status_code == 409
    for response in (disabled, unsupported, unavailable, conflict):
        assert not response.headers["content-type"].startswith("text/event-stream")
    agent.ainvoke.assert_not_called()
