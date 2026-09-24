"""Contract and lifecycle tests for the intelligent chat SSE endpoint."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
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
