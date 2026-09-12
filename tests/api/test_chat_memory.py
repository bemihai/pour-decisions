"""API contracts for opt-in durable conversation threads."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.schemas.chat import ChatResponse


THREAD_ID = "123e4567-e89b-12d3-a456-426614174000"


def _populate_state(app, *, agent=None, model=None, memory_manager=None) -> None:
    """Populate the application state used by chat dependencies and dispatch."""
    app.state.config = MagicMock()
    app.state.local_model = model
    app.state.cloud_model = model
    app.state.model = model
    app.state.local_intelligent_agent = agent
    app.state.cloud_intelligent_agent = agent
    app.state.intelligent_agent = agent
    app.state.conversation_memory_manager = memory_manager
    app.state.retriever = None
    app.state.reranker = None


@pytest.fixture()
def client() -> TestClient:
    """Return a client with isolated disabled-memory state."""
    from src.api.main import app

    _populate_state(app)
    return TestClient(app)


def _agent(answer: str = "Thread-aware answer") -> MagicMock:
    """Return an async intelligent-agent test double."""
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"final_answer": answer, "messages": []})
    return agent


def test_missing_thread_id_preserves_stateless_history(client: TestClient) -> None:
    """Requests without a thread ID retain the existing client-history contract."""
    from src.api.main import app

    agent = _agent()
    _populate_state(app, agent=agent)

    response = client.post(
        "/api/chat/",
        json={
            "message": "Current question",
            "message_history": [{"role": "human", "content": "Earlier question"}],
        },
    )

    assert response.status_code == 200
    assert ChatResponse(**response.json()).thread_id is None
    agent.ainvoke.assert_awaited_once()
    _, kwargs = agent.ainvoke.await_args
    assert kwargs["message_history"] == [{"role": "human", "content": "Earlier question"}]
    assert "thread_id" not in kwargs
    assert "thread_action" not in kwargs


def test_enabled_memory_routes_uuid_and_action_to_intelligent_agent(client: TestClient) -> None:
    """A valid UUID opts an intelligent request into the shared manager path."""
    from src.api.main import app

    agent = _agent()
    manager = MagicMock()
    _populate_state(app, agent=agent, memory_manager=manager)

    response = client.post(
        "/api/chat/",
        json={
            "message": "Try that again",
            "thread_id": THREAD_ID,
            "thread_action": "replace_last",
        },
    )

    assert response.status_code == 200
    assert response.json()["thread_id"] == THREAD_ID
    _, kwargs = agent.ainvoke.await_args
    assert kwargs["thread_id"] == THREAD_ID
    assert kwargs["thread_action"] == "replace_last"


def test_disabled_memory_ignores_thread_for_execution_but_echoes_it(client: TestClient) -> None:
    """Disabled memory keeps stateless behavior without changing the wire response."""
    from src.api.main import app

    agent = _agent()
    _populate_state(app, agent=agent, memory_manager=None)

    response = client.post(
        "/api/chat/",
        json={
            "message": "Continue",
            "thread_id": THREAD_ID,
            "message_history": [{"role": "ai", "content": "Prior answer"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["thread_id"] == THREAD_ID
    _, kwargs = agent.ainvoke.await_args
    assert kwargs["message_history"] == [{"role": "ai", "content": "Prior answer"}]
    assert "thread_id" not in kwargs


def test_rag_only_preserves_history_and_does_not_use_thread_manager(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG-only remains client-history based even when a thread ID is supplied."""
    from src.api import main
    from src.api.routes import chat

    model = MagicMock()
    manager = MagicMock()
    _populate_state(main.app, model=model, memory_manager=manager)
    invoke = AsyncMock(return_value=("RAG answer", [], []))
    monkeypatch.setattr(chat, "_ainvoke_rag_only", invoke)

    response = client.post(
        "/api/chat/",
        json={
            "message": "RAG question",
            "agent_mode": "rag_only",
            "thread_id": THREAD_ID,
            "message_history": [{"role": "human", "content": "Prior RAG question"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["thread_id"] == THREAD_ID
    assert invoke.await_args.kwargs["message_history"] == [
        {"role": "human", "content": "Prior RAG question"}
    ]
    manager.delete_thread.assert_not_called()


def test_replace_last_without_completed_turn_returns_conflict(client: TestClient) -> None:
    """The reviewed empty-thread replacement error maps to HTTP 409."""
    from src.api.main import app

    agent = _agent()
    agent.ainvoke.side_effect = RuntimeError(
        "Cannot replace a turn before the thread has a completed turn"
    )
    _populate_state(app, agent=agent, memory_manager=MagicMock())

    response = client.post(
        "/api/chat/",
        json={
            "message": "Replace",
            "thread_id": THREAD_ID,
            "thread_action": "replace_last",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Cannot replace a turn before the thread has a completed turn"
    )


@pytest.mark.parametrize("thread_id", ["", "not-a-uuid"])
def test_invalid_thread_ids_are_rejected(client: TestClient, thread_id: str) -> None:
    """Empty and malformed identifiers fail Pydantic validation."""
    response = client.post(
        "/api/chat/",
        json={"message": "Question", "thread_id": thread_id},
    )

    assert response.status_code == 422


def test_invalid_thread_action_is_rejected(client: TestClient) -> None:
    """Only the two reviewed lifecycle actions are accepted."""
    response = client.post(
        "/api/chat/",
        json={"message": "Question", "thread_action": "rewind"},
    )

    assert response.status_code == 422


def test_delete_thread_is_idempotent_when_enabled(client: TestClient) -> None:
    """Deletion delegates every request and returns 204 for an absent thread too."""
    from src.api.main import app

    manager = MagicMock()
    manager.delete_thread = AsyncMock(return_value=None)
    _populate_state(app, memory_manager=manager)

    first = client.delete(f"/api/chat/threads/{THREAD_ID}")
    second = client.delete(f"/api/chat/threads/{THREAD_ID}")

    assert first.status_code == 204
    assert second.status_code == 204
    assert manager.delete_thread.await_args_list[0].args == (THREAD_ID,)
    assert manager.delete_thread.await_count == 2


def test_delete_thread_is_noop_when_memory_disabled(client: TestClient) -> None:
    """The lifecycle endpoint remains available when storage is disabled."""
    response = client.delete(f"/api/chat/threads/{THREAD_ID}")

    assert response.status_code == 204


def test_delete_thread_failure_is_generic(client: TestClient) -> None:
    """Storage failures do not leak details through the deletion endpoint."""
    from src.api.main import app

    manager = MagicMock()
    manager.delete_thread = AsyncMock(side_effect=RuntimeError("sensitive database path"))
    _populate_state(app, memory_manager=manager)

    response = client.delete(f"/api/chat/threads/{THREAD_ID}")

    assert response.status_code == 500
    assert response.json() == {"detail": "Conversation thread deletion failed"}
    assert "sensitive" not in response.text


def test_thread_openapi_and_typescript_contracts_agree(client: TestClient) -> None:
    """Generated Python schema and the checked-in TypeScript mirror expose the same fields."""
    openapi = client.get("/openapi.json").json()
    request_properties = openapi["components"]["schemas"]["ChatRequest"]["properties"]
    response_properties = openapi["components"]["schemas"]["ChatResponse"]["properties"]

    assert request_properties["thread_id"]["anyOf"][0]["format"] == "uuid"
    assert request_properties["thread_action"]["default"] == "append"
    assert request_properties["thread_action"]["enum"] == ["append", "replace_last"]
    assert response_properties["thread_id"]["anyOf"][0]["format"] == "uuid"
    assert "thread_id" not in openapi["components"]["schemas"]["ChatResponse"]["required"]
    assert "/api/chat/threads/{thread_id}" in openapi["paths"]

    typescript = (Path(__file__).parents[2] / "frontend/src/lib/types.ts").read_text(
        encoding="utf-8"
    )
    assert 'export type ThreadAction = "append" | "replace_last";' in typescript
    assert "thread_id?: string | null;" in typescript
    assert "thread_action?: ThreadAction;" in typescript
