"""API tests for chat trace correlation and trace_id response behavior."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with lightweight app state for chat route tests."""
    from src.api.main import app

    app.state.model = MagicMock()
    app.state.intelligent_agent = None
    app.state.keyword_agent = None
    app.state.retriever = None
    app.state.reranker = None

    return TestClient(app)


@pytest.fixture(autouse=True)
def _mock_rag_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid invoking real LLM chain logic in trace correlation tests."""
    from src.api.routes import chat

    monkeypatch.setattr(
        chat,
        "_invoke_rag_only",
        lambda **_: ("ok", [], []),
    )


def test_chat_response_trace_id_disabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Response should include null trace_id when observability is disabled."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: False)

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["trace_id"] is None


def test_trace_id_none_when_provider_none(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider 'none' should not emit trace_id even if observability is enabled."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: False)

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["trace_id"] is None


def test_trace_id_none_when_provider_unsupported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported providers should not emit trace_id."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: False)

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["trace_id"] is None


def test_chat_response_includes_trace_id_when_enabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Response should include a request trace ID when observability is enabled."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
        },
    )

    assert resp.status_code == 200
    trace_id = resp.json()["trace_id"]
    assert isinstance(trace_id, str)
    assert len(trace_id) > 0


def test_chat_with_x_request_id_header(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Header-provided request ID should be echoed as trace_id."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)

    resp = client.post(
        "/api/chat/",
        headers={"X-Request-Id": "my-id-123"},
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["trace_id"] == "my-id-123"


def test_chat_without_x_request_id_generates_uuid(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing request header should produce a generated UUID trace_id."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
        },
    )

    assert resp.status_code == 200
    uuid.UUID(resp.json()["trace_id"])


def test_chat_with_x_session_id_propagates_to_trace_context(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Session ID header should be forwarded into trace context."""
    from src.api.main import app
    from src.api.routes import chat

    seen_trace_context: dict[str, str] = {}

    def _capture_intelligent(agent, prompt: str, message_history: list[dict], trace_context=None):
        if trace_context:
            seen_trace_context.update(trace_context)
        return "ok", [], []

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)
    monkeypatch.setattr(chat, "_invoke_intelligent_agent", _capture_intelligent)
    app.state.intelligent_agent = MagicMock()

    resp = client.post(
        "/api/chat/",
        headers={"X-Session-Id": "session-123"},
        json={
            "message": "Hello",
            "agent_mode": "intelligent",
        },
    )

    assert resp.status_code == 200
    assert seen_trace_context.get("session_id") == "session-123"

    app.state.intelligent_agent = None


def test_chat_response_shape_unchanged(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Baseline response fields should remain present after tracing changes."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: False)

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert "web_sources" in body
    assert "agent_mode" in body
    assert "error" in body


def test_empty_message_history_tracing_safe(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty message history should still return a correlated trace_id safely."""
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "rag_only",
            "message_history": [],
        },
    )

    assert resp.status_code == 200
    assert isinstance(resp.json()["trace_id"], str)


def test_error_response_still_returns_trace_id(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Graceful error responses should still include trace_id when enabled."""
    from src.api.main import app
    from src.api.routes import chat

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)

    failing_agent = MagicMock()
    failing_agent.invoke.side_effect = RuntimeError("boom")
    app.state.intelligent_agent = failing_agent

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "intelligent",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is not None
    assert isinstance(body["trace_id"], str)

    app.state.intelligent_agent = None


def test_error_path_sets_error_class_span_attribute(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Error path should record error_class metadata on the request span."""
    from src.api.main import app
    from src.api.routes import chat

    captured_span_attrs: list[dict[str, object]] = []

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)
    monkeypatch.setattr(chat, "set_span_attributes", lambda _span, attrs: captured_span_attrs.append(attrs))

    failing_agent = MagicMock()
    failing_agent.invoke.side_effect = RuntimeError("boom")
    app.state.intelligent_agent = failing_agent

    resp = client.post(
        "/api/chat/",
        json={
            "message": "Hello",
            "agent_mode": "intelligent",
        },
    )

    assert resp.status_code == 200
    assert any(attrs.get("error_class") == "RuntimeError" for attrs in captured_span_attrs)

    app.state.intelligent_agent = None


def test_all_three_modes_emit_trace_context(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace context should be passed into all three mode execution helpers."""
    from src.api.main import app
    from src.api.routes import chat

    seen_contexts: list[dict[str, str]] = []

    def _capture_intelligent(agent, prompt: str, message_history: list[dict], trace_context=None):
        seen_contexts.append(trace_context or {})
        return "ok", [], []

    def _capture_keyword(agent, prompt: str, message_history: list[dict], trace_context=None):
        seen_contexts.append(trace_context or {})
        return "ok", [], []

    def _capture_rag_only(
        prompt: str,
        model,
        retriever,
        reranker,
        message_history: list[dict],
        enable_rag: bool,
        n_results_override: int | None,
        trace_context=None,
    ):
        seen_contexts.append(trace_context or {})
        return "ok", [], []

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)
    monkeypatch.setattr(chat, "_invoke_intelligent_agent", _capture_intelligent)
    monkeypatch.setattr(chat, "_invoke_keyword_agent", _capture_keyword)
    monkeypatch.setattr(chat, "_invoke_rag_only", _capture_rag_only)

    app.state.intelligent_agent = MagicMock()
    app.state.keyword_agent = MagicMock()

    for mode in ("intelligent", "keyword", "rag_only"):
        resp = client.post(
            "/api/chat/",
            json={
                "message": "Hello",
                "agent_mode": mode,
            },
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["trace_id"], str)

    assert len(seen_contexts) == 3
    assert all(context.get("request_id") for context in seen_contexts)
    observed_modes = sorted(context.get("agent_mode", "") for context in seen_contexts)
    assert observed_modes == ["intelligent", "keyword", "rag_only"]

    app.state.intelligent_agent = None
    app.state.keyword_agent = None




