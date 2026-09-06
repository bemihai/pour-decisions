"""API tests for chat trace correlation and trace_id response behavior."""

from __future__ import annotations

from contextlib import contextmanager
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.prompt_registry import get_prompt_registry
from src.agents.provenance import (
    ExecutionProvenance,
    ModelProvenance,
    PromptProvenance,
)


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with lightweight app state for chat route tests."""
    from src.api.main import app

    mock_model = MagicMock()
    app.state.local_model = None
    app.state.cloud_model = mock_model
    app.state.model = mock_model
    app.state.local_intelligent_agent = None
    app.state.cloud_intelligent_agent = None
    app.state.intelligent_agent = None
    app.state.tool_registry = None
    app.state.retriever = None
    app.state.reranker = None
    app.state.config = MagicMock()

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


def test_chat_with_x_session_id_propagates_to_trace_context(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session ID header should be forwarded into trace context."""
    from src.api.main import app
    from src.api.routes import chat

    seen_trace_context: dict[str, str] = {}

    async def _capture_intelligent(
        agent: object,
        prompt: str,
        message_history: list[dict],
        trace_context: dict[str, str] | None = None,
    ) -> tuple[str, list[object], list[object]]:
        if trace_context:
            seen_trace_context.update(trace_context)
        return "ok", [], []

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)
    monkeypatch.setattr(chat, "_ainvoke_intelligent_agent", _capture_intelligent)
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
    failing_agent.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
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
    failing_agent.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
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


def test_all_modes_emit_trace_context(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace context should be passed into both mode execution helpers."""
    from src.api.main import app
    from src.api.routes import chat

    seen_contexts: list[dict[str, str]] = []

    async def _capture_intelligent(
        agent: object,
        prompt: str,
        message_history: list[dict],
        trace_context: dict[str, str] | None = None,
    ) -> tuple[str, list[object], list[object]]:
        seen_contexts.append(trace_context or {})
        return "ok", [], []

    def _capture_rag_only(
        prompt: str,
        cfg: object,
        model: object,
        retriever: object,
        reranker: object,
        message_history: list[dict],
        enable_rag: bool,
        n_results_override: int | None,
        trace_context: dict[str, str] | None = None,
    ) -> tuple[str, list[object], list[object]]:
        seen_contexts.append(trace_context or {})
        return "ok", [], []

    monkeypatch.setattr(chat, "_is_observability_enabled", lambda: True)
    monkeypatch.setattr(chat, "_ainvoke_intelligent_agent", _capture_intelligent)
    monkeypatch.setattr(chat, "_invoke_rag_only", _capture_rag_only)

    app.state.intelligent_agent = MagicMock()

    for mode in ("intelligent", "rag_only"):
        resp = client.post(
            "/api/chat/",
            json={
                "message": "Hello",
                "agent_mode": mode,
            },
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["trace_id"], str)

    assert len(seen_contexts) == 2
    assert all(context.get("request_id") for context in seen_contexts)
    observed_modes = sorted(context.get("agent_mode", "") for context in seen_contexts)
    assert observed_modes == ["intelligent", "rag_only"]

    app.state.intelligent_agent = None


def _intelligent_provenance() -> ExecutionProvenance:
    """Build bounded hybrid provenance without constructing an agent graph."""
    prompt = get_prompt_registry().get("intelligent_agent_system")
    return ExecutionProvenance(
        mode="intelligent",
        prompts=(
            PromptProvenance(
                name=prompt.name,
                source_hash=prompt.source_hash,
                rendered_hash="sha256:rendered",
            ),
        ),
        prompt_bundle_hash="sha256:intelligent-bundle",
        models=(
            ModelProvenance(
                role="planning",
                model_class="tests.CloudPlanner",
                provider="google",
                name="cloud-planner",
            ),
            ModelProvenance(
                role="generation",
                model_class="tests.LocalGenerator",
                provider="ollama",
                name="local-generator",
            ),
        ),
    )


def test_intelligent_request_span_receives_selected_agent_provenance(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request root should receive the actual intelligent agent's compact map."""
    from src.api.main import app
    from src.api.routes import chat

    provenance = _intelligent_provenance()
    agent = MagicMock()
    agent.execution_provenance = provenance
    app.state.cloud_intelligent_agent = agent
    captured: list[dict[str, str | int | float | bool]] = []

    @contextmanager
    def _span(_context: dict[str, str]):
        yield MagicMock()

    monkeypatch.setattr(chat, "start_request_span", _span)
    monkeypatch.setattr(
        chat,
        "set_execution_provenance_attributes",
        lambda _span, attributes: captured.append(dict(attributes)),
    )
    monkeypatch.setattr(
        chat,
        "_ainvoke_intelligent_agent",
        AsyncMock(return_value=("ok", [], [])),
    )

    response = client.post(
        "/api/chat/",
        json={"message": "Hello", "agent_mode": "intelligent", "model_provider": "cloud"},
    )

    assert response.status_code == 200
    assert captured == [provenance.to_trace_attributes()]
    assert captured[0]["pour_decisions.model.planning.name"] == "cloud-planner"
    assert captured[0]["pour_decisions.model.generation.name"] == "local-generator"
    assert not any("rag_only" in key for key in captured[0])
    app.state.cloud_intelligent_agent = None


def test_local_rag_fallback_span_reports_actual_cloud_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local request resolved to cloud should trace the cloud generation model."""
    from src.api.main import app
    from src.api.routes import chat

    cloud_model = ChatGoogleGenerativeAI(model="gemini-fallback", google_api_key="test-key")
    app.state.local_model = None
    app.state.cloud_model = cloud_model
    app.state.model = cloud_model
    captured: list[dict[str, str | int | float | bool]] = []

    @contextmanager
    def _span(_context: dict[str, str]):
        yield MagicMock()

    monkeypatch.setattr(chat, "start_request_span", _span)
    monkeypatch.setattr(
        chat,
        "set_execution_provenance_attributes",
        lambda _span, attributes: captured.append(dict(attributes)),
    )

    response = client.post(
        "/api/chat/",
        json={"message": "Hello", "agent_mode": "rag_only", "model_provider": "local"},
    )

    assert response.status_code == 200
    assert response.json()["model_provider"] == "cloud"
    assert captured[0]["pour_decisions.execution.mode"] == "rag"
    assert captured[0]["pour_decisions.model.generation.provider"] == "google"
    assert captured[0]["pour_decisions.model.generation.name"] == "gemini-fallback"
    assert "pour_decisions.prompt.rag_only_system.source_hash" in captured[0]
    assert "pour_decisions.prompt.rag_only_user.source_hash" in captured[0]
    assert "pour_decisions.prompt.intelligent_agent_system.source_hash" not in captured[0]
