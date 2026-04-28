"""Unit tests for observability setup and tracing helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest

from src.utils import tracing


def _build_cfg(enabled: bool, provider: str = "phoenix") -> SimpleNamespace:
    """Build a minimal config object for tracing tests.

    Args:
        enabled: Observability enabled flag.
        provider: Observability provider name.

    Returns:
        SimpleNamespace with nested observability settings.
    """
    return SimpleNamespace(
        observability=SimpleNamespace(
            enabled=enabled,
            provider=provider,
            phoenix=SimpleNamespace(
                endpoint="http://localhost:6006",
                endpoint_docker="http://phoenix:6006",
                project_name="pour-decisions",
            ),
        )
    )


@pytest.fixture(autouse=True)
def _reset_tracing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset module-level tracing state between tests."""
    monkeypatch.setattr(tracing, "_OBSERVABILITY_ENABLED", False)
    monkeypatch.setattr(tracing, "_LANGFUSE_STUB_WARNING_EMITTED", False)


def test_init_observability_disabled() -> None:
    """Disabled observability should be a no-op without raising."""
    cfg = _build_cfg(enabled=False)

    tracing.init_observability(cfg)

    assert tracing._OBSERVABILITY_ENABLED is False


def test_init_observability_none_provider() -> None:
    """Provider 'none' should keep observability disabled."""
    cfg = _build_cfg(enabled=True, provider="none")

    tracing.init_observability(cfg)

    assert tracing._OBSERVABILITY_ENABLED is False


def test_init_observability_phoenix_endpoint_unreachable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Phoenix setup failure should be fail-open with a warning."""
    cfg = _build_cfg(enabled=True, provider="phoenix")

    def _raise_register(endpoint: str, project_name: str) -> None:
        raise RuntimeError(f"cannot reach endpoint {endpoint} for project {project_name}")

    monkeypatch.setattr(tracing, "_register_phoenix", _raise_register)

    tracing.init_observability(cfg)

    assert tracing._OBSERVABILITY_ENABLED is False
    assert "Observability initialization failed" in caplog.text


def test_get_trace_context_with_ids() -> None:
    """Trace context should include request, session, and mode when present."""
    context = tracing.get_trace_context(request_id="req-1", session_id="sess-1", agent_mode="rag_only")

    assert context["request_id"] == "req-1"
    assert context["session_id"] == "sess-1"
    assert context["agent_mode"] == "rag_only"


def test_get_trace_context_without_session_id() -> None:
    """Trace context should omit session_id when not provided."""
    context = tracing.get_trace_context(request_id="req-1", session_id=None, agent_mode="intelligent")

    assert context["request_id"] == "req-1"
    assert context["agent_mode"] == "intelligent"
    assert "session_id" not in context


def test_start_request_span_disabled_returns_noop_context() -> None:
    """Disabled observability should return a no-op span context manager."""
    manager = tracing.start_request_span({"request_id": "req-1"})

    assert isinstance(manager, AbstractContextManager)
    with manager as span:
        assert span is None


def test_get_langfuse_callback_returns_none_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    """Compatibility stub should return None and emit a single warning."""
    result_first = tracing.get_langfuse_callback()
    result_second = tracing.get_langfuse_callback()

    assert result_first is None
    assert result_second is None
    assert caplog.text.count("get_langfuse_callback() is deprecated") == 1

