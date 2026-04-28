"""Unit tests for observability setup and tracing helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import cast

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


def test_normalize_phoenix_otlp_endpoint_appends_traces_path() -> None:
    """Base Phoenix URL should be normalized to OTLP traces endpoint."""
    endpoint = tracing._normalize_phoenix_otlp_endpoint("http://localhost:6006")

    assert endpoint == "http://localhost:6006/v1/traces"


def test_normalize_phoenix_otlp_endpoint_keeps_existing_traces_path() -> None:
    """Already-correct OTLP endpoint should be preserved."""
    endpoint = tracing._normalize_phoenix_otlp_endpoint("http://phoenix:6006/v1/traces")

    assert endpoint == "http://phoenix:6006/v1/traces"


def test_init_observability_registers_normalized_phoenix_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """init_observability should register OTLP traces endpoint even from base URL config."""
    cfg = _build_cfg(enabled=True, provider="phoenix")
    captured: dict[str, str] = {}

    monkeypatch.setattr(tracing, "_is_docker_runtime", lambda: False)

    def _capture_register(endpoint: str, project_name: str) -> None:
        captured["endpoint"] = endpoint
        captured["project_name"] = project_name

    monkeypatch.setattr(tracing, "_register_phoenix", _capture_register)
    monkeypatch.setattr(tracing, "_instrument_langchain", lambda: None)

    tracing.init_observability(cfg)

    assert captured["endpoint"] == "http://localhost:6006/v1/traces"
    assert captured["project_name"] == "pour-decisions"


def test_compute_equivalent_cost_zero_tokens() -> None:
    """Zero token usage should produce zero equivalent cost."""
    cost = tracing.compute_equivalent_cost(0, 0, "gemini-2.5-flash")

    assert cost["actual_billed_cost_usd"] == 0.0
    assert cost["equivalent_paid_cost_usd"] == 0.0


def test_compute_equivalent_cost_known_values() -> None:
    """Known token counts should match expected equivalent paid cost."""
    cost = tracing.compute_equivalent_cost(1_000_000, 1_000_000, "gemini-2.5-flash")

    assert cost["equivalent_paid_cost_usd"] == 0.75


def test_cost_tracking_callback_sets_span_cost_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callback should extract token usage and set cost attributes on current span."""

    class _FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

    fake_span = _FakeSpan()
    monkeypatch.setattr(tracing.trace, "get_current_span", lambda: fake_span)

    callback = tracing.CostTrackingCallback()
    response = SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            },
            "model_name": "gemini-2.5-flash",
        }
    )

    callback.on_llm_end(response)

    assert fake_span.attributes["llm_input_tokens"] == 100
    assert fake_span.attributes["llm_output_tokens"] == 50
    assert fake_span.attributes["actual_billed_cost_usd"] == 0.0
    equivalent_cost = cast(float, fake_span.attributes["equivalent_paid_cost_usd"])
    assert equivalent_cost > 0.0


def test_get_tracing_callbacks_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled observability should return a cost-tracking callback."""
    monkeypatch.setattr(tracing, "_OBSERVABILITY_ENABLED", True)

    callbacks = tracing.get_tracing_callbacks()

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], tracing.CostTrackingCallback)


