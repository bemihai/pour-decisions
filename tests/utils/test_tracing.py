"""Unit tests for observability setup and tracing helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

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
    monkeypatch.setattr(tracing, "_LANGCHAIN_INSTRUMENTED", False)


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


def test_init_observability_unsupported_provider_disables(caplog: pytest.LogCaptureFixture) -> None:
    """Unsupported providers should disable observability with a warning."""
    cfg = _build_cfg(enabled=True, provider="custom")

    tracing.init_observability(cfg)

    assert tracing._OBSERVABILITY_ENABLED is False
    assert "Unsupported observability provider" in caplog.text


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


def test_init_observability_phoenix_success_sets_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful Phoenix init should enable observability."""
    cfg = _build_cfg(enabled=True, provider="phoenix")

    monkeypatch.setattr(tracing, "_register_phoenix", lambda **_kwargs: None)
    monkeypatch.setattr(tracing, "_instrument_langchain", lambda: None)

    tracing.init_observability(cfg)

    assert tracing._OBSERVABILITY_ENABLED is True


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


def test_init_observability_uses_docker_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker runtime should select endpoint_docker for Phoenix registration."""
    cfg = _build_cfg(enabled=True, provider="phoenix")
    captured: dict[str, str] = {}

    monkeypatch.setattr(tracing, "_is_docker_runtime", lambda: True)

    def _capture_register(endpoint: str, project_name: str) -> None:
        captured["endpoint"] = endpoint
        captured["project_name"] = project_name

    monkeypatch.setattr(tracing, "_register_phoenix", _capture_register)
    monkeypatch.setattr(tracing, "_instrument_langchain", lambda: None)

    tracing.init_observability(cfg)

    assert tracing._OBSERVABILITY_ENABLED is True
    assert captured["endpoint"] == "http://phoenix:6006/v1/traces"
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


def test_cost_tracking_callback_usage_metadata_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callback should fall back to usage_metadata when llm_output token_usage is absent."""

    class _FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

    fake_span = _FakeSpan()
    monkeypatch.setattr(tracing.trace, "get_current_span", lambda: fake_span)

    callback = tracing.CostTrackingCallback()
    generation = SimpleNamespace(
        message=SimpleNamespace(
            usage_metadata={
                "input_tokens": 30,
                "output_tokens": 12,
            }
        )
    )
    response = SimpleNamespace(llm_output={}, generations=[[generation]])

    callback.on_llm_end(response)

    assert fake_span.attributes["llm_input_tokens"] == 30
    assert fake_span.attributes["llm_output_tokens"] == 12


def test_get_tracing_callbacks_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled observability should return a cost-tracking callback."""
    monkeypatch.setattr(tracing, "_OBSERVABILITY_ENABLED", True)

    callbacks = tracing.get_tracing_callbacks()

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], tracing.CostTrackingCallback)


def test_get_tracing_callbacks_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled observability should return no tracing callbacks."""
    monkeypatch.setattr(tracing, "_OBSERVABILITY_ENABLED", False)

    callbacks = tracing.get_tracing_callbacks()

    assert callbacks == []


def test_start_request_span_enabled_sets_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabled request span should call attribute setter with trace metadata."""

    class _FakeSpan:
        pass

    from contextlib import contextmanager

    class _FakeTracer:
        @contextmanager
        def start_as_current_span(self, _name: str):
            yield _FakeSpan()

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(tracing, "_OBSERVABILITY_ENABLED", True)
    monkeypatch.setattr(tracing, "_TRACER", _FakeTracer())
    monkeypatch.setattr(tracing, "set_span_attributes", lambda _span, attrs: captured.append(attrs))

    with tracing.start_request_span({"request_id": "req-enabled", "agent_mode": "rag_only"}) as span:
        assert span is not None

    assert captured == [{"request_id": "req-enabled", "agent_mode": "rag_only"}]


def test_set_span_attributes_skips_none() -> None:
    """set_span_attributes should not write None-valued keys."""

    class _FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

    span = _FakeSpan()
    tracing.set_span_attributes(cast(Any, span), {"request_id": "abc", "session_id": None})

    assert span.attributes == {"request_id": "abc"}


def test_set_current_span_attributes_uses_active_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Current-span helper should reuse the active request span."""
    span = MagicMock()
    monkeypatch.setattr(tracing.trace, "get_current_span", lambda: span)

    tracing.set_current_span_attributes({"guardrail.llm_calls": 2})

    span.set_attribute.assert_called_once_with("guardrail.llm_calls", 2)


def test_set_execution_provenance_attributes_delegates_flat_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution provenance should reuse the existing safe batch setter."""
    span = MagicMock()
    captured: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        tracing,
        "set_span_attributes",
        lambda target, attributes: captured.append((target, attributes)),
    )
    attributes = {
        "pour_decisions.execution.mode": "intelligent",
        "pour_decisions.tools.selected_count": 3,
    }

    tracing.set_execution_provenance_attributes(span, attributes)

    assert captured == [(span, attributes)]


def test_set_execution_provenance_attributes_accepts_disabled_span() -> None:
    """A disabled request span should remain a no-op without raising."""
    tracing.set_execution_provenance_attributes(
        None,
        {"pour_decisions.execution.mode": "rag"},
    )
