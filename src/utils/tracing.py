"""Observability helpers for API request tracing and LangChain instrumentation.

This module provides provider-aware observability setup with a Phoenix v1
implementation and OpenTelemetry span helper utilities.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

from src.utils.logger import logger

_OBSERVABILITY_ENABLED = False
_TRACER = trace.get_tracer(__name__)


def _is_docker_runtime() -> bool:
    """Return True when the API is running inside Docker Compose.

    Returns:
        True when the process can resolve dependencies via Docker service names.
    """
    return os.environ.get("CHROMA_HOST") == "chromadb"


def _register_phoenix(endpoint: str, project_name: str) -> None:
    """Register Phoenix OpenTelemetry exporter.

    Args:
        endpoint: Phoenix OTLP endpoint.
        project_name: Project name shown in Phoenix.
    """
    from phoenix.otel import register

    register(endpoint=endpoint, project_name=project_name)


def _instrument_langchain() -> None:
    """Enable LangChain/LangGraph auto-instrumentation."""
    from openinference.instrumentation.langchain import LangChainInstrumentor

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument()


def init_observability(cfg: Any) -> None:
    """Initialize observability wiring once at startup.

    The function is fail-open by design: any setup error disables
    instrumentation and allows the API to keep serving requests.

    Args:
        cfg: Application config object (OmegaConf).
    """
    global _OBSERVABILITY_ENABLED

    observability_cfg = getattr(cfg, "observability", None)
    enabled = bool(getattr(observability_cfg, "enabled", False))
    provider = str(getattr(observability_cfg, "provider", "none")).lower()

    if not enabled or provider == "none":
        _OBSERVABILITY_ENABLED = False
        return

    if provider != "phoenix":
        logger.warning(f"Unsupported observability provider '{provider}'. Observability is disabled.")
        _OBSERVABILITY_ENABLED = False
        return

    phoenix_cfg = getattr(observability_cfg, "phoenix", None)
    endpoint_key = "endpoint_docker" if _is_docker_runtime() else "endpoint"
    endpoint = str(getattr(phoenix_cfg, endpoint_key, "http://localhost:6006"))
    project_name = str(getattr(phoenix_cfg, "project_name", "pour-decisions"))

    try:
        _register_phoenix(endpoint=endpoint, project_name=project_name)
        _instrument_langchain()
        _OBSERVABILITY_ENABLED = True
        logger.info(f"Observability initialized with Phoenix ({endpoint})")
    except Exception as err:
        _OBSERVABILITY_ENABLED = False
        logger.warning(f"Observability initialization failed. Tracing disabled: {err}")


def get_trace_context(request_id: str, session_id: str | None, agent_mode: str) -> dict[str, str]:
    """Build normalized request trace metadata.

    Args:
        request_id: Correlation ID for the current request.
        session_id: Optional client session ID.
        agent_mode: Selected execution mode.

    Returns:
        Request metadata safe to propagate through call stacks.
    """
    context: dict[str, str] = {
        "request_id": request_id,
        "agent_mode": agent_mode,
    }
    if session_id:
        context["session_id"] = session_id
    return context


@contextmanager
def start_request_span(trace_context: dict[str, Any]):
    """Start a request-level span when observability is enabled.

    Args:
        trace_context: Request metadata that will be attached as span attributes.

    Yields:
        Active OpenTelemetry span or None when instrumentation is disabled.
    """
    if not _OBSERVABILITY_ENABLED:
        with nullcontext(None) as no_op_span:
            yield no_op_span
        return

    with _TRACER.start_as_current_span("chat_request") as span:
        set_span_attributes(span, trace_context)
        yield span


def set_span_attributes(span: Span | None, attributes: dict[str, Any]) -> None:
    """Set a batch of attributes on an active span.

    Args:
        span: OpenTelemetry span or None.
        attributes: Mapping of attribute keys and values.
    """
    if span is None:
        return

    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            span.set_attribute(key, str(value))


