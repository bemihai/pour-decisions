"""Observability helpers for API request tracing and LangChain instrumentation.

This module provides provider-aware observability setup with a Phoenix v1
implementation and OpenTelemetry span helper utilities.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os
from typing import Any
from urllib.parse import urlsplit

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace
from opentelemetry.trace import Span

from src.utils.logger import logger

_OBSERVABILITY_ENABLED = False
_LANGCHAIN_INSTRUMENTED = False
_TRACER = trace.get_tracer(__name__)


def _parse_bool_config(value: Any) -> bool:
    """Convert a config value to bool using explicit string parsing.

    Args:
        value: Raw config value that may be a bool, string, numeric, or None.

    Returns:
        The parsed boolean value. Unknown or empty values default to False.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "t", "yes", "y", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def is_observability_active() -> bool:
    """Return True when observability was successfully initialized at startup.

    This reflects the runtime state set by :func:`init_observability` and is
    the single authoritative flag for both span creation and trace-ID emission.

    Returns:
        True when the observability backend is active and spans are being emitted.
    """
    return _OBSERVABILITY_ENABLED


_GEMINI_FLASH_PRICING = {
    "input_per_million": 0.15,
    "output_per_million": 0.60,
}


def _to_int(value: Any) -> int:
    """Convert a token-count candidate to int safely.

    Args:
        value: Any token count value.

    Returns:
        Parsed integer or 0 on conversion failure.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def compute_equivalent_cost(input_tokens: int, output_tokens: int, model_name: str) -> dict[str, Any]:
    """Compute equivalent paid cost for token usage.

    Args:
        input_tokens: Prompt tokens.
        output_tokens: Completion tokens.
        model_name: Model identifier.

    Returns:
        Cost attributes suitable for span metadata.
    """
    equivalent_cost_usd = (
        input_tokens / 1_000_000 * _GEMINI_FLASH_PRICING["input_per_million"]
    ) + (
        output_tokens / 1_000_000 * _GEMINI_FLASH_PRICING["output_per_million"]
    )

    return {
        "actual_billed_cost_usd": 0.0,
        "equivalent_paid_cost_usd": round(equivalent_cost_usd, 8),
        "model_name": model_name,
    }


class CostTrackingCallback(BaseCallbackHandler):
    """LangChain callback that attaches token and equivalent-cost span attributes."""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Attach token/cost attributes to the current active span.

        Args:
            response: LLM result object from LangChain callbacks.
            **kwargs: Additional callback args.
        """
        current_span = trace.get_current_span()
        if current_span is None:
            return

        llm_output = getattr(response, "llm_output", {}) or {}
        token_usage = llm_output.get("token_usage", {}) if isinstance(llm_output, dict) else {}

        input_tokens = _to_int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)))
        output_tokens = _to_int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)))
        model_name = "unknown"

        if isinstance(llm_output, dict):
            model_name = str(llm_output.get("model_name", kwargs.get("model_name", "unknown")))

        if input_tokens == 0 and output_tokens == 0:
            usage_metadata = None
            generations = getattr(response, "generations", None)
            if generations and isinstance(generations, list) and generations and generations[0]:
                first_generation = generations[0][0]
                message = getattr(first_generation, "message", None)
                usage_metadata = getattr(message, "usage_metadata", None)

            if isinstance(usage_metadata, dict):
                input_tokens = _to_int(usage_metadata.get("input_tokens", usage_metadata.get("prompt_token_count", 0)))
                output_tokens = _to_int(
                    usage_metadata.get("output_tokens", usage_metadata.get("candidates_token_count", 0))
                )

        set_span_attributes(
            current_span,
            {
                "llm_input_tokens": input_tokens,
                "llm_output_tokens": output_tokens,
            },
        )
        set_span_attributes(current_span, compute_equivalent_cost(input_tokens, output_tokens, model_name))


def get_tracing_callbacks() -> list[BaseCallbackHandler]:
    """Return tracing callbacks for LangChain invocations.

    Returns:
        Callback list for cost tagging when observability is enabled.
    """
    if not _OBSERVABILITY_ENABLED:
        return []
    return [CostTrackingCallback()]


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


def _normalize_phoenix_otlp_endpoint(endpoint: str) -> str:
    """Normalize Phoenix endpoint to an OTLP traces URL.

    The OTLP HTTP exporter posts traces to ``/v1/traces``. If configuration
    provides only the Phoenix base URL (for example ``http://localhost:6006``),
    this helper appends the OTLP path.

    Args:
        endpoint: Configured Phoenix endpoint.

    Returns:
        Endpoint URL safe for OTLP trace export.
    """
    cleaned = endpoint.strip().rstrip("/")
    if not cleaned:
        return "http://localhost:6006/v1/traces"

    path = urlsplit(cleaned).path
    if path.endswith("/v1/traces"):
        return cleaned
    if path in ("", "/"):
        return f"{cleaned}/v1/traces"
    return cleaned


def _instrument_langchain() -> None:
    """Enable LangChain/LangGraph auto-instrumentation."""
    global _LANGCHAIN_INSTRUMENTED

    if _LANGCHAIN_INSTRUMENTED:
        return

    from openinference.instrumentation.langchain import LangChainInstrumentor

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument()
    _LANGCHAIN_INSTRUMENTED = True


def init_observability(cfg: Any) -> None:
    """Initialize observability wiring once at startup.

    The function is fail-open by design: any setup error disables
    instrumentation and allows the API to keep serving requests.

    Args:
        cfg: Application config object (OmegaConf).
    """
    global _OBSERVABILITY_ENABLED

    observability_cfg = getattr(cfg, "observability", None)
    enabled = _parse_bool_config(getattr(observability_cfg, "enabled", False))
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
    normalized_endpoint = _normalize_phoenix_otlp_endpoint(endpoint)
    project_name = str(getattr(phoenix_cfg, "project_name", "pour-decisions"))

    try:
        _register_phoenix(endpoint=normalized_endpoint, project_name=project_name)
        _instrument_langchain()
        _OBSERVABILITY_ENABLED = True
        logger.info(f"Observability initialized with Phoenix ({normalized_endpoint})")
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


