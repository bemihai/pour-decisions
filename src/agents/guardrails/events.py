"""Deterministic terminal behavior and observability for agent guardrails."""

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .budget import CALL_BUDGET_EVENT_CODE
from .loop_detector import LOOP_DETECTED_EVENT_CODE
from .relevance import RELEVANCE_DEFLECTED_EVENT_CODE
from .safe_errors import SafeToolErrorCode
from .tool_execution import ToolExecutionEventCode


FAIL_SOFT_RESPONSE = "I couldn't complete this request safely. Please retry with a narrower question."
FAIL_SOFT_NOTE = "I reached a processing limit, so some requested details may be incomplete."

_SAFE_TOOL_ERROR_PREFIXES = tuple(f"[{code.value}]" for code in SafeToolErrorCode)


def build_fail_soft_message(messages: Sequence[BaseMessage]) -> AIMessage:
    """Build a deterministic terminal message without exposing tool content.

    Args:
        messages: Current graph messages in chronological order.

    Returns:
        A safe terminal AI message. Existing final AI text is preserved with a
        bounded note; unresolved calls and non-AI messages use the generic text.
    """
    last_message = messages[-1] if messages else None
    content = getattr(last_message, "content", "")
    tool_calls = getattr(last_message, "tool_calls", [])
    if isinstance(last_message, AIMessage) and isinstance(content, str) and content.strip() and not tool_calls:
        return AIMessage(content=f"{content.strip()}\n\n{FAIL_SOFT_NOTE}")
    return AIMessage(content=FAIL_SOFT_RESPONSE)


def build_guardrail_trace_attributes(
    response: dict[str, Any],
    graph_limit: int,
    output_redaction_count: int,
    tool_concurrency_limit: int,
) -> dict[str, str | int | bool]:
    """Build low-cardinality guardrail attributes for the active request span.

    Args:
        response: Final graph state returned by the intelligent agent.
        graph_limit: Configured LangGraph recursion limit for the request.
        output_redaction_count: Number of final-output sanitizer replacements.
        tool_concurrency_limit: Configured app-worker tool admission limit.

    Returns:
        Attributes containing only bounded codes, counts, booleans, and a
        catalogue tool name when an exact duplicate was detected.
    """
    events = response.get("guardrail_events", [])
    event_codes = {
        str(event.get("code", ""))
        for event in events
        if isinstance(event, dict)
    }
    loop_event = next(
        (
            event
            for event in events
            if isinstance(event, dict) and event.get("code") == LOOP_DETECTED_EVENT_CODE
        ),
        None,
    )

    attributes: dict[str, str | int | bool] = {
        "guardrail.call_budget.triggered": CALL_BUDGET_EVENT_CODE in event_codes,
        "guardrail.llm_calls": _non_negative_int(response.get("llm_call_count", 0)),
        "guardrail.graph_limit": graph_limit,
        "guardrail.loop.triggered": LOOP_DETECTED_EVENT_CODE in event_codes,
        "guardrail.relevance.triggered": RELEVANCE_DEFLECTED_EVENT_CODE in event_codes,
        "guardrail.tool_error.count": count_safe_tool_errors(response.get("messages", [])),
        "guardrail.tool.timeout.count": _count_event_code(
            events,
            ToolExecutionEventCode.DEADLINE_EXCEEDED.value,
        ),
        "guardrail.tool.sync_timeout.count": _count_event_code(
            events,
            ToolExecutionEventCode.SYNC_TIMEOUT.value,
        ),
        "guardrail.tool.retry.count": _count_event_code(
            events,
            ToolExecutionEventCode.RETRY_STARTED.value,
        ),
        "guardrail.tool.retry_success.count": _count_event_code(
            events,
            ToolExecutionEventCode.RETRY_SUCCEEDED.value,
        ),
        "guardrail.tool.terminal_failure.count": _count_event_code(
            events,
            ToolExecutionEventCode.TERMINAL_FAILURE.value,
        ),
        "guardrail.tool.concurrency.limit": _non_negative_int(tool_concurrency_limit),
        "guardrail.output_redaction.count": max(output_redaction_count, 0),
    }
    if loop_event is not None:
        tool_name = loop_event.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            attributes["guardrail.loop.tool_name"] = tool_name
    return attributes


def count_safe_tool_errors(messages: Sequence[BaseMessage]) -> int:
    """Count stable safe-error tool messages without retaining their content."""
    return sum(
        1
        for message in messages
        if isinstance(message, ToolMessage)
        and message.status == "error"
        and isinstance(message.content, str)
        and message.content.startswith(_SAFE_TOOL_ERROR_PREFIXES)
    )


def _count_event_code(events: object, expected_code: str) -> int:
    """Count exact stable codes while ignoring malformed event containers."""
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return 0
    return sum(
        1
        for event in events
        if isinstance(event, dict) and event.get("code") == expected_code
    )


def _non_negative_int(value: Any) -> int:
    """Return a non-negative integer for an internal counter candidate."""
    return value if type(value) is int and value >= 0 else 0
