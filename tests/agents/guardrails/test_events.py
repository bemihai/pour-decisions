"""Tests for deterministic guardrail terminal responses."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.guardrails.events import (
    FAIL_SOFT_NOTE,
    FAIL_SOFT_RESPONSE,
    build_fail_soft_message,
    build_guardrail_trace_attributes,
    count_safe_tool_errors,
)


def test_existing_final_ai_answer_is_preserved_with_note() -> None:
    """A completed AI answer should survive deterministic termination."""
    result = build_fail_soft_message([AIMessage(content="A safe partial answer.")])

    assert result.content == f"A safe partial answer.\n\n{FAIL_SOFT_NOTE}"


def test_unresolved_tool_call_uses_generic_response() -> None:
    """Pending tool calls should never be rendered as a user answer."""
    message = AIMessage(
        content="internal planning text",
        tool_calls=[
            {
                "name": "search_wine",
                "args": {"query": "Barolo"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    assert build_fail_soft_message([message]).content == FAIL_SOFT_RESPONSE


def test_tool_message_content_is_never_serialized() -> None:
    """Raw tool output should not be copied into deterministic terminal text."""
    message = ToolMessage(content="SECRET_TOOL_OUTPUT", tool_call_id="call-1")

    result = build_fail_soft_message([message])

    assert result.content == FAIL_SOFT_RESPONSE
    assert "SECRET_TOOL_OUTPUT" not in str(result.content)


def test_missing_final_answer_uses_generic_response() -> None:
    """Human-only or empty state should use the bounded retry response."""
    assert build_fail_soft_message([]).content == FAIL_SOFT_RESPONSE
    assert build_fail_soft_message([HumanMessage(content="question")]).content == FAIL_SOFT_RESPONSE


def test_guardrail_trace_attributes_are_bounded_counts_and_triggers() -> None:
    """Trace metadata should summarize events without retaining sensitive payloads."""
    messages = [
        ToolMessage(
            content="[web_search_tool_failed] Web search is temporarily unavailable.",
            tool_call_id="call-1",
            status="error",
        ),
        AIMessage(content="Safe answer."),
    ]
    response = {
        "messages": messages,
        "llm_call_count": 2,
        "guardrail_events": [
            {
                "code": "exact_tool_call_duplicate",
                "tool_name": "search_web_for_wine",
                "duplicate_scope": "history",
            }
        ],
    }

    attributes = build_guardrail_trace_attributes(response, graph_limit=30, output_redaction_count=1)

    assert attributes == {
        "guardrail.call_budget.triggered": False,
        "guardrail.llm_calls": 2,
        "guardrail.graph_limit": 30,
        "guardrail.loop.triggered": True,
        "guardrail.relevance.triggered": False,
        "guardrail.tool_error.count": 1,
        "guardrail.output_redaction.count": 1,
        "guardrail.loop.tool_name": "search_web_for_wine",
    }
    assert "duplicate_scope" not in attributes
    assert "Web search" not in str(attributes)


def test_safe_tool_error_count_ignores_unrecognized_error_messages() -> None:
    """Only stable safe-error codes should contribute to the trace count."""
    messages = [
        ToolMessage(content="raw exception text", tool_call_id="call-1", status="error"),
        ToolMessage(content="[tool_execution_failed] Safe fallback.", tool_call_id="call-2", status="error"),
        ToolMessage(content="[tool_execution_failed] Non-error status.", tool_call_id="call-3"),
    ]

    assert count_safe_tool_errors(messages) == 1
