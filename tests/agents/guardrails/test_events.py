"""Tests for deterministic guardrail terminal responses."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.guardrails.events import (
    FAIL_SOFT_NOTE,
    FAIL_SOFT_RESPONSE,
    build_fail_soft_message,
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
