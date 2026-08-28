"""Deterministic terminal behavior shared by intelligent-agent guardrails."""

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage


FAIL_SOFT_RESPONSE = "I couldn't complete this request safely. Please retry with a narrower question."
FAIL_SOFT_NOTE = "I reached a processing limit, so some requested details may be incomplete."


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
