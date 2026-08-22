"""Deterministic agent trajectory metrics and tool-output classification."""

from collections import Counter
from typing import Any

from src.eval.models import AgentToolOutput

TOOL_TRAJECTORY_METRICS = (
    "tool_recall",
    "tool_precision",
    "tool_exact_match",
    "tool_ordered_match",
)

_RAG_TOOLS = frozenset(
    {
        "search_wine_knowledge",
        "search_wine_region_info",
        "search_grape_variety_info",
        "search_wine_term_definition",
        "search_wine_producer_info",
    }
)
_CELLAR_TOOLS = frozenset(
    {
        "get_cellar_wines",
        "get_wine_details",
        "get_cellar_statistics",
    }
)
_PAIRING_TOOLS = frozenset(
    {
        "get_food_pairing_wines",
        "get_pairing_for_wine",
        "get_wine_and_cheese_pairings",
    }
)
_WEB_TOOLS = frozenset(
    {
        "search_web_for_wine",
        "search_wine_price",
        "search_wine_reviews",
    }
)


def score_expected_tool_calls(expected: list[str], actual: list[str]) -> dict[str, float]:
    """Score an observed tool trajectory against explicitly required calls.

    Precision and recall use multiset overlap so redundant calls are penalized.
    Ordered match accepts extra calls while requiring the expected calls to occur
    in order. Exact match requires identical names, order, and call counts.

    Args:
        expected: Required tool names in expected order.
        actual: Observed tool names in execution order.

    Returns:
        Tool metric scores, or an empty mapping when no requirements were given.
    """
    if not expected:
        return {}

    overlap_count = sum((Counter(expected) & Counter(actual)).values())
    recall = overlap_count / len(expected)
    precision = overlap_count / len(actual) if actual else 0.0

    return {
        "tool_recall": recall,
        "tool_precision": precision,
        "tool_exact_match": float(actual == expected),
        "tool_ordered_match": float(_is_ordered_subsequence(expected, actual)),
    }


def extract_agent_tool_calls(messages: list[Any], fallback: list[str] | None = None) -> list[str]:
    """Extract tool calls in execution order from agent messages.

    AI tool-call requests are authoritative when present. Tool messages provide
    a fallback for providers that omit structured calls, followed by the agent's
    summary list as a final compatibility fallback.
    """
    requested_calls: list[str] = []
    completed_calls: list[str] = []

    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            name = _extract_tool_call_name(tool_call)
            if name:
                requested_calls.append(name)

        if getattr(message, "type", None) == "tool":
            name = getattr(message, "name", None)
            if name:
                completed_calls.append(str(name))

    if requested_calls:
        return requested_calls
    if completed_calls:
        return completed_calls
    return [str(name) for name in (fallback or []) if str(name)]


def extract_agent_tool_outputs(messages: list[Any]) -> list[AgentToolOutput]:
    """Capture typed, normalized outputs from agent tool messages."""
    outputs: list[AgentToolOutput] = []
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue

        tool_name = str(getattr(message, "name", "") or "")
        outputs.append(
            AgentToolOutput(
                tool_name=tool_name,
                output_type=classify_tool_output(tool_name),
                content=_normalize_tool_content(getattr(message, "content", "")),
            )
        )
    return outputs


def classify_tool_output(tool_name: str) -> str:
    """Map an agent tool name to its evaluation evidence type."""
    if tool_name in _RAG_TOOLS:
        return "rag_context"
    if tool_name in _CELLAR_TOOLS:
        return "cellar_result"
    if tool_name in _PAIRING_TOOLS:
        return "pairing_result"
    if tool_name in _WEB_TOOLS:
        return "web_result"
    return "other_result"


def _is_ordered_subsequence(expected: list[str], actual: list[str]) -> bool:
    """Return whether expected calls appear in order within actual calls."""
    expected_index = 0
    for actual_name in actual:
        if actual_name == expected[expected_index]:
            expected_index += 1
            if expected_index == len(expected):
                return True
    return False


def _extract_tool_call_name(tool_call: Any) -> str:
    """Extract a tool name from mapping- or object-shaped call metadata."""
    if isinstance(tool_call, dict):
        return str(tool_call.get("name", "") or "")
    if hasattr(tool_call, "get"):
        return str(tool_call.get("name", "") or "")
    return str(getattr(tool_call, "name", "") or "")


def _normalize_tool_content(content: Any) -> str:
    """Normalize supported LangChain tool-message content shapes to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content if str(item).strip())
    if content is None:
        return ""
    return str(content)
