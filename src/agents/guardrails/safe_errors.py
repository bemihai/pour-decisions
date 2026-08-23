"""Stable, non-disclosing messages for unexpected tool failures."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TypeAlias

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from src.agents.tools.registry import ToolCategory, ToolSelectionSnapshot


ToolCallResult: TypeAlias = ToolMessage | Command
ToolCallExecutor: TypeAlias = Callable[[Any], ToolCallResult]
ToolCallWrapper: TypeAlias = Callable[[Any, ToolCallExecutor], ToolCallResult]


class SafeToolErrorCode(str, Enum):
    """Stable internal codes for intelligent-agent tool failures."""

    CELLAR_UNAVAILABLE = "cellar_tool_failed"
    TASTE_PROFILE_UNAVAILABLE = "taste_profile_tool_failed"
    PAIRING_UNAVAILABLE = "pairing_tool_failed"
    WINE_KNOWLEDGE_UNAVAILABLE = "wine_knowledge_tool_failed"
    WEB_SEARCH_UNAVAILABLE = "web_search_tool_failed"
    TOOL_UNAVAILABLE = "tool_execution_failed"


@dataclass(frozen=True)
class SafeToolError:
    """One stable safe code and recovery message."""

    code: SafeToolErrorCode
    message: str


_SAFE_ERRORS_BY_CATEGORY = {
    ToolCategory.CELLAR: SafeToolError(
        code=SafeToolErrorCode.CELLAR_UNAVAILABLE,
        message="Cellar data is temporarily unavailable. Continue without it.",
    ),
    ToolCategory.TASTE_PROFILE: SafeToolError(
        code=SafeToolErrorCode.TASTE_PROFILE_UNAVAILABLE,
        message="Taste profile data is temporarily unavailable. Continue without it.",
    ),
    ToolCategory.PAIRING: SafeToolError(
        code=SafeToolErrorCode.PAIRING_UNAVAILABLE,
        message="Wine pairing guidance is temporarily unavailable. Continue without it.",
    ),
    ToolCategory.RAG: SafeToolError(
        code=SafeToolErrorCode.WINE_KNOWLEDGE_UNAVAILABLE,
        message="Wine knowledge search is temporarily unavailable. Continue without it.",
    ),
    ToolCategory.WEB_SEARCH: SafeToolError(
        code=SafeToolErrorCode.WEB_SEARCH_UNAVAILABLE,
        message="Web search is temporarily unavailable. Continue without it.",
    ),
}

_GENERIC_SAFE_TOOL_ERROR = SafeToolError(
    code=SafeToolErrorCode.TOOL_UNAVAILABLE,
    message="This tool is temporarily unavailable. Continue without it.",
)


def get_safe_tool_error(category: ToolCategory | None) -> SafeToolError:
    """Return the stable safe error for a tool category or the generic fallback."""
    if category is None:
        return _GENERIC_SAFE_TOOL_ERROR
    return _SAFE_ERRORS_BY_CATEGORY.get(category, _GENERIC_SAFE_TOOL_ERROR)


def format_safe_tool_error(error: SafeToolError) -> str:
    """Format a safe tool error for LangGraph ``ToolMessage`` content."""
    return f"[{error.code.value}] {error.message}"


def build_safe_tool_call_wrapper(snapshot: ToolSelectionSnapshot) -> ToolCallWrapper:
    """Build a category-aware error boundary from one immutable M6 snapshot."""
    categories_by_name = {
        definition.metadata.name: definition.metadata.category
        for definition in snapshot.definitions
    }

    def wrap_tool_call(request: Any, execute: ToolCallExecutor) -> ToolCallResult:
        """Execute one tool call and convert unexpected failures to safe output."""
        try:
            return execute(request)
        except GraphBubbleUp:
            raise
        except Exception:
            tool_call = request.tool_call
            tool_name = str(tool_call.get("name", ""))
            safe_error = get_safe_tool_error(categories_by_name.get(tool_name))
            return ToolMessage(
                content=format_safe_tool_error(safe_error),
                name=tool_name,
                tool_call_id=str(tool_call.get("id", "")),
                status="error",
            )

    return wrap_tool_call
