"""Tests for stable, non-disclosing tool-error messages."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from src.agents.guardrails.safe_errors import (
    SafeToolErrorCode,
    build_async_safe_tool_call_wrapper,
    build_safe_tool_call_wrapper,
    format_safe_tool_error,
    get_safe_tool_error,
)
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolCategory, ToolSelectionSnapshot


RAW_FAILURE = "M06A_SYNTHETIC_PRIVATE_FAILURE"


def _snapshot() -> ToolSelectionSnapshot:
    """Return the complete immutable tool snapshot for wrapper tests."""
    return ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=())


def _raise_unexpected(_request: object) -> ToolMessage:
    """Raise one synthetic failure from a synchronous handler."""
    raise RuntimeError(RAW_FAILURE)


async def _raise_unexpected_async(_request: object) -> ToolMessage:
    """Raise one synthetic failure from an asynchronous handler."""
    raise RuntimeError(RAW_FAILURE)


def _raise_graph_bubble_up(_request: object) -> ToolMessage:
    """Raise LangGraph control flow from a synchronous handler."""
    raise GraphBubbleUp()


async def _raise_graph_bubble_up_async(_request: object) -> ToolMessage:
    """Raise LangGraph control flow from an asynchronous handler."""
    raise GraphBubbleUp()


async def _raise_upstream_timeout_async(_request: object) -> ToolMessage:
    """Raise a timeout owned by an upstream tool implementation."""
    raise TimeoutError("synthetic upstream timeout")


async def _raise_caller_cancellation_async(_request: object) -> ToolMessage:
    """Raise cancellation originating outside normal tool failure handling."""
    raise asyncio.CancelledError()


@pytest.mark.parametrize(
    ("category", "expected_code", "expected_message"),
    (
        (
            ToolCategory.CELLAR,
            SafeToolErrorCode.CELLAR_UNAVAILABLE,
            "Cellar data is temporarily unavailable. Continue without it.",
        ),
        (
            ToolCategory.TASTE_PROFILE,
            SafeToolErrorCode.TASTE_PROFILE_UNAVAILABLE,
            "Taste profile data is temporarily unavailable. Continue without it.",
        ),
        (
            ToolCategory.PAIRING,
            SafeToolErrorCode.PAIRING_UNAVAILABLE,
            "Wine pairing guidance is temporarily unavailable. Continue without it.",
        ),
        (
            ToolCategory.RAG,
            SafeToolErrorCode.WINE_KNOWLEDGE_UNAVAILABLE,
            "Wine knowledge search is temporarily unavailable. Continue without it.",
        ),
        (
            ToolCategory.WEB_SEARCH,
            SafeToolErrorCode.WEB_SEARCH_UNAVAILABLE,
            "Web search is temporarily unavailable. Continue without it.",
        ),
    ),
)
def test_category_safe_errors_are_stable(
    category: ToolCategory,
    expected_code: SafeToolErrorCode,
    expected_message: str,
) -> None:
    """Every active M6 category should resolve to its reviewed safe output."""
    error = get_safe_tool_error(category)

    assert error.code is expected_code
    assert error.message == expected_message
    assert format_safe_tool_error(error) == f"[{expected_code.value}] {expected_message}"


def test_missing_category_uses_stable_generic_fallback() -> None:
    """Unknown tool metadata should not prevent safe failure handling."""
    error = get_safe_tool_error(None)

    assert error.code is SafeToolErrorCode.TOOL_UNAVAILABLE
    assert error.message == "This tool is temporarily unavailable. Continue without it."


def test_safe_error_content_cannot_include_raw_exception_text() -> None:
    """Safe outputs should be independent of provider failure details."""
    raw_failure = "Set M09A_SYNTHETIC_PROVIDER_TOKEN=/private/secret"
    content = format_safe_tool_error(get_safe_tool_error(ToolCategory.WEB_SEARCH))

    assert raw_failure not in content
    assert "M09A_SYNTHETIC_PROVIDER_TOKEN" not in content
    assert "/private/secret" not in content


@pytest.mark.asyncio
@pytest.mark.parametrize("category", tuple(ToolCategory))
async def test_sync_and_async_wrappers_return_equivalent_category_errors(category: ToolCategory) -> None:
    """Both wrappers should preserve every category-specific ToolMessage field."""
    definition = next(
        definition for definition in TOOL_DEFINITIONS if definition.metadata.category is category
    )
    request = SimpleNamespace(
        tool_call={"name": definition.metadata.name, "id": f"{category.value}-call"}
    )

    sync_result = build_safe_tool_call_wrapper(_snapshot())(request, _raise_unexpected)
    async_result = await build_async_safe_tool_call_wrapper(_snapshot())(
        request,
        _raise_unexpected_async,
    )

    assert isinstance(sync_result, ToolMessage)
    assert isinstance(async_result, ToolMessage)
    assert async_result.content == sync_result.content
    assert async_result.name == sync_result.name
    assert async_result.tool_call_id == sync_result.tool_call_id
    assert async_result.status == sync_result.status == "error"
    assert RAW_FAILURE not in str(sync_result.content)


@pytest.mark.asyncio
async def test_sync_and_async_wrappers_return_equivalent_generic_errors() -> None:
    """Unknown tools should use the same non-disclosing generic fallback."""
    request = SimpleNamespace(tool_call={"name": "unknown_tool", "id": "unknown-call"})

    sync_result = build_safe_tool_call_wrapper(_snapshot())(request, _raise_unexpected)
    async_result = await build_async_safe_tool_call_wrapper(_snapshot())(
        request,
        _raise_unexpected_async,
    )

    assert isinstance(sync_result, ToolMessage)
    assert isinstance(async_result, ToolMessage)
    assert async_result.content == sync_result.content
    assert async_result.name == sync_result.name == "unknown_tool"
    assert async_result.tool_call_id == sync_result.tool_call_id == "unknown-call"
    assert async_result.status == sync_result.status == "error"


@pytest.mark.asyncio
async def test_sync_and_async_wrappers_preserve_graph_bubble_up() -> None:
    """Both wrappers must re-raise LangGraph control-flow exceptions unchanged."""
    request = SimpleNamespace(tool_call={"name": "unknown_tool", "id": "bubble-call"})

    with pytest.raises(GraphBubbleUp):
        build_safe_tool_call_wrapper(_snapshot())(request, _raise_graph_bubble_up)

    with pytest.raises(GraphBubbleUp):
        await build_async_safe_tool_call_wrapper(_snapshot())(
            request,
            _raise_graph_bubble_up_async,
        )


@pytest.mark.asyncio
async def test_async_wrapper_preserves_caller_cancellation() -> None:
    """Caller cancellation must not be converted into a safe ToolMessage."""
    request = SimpleNamespace(tool_call={"name": "unknown_tool", "id": "cancel-call"})

    with pytest.raises(asyncio.CancelledError):
        await build_async_safe_tool_call_wrapper(_snapshot())(
            request,
            _raise_caller_cancellation_async,
        )


@pytest.mark.asyncio
async def test_async_wrapper_safely_handles_upstream_timeout_error() -> None:
    """Before M9B, an upstream timeout remains an ordinary safe tool failure."""
    definition = TOOL_DEFINITIONS[0]
    request = SimpleNamespace(
        tool_call={"name": definition.metadata.name, "id": "upstream-timeout-call"}
    )

    result = await build_async_safe_tool_call_wrapper(_snapshot())(
        request,
        _raise_upstream_timeout_async,
    )

    assert isinstance(result, ToolMessage)
    assert result.name == definition.metadata.name
    assert result.tool_call_id == "upstream-timeout-call"
    assert result.status == "error"
    assert "synthetic upstream timeout" not in str(result.content)
