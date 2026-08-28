"""Tests for stable, non-disclosing tool-error messages."""

import pytest

from src.agents.guardrails.safe_errors import (
    SafeToolErrorCode,
    format_safe_tool_error,
    get_safe_tool_error,
)
from src.agents.tools.registry import ToolCategory


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
