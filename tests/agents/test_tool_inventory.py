"""Regression tests for the pre-M6 intelligent-agent tool inventory."""

from collections.abc import Iterable

from langchain_core.tools import BaseTool

from src.agents import tools


EXPECTED_CORE_TOOL_NAMES = (
    "get_cellar_wines",
    "get_wine_details",
    "get_user_taste_profile",
    "search_wine_knowledge",
    "get_food_pairing_wines",
)

EXPECTED_EXTENDED_TOOL_NAMES = (
    "get_cellar_statistics",
    "get_top_rated_wines",
    "get_wine_recommendations_from_profile",
    "compare_wine_to_profile",
    "get_pairing_for_wine",
    "get_wine_and_cheese_pairings",
    "search_wine_region_info",
    "search_grape_variety_info",
    "search_wine_term_definition",
    "search_wine_producer_info",
    "search_web_for_wine",
    "search_wine_price",
    "search_wine_reviews",
)

EXPECTED_ALL_TOOL_NAMES = EXPECTED_CORE_TOOL_NAMES + EXPECTED_EXTENDED_TOOL_NAMES


def _tool_names(tool_inventory: Iterable[BaseTool]) -> tuple[str, ...]:
    """Return tool names in their configured order."""
    return tuple(tool.name for tool in tool_inventory)


def test_core_tool_inventory_is_frozen() -> None:
    """Core membership and ordering must remain stable across the M6 migration."""
    assert _tool_names(tools.CORE_TOOLS) == EXPECTED_CORE_TOOL_NAMES
    assert len(tools.CORE_TOOLS) == 5


def test_extended_tool_inventory_is_frozen() -> None:
    """Extended membership and ordering must remain stable across the M6 migration."""
    assert _tool_names(tools.EXTENDED_TOOLS) == EXPECTED_EXTENDED_TOOL_NAMES
    assert len(tools.EXTENDED_TOOLS) == 13


def test_all_tools_preserve_core_then_extended_order() -> None:
    """The active agent inventory must remain the ordered 18-tool baseline."""
    assert tools.ALL_TOOLS == tools.CORE_TOOLS + tools.EXTENDED_TOOLS
    assert _tool_names(tools.ALL_TOOLS) == EXPECTED_ALL_TOOL_NAMES
    assert len(tools.ALL_TOOLS) == 18


def test_get_tools_preserves_core_and_extended_contract() -> None:
    """The public selector must retain its current core/all behavior and ordering."""
    assert _tool_names(tools.get_tools(extended=False)) == EXPECTED_CORE_TOOL_NAMES
    assert _tool_names(tools.get_tools(extended=True)) == EXPECTED_ALL_TOOL_NAMES
