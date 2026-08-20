"""
Wine agent tools package.

This package provides all tools for the wine agent, organized by functionality:
- cellar_tools: Wine cellar inventory and management
- taste_profile_tools: User preference analysis and recommendations
- pairing_tools: Food and wine pairing recommendations
- rag_tools: Wine knowledge base search (RAG)
"""

from langchain_core.tools import BaseTool

from .cellar_tools import (
    get_cellar_wines,
    get_wine_details,
    get_cellar_statistics,
)

from .taste_profile_tools import (
    get_user_taste_profile,
    get_top_rated_wines,
    get_wine_recommendations_from_profile,
    compare_wine_to_profile,
)

from .pairing_tools import (
    get_food_pairing_wines,
    get_pairing_for_wine,
    get_wine_and_cheese_pairings,
    suggest_dinner_menu_with_wines,
)

from .rag_tools import (
    search_wine_knowledge,
    search_wine_region_info,
    search_grape_variety_info,
    search_wine_term_definition,
    search_wine_producer_info,
)

from .web_search_tools import (
    search_web_for_wine,
    search_wine_price,
    search_wine_reviews,
)
from .catalog import (
    CORE_DEFINITIONS,
    EXTENDED_DEFINITIONS,
    TOOL_DEFINITIONS,
    build_tool_registry,
)


# Core tools
CORE_TOOLS: list[BaseTool] = [definition.tool for definition in CORE_DEFINITIONS]

# Extended tools
EXTENDED_TOOLS: list[BaseTool] = [definition.tool for definition in EXTENDED_DEFINITIONS]

ALL_TOOLS: list[BaseTool] = [definition.tool for definition in TOOL_DEFINITIONS]


def get_tools(extended: bool = True) -> list[BaseTool]:
    """Get tools for specific implementation phase.

    Args:
        extended: If True, return all tools. If False, return core tools only.

    Returns:
        List of tool instances
    """
    if not extended:
        return CORE_TOOLS
    else:
        return ALL_TOOLS


__all__ = [
    # Core tools
    "get_cellar_wines",
    "get_wine_details",
    "get_user_taste_profile",
    "search_wine_knowledge",
    "get_food_pairing_wines",

    # Extended tools
    "get_cellar_statistics",
    "get_top_rated_wines",
    "get_wine_recommendations_from_profile",
    "compare_wine_to_profile",
    "get_pairing_for_wine",
    "get_wine_and_cheese_pairings",
    "suggest_dinner_menu_with_wines",
    "search_wine_region_info",
    "search_grape_variety_info",
    "search_wine_term_definition",
    "search_wine_producer_info",

    # Web search tools
    "search_web_for_wine",
    "search_wine_price",
    "search_wine_reviews",

    # Tool collections
    "CORE_TOOLS",
    "EXTENDED_TOOLS",
    "ALL_TOOLS",
    "get_tools",
    "TOOL_DEFINITIONS",
    "build_tool_registry",
]
