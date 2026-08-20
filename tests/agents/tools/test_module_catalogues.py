"""Contract tests for the five explicit M6 module catalogues."""

import pytest

from src.agents.tools import (
    cellar_tools,
    pairing_tools,
    rag_tools,
    taste_profile_tools,
    web_search_tools,
)
from src.agents.tools.registry import (
    CostClass,
    LatencyClass,
    ToolCategory,
    ToolDefinition,
    ToolPrerequisite,
    ToolTier,
)


EXPECTED_MODULE_TOOL_NAMES = {
    "cellar": (
        "get_cellar_wines",
        "get_wine_details",
        "get_cellar_statistics",
    ),
    "taste_profile": (
        "get_user_taste_profile",
        "get_top_rated_wines",
        "get_wine_recommendations_from_profile",
        "compare_wine_to_profile",
    ),
    "pairing": (
        "get_food_pairing_wines",
        "get_pairing_for_wine",
        "get_wine_and_cheese_pairings",
    ),
    "rag": (
        "search_wine_knowledge",
        "search_wine_region_info",
        "search_grape_variety_info",
        "search_wine_term_definition",
        "search_wine_producer_info",
    ),
    "web_search": (
        "search_web_for_wine",
        "search_wine_price",
        "search_wine_reviews",
    ),
}

MODULE_CATALOGUES = {
    "cellar": cellar_tools.TOOL_DEFINITIONS,
    "taste_profile": taste_profile_tools.TOOL_DEFINITIONS,
    "pairing": pairing_tools.TOOL_DEFINITIONS,
    "rag": rag_tools.TOOL_DEFINITIONS,
    "web_search": web_search_tools.TOOL_DEFINITIONS,
}

EXPECTED_CORE_TOOL_NAMES = {
    "get_cellar_wines",
    "get_wine_details",
    "get_user_taste_profile",
    "search_wine_knowledge",
    "get_food_pairing_wines",
}


@pytest.mark.parametrize(("module_name", "expected_names"), EXPECTED_MODULE_TOOL_NAMES.items())
def test_module_catalogue_membership_and_order(
    module_name: str,
    expected_names: tuple[str, ...],
) -> None:
    """Each module should expose its reviewed active tools in local order."""
    definitions = MODULE_CATALOGUES[module_name]

    assert isinstance(definitions, tuple)
    assert tuple(definition.tool.name for definition in definitions) == expected_names


def test_combined_module_catalogues_contain_exactly_18_unique_active_tools() -> None:
    """Module metadata must cover the frozen inventory without duplication."""
    definitions = tuple(
        definition
        for catalogue in MODULE_CATALOGUES.values()
        for definition in catalogue
    )
    names = [definition.tool.name for definition in definitions]

    assert len(definitions) == 18
    assert len(set(names)) == 18
    assert set(names) == {
        name
        for expected_names in EXPECTED_MODULE_TOOL_NAMES.values()
        for name in expected_names
    }
    assert pairing_tools.suggest_dinner_menu_with_wines.name not in names


def test_metadata_names_categories_and_tiers_match_tools() -> None:
    """Every definition should match its tool and preserve the Gate 0 tier split."""
    for module_name, definitions in MODULE_CATALOGUES.items():
        for definition in definitions:
            assert isinstance(definition, ToolDefinition)
            assert definition.metadata.name == definition.tool.name
            assert definition.metadata.category == ToolCategory(module_name)
            expected_tier = (
                ToolTier.CORE
                if definition.tool.name in EXPECTED_CORE_TOOL_NAMES
                else ToolTier.EXTENDED
            )
            assert definition.metadata.tier == expected_tier


def test_database_prerequisites_match_reviewed_capabilities() -> None:
    """Database tools should distinguish inventory schema from pairing rules."""
    definitions = {
        definition.tool.name: definition
        for module_name in ("cellar", "taste_profile", "pairing")
        for definition in MODULE_CATALOGUES[module_name]
    }

    for name, definition in definitions.items():
        expected = (ToolPrerequisite.CELLAR_SCHEMA,)
        if name == "get_food_pairing_wines":
            expected = (
                ToolPrerequisite.CELLAR_SCHEMA,
                ToolPrerequisite.PAIRING_RULES,
            )
        assert definition.metadata.prerequisites == expected


def test_rag_and_web_metadata_preserve_cost_and_latency_contracts() -> None:
    """Local RAG stays free while metered web search is marked cheap."""
    for definition in MODULE_CATALOGUES["rag"]:
        assert definition.metadata.prerequisites == (ToolPrerequisite.CHROMA_COLLECTION,)
        assert definition.metadata.cost_class == CostClass.FREE
        assert definition.metadata.latency_class == LatencyClass.SLOW

    for definition in MODULE_CATALOGUES["web_search"]:
        assert definition.metadata.prerequisites == (ToolPrerequisite.WEB_SEARCH_CONFIG,)
        assert definition.metadata.cost_class == CostClass.CHEAP
        assert definition.metadata.latency_class == LatencyClass.SLOW


def test_every_capability_description_is_concise_and_non_blank() -> None:
    """Capability text should be suitable for later prompt and API rendering."""
    for definitions in MODULE_CATALOGUES.values():
        for definition in definitions:
            capability = definition.metadata.capability
            assert capability == capability.strip()
            assert capability
            assert len(capability) <= 120
