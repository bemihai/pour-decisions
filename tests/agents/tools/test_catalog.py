"""Tests for authoritative M6 tool catalogue composition."""

from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.agents import tools
from src.agents.tools.catalog import (
    CORE_DEFINITIONS,
    EXTENDED_DEFINITIONS,
    TOOL_DEFINITIONS,
    build_tool_registry,
)
from src.agents.tools.registry import ToolDefinition, ToolRegistry


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


def _definition_names(definitions: Iterable[ToolDefinition]) -> tuple[str, ...]:
    """Return definition names in catalogue order."""
    return tuple(definition.metadata.name for definition in definitions)


def test_authoritative_catalogue_matches_gate0_order() -> None:
    """Catalogue composition must exactly reproduce the frozen 5/13/18 order."""
    assert _definition_names(CORE_DEFINITIONS) == EXPECTED_CORE_TOOL_NAMES
    assert _definition_names(EXTENDED_DEFINITIONS) == EXPECTED_EXTENDED_TOOL_NAMES
    assert _definition_names(TOOL_DEFINITIONS) == (
        EXPECTED_CORE_TOOL_NAMES + EXPECTED_EXTENDED_TOOL_NAMES
    )


def test_legacy_lists_derive_from_authoritative_definitions() -> None:
    """Static compatibility exports should contain the catalogue tool objects."""
    assert tools.CORE_TOOLS == [definition.tool for definition in CORE_DEFINITIONS]
    assert tools.EXTENDED_TOOLS == [definition.tool for definition in EXTENDED_DEFINITIONS]
    assert tools.ALL_TOOLS == [definition.tool for definition in TOOL_DEFINITIONS]


def test_build_tool_registry_returns_fresh_validated_instances() -> None:
    """Registry construction must remain explicit and avoid a hidden singleton."""
    config = OmegaConf.create({})

    first = build_tool_registry(config)
    second = build_tool_registry(config)

    assert isinstance(first, ToolRegistry)
    assert isinstance(second, ToolRegistry)
    assert first is not second
    assert first.definitions() == TOOL_DEFINITIONS
    assert second.definitions() == TOOL_DEFINITIONS


def test_static_get_tools_does_not_run_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolving legacy lists must not touch future dependency probes."""
    readiness = MagicMock(side_effect=AssertionError("readiness must not run"))
    monkeypatch.setattr(ToolRegistry, "check_readiness", readiness)

    assert tuple(tool.name for tool in tools.get_tools(extended=False)) == EXPECTED_CORE_TOOL_NAMES
    assert tuple(tool.name for tool in tools.get_tools(extended=True)) == (
        EXPECTED_CORE_TOOL_NAMES + EXPECTED_EXTENDED_TOOL_NAMES
    )
    readiness.assert_not_called()
