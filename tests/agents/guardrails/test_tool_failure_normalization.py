"""Regression tests for unexpected failures across the active tool catalogue."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool
from pytest_mock import MockerFixture

from src.agents.tools import cellar_tools, pairing_tools, rag_tools, taste_profile_tools, web_search_tools
from src.agents.tools.catalog import TOOL_DEFINITIONS


RAW_FAILURE = "M09A_SYNTHETIC_INTERNAL_FAILURE"


@dataclass(frozen=True)
class ToolFailureCase:
    """One active tool invocation and the dependency patched to fail."""

    tool: BaseTool
    arguments: dict[str, object]
    failure_target: str
    failure_method: str | None = None


TOOL_FAILURE_CASES = (
    ToolFailureCase(
        cellar_tools.get_cellar_wines,
        {},
        "src.agents.tools.cellar_tools.WineRepository",
    ),
    ToolFailureCase(
        cellar_tools.get_wine_details,
        {"wine_name": "Synthetic Wine"},
        "src.agents.tools.cellar_tools.WineRepository",
    ),
    ToolFailureCase(
        cellar_tools.get_cellar_statistics,
        {},
        "src.agents.tools.cellar_tools.StatsRepository",
    ),
    ToolFailureCase(
        taste_profile_tools.get_user_taste_profile,
        {},
        "src.agents.tools.taste_profile_tools.TastingRepository",
    ),
    ToolFailureCase(
        taste_profile_tools.get_top_rated_wines,
        {},
        "src.agents.tools.taste_profile_tools.TastingRepository",
    ),
    ToolFailureCase(
        taste_profile_tools.get_wine_recommendations_from_profile,
        {},
        "src.agents.tools.taste_profile_tools.get_user_taste_profile",
        "invoke",
    ),
    ToolFailureCase(
        taste_profile_tools.compare_wine_to_profile,
        {"wine_name": "Synthetic Wine"},
        "src.agents.tools.taste_profile_tools.WineRepository",
    ),
    ToolFailureCase(
        pairing_tools.get_food_pairing_wines,
        {"food": "synthetic dish"},
        "src.agents.tools.pairing_tools.FoodPairingRepository",
    ),
    ToolFailureCase(
        pairing_tools.get_pairing_for_wine,
        {"wine_name": "Synthetic Wine"},
        "src.agents.tools.pairing_tools.WineRepository",
    ),
    ToolFailureCase(
        pairing_tools.get_wine_and_cheese_pairings,
        {"cheese_type": "blue"},
        "src.agents.tools.pairing_tools.WineRepository",
    ),
    ToolFailureCase(
        rag_tools.search_wine_knowledge,
        {"query": "Synthetic query"},
        "src.agents.tools.rag_tools._execute_rag_query",
    ),
    ToolFailureCase(
        rag_tools.search_wine_region_info,
        {"region": "Synthetic Region"},
        "src.agents.tools.rag_tools._execute_rag_query",
    ),
    ToolFailureCase(
        rag_tools.search_grape_variety_info,
        {"varietal": "Synthetic Grape"},
        "src.agents.tools.rag_tools._execute_rag_query",
    ),
    ToolFailureCase(
        rag_tools.search_wine_term_definition,
        {"term": "synthetic term"},
        "src.agents.tools.rag_tools._execute_rag_query",
    ),
    ToolFailureCase(
        rag_tools.search_wine_producer_info,
        {"producer": "Synthetic Producer"},
        "src.agents.tools.rag_tools._execute_rag_query",
    ),
    ToolFailureCase(
        web_search_tools.search_web_for_wine,
        {"query": "Synthetic query"},
        "src.agents.tools.web_search_tools._get_engine",
    ),
    ToolFailureCase(
        web_search_tools.search_wine_price,
        {"wine_name": "Synthetic Wine"},
        "src.agents.tools.web_search_tools._get_engine",
    ),
    ToolFailureCase(
        web_search_tools.search_wine_reviews,
        {"wine_name": "Synthetic Wine"},
        "src.agents.tools.web_search_tools._get_engine",
    ),
)


def test_failure_cases_cover_every_active_catalogue_tool() -> None:
    """The normalization audit must remain aligned with the authoritative M6 catalogue."""
    case_names = {case.tool.name for case in TOOL_FAILURE_CASES}
    catalogue_names = {definition.metadata.name for definition in TOOL_DEFINITIONS}

    assert len(TOOL_FAILURE_CASES) == len(TOOL_DEFINITIONS) == 18
    assert case_names == catalogue_names


@pytest.mark.parametrize(
    "case",
    TOOL_FAILURE_CASES,
    ids=lambda case: case.tool.name,
)
def test_unexpected_tool_failures_escape_without_raw_return(
    case: ToolFailureCase,
    mocker: MockerFixture,
) -> None:
    """Every active tool should re-raise unexpected dependency failures."""
    failure = RuntimeError(RAW_FAILURE)
    if case.failure_method is None:
        mocker.patch(case.failure_target, side_effect=failure)
    else:
        dependency = MagicMock()
        getattr(dependency, case.failure_method).side_effect = failure
        mocker.patch(case.failure_target, new=dependency)

    with pytest.raises(RuntimeError, match=RAW_FAILURE):
        case.tool.invoke(case.arguments)
