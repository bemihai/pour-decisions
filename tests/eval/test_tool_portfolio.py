"""Tests for the bounded intelligent-agent portfolio experiment."""

from omegaconf import OmegaConf

from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.eval.tool_portfolio import (
    arguments_match,
    build_stub_tool_registry,
    estimate_tool_portfolio_pair,
    load_tool_portfolio_cohort,
)


def test_portfolio_cohort_covers_every_active_tool_once() -> None:
    """The discriminating cohort should stay synchronized with the catalogue."""
    cohort = load_tool_portfolio_cohort()

    assert len(cohort.cases) == len(TOOL_DEFINITIONS) == 19
    assert {case.expected_tool for case in cohort.cases} == {
        definition.metadata.name for definition in TOOL_DEFINITIONS
    }


def test_portfolio_pair_estimate_is_bounded_and_uses_no_external_tools() -> None:
    """The one-repetition pair should cap model attempts before capture."""
    estimate = estimate_tool_portfolio_pair(load_tool_portfolio_cohort())

    assert estimate == {
        "cases": 19,
        "repetitions": 1,
        "paired_executions": 38,
        "maximum_logical_model_calls": 190,
        "external_tool_calls": 0,
    }


def test_argument_matching_requires_expected_semantic_subset() -> None:
    """Automatic scoring should accept extra defaults but reject missing semantics."""
    assert arguments_match(
        {"wine_name": "Test Barolo", "vintage": 2019, "query": "__present__"},
        {
            "wine_name": "2019 Test Barolo Riserva",
            "vintage": 2019,
            "query": "current Test Barolo price",
            "domain": None,
        },
    )
    assert arguments_match({"domain": None}, {})
    assert not arguments_match({"ready_to_drink": False}, {})
    assert not arguments_match({"domain": None}, {"domain": "example.com"})


def test_stub_registry_preserves_contracts_without_real_dependencies() -> None:
    """Experiment tools should validate real arguments but execute local fixture bodies."""
    registry = build_stub_tool_registry(OmegaConf.create({}))
    definition = next(item for item in registry.definitions() if item.metadata.name == "get_cellar_wines")

    result = definition.tool.invoke({"ready_to_drink": False})

    assert '"fixture_tool": "get_cellar_wines"' in result
    assert '"ready_to_drink": false' in result
    assert definition.metadata.prerequisites == ()
