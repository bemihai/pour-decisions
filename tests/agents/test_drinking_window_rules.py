"""
Unit tests for src/agents/drinking_window_rules.py.

Tests cover rule matching, priority ordering, regex patterns, and edge cases
for the estimate_from_rules function.
"""

import pytest

from src.agents.drinking_window_rules import (
    RULES,
    DrinkingWindowRule,
    _match,
    _matches_rule,
    estimate_from_rules,
)
from src.database.models import Wine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _wine(
    wine_type: str = "Red",
    varietal: str | None = None,
    region_name: str | None = None,
    country: str | None = None,
    designation: str | None = None,
    appellation: str | None = None,
    vintage: int | None = 2018,
) -> Wine:
    """Build a minimal Wine instance for rule testing."""
    return Wine(
        wine_name="Test Wine",
        wine_type=wine_type,
        varietal=varietal,
        region_name=region_name,
        country=country,
        designation=designation,
        appellation=appellation,
        vintage=vintage,
    )


# ---------------------------------------------------------------------------
# _match helper
# ---------------------------------------------------------------------------


class TestMatch:
    def test_none_pattern_is_wildcard(self):
        assert _match(None, "anything") is True
        assert _match(None, None) is True
        assert _match(None, "") is True

    def test_matching_pattern(self):
        assert _match(r"Nebbiolo", "Nebbiolo") is True

    def test_pattern_case_insensitive(self):
        assert _match(r"nebbiolo", "Nebbiolo") is True
        assert _match(r"NEBBIOLO", "nebbiolo") is True

    def test_partial_match(self):
        assert _match(r"Cab", "Cabernet Sauvignon") is True

    def test_no_match(self):
        assert _match(r"Riesling", "Chardonnay") is False

    def test_none_value_with_pattern(self):
        assert _match(r"Nebbiolo", None) is False

    def test_empty_value_with_pattern(self):
        assert _match(r"Nebbiolo", "") is False


# ---------------------------------------------------------------------------
# _matches_rule
# ---------------------------------------------------------------------------


class TestMatchesRule:
    def test_wine_type_mismatch_fails(self):
        rule = DrinkingWindowRule(priority=50, wine_type="White", offset_start=1, offset_end=5)
        assert _matches_rule(rule, _wine(wine_type="Red")) is False

    def test_wine_type_case_insensitive(self):
        rule = DrinkingWindowRule(priority=50, wine_type="red", offset_start=1, offset_end=5)
        assert _matches_rule(rule, _wine(wine_type="Red")) is True

    def test_varietal_regex_required(self):
        rule = DrinkingWindowRule(
            priority=50, wine_type="Red", varietal_re=r"Nebbiolo", offset_start=5, offset_end=20
        )
        assert _matches_rule(rule, _wine(varietal="Cabernet Sauvignon")) is False
        assert _matches_rule(rule, _wine(varietal="Nebbiolo")) is True

    def test_region_regex_matches_country_fallback(self):
        rule = DrinkingWindowRule(
            priority=50, wine_type="Red", region_re=r"Italy", offset_start=2, offset_end=8
        )
        wine = _wine(region_name=None, country="Italy")
        assert _matches_rule(rule, wine) is True

    def test_designation_regex_matches_appellation(self):
        rule = DrinkingWindowRule(
            priority=50, wine_type="Red", designation_re=r"Riserva", offset_start=5, offset_end=15
        )
        wine = _wine(designation=None, appellation="Chianti Classico Riserva")
        assert _matches_rule(rule, wine) is True

    def test_all_none_wildcards_match_any_wine(self):
        rule = DrinkingWindowRule(priority=10, wine_type="Red", offset_start=2, offset_end=8)
        assert _matches_rule(rule, _wine()) is True


# ---------------------------------------------------------------------------
# estimate_from_rules
# ---------------------------------------------------------------------------


class TestEstimateFromRules:
    def test_returns_none_for_non_vintage_wine(self):
        wine = _wine(vintage=None)
        assert estimate_from_rules(wine) is None

    def test_returns_tuple_for_matched_wine(self):
        wine = _wine(wine_type="Red", varietal="Nebbiolo", region_name="Piedmont", vintage=2018)
        result = estimate_from_rules(wine)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_barolo_window_from_vintage(self):
        wine = _wine(wine_type="Red", varietal="Nebbiolo", region_name="Barolo", vintage=2015)
        result = estimate_from_rules(wine)
        assert result is not None
        from_year, to_year = result
        # Barolo standard: offset_start=5, offset_end=20
        assert from_year == 2015 + 5
        assert to_year == 2015 + 20

    def test_barolo_riserva_takes_priority_over_standard(self):
        wine = _wine(
            wine_type="Red",
            varietal="Nebbiolo",
            region_name="Barolo",
            designation="Riserva",
            vintage=2015,
        )
        result = estimate_from_rules(wine)
        assert result is not None
        from_year, to_year = result
        # Riserva rule: offset_start=8, offset_end=30
        assert from_year == 2015 + 8
        assert to_year == 2015 + 30

    def test_cabernet_classified_bordeaux(self):
        wine = _wine(
            wine_type="Red",
            varietal="Cabernet Sauvignon",
            region_name="Pauillac",
            designation="Grand Cru Classé",
            vintage=2010,
        )
        result = estimate_from_rules(wine)
        assert result is not None
        # Classified Bordeaux: offset_start=8, offset_end=30
        assert result[0] == 2010 + 8
        assert result[1] == 2010 + 30

    def test_generic_red_fallback(self):
        wine = _wine(wine_type="Red", varietal="Montepulciano", region_name="Abruzzo", vintage=2020)
        result = estimate_from_rules(wine)
        assert result is not None
        # Generic red fallback: offset_start=2, offset_end=8
        assert result[0] == 2020 + 2
        assert result[1] == 2020 + 8

    def test_generic_white_fallback(self):
        wine = _wine(wine_type="White", varietal="Vermentino", vintage=2022)
        result = estimate_from_rules(wine)
        assert result is not None
        # Generic white fallback: offset_start=1, offset_end=4
        assert result[0] == 2022 + 1
        assert result[1] == 2022 + 4

    def test_rose_fallback(self):
        wine = _wine(wine_type="Rosé", vintage=2023)
        result = estimate_from_rules(wine)
        assert result is not None
        # Rosé: offset_start=0, offset_end=2
        assert result[0] == 2023 + 0
        assert result[1] == 2023 + 2

    def test_sparkling_generic(self):
        wine = _wine(wine_type="Sparkling", vintage=2020)
        result = estimate_from_rules(wine)
        assert result is not None
        assert result[0] == 2020 + 0
        assert result[1] == 2020 + 3

    def test_vintage_champagne_prestige(self):
        wine = _wine(
            wine_type="Sparkling",
            region_name="Champagne",
            designation="Prestige Cuvée",
            vintage=2015,
        )
        result = estimate_from_rules(wine)
        assert result is not None
        # Prestige Champagne: offset_start=2, offset_end=12
        assert result[0] == 2015 + 2
        assert result[1] == 2015 + 12

    def test_sauvignon_blanc_drink_fresh(self):
        wine = _wine(wine_type="White", varietal="Sauvignon Blanc", vintage=2023)
        result = estimate_from_rules(wine)
        assert result is not None
        # Generic SB: offset_start=0, offset_end=3
        assert result[0] == 2023 + 0
        assert result[1] == 2023 + 3

    def test_gamay_drink_fresh(self):
        wine = _wine(wine_type="Red", varietal="Gamay", vintage=2024)
        result = estimate_from_rules(wine)
        assert result is not None
        # Gamay: offset_start=0, offset_end=3
        assert result[0] == 2024 + 0
        assert result[1] == 2024 + 3

    def test_pinot_noir_burgundy_grand_cru(self):
        wine = _wine(
            wine_type="Red",
            varietal="Pinot Noir",
            region_name="Côte de Nuits",
            designation="Grand Cru",
            vintage=2016,
        )
        result = estimate_from_rules(wine)
        assert result is not None
        # Grand Cru Burgundy PN: offset_start=5, offset_end=20
        assert result[0] == 2016 + 5
        assert result[1] == 2016 + 20

    def test_dessert_sauternes(self):
        wine = _wine(wine_type="Dessert", varietal="Sauternes", vintage=2001)
        result = estimate_from_rules(wine)
        assert result is not None
        # Classic noble rot: offset_start=3, offset_end=20
        assert result[0] == 2001 + 3
        assert result[1] == 2001 + 20

    def test_fortified_vintage_port(self):
        wine = _wine(wine_type="Fortified", designation="Vintage Port", vintage=2003)
        result = estimate_from_rules(wine)
        assert result is not None
        # Vintage Port: offset_start=0, offset_end=20
        assert result[0] == 2003 + 0
        assert result[1] == 2003 + 20


# ---------------------------------------------------------------------------
# RULES catalogue integrity
# ---------------------------------------------------------------------------


class TestRulesCatalogue:
    def test_rules_sorted_descending_by_priority(self):
        priorities = [r.priority for r in RULES]
        assert priorities == sorted(priorities, reverse=True), "RULES must be sorted by descending priority"

    def test_all_rules_have_positive_offsets(self):
        for rule in RULES:
            assert rule.offset_start >= 0, f"Rule {rule.note!r} has negative offset_start"
            assert rule.offset_end > 0, f"Rule {rule.note!r} has non-positive offset_end"
            assert rule.offset_end > rule.offset_start, (
                f"Rule {rule.note!r}: offset_end must be > offset_start"
            )

    def test_each_wine_type_has_fallback_rule(self):
        wine_types_covered = {r.wine_type for r in RULES if r.varietal_re is None and r.region_re is None and r.designation_re is None}
        for wine_type in ("Red", "White", "Rosé", "Sparkling", "Dessert", "Fortified"):
            assert wine_type in wine_types_covered, f"No fallback rule for wine type '{wine_type}'"

