"""Successful behavior tests for the intelligent-agent pairing tools."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agents.tools import pairing_tools


def test_get_food_pairing_wines_returns_general_rule_without_cellar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """General pairing mode should use local rules without querying cellar wines."""
    rule = SimpleNamespace(
        get_wine_types_list=lambda: ["Red"],
        get_varietals_list=lambda: ["Syrah"],
        pairing_explanation="Match the dish's intensity.",
    )
    repository = MagicMock()
    repository.find_matching_rule.return_value = rule
    wine_repository = MagicMock(side_effect=AssertionError("cellar must not be queried"))
    monkeypatch.setattr(pairing_tools, "FoodPairingRepository", lambda _path: repository)
    monkeypatch.setattr(pairing_tools, "WineRepository", wine_repository)

    result = pairing_tools.get_food_pairing_wines.invoke({"food": "lamb", "from_cellar_only": False})

    assert result["recommended_wine_types"] == ["Red"]
    assert result["recommended_varietals"] == ["Syrah"]
    assert result["cellar_matches"] == []
    wine_repository.assert_not_called()


def test_get_pairing_for_wine_returns_specific_foods(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known cellar wine should produce successful wine-to-food guidance."""
    repository = MagicMock()
    repository.get_by_name.return_value = SimpleNamespace(
        wine_name="Test Barolo",
        wine_type="Red",
        varietal="Nebbiolo",
        region_name="Piedmont",
        country="Italy",
    )
    monkeypatch.setattr(pairing_tools, "WineRepository", lambda _path: repository)

    result = pairing_tools.get_pairing_for_wine.invoke({"wine_name": "Test Barolo"})

    assert result["wine_name"] == "Test Barolo"
    assert "Osso buco" in result["primary_pairings"]
    assert "Italian cuisine" in result["traditional_pairings"]


def test_get_wine_and_cheese_pairings_supports_general_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """General cheese guidance should succeed without accessing the cellar."""
    wine_repository = MagicMock(side_effect=AssertionError("cellar must not be queried"))
    monkeypatch.setattr(pairing_tools, "WineRepository", wine_repository)

    result = pairing_tools.get_wine_and_cheese_pairings.invoke({"cheese_type": "blue", "from_cellar_only": False})

    assert result["cheese_category"] == "Blue Cheese"
    assert "Sauternes" in result["classic_pairings"]
    assert result["cellar_suggestions"] == []
    wine_repository.assert_not_called()
