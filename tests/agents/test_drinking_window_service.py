"""
Unit tests for src/agents/drinking_window_service.py.

Covers:
- compute_drink_index bell-curve formula
- DrinkingWindowService.estimate_heuristic (with mocked repository)
- DrinkingWindowService.estimate_batch_heuristic
- DrinkingWindowService.update_drinking_window
- DrinkingWindowService.get_effective_drink_index
- Source-priority enforcement
"""

from unittest.mock import MagicMock

import pytest

from src.agents.drinking_window_service import DrinkingWindowService, compute_drink_index
from src.database.models import Wine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wine(
    wine_id: int = 1,
    wine_type: str = "Red",
    varietal: str | None = "Nebbiolo",
    region_name: str | None = "Barolo",
    vintage: int | None = 2015,
    drink_from_year: int | None = None,
    drink_to_year: int | None = None,
    drink_index: float | None = None,
    drink_window_source: str | None = None,
) -> Wine:
    """Build a minimal Wine with id set."""
    w = Wine(
        wine_name="Test Wine",
        wine_type=wine_type,
        varietal=varietal,
        region_name=region_name,
        vintage=vintage,
        drink_from_year=drink_from_year,
        drink_to_year=drink_to_year,
        drink_index=drink_index,
        drink_window_source=drink_window_source,
    )
    w.id = wine_id
    return w


# ---------------------------------------------------------------------------
# compute_drink_index
# ---------------------------------------------------------------------------


class TestComputeDrinkIndex:
    def test_midpoint_returns_100(self):
        result = compute_drink_index(2020, 2030, current_year=2025)
        assert result == 100.0

    def test_at_window_open_returns_zero(self):
        # Cosine bell: position=0 -> cos(-pi)=-1 -> (1-1)/2*100=0
        result = compute_drink_index(2020, 2030, current_year=2020)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_at_window_close_returns_zero(self):
        # Cosine bell: position=1 -> cos(pi)=-1 -> (1-1)/2*100=0
        result = compute_drink_index(2020, 2030, current_year=2030)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_before_window_returns_small_positive(self):
        result = compute_drink_index(2025, 2035, current_year=2015)
        assert 0 < result < 50

    def test_after_window_returns_small_positive(self):
        result = compute_drink_index(2015, 2020, current_year=2030)
        assert 0 <= result < 50

    def test_equal_years_returns_100_when_current_gte_from(self):
        result = compute_drink_index(2020, 2020, current_year=2021)
        assert result == 100.0

    def test_equal_years_returns_0_when_current_lt_from(self):
        result = compute_drink_index(2025, 2025, current_year=2020)
        assert result == 0.0

    def test_result_always_non_negative(self):
        for year in range(2000, 2050):
            assert compute_drink_index(2015, 2025, year) >= 0.0

    def test_result_always_lte_100(self):
        for year in range(2000, 2050):
            assert compute_drink_index(2015, 2025, year) <= 100.0

    def test_uses_current_year_by_default(self):
        """Calling without current_year should not raise."""
        result = compute_drink_index(2015, 2030)
        assert 0 <= result <= 100


# ---------------------------------------------------------------------------
# DrinkingWindowService.estimate_heuristic
# ---------------------------------------------------------------------------


class TestEstimateHeuristic:
    def _make_service(self, update_return: bool = True) -> tuple[DrinkingWindowService, MagicMock]:
        service = DrinkingWindowService.__new__(DrinkingWindowService)
        repo = MagicMock()
        repo.update_drinking_window.return_value = update_return
        service.wine_repo = repo
        return service, repo

    def test_estimates_and_persists(self):
        service, repo = self._make_service()
        wine = _wine(drink_window_source=None)
        result = service.estimate_heuristic(wine)
        assert result is True
        repo.update_drinking_window.assert_called_once()
        call_args = repo.update_drinking_window.call_args
        wine_id, from_year, to_year, index, source = call_args[0]
        assert wine_id == wine.id
        assert from_year == 2015 + 5   # Barolo standard offset_start
        assert to_year == 2015 + 20    # Barolo standard offset_end
        assert 0 <= index <= 100
        assert source == "heuristic"

    def test_skips_when_higher_priority_source_exists(self):
        service, repo = self._make_service()
        wine = _wine(drink_window_source="cellar_tracker")
        result = service.estimate_heuristic(wine)
        assert result is False
        repo.update_drinking_window.assert_not_called()

    def test_skips_when_manual_source(self):
        service, repo = self._make_service()
        wine = _wine(drink_window_source="manual")
        result = service.estimate_heuristic(wine)
        assert result is False

    def test_skips_non_vintage_wine(self):
        service, repo = self._make_service()
        wine = _wine(vintage=None)
        result = service.estimate_heuristic(wine)
        assert result is False
        repo.update_drinking_window.assert_not_called()

    def test_skips_wine_with_no_rule_match(self):
        service, repo = self._make_service()
        # A wine type that has no rules won't match any rule
        wine = _wine(wine_type="Unknown", varietal=None, region_name=None)
        result = service.estimate_heuristic(wine)
        assert result is False


# ---------------------------------------------------------------------------
# DrinkingWindowService.estimate_batch_heuristic
# ---------------------------------------------------------------------------


class TestEstimateBatchHeuristic:
    def _make_service(self) -> DrinkingWindowService:
        service = DrinkingWindowService.__new__(DrinkingWindowService)
        repo = MagicMock()
        repo.update_drinking_window.return_value = True
        service.wine_repo = repo
        return service

    def test_returns_correct_counts(self):
        service = self._make_service()
        wines = [
            _wine(wine_id=1, vintage=2015),          # Barolo -> estimated
            _wine(wine_id=2, vintage=None),           # no vintage -> skipped
            _wine(wine_id=3, drink_window_source="manual"),  # high priority -> skipped
        ]
        stats = service.estimate_batch_heuristic(wines)
        assert stats["estimated"] == 1
        assert stats["skipped"] == 2

    def test_empty_list_returns_zero_counts(self):
        service = self._make_service()
        stats = service.estimate_batch_heuristic([])
        assert stats == {"estimated": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# DrinkingWindowService.update_drinking_window
# ---------------------------------------------------------------------------


class TestUpdateDrinkingWindow:
    def _make_service(self, update_return: bool = True) -> tuple[DrinkingWindowService, MagicMock]:
        service = DrinkingWindowService.__new__(DrinkingWindowService)
        repo = MagicMock()
        repo.update_drinking_window.return_value = update_return
        service.wine_repo = repo
        return service, repo

    def test_computes_index_when_not_supplied(self):
        service, repo = self._make_service()
        service.update_drinking_window(42, 2020, 2030, "heuristic")
        _, _, _, index, _ = repo.update_drinking_window.call_args[0]
        assert 0 <= index <= 100

    def test_passes_supplied_index(self):
        service, repo = self._make_service()
        service.update_drinking_window(42, 2020, 2030, "manual", drink_index=99.0)
        _, _, _, index, _ = repo.update_drinking_window.call_args[0]
        assert index == 99.0

    def test_delegates_to_repository(self):
        service, repo = self._make_service(update_return=False)
        result = service.update_drinking_window(99, 2020, 2030, "llm")
        assert result is False


# ---------------------------------------------------------------------------
# DrinkingWindowService.get_effective_drink_index
# ---------------------------------------------------------------------------


class TestGetEffectiveDrinkIndex:
    def _make_service(self) -> DrinkingWindowService:
        service = DrinkingWindowService.__new__(DrinkingWindowService)
        service.wine_repo = MagicMock()
        return service

    def test_returns_ct_index_when_source_is_ct(self):
        service = self._make_service()
        wine = _wine(drink_index=75.0, drink_window_source="cellar_tracker")
        assert service.get_effective_drink_index(wine) == 75.0

    def test_computes_local_index_when_ct_missing(self):
        service = self._make_service()
        wine = _wine(
            drink_index=None,
            drink_window_source="heuristic",
            drink_from_year=2020,
            drink_to_year=2030,
        )
        result = service.get_effective_drink_index(wine)
        assert result is not None
        assert 0 <= result <= 100

    def test_returns_none_when_no_window_or_index(self):
        service = self._make_service()
        wine = _wine(drink_index=None, drink_from_year=None, drink_to_year=None)
        assert service.get_effective_drink_index(wine) is None

    def test_prefers_ct_index_over_local_computation(self):
        """CT index is returned directly without recomputing, even if window is available."""
        service = self._make_service()
        wine = _wine(
            drink_index=42.0,
            drink_window_source="cellar_tracker",
            drink_from_year=2015,
            drink_to_year=2025,
        )
        assert service.get_effective_drink_index(wine) == 42.0


# ---------------------------------------------------------------------------
# Source priority helper (_can_overwrite)
# ---------------------------------------------------------------------------


class TestCanOverwrite:
    def test_null_existing_always_overwritable(self):
        assert DrinkingWindowService._can_overwrite(None, "heuristic") is True
        assert DrinkingWindowService._can_overwrite(None, "llm") is True
        assert DrinkingWindowService._can_overwrite(None, "manual") is True

    def test_manual_cannot_be_overwritten_by_lower_priority(self):
        assert DrinkingWindowService._can_overwrite("manual", "llm") is False
        assert DrinkingWindowService._can_overwrite("manual", "heuristic") is False
        assert DrinkingWindowService._can_overwrite("manual", "cellar_tracker") is False

    def test_manual_can_overwrite_manual(self):
        assert DrinkingWindowService._can_overwrite("manual", "manual") is True

    def test_cellar_tracker_can_be_overwritten_only_by_manual(self):
        assert DrinkingWindowService._can_overwrite("cellar_tracker", "manual") is True
        assert DrinkingWindowService._can_overwrite("cellar_tracker", "llm") is False
        assert DrinkingWindowService._can_overwrite("cellar_tracker", "heuristic") is False

    def test_llm_overwritten_by_manual_or_ct(self):
        assert DrinkingWindowService._can_overwrite("llm", "manual") is True
        assert DrinkingWindowService._can_overwrite("llm", "cellar_tracker") is True
        assert DrinkingWindowService._can_overwrite("llm", "llm") is True
        assert DrinkingWindowService._can_overwrite("llm", "heuristic") is False

    def test_heuristic_overwritten_by_any_source(self):
        for source in ("manual", "cellar_tracker", "llm", "heuristic"):
            assert DrinkingWindowService._can_overwrite("heuristic", source) is True


