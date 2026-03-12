"""
Service for estimating and managing wine drinking windows.

Provides:
- Heuristic estimation from the rule engine (local, free, no LLM).
- Source-priority-aware persistence of drinking windows.
- Bell-curve drinking index computation as fallback for CellarTracker index.

Usage:
    >>> from src.agents.drinking_window_service import DrinkingWindowService
    >>> service = DrinkingWindowService()
    >>> wine = wine_repo.get_by_id(42)
    >>> service.estimate_heuristic(wine)  # persists window + index
    True
"""

from datetime import datetime
from math import cos, exp, pi

from src.agents.drinking_window_rules import estimate_from_rules
from src.database.models import Wine
from src.database.repository import WineRepository
from src.utils import get_default_db_path, logger

# Source priority map: lower number = higher priority.
_SOURCE_PRIORITY: dict[str, int] = {
    "manual": 1,
    "cellar_tracker": 2,
    "llm": 3,
    "heuristic": 4,
}


def compute_drink_index(drink_from_year: int, drink_to_year: int, current_year: int | None = None) -> float:
    """Compute a 0-100 drinking index using a bell-curve model.

    Mirrors the CellarTracker approach: the index peaks in the middle of the
    drinking window and tapers off symmetrically. Wines outside the window get
    non-zero values to support a smooth gradient in the UI.

    Args:
        drink_from_year: Year the window opens.
        drink_to_year: Year the window closes.
        current_year: Year to evaluate for (defaults to the current calendar year).

    Returns:
        Float in range [0, 100], with 100 at the midpoint of the window.

    Example:
        >>> compute_drink_index(2020, 2030, 2025)
        100.0
        >>> compute_drink_index(2020, 2030, 2020)
        50.0
        >>> compute_drink_index(2020, 2030, 2015)  # before window
        # small positive value
    """
    if current_year is None:
        current_year = datetime.now().year

    if drink_to_year <= drink_from_year:
        return 100.0 if current_year >= drink_from_year else 0.0

    window_length = drink_to_year - drink_from_year
    # Normalised position within the window: 0 = open, 1 = close
    position = (current_year - drink_from_year) / window_length

    if 0.0 <= position <= 1.0:
        # Cosine bell: 0 at edges, 100 at midpoint (position=0.5)
        return round((1 + cos(pi * (2 * position - 1))) / 2 * 100, 2)

    if position < 0.0:
        # Before window: exponential ramp-up (halves every 4 years before open)
        years_before = drink_from_year - current_year
        return round(max(0.0, 50 * exp(-0.18 * years_before)), 2)

    # After window: exponential decay (halves every ~3 years after close)
    years_past = current_year - drink_to_year
    return round(max(0.0, 50 * exp(-0.23 * years_past)), 2)


class DrinkingWindowService:
    """Service for estimating and managing wine drinking windows.

    Handles heuristic estimation, source-priority-aware persistence, and drinking
    index computation. Does NOT make LLM calls; LLM-based estimation is handled by
    DescriptionService which calls update_drinking_window() here after parsing its
    structured output.

    Example:
        >>> service = DrinkingWindowService()
        >>> stats = service.estimate_batch_heuristic(wine_repo.get_without_drinking_window())
        >>> print(f"Estimated {stats['estimated']} windows")
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialise the service.

        Args:
            db_path: Path to the SQLite database. Defaults to the configured path.
        """
        self.wine_repo = WineRepository(db_path or get_default_db_path())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_heuristic(self, wine: Wine) -> bool:
        """Estimate and persist the drinking window for a single wine using heuristics.

        Skips wines that already have a higher-priority source.

        Args:
            wine: Wine model with at least vintage, wine_type set.

        Returns:
            True if the window was estimated and persisted, False otherwise.
        """
        if not self._can_overwrite(wine.drink_window_source, "heuristic"):
            logger.debug(f"Skipping heuristic for wine {wine.id}: source '{wine.drink_window_source}' has higher priority")
            return False

        result = estimate_from_rules(wine)
        if result is None:
            logger.debug(f"No heuristic rule matched wine {wine.id}: {wine.wine_name}")
            return False

        from_year, to_year = result
        index = compute_drink_index(from_year, to_year)
        return self.wine_repo.update_drinking_window(wine.id, from_year, to_year, index, "heuristic")

    def estimate_batch_heuristic(self, wines: list[Wine]) -> dict[str, int]:
        """Estimate drinking windows for a batch of wines using heuristics.

        Intended to run after a CellarTracker sync to fill in missing windows.

        Args:
            wines: List of Wine models to process.

        Returns:
            Dict with keys 'estimated' and 'skipped'.

        Example:
            >>> wines = wine_repo.get_without_drinking_window()
            >>> stats = service.estimate_batch_heuristic(wines)
            >>> print(stats)
            {'estimated': 45, 'skipped': 12}
        """
        estimated = skipped = 0
        for wine in wines:
            if self.estimate_heuristic(wine):
                estimated += 1
            else:
                skipped += 1

        logger.info(f"Heuristic batch: {estimated} estimated, {skipped} skipped")
        return {"estimated": estimated, "skipped": skipped}

    def update_drinking_window(
        self,
        wine_id: int,
        from_year: int,
        to_year: int,
        source: str,
        drink_index: float | None = None,
    ) -> bool:
        """Persist a drinking window, recomputing the local index when not supplied.

        Delegates priority enforcement to WineRepository.update_drinking_window().

        Args:
            wine_id: Target wine ID.
            from_year: Start of the drinking window.
            to_year: End of the drinking window.
            source: Provenance: 'manual', 'cellar_tracker', 'llm', or 'heuristic'.
            drink_index: Pre-computed index. If None, computed locally.

        Returns:
            True if the row was updated.
        """
        if drink_index is None:
            drink_index = compute_drink_index(from_year, to_year)
        return self.wine_repo.update_drinking_window(wine_id, from_year, to_year, drink_index, source)

    def get_effective_drink_index(self, wine: Wine) -> float | None:
        """Return the best available drinking index for a wine.

        Prefers the CellarTracker-sourced index. Falls back to local computation
        when CT index is absent but a drinking window is available.

        Args:
            wine: Wine model.

        Returns:
            Drinking index float, or None if no window is available.
        """
        if wine.drink_index is not None and wine.drink_window_source == "cellar_tracker":
            return wine.drink_index

        if wine.drink_from_year and wine.drink_to_year:
            return compute_drink_index(wine.drink_from_year, wine.drink_to_year)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _can_overwrite(existing_source: str | None, new_source: str) -> bool:
        """Return True when new_source is allowed to overwrite existing_source."""
        if existing_source is None:
            return True
        existing_priority = _SOURCE_PRIORITY.get(existing_source, 99)
        new_priority = _SOURCE_PRIORITY.get(new_source, 99)
        return new_priority <= existing_priority

