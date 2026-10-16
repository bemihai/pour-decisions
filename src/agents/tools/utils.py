"""Utility functions for agent tools."""

from src.agents.drinking_window_service import DrinkStatus, compute_drinking_readiness


def get_drink_status(drink_from_year: int | None, drink_to_year: int | None) -> DrinkStatus:
    """Determine drink status for a wine based on drinking window and current year.
    Args:
        drink_from_year: Year wine becomes ready to drink
        drink_to_year: Year wine is past peak
    Returns:
        ``"hold"``, ``"ready"``, ``"past_peak"``, or ``"unknown"``.
    """
    return compute_drinking_readiness(drink_from_year, drink_to_year).drink_status
