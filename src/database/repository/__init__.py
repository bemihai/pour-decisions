"""
Database access layer for wine cellar.

Provides high-level interface for database operations without exposing SQL queries.
Follows repository pattern for clean separation of concerns.
"""
from .bottle import BottleRepository
from .declared_preference import (
    DeclaredPreferenceRepository,
    MAX_DECLARED_PREFERENCES,
    PreferenceCombinationError,
    PreferenceCountChangedError,
    PreferenceDuplicateError,
    PreferenceLimitReachedError,
    PreferenceNotFoundError,
    PreferenceRepositoryError,
    PreferenceStoreBusyError,
    PreferenceValueUnresolvedError,
    PreferenceVersionConflictError,
    normalize_preference_text,
)
from .producer import ProducerRepository
from .region import RegionRepository
from .stats import StatsRepository
from .tasting import TastingRepository
from .wine import WineRepository
from .sync_logs import SyncLogRepository
from .food_pairing import FoodPairingRepository

__all__ = [
    "BottleRepository",
    "DeclaredPreferenceRepository",
    "MAX_DECLARED_PREFERENCES",
    "PreferenceCombinationError",
    "PreferenceCountChangedError",
    "PreferenceDuplicateError",
    "PreferenceLimitReachedError",
    "PreferenceNotFoundError",
    "PreferenceRepositoryError",
    "PreferenceStoreBusyError",
    "PreferenceValueUnresolvedError",
    "PreferenceVersionConflictError",
    "normalize_preference_text",
    "ProducerRepository",
    "RegionRepository",
    "StatsRepository",
    "TastingRepository",
    "WineRepository",
    "SyncLogRepository",
    "FoodPairingRepository",
]
