"""
Database access layer for wine cellar.

Provides high-level interface for database operations without exposing SQL queries.
Follows repository pattern for clean separation of concerns.
"""

from .bottle import BottleRepository
from .declared_preference import (
    MAX_DECLARED_PREFERENCES,
    DeclaredPreferenceRepository,
    PreferenceCombinationError,
    PreferenceCountChangedError,
    PreferenceDuplicateError,
    PreferenceLimitReachedError,
    PreferenceNotFoundError,
    PreferenceRepositoryError,
    PreferenceStoreBusyError,
    PreferenceValueUnresolvedError,
    PreferenceVersionConflictError,
    canonicalize_preference_identity,
    normalize_preference_text,
)
from .food_pairing import FoodPairingRepository
from .producer import ProducerRepository
from .region import RegionRepository
from .shopping_list import (
    ShoppingListDuplicateError,
    ShoppingListNotFoundError,
    ShoppingListRepository,
    ShoppingListRepositoryError,
    ShoppingListStoreBusyError,
    ShoppingListStoreUnavailableError,
    ShoppingListValidationError,
    ShoppingListVersionConflictError,
    normalize_shopping_list_name,
)
from .stats import StatsRepository
from .sync_logs import SyncLogRepository
from .tasting import TastingRepository
from .wine import WineRepository

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
    "canonicalize_preference_identity",
    "normalize_preference_text",
    "ProducerRepository",
    "RegionRepository",
    "ShoppingListRepository",
    "ShoppingListRepositoryError",
    "ShoppingListValidationError",
    "ShoppingListNotFoundError",
    "ShoppingListDuplicateError",
    "ShoppingListVersionConflictError",
    "ShoppingListStoreBusyError",
    "ShoppingListStoreUnavailableError",
    "normalize_shopping_list_name",
    "StatsRepository",
    "TastingRepository",
    "WineRepository",
    "SyncLogRepository",
    "FoodPairingRepository",
]
