"""Database package for wine cellar management."""

from .db import (
    create_declared_preferences_schema,
    create_shopping_list_schema,
    drop_all_tables,
    get_db_connection,
    initialize_database,
    verify_declared_preferences_schema,
    verify_shopping_list_schema,
)
from .models import (
    Bottle,
    DeclaredPreference,
    FoodPairingRule,
    PreferenceCurrency,
    PreferenceStance,
    PreferenceSubjectKind,
    Producer,
    Region,
    SyncLog,
    Tasting,
    Wine,
    WineStyle,
)
from .utils import build_update_query

__all__ = [
    "get_db_connection",
    "initialize_database",
    "create_declared_preferences_schema",
    "verify_declared_preferences_schema",
    "create_shopping_list_schema",
    "verify_shopping_list_schema",
    "drop_all_tables",
    "build_update_query",
    "Wine",
    "Bottle",
    "Producer",
    "Region",
    "Tasting",
    "SyncLog",
    "FoodPairingRule",
    "DeclaredPreference",
    "PreferenceCurrency",
    "PreferenceStance",
    "PreferenceSubjectKind",
    "WineStyle",
]
