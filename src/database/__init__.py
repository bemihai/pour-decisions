"""Database package for wine cellar management."""

from .db import create_declared_preferences_schema, get_db_connection, initialize_database
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
    'get_db_connection',
    'initialize_database',
    'create_declared_preferences_schema',
    'build_update_query',
    'Wine',
    'Bottle',
    'Producer',
    'Region',
    'Tasting',
    'SyncLog',
    'FoodPairingRule',
    'DeclaredPreference',
    'PreferenceCurrency',
    'PreferenceStance',
    'PreferenceSubjectKind',
    'WineStyle',
]
