"""Database package for wine cellar management."""

from .db import get_db_connection, initialize_database
from .models import Wine, Bottle, Producer, Region, Tasting, SyncLog, FoodPairingRule
from .utils import build_update_query

__all__ = [
    'get_db_connection',
    'initialize_database',
    'build_update_query',
    'Wine',
    'Bottle',
    'Producer',
    'Region',
    'Tasting',
    'SyncLog',
    'FoodPairingRule',
]
