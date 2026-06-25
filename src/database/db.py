"""Database connection and initialization."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from src.utils import logger, get_default_db_path


def _resolve_db_path(db_path: str | Path | None) -> str | Path:
    """Resolve the effective database path lazily.

    Args:
        db_path: Explicit path override, or ``None`` to use the configured default.

    Returns:
        Database path to use for the current operation.
    """
    if db_path is not None:
        return db_path
    return get_default_db_path()


@contextmanager
def get_db_connection(db_path: str | Path | None = None):
    """
    Context manager for database connections.

    Args:
        db_path: Path to SQLite database file. When ``None``, the configured
            default path is resolved lazily.

    Yields:
        sqlite3.Connection: Database connection
    """
    conn = None
    try:
        resolved_db_path = _resolve_db_path(db_path)
        conn = sqlite3.connect(resolved_db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()


def initialize_database(db_path: str | Path | None = None) -> bool:
    """
    Initialize the wine cellar database with schema.

    Args:
        db_path: Path to SQLite database file. When ``None``, the configured
            default path is resolved lazily.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        resolved_db_path = _resolve_db_path(db_path)
        Path(resolved_db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing database at: {resolved_db_path}")

        with get_db_connection(resolved_db_path) as conn:
            cursor = conn.cursor()

            # Create tables in order of dependencies
            _create_producers_table(cursor)
            _create_regions_table(cursor)
            _create_wines_table(cursor)
            _create_tastings_table(cursor)
            _create_bottles_table(cursor)
            _create_sync_log_table(cursor)
            _create_views(cursor)

            conn.commit()

        logger.info(f"Database initialized successfully at: {resolved_db_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def _create_producers_table(cursor: sqlite3.Cursor):
    """Create producers table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS producers (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL UNIQUE,
            country             TEXT,
            region              TEXT,
            description         TEXT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_producers_name ON producers(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_producers_country ON producers(country)")


def _create_regions_table(cursor: sqlite3.Cursor):
    """Create regions table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            primary_name        TEXT NOT NULL,
            secondary_name      TEXT,
            country             TEXT NOT NULL,
            description         TEXT,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(primary_name, secondary_name, country)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_regions_country ON regions(country)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_regions_primary ON regions(primary_name)")


def _create_tastings_table(cursor: sqlite3.Cursor):
    """Create tastings table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tastings (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            wine_id                 INTEGER NOT NULL REFERENCES wines(id) ON DELETE CASCADE,
            is_defective            BOOLEAN NOT NULL DEFAULT 0,
            personal_rating         INTEGER,
            tasting_notes           TEXT,
            do_like                 BOOLEAN,
            community_rating        DECIMAL(5,2),
            like_votes              INTEGER DEFAULT 0,
            like_percentage         DECIMAL(5,2),
            last_tasted_date        DATE,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(wine_id),
            CHECK(personal_rating IS NULL OR (personal_rating >= 0 AND personal_rating <= 100)),
            CHECK(community_rating IS NULL OR (community_rating >= 0 AND community_rating <= 100))
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tastings_wine ON tastings(wine_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tastings_personal_rating ON tastings(personal_rating)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tastings_last_tasted_date ON tastings(last_tasted_date)")


def _create_wines_table(cursor: sqlite3.Cursor):
    """Create wines table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wines (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            source                  TEXT NOT NULL CHECK(source IN ('cellar_tracker', 'vivino', 'manual')),
            external_id             TEXT,
            wine_name               TEXT NOT NULL,
            producer_id             INTEGER REFERENCES producers(id),
            vintage                 INTEGER,
            wine_type               TEXT NOT NULL,
            varietal                TEXT,
            designation             TEXT,
            region_id               INTEGER REFERENCES regions(id),
            appellation             TEXT,
            vineyard                TEXT,
            bottle_size             TEXT DEFAULT '750ml',
            drink_from_year         INTEGER,
            drink_to_year           INTEGER,
            drink_index             DECIMAL(5,2),
            drink_window_source     TEXT,
            description             TEXT,
            q_purchased             INTEGER DEFAULT 0,
            q_quantity              INTEGER DEFAULT 0,
            q_consumed              INTEGER DEFAULT 0,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source, external_id)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wines_producer ON wines(producer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wines_region ON wines(region_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wines_vintage ON wines(vintage)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wines_type ON wines(wine_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wines_name ON wines(wine_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wines_external_id ON wines(source, external_id)")


def _create_bottles_table(cursor: sqlite3.Cursor):
    """Create bottles table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bottles (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            wine_id                 INTEGER NOT NULL REFERENCES wines(id) ON DELETE CASCADE,
            source                  TEXT NOT NULL CHECK(source IN ('cellar_tracker', 'vivino', 'manual')),
            external_bottle_id      TEXT,
            quantity                INTEGER NOT NULL DEFAULT 1,
            status                  TEXT NOT NULL DEFAULT 'in_cellar' 
                                    CHECK(status IN ('in_cellar', 'consumed', 'gifted', 'lost')),
            location                TEXT,
            bin                     TEXT,
            purchase_date           DATE,
            purchase_price          DECIMAL(10,2),
            valuation_price         DECIMAL(10,2),
            currency                TEXT DEFAULT 'RON',
            store_name              TEXT,
            consumed_date           DATE,
            bottle_note             TEXT,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CHECK(
                (status = 'in_cellar' AND consumed_date IS NULL) OR
                (status = 'consumed' AND consumed_date IS NOT NULL) OR
                (status IN ('gifted', 'lost'))
            )
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bottles_wine ON bottles(wine_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bottles_status ON bottles(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bottles_location ON bottles(location)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bottles_consumed_date ON bottles(consumed_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bottles_location_status ON bottles(location, status)")


def _create_sync_log_table(cursor: sqlite3.Cursor):
    """Create sync_log table."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            source                  TEXT NOT NULL,
            sync_type               TEXT NOT NULL,
            sync_started_at         TIMESTAMP NOT NULL,
            sync_completed_at       TIMESTAMP,
            status                  TEXT NOT NULL,
            records_processed       INTEGER DEFAULT 0,
            records_imported        INTEGER DEFAULT 0,
            records_updated         INTEGER DEFAULT 0,
            records_skipped         INTEGER DEFAULT 0,
            records_failed          INTEGER DEFAULT 0,
            error_message           TEXT,
            error_details           TEXT,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sync_log_source ON sync_log(source)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sync_log_status ON sync_log(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sync_log_date ON sync_log(sync_started_at)")


def _create_views(cursor: sqlite3.Cursor):
    """Create database views."""

    # Current Inventory View
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS current_inventory AS
        SELECT 
            w.id as wine_id,
            w.wine_name,
            p.name as producer,
            w.vintage,
            w.wine_type,
            r.country,
            COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region,
            b.location,
            b.bin,
            SUM(b.quantity) as total_bottles,
            t.personal_rating,
            t.community_rating,
            w.drink_from_year,
            w.drink_to_year,
            MAX(b.purchase_date) as last_purchase_date
        FROM wines w
        JOIN bottles b ON w.id = b.wine_id
        LEFT JOIN producers p ON w.producer_id = p.id
        LEFT JOIN regions r ON w.region_id = r.id
        LEFT JOIN tastings t ON w.id = t.wine_id
        WHERE b.status = 'in_cellar'
        GROUP BY w.id, b.location, b.bin
        ORDER BY p.name, w.vintage DESC
    """)

    # Top Rated Wines View
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS top_rated_wines AS
        SELECT 
            w.wine_name,
            p.name as producer,
            w.vintage,
            w.wine_type,
            r.country,
            t.personal_rating,
            t.community_rating,
            COUNT(b.id) as bottles_owned
        FROM wines w
        LEFT JOIN producers p ON w.producer_id = p.id
        LEFT JOIN regions r ON w.region_id = r.id
        LEFT JOIN tastings t ON w.id = t.wine_id
        LEFT JOIN bottles b ON w.id = b.wine_id AND b.status = 'in_cellar'
        WHERE t.personal_rating IS NOT NULL
        GROUP BY w.id
        ORDER BY t.personal_rating DESC, t.community_rating DESC
    """)

    # Cellar Statistics View
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS cellar_stats AS
        SELECT 
            w.wine_type,
            r.country,
            COUNT(DISTINCT w.id) as unique_wines,
            SUM(b.quantity) as total_bottles,
            AVG(t.personal_rating) as avg_personal_rating,
            AVG(t.community_rating) as avg_community_rating
        FROM wines w
        JOIN bottles b ON w.id = b.wine_id
        LEFT JOIN regions r ON w.region_id = r.id
        LEFT JOIN tastings t ON w.id = t.wine_id
        WHERE b.status = 'in_cellar'
        GROUP BY w.wine_type, r.country
        ORDER BY total_bottles DESC
    """)


def drop_all_tables(db_path: str | Path | None = None) -> bool:
    """
    Drop all tables (for testing/reset purposes).

    Args:
        db_path: Path to SQLite database file. When ``None``, the configured
            default path is resolved lazily.

    Returns:
        bool: True if successful
    """
    try:
        with get_db_connection(_resolve_db_path(db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute("DROP VIEW IF EXISTS current_inventory")
            cursor.execute("DROP VIEW IF EXISTS top_rated_wines")
            cursor.execute("DROP VIEW IF EXISTS cellar_stats")
            cursor.execute("DROP TABLE IF EXISTS sync_log")
            cursor.execute("DROP TABLE IF EXISTS bottles")
            cursor.execute("DROP TABLE IF EXISTS tastings")
            cursor.execute("DROP TABLE IF EXISTS wines")
            cursor.execute("DROP TABLE IF EXISTS regions")
            cursor.execute("DROP TABLE IF EXISTS producers")

            conn.commit()

        logger.info("All tables dropped successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        return False
