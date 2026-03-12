"""
Migration script to add drink_window_source column to wines table.

Tracks the provenance of a wine's drinking window so higher-priority sources
(manual, cellar_tracker) are never overwritten by lower-priority ones (llm, heuristic).

Priority order (highest first): manual > cellar_tracker > llm > heuristic > NULL
"""

import sqlite3
from pathlib import Path

ALLOWED_TABLES: set[str] = {"wines"}


def check_column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    """Check if a column already exists in a table.

    Args:
        cursor: Database cursor.
        table_name: Table to inspect. Must be in ALLOWED_TABLES.
        column_name: Column to look for.

    Returns:
        True if column exists.

    Raises:
        ValueError: If table_name is not whitelisted.
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name '{table_name}' for schema inspection.")
    cursor.execute(f"PRAGMA table_info({table_name})")
    return column_name in [row[1] for row in cursor.fetchall()]


def apply_migration(conn: sqlite3.Connection) -> None:
    """Add drink_window_source column and backfill existing CellarTracker rows.

    Args:
        conn: Open SQLite connection.
    """
    cursor = conn.cursor()

    if check_column_exists(cursor, "wines", "drink_window_source"):
        print("  - Column 'drink_window_source' already exists in wines table, skipping")
        return

    cursor.execute("ALTER TABLE wines ADD COLUMN drink_window_source TEXT")
    print("  + Added 'drink_window_source' column to wines table")

    # Backfill: existing CellarTracker wines that already have a drinking window
    cursor.execute("""
        UPDATE wines
        SET drink_window_source = 'cellar_tracker'
        WHERE source = 'cellar_tracker'
          AND drink_from_year IS NOT NULL
    """)
    print(f"  + Backfilled {cursor.rowcount} CellarTracker rows with drink_window_source='cellar_tracker'")

    conn.commit()
    print("  Migration committed successfully")


def run_migration(db_path: str) -> None:
    """Run the migration.

    Args:
        db_path: Path to the SQLite database file.
    """
    print(f"Running migration on database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        apply_migration(conn)
        print("Migration completed successfully")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from src.utils import get_default_db_path

    db_path = get_default_db_path()
    run_migration(str(db_path))

