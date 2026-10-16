"""Add raw CellarTracker availability storage to an existing cellar database.

This migration is explicit and idempotent. API startup never imports or runs it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.utils import get_default_db_path, logger


def _column_exists(cursor: sqlite3.Cursor) -> bool:
    """Return whether the target column already exists."""
    columns = cursor.execute("PRAGMA table_info(wines)").fetchall()
    return any(row[1] == "cellar_tracker_available" for row in columns)


def _validate_database(cursor: sqlite3.Cursor) -> None:
    """Raise when database integrity or foreign-key validation fails."""
    integrity_rows = [row[0] for row in cursor.execute("PRAGMA integrity_check").fetchall()]
    if integrity_rows != ["ok"]:
        raise sqlite3.DatabaseError("Database integrity validation failed.")
    if cursor.execute("PRAGMA foreign_key_check").fetchall():
        raise sqlite3.IntegrityError("Database foreign-key validation failed.")


def run_migration(db_path: str | Path) -> None:
    """Add and backfill ``cellar_tracker_available`` transactionally.

    Args:
        db_path: Explicit path to an existing cellar database.

    Raises:
        FileNotFoundError: If the explicit database path does not exist.
        sqlite3.DatabaseError: If SQLite cannot apply or validate the migration.
    """
    resolved_path = Path(db_path)
    if not resolved_path.is_file():
        raise FileNotFoundError("Cellar database does not exist at the requested migration path.")

    connection = sqlite3.connect(resolved_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.cursor()

        if not _column_exists(cursor):
            cursor.execute("ALTER TABLE wines ADD COLUMN cellar_tracker_available DECIMAL")

        cursor.execute("""
            UPDATE wines
            SET cellar_tracker_available = drink_index
            WHERE cellar_tracker_available IS NULL
              AND drink_index IS NOT NULL
              AND source = 'cellar_tracker'
              AND (drink_window_source = 'cellar_tracker' OR drink_window_source IS NULL)
            """)

        _validate_database(cursor)
        connection.commit()
        logger.info("CellarTracker availability migration completed and verified")
    except Exception:
        connection.rollback()
        logger.exception("CellarTracker availability migration failed; transaction rolled back")
        raise
    finally:
        connection.close()


def main() -> None:
    """Run the migration against the configured cellar database."""
    run_migration(get_default_db_path())


if __name__ == "__main__":
    main()
