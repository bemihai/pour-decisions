"""Explicitly clear wine descriptions after verifying a restorable backup.

This data migration is idempotent and is never run during application startup.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.utils import logger


def _validate_database(connection: sqlite3.Connection, label: str) -> None:
    """Raise when a database fails integrity or foreign-key validation."""
    integrity_rows = [row[0] for row in connection.execute("PRAGMA integrity_check").fetchall()]
    if integrity_rows != ["ok"]:
        raise sqlite3.DatabaseError(f"{label} database integrity validation failed.")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise sqlite3.IntegrityError(f"{label} database foreign-key validation failed.")


def _wine_snapshot(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    """Return all persisted wine fields in stable order."""
    return connection.execute("""SELECT id, source, external_id, wine_name, producer_id, vintage, wine_type, varietal,
                  designation, region_id, appellation, vineyard, bottle_size, drink_from_year,
                  drink_to_year, drink_index, q_purchased, q_quantity, q_consumed, created_at,
                  updated_at, description, drink_window_source, cellar_tracker_available
           FROM wines ORDER BY id""").fetchall()


def _without_description(row: tuple[object, ...]) -> tuple[object, ...]:
    """Return a wine snapshot row without its description field."""
    return row[:21] + row[22:]


def _require_verified_backup(
    database_connection: sqlite3.Connection,
    database_path: Path,
    backup_path: Path,
) -> None:
    """Verify that the backup is healthy and can restore current wine descriptions."""
    if not backup_path.is_file() or backup_path.stat().st_size == 0:
        raise FileNotFoundError("Verified backup does not exist or is empty.")
    if database_path.resolve() == backup_path.resolve():
        raise ValueError("Backup path must differ from the database path.")

    with sqlite3.connect(backup_path) as backup_connection:
        _validate_database(database_connection, "Cellar")
        _validate_database(backup_connection, "Backup")
        database_rows = _wine_snapshot(database_connection)
        backup_rows = _wine_snapshot(backup_connection)
        identities_match = [_without_description(row) for row in database_rows] == [
            _without_description(row) for row in backup_rows
        ]
        descriptions_are_restorable = identities_match and all(
            database_row[21] is None or database_row[21] == backup_row[21]
            for database_row, backup_row in zip(database_rows, backup_rows, strict=True)
        )
        if not descriptions_are_restorable:
            raise ValueError("Backup wine data does not match the current cellar database.")


def run_migration(
    database_path: str | Path,
    backup_path: str | Path,
    *,
    writers_stopped: bool,
) -> int:
    """Clear only wine descriptions in one validated transaction."""
    resolved_database_path = Path(database_path)
    resolved_backup_path = Path(backup_path)
    if not resolved_database_path.is_file():
        raise FileNotFoundError("Cellar database does not exist at the requested migration path.")
    if not writers_stopped:
        raise ValueError("Stop API, import, and sync writers and confirm before running this migration.")

    connection = sqlite3.connect(resolved_database_path, isolation_level=None, timeout=0)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN EXCLUSIVE")
        _require_verified_backup(connection, resolved_database_path, resolved_backup_path)
        cursor = connection.execute("UPDATE wines SET description = NULL WHERE description IS NOT NULL")
        cleared_count = cursor.rowcount
        _validate_database(connection, "Migrated cellar")
        connection.commit()
        logger.info("Wine description cleanup completed and verified: cleared=%s", cleared_count)
        return cleared_count
    except Exception:
        connection.rollback()
        logger.exception("Wine description cleanup failed; transaction rolled back")
        raise
    finally:
        connection.close()


def main() -> None:
    """Parse explicit safety inputs and run the wine-description cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path, help="Existing cellar database path")
    parser.add_argument("--backup", required=True, type=Path, help="Verified pre-cleanup backup path")
    parser.add_argument(
        "--confirm-writers-stopped",
        action="store_true",
        help="Confirm API, import, and sync writers are stopped",
    )
    arguments = parser.parse_args()
    cleared_count = run_migration(
        arguments.database,
        arguments.backup,
        writers_stopped=arguments.confirm_writers_stopped,
    )
    logger.info("Wine descriptions cleared: %s", cleared_count)


if __name__ == "__main__":
    main()
