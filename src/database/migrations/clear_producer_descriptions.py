"""Explicitly clear producer descriptions after verifying a restorable backup.

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


def _producer_snapshot(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
    """Return producer identity fields and descriptions in stable order."""
    return connection.execute("SELECT id, name, country, region, description FROM producers ORDER BY id").fetchall()


def _require_verified_backup(
    database_connection: sqlite3.Connection,
    database_path: Path,
    backup_path: Path,
) -> None:
    """Verify that the backup is healthy and can restore current producer descriptions."""
    if not backup_path.is_file() or backup_path.stat().st_size == 0:
        raise FileNotFoundError("Verified backup does not exist or is empty.")
    if database_path.resolve() == backup_path.resolve():
        raise ValueError("Backup path must differ from the database path.")

    with sqlite3.connect(backup_path) as backup_connection:
        _validate_database(database_connection, "Cellar")
        _validate_database(backup_connection, "Backup")
        database_rows = _producer_snapshot(database_connection)
        backup_rows = _producer_snapshot(backup_connection)
        identities_match = [row[:4] for row in database_rows] == [row[:4] for row in backup_rows]
        descriptions_are_restorable = identities_match and all(
            database_row[4] is None or database_row[4] == backup_row[4]
            for database_row, backup_row in zip(database_rows, backup_rows, strict=True)
        )
        if not descriptions_are_restorable:
            raise ValueError("Backup producer data does not match the current cellar database.")


def run_migration(
    database_path: str | Path,
    backup_path: str | Path,
    *,
    writers_stopped: bool,
) -> int:
    """Clear all producer descriptions in one validated transaction.

    Args:
        database_path: Explicit path to the existing cellar database.
        backup_path: Explicit path to a verified pre-cleanup backup.
        writers_stopped: Explicit confirmation that API, sync, and import writers are stopped.

    Returns:
        Number of non-null producer descriptions cleared.

    Raises:
        FileNotFoundError: If the database or backup is missing.
        ValueError: If writer confirmation or backup verification fails.
        sqlite3.DatabaseError: If locking or validation fails.
    """
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
        cursor = connection.execute("UPDATE producers SET description = NULL WHERE description IS NOT NULL")
        cleared_count = cursor.rowcount
        _validate_database(connection, "Migrated cellar")
        connection.commit()
        logger.info("Producer description cleanup completed and verified: cleared=%s", cleared_count)
        return cleared_count
    except Exception:
        connection.rollback()
        logger.exception("Producer description cleanup failed; transaction rolled back")
        raise
    finally:
        connection.close()


def main() -> None:
    """Parse explicit safety inputs and run the producer-description cleanup."""
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
    logger.info("Producer descriptions cleared: %s", cleared_count)


if __name__ == "__main__":
    main()
