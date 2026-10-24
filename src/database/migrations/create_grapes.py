"""Create the grape schema in an existing cellar database.

This migration is explicit and idempotent. API startup never imports or runs it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.database.db import create_grapes_schema, get_db_connection, verify_grapes_schema
from src.database.repository.grape import link_wine_grapes
from src.utils import get_default_db_path, logger


def run_migration(db_path: str | Path) -> None:
    """Create or verify the grape schema transactionally.

    Args:
        db_path: Explicit path to an existing cellar database.

    Raises:
        FileNotFoundError: If the explicit database path does not exist.
        ValueError: If an existing partial schema is incompatible.
        sqlite3.DatabaseError: If SQLite cannot complete validation or commit.
    """
    resolved_path = Path(db_path)
    if not resolved_path.is_file():
        raise FileNotFoundError("Cellar database does not exist at the requested migration path.")

    with get_db_connection(resolved_path) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.cursor()
            create_grapes_schema(cursor)
            verify_grapes_schema(cursor)
            wines = cursor.execute("SELECT id, varietal FROM wines").fetchall()
            for wine in wines:
                link_wine_grapes(cursor, wine["id"], wine["varietal"])

            integrity_rows = [row[0] for row in cursor.execute("PRAGMA integrity_check").fetchall()]
            if integrity_rows != ["ok"]:
                raise sqlite3.DatabaseError("Database integrity validation failed.")
            if cursor.execute("PRAGMA foreign_key_check").fetchall():
                raise sqlite3.IntegrityError("Database foreign-key validation failed.")

            connection.commit()
            logger.info("Grape migration completed and verified")
        except Exception:
            connection.rollback()
            logger.exception("Grape migration failed; transaction rolled back")
            raise


def main() -> None:
    """Run the migration against the configured cellar database."""
    run_migration(get_default_db_path())


if __name__ == "__main__":
    main()
