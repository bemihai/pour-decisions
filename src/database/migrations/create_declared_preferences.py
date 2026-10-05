"""Create the declared-preference schema in an existing cellar database.

This migration is explicit and idempotent. API startup never imports or runs it.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

from src.database import create_declared_preferences_schema, verify_declared_preferences_schema
from src.utils import get_default_db_path, logger


def run_migration(db_path: str | Path) -> None:
    """Create or verify the declared-preference schema transactionally.

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

    connection = sqlite3.connect(resolved_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.cursor()
        create_declared_preferences_schema(cursor)
        verify_declared_preferences_schema(cursor)

        integrity_rows = [row[0] for row in cursor.execute("PRAGMA integrity_check").fetchall()]
        if integrity_rows != ["ok"]:
            raise sqlite3.DatabaseError("Database integrity validation failed.")
        if cursor.execute("PRAGMA foreign_key_check").fetchall():
            raise sqlite3.IntegrityError("Database foreign-key validation failed.")

        connection.commit()
        logger.info("Declared-preference migration completed and verified")
    except Exception:
        connection.rollback()
        logger.exception("Declared-preference migration failed; transaction rolled back")
        raise
    finally:
        connection.close()


def main() -> None:
    """Run the migration against the configured cellar database."""
    run_migration(get_default_db_path())


if __name__ == "__main__":
    main()
