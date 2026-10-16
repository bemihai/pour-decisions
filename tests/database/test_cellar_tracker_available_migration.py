"""Tests for the explicit CellarTracker availability migration."""

import sqlite3
from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.database.migrations import add_cellar_tracker_available as migration


def _create_legacy_database(path: Path) -> None:
    """Create a populated database without the new availability column."""
    assert initialize_database(path)
    with get_db_connection(path) as conn:
        conn.execute("ALTER TABLE wines DROP COLUMN cellar_tracker_available")
        conn.executemany(
            """
            INSERT INTO wines (
                id, source, external_id, wine_name, wine_type, drink_index, drink_window_source
            ) VALUES (?, ?, ?, ?, 'Red', ?, ?)
            """,
            [
                (1, "cellar_tracker", "ct-null", "CT legacy null source", 0.375, None),
                (2, "cellar_tracker", "ct-source", "CT legacy source", -0.25, "cellar_tracker"),
                (3, "cellar_tracker", "ct-manual", "CT manual window", 88.0, "manual"),
                (4, "vivino", "vivino", "Vivino wine", 55.0, None),
            ],
        )
        conn.commit()


def _availability_values(path: Path) -> list[tuple[int, float | None]]:
    """Return migrated availability values in stable row order."""
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT id, cellar_tracker_available FROM wines ORDER BY id").fetchall()


def test_migration_backfills_only_recoverable_cellartracker_values(temp_dir: Path) -> None:
    """Legacy CT values are preserved without treating unrelated indices as availability."""
    db_path = temp_dir / "legacy.db"
    _create_legacy_database(db_path)

    migration.run_migration(db_path)

    assert _availability_values(db_path) == [(1, 0.375), (2, -0.25), (3, None), (4, None)]
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_migration_is_idempotent_and_preserves_refreshed_values(temp_dir: Path) -> None:
    """Repeated runs neither duplicate schema changes nor overwrite refreshed data."""
    db_path = temp_dir / "legacy.db"
    _create_legacy_database(db_path)
    migration.run_migration(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE wines SET cellar_tracker_available = 1.75 WHERE id = 1")
        connection.commit()

    migration.run_migration(db_path)

    assert _availability_values(db_path)[0] == (1, 1.75)
    with sqlite3.connect(db_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(wines)")]
    assert columns.count("cellar_tracker_available") == 1


def test_migration_rolls_back_schema_and_data_on_validation_failure(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validation failure leaves the legacy database unchanged."""
    db_path = temp_dir / "legacy.db"
    _create_legacy_database(db_path)

    def fail_validation(_cursor: sqlite3.Cursor) -> None:
        raise sqlite3.DatabaseError("synthetic validation failure")

    monkeypatch.setattr(migration, "_validate_database", fail_validation)

    with pytest.raises(sqlite3.DatabaseError, match="synthetic validation failure"):
        migration.run_migration(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(wines)")]
        drink_indices = connection.execute("SELECT id, drink_index FROM wines ORDER BY id").fetchall()
    assert "cellar_tracker_available" not in columns
    assert drink_indices == [(1, 0.375), (2, -0.25), (3, 88), (4, 55)]


def test_migration_requires_existing_database(temp_dir: Path) -> None:
    """A mistyped path cannot silently create a partial cellar database."""
    missing_path = temp_dir / "missing.db"

    with pytest.raises(FileNotFoundError):
        migration.run_migration(missing_path)

    assert not missing_path.exists()
