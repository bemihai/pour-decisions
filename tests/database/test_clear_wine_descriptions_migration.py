"""Tests for the explicit wine-description cleanup migration."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.database.migrations.clear_wine_descriptions import run_migration


def _create_populated_database(path: Path) -> None:
    """Create a cellar containing wine and producer descriptions."""
    assert initialize_database(path)
    with get_db_connection(path) as connection:
        connection.execute(
            "INSERT INTO producers (id, name, country, region, description) VALUES (1, ?, ?, ?, ?)",
            ("Crama Bauer", "Romania", "Dragasani", "Producer description remains."),
        )
        connection.executemany(
            """INSERT INTO wines (
                   id, source, external_id, wine_name, wine_type, producer_id, description,
                   drink_from_year, drink_to_year, drink_index, drink_window_source
               ) VALUES (?, 'manual', ?, ?, 'Red', 1, ?, ?, ?, ?, ?)""",
            [
                (10, "wine-10", "Test Wine", "Old wine description.", 2025, 2032, 0.6, "manual"),
                (11, "wine-11", "No Description", None, None, None, None, None),
            ],
        )
        connection.commit()


def _wine_rows(path: Path) -> list[tuple[object, ...]]:
    """Return wine fields that must survive cleanup or backup restore."""
    with sqlite3.connect(path) as connection:
        return connection.execute("""SELECT id, wine_name, producer_id, description, drink_from_year, drink_to_year,
                      drink_index, drink_window_source FROM wines ORDER BY id""").fetchall()


def test_migration_clears_only_wine_descriptions_and_is_idempotent(temp_dir: Path) -> None:
    """Cleanup preserves identities, producer prose, and drinking windows."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    _create_populated_database(database_path)
    shutil.copy2(database_path, backup_path)

    assert run_migration(database_path, backup_path, writers_stopped=True) == 1
    assert run_migration(database_path, backup_path, writers_stopped=True) == 0

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT description FROM wines ORDER BY id").fetchall() == [(None,), (None,)]
        assert connection.execute("SELECT description FROM producers WHERE id = 1").fetchone() == (
            "Producer description remains.",
        )
        assert connection.execute(
            "SELECT drink_from_year, drink_to_year, drink_index, drink_window_source FROM wines WHERE id = 10"
        ).fetchone() == (2025, 2032, 0.6, "manual")
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_verified_backup_restores_pre_cleanup_wines(temp_dir: Path) -> None:
    """The required backup restores the exact pre-cleanup wine state."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    restored_path = temp_dir / "restored.db"
    _create_populated_database(database_path)
    expected_rows = _wine_rows(database_path)
    shutil.copy2(database_path, backup_path)

    run_migration(database_path, backup_path, writers_stopped=True)
    shutil.copy2(backup_path, restored_path)

    assert _wine_rows(restored_path) == expected_rows


def test_migration_requires_explicit_backup_and_stopped_writers(temp_dir: Path) -> None:
    """Missing safety prerequisites fail without changing descriptions."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    _create_populated_database(database_path)
    original_rows = _wine_rows(database_path)

    with pytest.raises(FileNotFoundError):
        run_migration(database_path, backup_path, writers_stopped=True)
    shutil.copy2(database_path, backup_path)
    with pytest.raises(ValueError, match="Stop API"):
        run_migration(database_path, backup_path, writers_stopped=False)

    assert _wine_rows(database_path) == original_rows


def test_mismatched_backup_is_rejected(temp_dir: Path) -> None:
    """A healthy but stale backup cannot authorize destructive cleanup."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    _create_populated_database(database_path)
    shutil.copy2(database_path, backup_path)
    with sqlite3.connect(backup_path) as connection:
        connection.execute("UPDATE wines SET description = 'Stale backup' WHERE id = 10")
        connection.commit()

    with pytest.raises(ValueError, match="does not match"):
        run_migration(database_path, backup_path, writers_stopped=True)

    assert _wine_rows(database_path)[0][3] == "Old wine description."
