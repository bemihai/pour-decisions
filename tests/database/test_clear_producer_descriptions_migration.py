"""Tests for the explicit producer-description cleanup migration."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database
from src.database.migrations.clear_producer_descriptions import run_migration


def _create_populated_database(path: Path) -> None:
    """Create a cellar with producer descriptions and related wine data."""
    assert initialize_database(path)
    with get_db_connection(path) as connection:
        connection.executemany(
            "INSERT INTO producers (id, name, country, region, description) VALUES (?, ?, ?, ?, ?)",
            [
                (1, "Crama Bauer", "Romania", "Dragasani", "Existing Bauer description."),
                (2, "No Description", "France", "Loire", None),
            ],
        )
        connection.execute("""
            INSERT INTO wines (id, source, external_id, wine_name, wine_type, producer_id, description)
            VALUES (10, 'manual', 'wine-10', 'Test Wine', 'Red', 1, 'Wine description remains.')
            """)
        connection.commit()


def _producer_rows(path: Path) -> list[tuple[object, ...]]:
    """Return all producer fields relevant to cleanup and recovery."""
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT id, name, country, region, description FROM producers ORDER BY id").fetchall()


def test_migration_clears_only_producer_descriptions_and_is_idempotent(temp_dir: Path) -> None:
    """Cleanup reports affected rows, preserves other data, and clears zero on repetition."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    _create_populated_database(database_path)
    shutil.copy2(database_path, backup_path)

    assert run_migration(database_path, backup_path, writers_stopped=True) == 1
    assert run_migration(database_path, backup_path, writers_stopped=True) == 0

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT description FROM producers ORDER BY id").fetchall() == [(None,), (None,)]
        assert connection.execute("SELECT producer_id, description FROM wines WHERE id = 10").fetchone() == (
            1,
            "Wine description remains.",
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_verified_backup_restores_pre_cleanup_descriptions(temp_dir: Path) -> None:
    """The required backup can restore the exact pre-cleanup producer state."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    restored_path = temp_dir / "restored.db"
    _create_populated_database(database_path)
    expected_rows = _producer_rows(database_path)
    shutil.copy2(database_path, backup_path)

    run_migration(database_path, backup_path, writers_stopped=True)
    shutil.copy2(backup_path, restored_path)

    assert _producer_rows(restored_path) == expected_rows


def test_migration_requires_explicit_existing_paths_and_stopped_writers(temp_dir: Path) -> None:
    """Missing safety prerequisites fail without modifying the cellar."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    _create_populated_database(database_path)
    original_rows = _producer_rows(database_path)

    with pytest.raises(FileNotFoundError):
        run_migration(database_path, backup_path, writers_stopped=True)
    shutil.copy2(database_path, backup_path)
    with pytest.raises(ValueError, match="Stop API"):
        run_migration(database_path, backup_path, writers_stopped=False)

    assert _producer_rows(database_path) == original_rows


def test_mismatched_backup_is_rejected(temp_dir: Path) -> None:
    """A healthy but stale backup cannot authorize destructive cleanup."""
    database_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    _create_populated_database(database_path)
    shutil.copy2(database_path, backup_path)
    with sqlite3.connect(backup_path) as connection:
        connection.execute("UPDATE producers SET description = 'Stale backup' WHERE id = 1")
        connection.commit()

    with pytest.raises(ValueError, match="does not match"):
        run_migration(database_path, backup_path, writers_stopped=True)

    assert _producer_rows(database_path)[0][-1] == "Existing Bauer description."
