"""Tests for the explicit declared-preference migration and recovery contract."""

from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3

import pytest

from src.database import get_db_connection, initialize_database, verify_declared_preferences_schema
from src.database.migrations.create_declared_preferences import run_migration

CORE_TABLES = ("producers", "regions", "wines", "bottles", "tastings", "sync_log")


def _create_legacy_database(path: Path) -> None:
    """Create a populated database representing the schema before this migration."""
    assert initialize_database(path)
    with get_db_connection(path) as conn:
        conn.execute("DROP TABLE declared_preferences")
        conn.execute("INSERT INTO producers (name, country) VALUES ('Legacy Producer', 'France')")
        conn.execute("INSERT INTO regions (primary_name, country) VALUES ('Legacy Region', 'France')")
        conn.commit()


def _schema_signature(path: Path) -> tuple[object, ...]:
    """Return exact table SQL, columns, and index definitions."""
    with sqlite3.connect(path) as conn:
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'declared_preferences'"
        ).fetchone()[0]
        columns = tuple(conn.execute("PRAGMA table_info(declared_preferences)").fetchall())
        indexes = []
        for row in conn.execute("PRAGMA index_list(declared_preferences)").fetchall():
            index_name = row[1]
            escaped_name = index_name.replace('"', '""')
            index_columns = tuple(
                index_row[2] for index_row in conn.execute(f'PRAGMA index_info("{escaped_name}")').fetchall()
            )
            indexes.append((index_name, row[2], index_columns))
        return " ".join(table_sql.casefold().split()), columns, tuple(sorted(indexes))


def _core_counts(path: Path) -> dict[str, int]:
    """Return counts for existing cellar tables using a fixed whitelist."""
    with sqlite3.connect(path) as conn:
        return {table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in CORE_TABLES}


def test_migration_creates_schema_and_preserves_core_rows(temp_dir: Path) -> None:
    """An existing cellar gains only the compatible preference schema."""
    db_path = temp_dir / "legacy.db"
    _create_legacy_database(db_path)
    before_counts = _core_counts(db_path)

    run_migration(db_path)

    with sqlite3.connect(db_path) as conn:
        verify_declared_preferences_schema(conn.cursor())
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert _core_counts(db_path) == before_counts


def test_migration_is_idempotent_on_fresh_and_migrated_databases(temp_dir: Path) -> None:
    """Compatible schemas remain byte-for-byte equivalent across repeated runs."""
    fresh_path = temp_dir / "fresh.db"
    legacy_path = temp_dir / "legacy.db"
    assert initialize_database(fresh_path)
    _create_legacy_database(legacy_path)

    run_migration(fresh_path)
    run_migration(legacy_path)
    first_migrated_signature = _schema_signature(legacy_path)
    run_migration(legacy_path)

    assert _schema_signature(fresh_path) == first_migrated_signature
    assert _schema_signature(legacy_path) == first_migrated_signature


def test_incompatible_partial_schema_rolls_back_without_repair(temp_dir: Path) -> None:
    """A lookalike table without constraints fails and gains no indexes."""
    db_path = temp_dir / "partial.db"
    _create_legacy_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE declared_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_kind TEXT NOT NULL,
                stance TEXT,
                normalized_value TEXT NOT NULL,
                display_value TEXT,
                price_minor_units INTEGER,
                currency TEXT,
                provenance TEXT NOT NULL DEFAULT 'explicit_user',
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00.000Z',
                updated_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00.000Z'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO declared_preferences (
                subject_kind, stance, normalized_value, display_value
            ) VALUES ('grape', 'like', 'legacy', 'Legacy')
            """
        )
        conn.commit()
        original_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'declared_preferences'"
        ).fetchone()[0]

    with pytest.raises(ValueError, match="incompatible"):
        run_migration(db_path)

    with sqlite3.connect(db_path) as conn:
        current_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'declared_preferences'"
        ).fetchone()[0]
        assert current_sql == original_sql
        assert conn.execute("PRAGMA index_list(declared_preferences)").fetchall() == []
        assert conn.execute("SELECT normalized_value FROM declared_preferences").fetchall() == [("legacy",)]


def test_verified_backup_can_restore_pre_migration_state(temp_dir: Path) -> None:
    """A temporary backup remains non-empty, valid, and restorable before first use."""
    db_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    restored_path = temp_dir / "cellar-restored.db"
    _create_legacy_database(db_path)
    expected_counts = _core_counts(db_path)

    shutil.copy2(db_path, backup_path)
    assert backup_path.stat().st_size > 0
    with sqlite3.connect(backup_path) as backup_connection:
        assert backup_connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"

    run_migration(db_path)
    shutil.copy2(backup_path, restored_path)

    with sqlite3.connect(restored_path) as restored_connection:
        assert restored_connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert restored_connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert restored_connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'declared_preferences'"
        ).fetchone()[0] == 0
    assert _core_counts(restored_path) == expected_counts


def test_migration_requires_an_existing_explicit_database(temp_dir: Path) -> None:
    """A typo cannot silently create a preference-only database."""
    missing_path = temp_dir / "missing.db"

    with pytest.raises(FileNotFoundError):
        run_migration(missing_path)

    assert not missing_path.exists()
