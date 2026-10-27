"""Tests for the explicit Shopping List migration and recovery contract."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from src.database import get_db_connection, initialize_database, verify_shopping_list_schema
from src.database.migrations.create_shopping_list import run_migration

CORE_TABLES = ("producers", "regions", "wines", "bottles", "tastings", "sync_log", "declared_preferences")


def _create_legacy_database(path: Path) -> None:
    """Create a populated database representing the schema before this migration."""
    assert initialize_database(path)
    with get_db_connection(path) as conn:
        conn.execute("DROP TABLE shopping_list_items")
        conn.execute("INSERT INTO producers (name, country) VALUES ('Legacy Producer', 'France')")
        conn.execute("INSERT INTO regions (primary_name, country) VALUES ('Legacy Region', 'France')")
        conn.commit()


def _schema_signature(path: Path) -> tuple[object, ...]:
    """Return exact table, columns, foreign keys, and index definitions."""
    with sqlite3.connect(path) as conn:
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'shopping_list_items'"
        ).fetchone()[0]
        columns = tuple(conn.execute("PRAGMA table_info(shopping_list_items)").fetchall())
        foreign_keys = tuple(conn.execute("PRAGMA foreign_key_list(shopping_list_items)").fetchall())
        indexes = []
        for row in conn.execute("PRAGMA index_list(shopping_list_items)").fetchall():
            name = row[1]
            escaped_name = name.replace('"', '""')
            info = tuple(conn.execute(f'PRAGMA index_xinfo("{escaped_name}")').fetchall())
            sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)).fetchone()[0]
            indexes.append((name, row[2], row[4], info, " ".join(sql.casefold().split())))
        return " ".join(table_sql.casefold().split()), columns, foreign_keys, tuple(sorted(indexes))


def _core_counts(path: Path) -> dict[str, int]:
    """Return counts for the pre-existing cellar tables."""
    with sqlite3.connect(path) as conn:
        return {table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in CORE_TABLES}


def test_migration_creates_schema_and_preserves_core_rows(temp_dir: Path) -> None:
    """An existing cellar gains only the compatible Shopping List schema."""
    db_path = temp_dir / "legacy.db"
    _create_legacy_database(db_path)
    before_counts = _core_counts(db_path)

    run_migration(db_path)

    with sqlite3.connect(db_path) as conn:
        verify_shopping_list_schema(conn.cursor())
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert _core_counts(db_path) == before_counts


def test_migration_is_idempotent_and_matches_fresh_schema(temp_dir: Path) -> None:
    """Fresh and migrated databases receive the same schema."""
    fresh_path = temp_dir / "fresh.db"
    legacy_path = temp_dir / "legacy.db"
    assert initialize_database(fresh_path)
    _create_legacy_database(legacy_path)

    run_migration(fresh_path)
    run_migration(legacy_path)
    first_signature = _schema_signature(legacy_path)
    run_migration(legacy_path)

    assert _schema_signature(fresh_path) == first_signature
    assert _schema_signature(legacy_path) == first_signature


def test_incompatible_partial_schema_rolls_back_without_repair(temp_dir: Path) -> None:
    """A lookalike table is rejected without adding indexes or losing its row."""
    db_path = temp_dir / "partial.db"
    _create_legacy_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE shopping_list_items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("INSERT INTO shopping_list_items (id, name) VALUES (7, 'Legacy intent')")
        conn.commit()
        original_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'shopping_list_items'"
        ).fetchone()[0]

    with pytest.raises(ValueError, match="incompatible"):
        run_migration(db_path)

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'shopping_list_items'"
            ).fetchone()[0]
            == original_sql
        )
        assert conn.execute("PRAGMA index_list(shopping_list_items)").fetchall() == []
        assert conn.execute("SELECT id, name FROM shopping_list_items").fetchall() == [(7, "Legacy intent")]


def test_whole_database_backup_restores_shopping_list_identity(temp_dir: Path) -> None:
    """A whole-database backup restores catalogue IDs and Shopping List links together."""
    db_path = temp_dir / "cellar.db"
    backup_path = temp_dir / "cellar-backup.db"
    restored_path = temp_dir / "cellar-restored.db"
    assert initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        wine_id = conn.execute(
            "INSERT INTO wines (source, external_id, wine_name, wine_type) VALUES ('manual', 'backup-wine', 'Backup Wine', 'Red')"
        ).lastrowid
        conn.execute(
            "INSERT INTO shopping_list_items (wine_id, identity_kind, catalogue_identity_id, name) VALUES (?, 'catalogue', ?, 'Backup Wine')",
            (wine_id, wine_id),
        )
        conn.commit()

    shutil.copy2(db_path, backup_path)
    with get_db_connection(db_path) as conn:
        conn.execute("DELETE FROM wines WHERE id = ?", (wine_id,))
        conn.commit()
    shutil.copy2(backup_path, restored_path)

    with sqlite3.connect(restored_path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT wine_id, catalogue_identity_id, name FROM shopping_list_items").fetchone() == (
            wine_id,
            wine_id,
            "Backup Wine",
        )


def test_pre_migration_backup_restores_legacy_state_and_additive_schema_is_legacy_readable(temp_dir: Path) -> None:
    """A disposable rehearsal proves both pre-write restore and additive-table compatibility."""
    db_path = temp_dir / "legacy.db"
    backup_path = temp_dir / "legacy-backup.db"
    restored_path = temp_dir / "legacy-restored.db"
    _create_legacy_database(db_path)
    expected_counts = _core_counts(db_path)
    shutil.copy2(db_path, backup_path)

    run_migration(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO shopping_list_items (identity_kind, name, normalized_name) "
            "VALUES ('manual', 'Disposable Intent', 'disposable intent')"
        )
        connection.commit()

        # A pre-feature reader can continue querying its known tables while ignoring
        # the additive Shopping List table.
        legacy_rows = connection.execute("SELECT name, country FROM producers ORDER BY id").fetchall()
        assert legacy_rows == [("Legacy Producer", "France")]
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert _core_counts(db_path) == expected_counts

    shutil.copy2(backup_path, restored_path)
    with sqlite3.connect(restored_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'shopping_list_items'"
            ).fetchone()
            is None
        )
    assert _core_counts(restored_path) == expected_counts


def test_migration_requires_existing_explicit_database(temp_dir: Path) -> None:
    """A mistyped path cannot create a Shopping List-only database."""
    missing_path = temp_dir / "missing.db"
    with pytest.raises(FileNotFoundError):
        run_migration(missing_path)
    assert not missing_path.exists()
