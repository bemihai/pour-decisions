"""Migration parity and rollback checks using disposable databases."""

import sqlite3
from pathlib import Path

import pytest

from src.database.migrations.create_chat_sessions import SCHEMA, run_migration


def schema(path: Path) -> dict[str, str]:
    """Read only application-owned schema definitions."""
    with sqlite3.connect(path) as conn:
        return {name: sql for name, sql in conn.execute("SELECT name, sql FROM sqlite_master") if name in SCHEMA}


def test_fresh_and_legacy_migration_are_identical_and_repeatable(tmp_path: Path) -> None:
    """Migration preserves legacy content and SQLite's unrelated version metadata."""
    fresh, legacy = tmp_path / "fresh.db", tmp_path / "legacy.db"
    fresh.touch()
    with sqlite3.connect(legacy) as conn:
        conn.execute("CREATE TABLE conversation_threads (thread_id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO conversation_threads VALUES ('preserved')")
        conn.execute("PRAGMA user_version = 17")
    run_migration(fresh)
    run_migration(legacy)
    run_migration(legacy)
    assert schema(fresh) == schema(legacy)
    with sqlite3.connect(legacy) as conn:
        assert conn.execute("SELECT * FROM conversation_threads").fetchall() == [("preserved",)]
        assert conn.execute("PRAGMA user_version").fetchone() == (17,)
        assert conn.execute("SELECT * FROM chat_schema_version").fetchall() == [(1, 1)]


@pytest.mark.parametrize(
    "partial",
    ["CREATE TABLE chat_sessions (id TEXT)", "CREATE TABLE chat_schema_version (id INTEGER, version INTEGER)"],
)
def test_incompatible_partial_schema_is_not_repaired(tmp_path: Path, partial: str) -> None:
    """An unknown partial schema fails and no other application objects appear."""
    path = tmp_path / "partial.db"
    with sqlite3.connect(path) as conn:
        conn.execute(partial)
    before = schema(path)
    with pytest.raises((ValueError, sqlite3.OperationalError)):
        run_migration(path)
    assert schema(path) == before


def test_schema_constraints_enforce_ownership_and_state(tmp_path: Path) -> None:
    """Invalid references, versions and incomplete tombstones cannot be stored."""
    path = tmp_path / "constraints.db"
    path.touch()
    run_migration(path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO chat_sessions (id, state) VALUES ('bad', 'deleted')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO chat_turns VALUES ('missing', 1, 'input', 'display', NULL, '[]', '[]', 'running', 'now', 'now')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE chat_schema_version SET version = 2")


def test_migration_requires_existing_explicit_target(tmp_path: Path) -> None:
    """A mistyped path cannot silently create a second conversation database."""
    with pytest.raises(FileNotFoundError):
        run_migration(tmp_path / "missing.db")
