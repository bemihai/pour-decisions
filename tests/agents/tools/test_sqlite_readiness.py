"""Tests for read-only SQLite prerequisite readiness."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.tools.registry import ToolPrerequisite, ToolRegistry


CELLAR_TABLES = ("wines", "bottles", "producers", "regions", "tastings")


def _create_database(path: Path, tables: tuple[str, ...]) -> None:
    """Create empty test tables in a temporary SQLite database."""
    with sqlite3.connect(path) as connection:
        for table in tables:
            connection.execute(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY)')


def _registry_for_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> ToolRegistry:
    """Build a registry whose database resolver returns the supplied path."""
    monkeypatch.setattr("src.agents.tools.registry.get_default_db_path", lambda: path)
    return ToolRegistry(())


@pytest.mark.parametrize(
    ("prerequisite", "tables"),
    [
        (ToolPrerequisite.CELLAR_SCHEMA, CELLAR_TABLES),
        (ToolPrerequisite.PAIRING_RULES, ("food_pairing_rules",)),
    ],
)
def test_complete_empty_sqlite_schema_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prerequisite: ToolPrerequisite,
    tables: tuple[str, ...],
) -> None:
    """Required tables should be sufficient even when they contain no rows."""
    database_path = tmp_path / "cellar.db"
    _create_database(database_path, tables)

    result = _registry_for_path(monkeypatch, database_path)._check_sqlite_schema(prerequisite)

    assert result.prerequisite == prerequisite
    assert result.available is True
    assert result.reason_code is None


def test_missing_database_is_unavailable_without_creating_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing cellar database should remain absent after the probe."""
    database_path = tmp_path / "missing.db"

    result = _registry_for_path(monkeypatch, database_path)._check_sqlite_schema(
        ToolPrerequisite.CELLAR_SCHEMA
    )

    assert result.available is False
    assert result.reason_code == "database_missing"
    assert result.reason == "Cellar database is missing."
    assert not database_path.exists()


@pytest.mark.parametrize(
    "prerequisite",
    [ToolPrerequisite.CELLAR_SCHEMA, ToolPrerequisite.PAIRING_RULES],
)
def test_incomplete_sqlite_schema_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prerequisite: ToolPrerequisite,
) -> None:
    """Missing required tables should produce a stable schema reason."""
    database_path = tmp_path / "incomplete.db"
    _create_database(database_path, ("unrelated",))

    result = _registry_for_path(monkeypatch, database_path)._check_sqlite_schema(prerequisite)

    assert result.available is False
    assert result.reason_code == "database_schema_incomplete"
    assert result.reason == "Cellar database schema is incomplete."


def test_sqlite_probe_opens_database_in_read_only_uri_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The readiness connection should explicitly use SQLite read-only mode."""
    database_path = tmp_path / "cellar.db"
    _create_database(database_path, CELLAR_TABLES)
    original_connect = sqlite3.connect
    connect_spy = MagicMock(wraps=original_connect)
    monkeypatch.setattr("src.agents.tools.registry.sqlite3.connect", connect_spy)

    result = _registry_for_path(monkeypatch, database_path)._check_sqlite_schema(
        ToolPrerequisite.CELLAR_SCHEMA
    )

    assert result.available is True
    database_uri = connect_spy.call_args.args[0]
    assert database_uri.endswith("?mode=ro")
    assert connect_spy.call_args.kwargs == {"uri": True}


def test_sqlite_probe_rejects_non_sqlite_prerequisite() -> None:
    """The shared SQLite helper should reject unrelated prerequisites clearly."""
    with pytest.raises(ValueError, match="Unsupported SQLite prerequisite"):
        ToolRegistry(())._check_sqlite_schema(ToolPrerequisite.WEB_SEARCH_CONFIG)


def test_sqlite_probe_contains_unexpected_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected SQLite failures should become safe unavailable evidence."""
    database_path = tmp_path / "cellar.db"
    _create_database(database_path, CELLAR_TABLES)
    monkeypatch.setattr(
        "src.agents.tools.registry.sqlite3.connect",
        MagicMock(side_effect=sqlite3.DatabaseError("database internals")),
    )

    result = _registry_for_path(monkeypatch, database_path)._check_sqlite_schema(
        ToolPrerequisite.CELLAR_SCHEMA
    )

    assert result.available is False
    assert result.reason_code == "readiness_check_failed"
    assert result.reason == "Cellar database readiness check failed."
    assert "Unexpected failure while checking cellar database schema" in caplog.text
    assert "database internals" not in result.reason
