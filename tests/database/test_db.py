"""Tests for lazy database path resolution in src.database.db."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.database import drop_all_tables, get_db_connection, initialize_database


def _reload_db_module():
    """Reload src.database.db so import-time behavior can be asserted."""
    sys.modules.pop("src.database.db", None)
    return importlib.import_module("src.database.db")


def test_importing_db_module_does_not_resolve_default_path(monkeypatch) -> None:
    """Importing src.database.db should not call get_default_db_path()."""
    import src.utils as utils_module

    calls: list[str] = []

    def _fail_if_called() -> Path:
        calls.append("called")
        raise AssertionError("get_default_db_path() should not run during module import")

    monkeypatch.setattr(utils_module, "get_default_db_path", _fail_if_called)

    module = _reload_db_module()

    assert module is not None
    assert calls == []


def test_get_db_connection_resolves_default_path_lazily(monkeypatch) -> None:
    """get_db_connection() should resolve the default path only when invoked."""
    import src.utils as utils_module

    calls: list[str] = []
    expected_path = Path("/tmp/test-lazy-default.db")

    def _fake_default_db_path() -> Path:
        calls.append("called")
        return expected_path

    monkeypatch.setattr(utils_module, "get_default_db_path", _fake_default_db_path)

    module = _reload_db_module()

    fake_connection = MagicMock()
    with patch.object(module.sqlite3, "connect", return_value=fake_connection) as mock_connect:
        with module.get_db_connection():
            pass

    assert calls == ["called"]
    mock_connect.assert_called_once_with(expected_path)
    fake_connection.execute.assert_called_once_with("PRAGMA foreign_keys = ON")
    fake_connection.close.assert_called_once()


def test_drop_all_tables_removes_shopping_list_before_wines(temp_dir: Path) -> None:
    """Full reset respects the Shopping List foreign-key dependency."""
    db_path = temp_dir / "reset.db"
    assert initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        wine_id = conn.execute(
            "INSERT INTO wines (source, external_id, wine_name, wine_type) VALUES ('manual', 'reset-wine', 'Reset Wine', 'Red')"
        ).lastrowid
        conn.execute(
            "INSERT INTO shopping_list_items (wine_id, identity_kind, catalogue_identity_id, name) VALUES (?, 'catalogue', ?, 'Reset Wine')",
            (wine_id, wine_id),
        )
        conn.commit()

    assert drop_all_tables(db_path)
    with get_db_connection(db_path) as conn:
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall()
            == []
        )
