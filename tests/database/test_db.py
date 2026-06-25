"""Tests for lazy database path resolution in src.database.db."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


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
