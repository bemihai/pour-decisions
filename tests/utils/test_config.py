"""Tests for cached config loading helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.utils.utils import clear_config_cache, get_config


def test_get_config_caches_loaded_config() -> None:
    """Repeated get_config() calls should reuse the same loaded object."""
    clear_config_cache()
    fake_cfg = SimpleNamespace(cellar=SimpleNamespace(db_path="cellar-data/test.db"))

    with patch("src.utils.utils.OmegaConf.load", return_value=fake_cfg) as mock_load:
        first = get_config()
        second = get_config()

    assert first is second
    mock_load.assert_called_once()
    clear_config_cache()


def test_clear_config_cache_forces_reload() -> None:
    """clear_config_cache() should force the next get_config() call to reload."""
    clear_config_cache()
    first_cfg = SimpleNamespace(name="first")
    second_cfg = SimpleNamespace(name="second")

    with patch("src.utils.utils.OmegaConf.load", side_effect=[first_cfg, second_cfg]) as mock_load:
        first = get_config()
        clear_config_cache()
        second = get_config()

    assert first is first_cfg
    assert second is second_cfg
    assert mock_load.call_count == 2
    clear_config_cache()


def test_get_config_loads_project_app_config() -> None:
    """get_config() should still resolve app_config.yml from the project root."""
    clear_config_cache()
    fake_cfg = SimpleNamespace()

    with patch("src.utils.utils.find_project_root", return_value="/tmp/project"):
        with patch("src.utils.utils.OmegaConf.load", return_value=fake_cfg) as mock_load:
            get_config()

    mock_load.assert_called_once_with(Path("/tmp/project") / "app_config.yml")
    clear_config_cache()
