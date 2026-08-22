"""Tests for tool-registry rollout and rollback configuration."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.agents.tools.catalog import build_tool_registry


def test_checked_in_registry_config_is_enabled() -> None:
    """Production configuration should enable the registry with the reviewed TTL."""
    config_path = Path(__file__).resolve().parents[3] / "app_config.yml"

    registry = build_tool_registry(OmegaConf.load(config_path))

    assert registry.registry_enabled is True
    assert registry.health_check_ttl_seconds == 60


def test_registry_accepts_explicit_valid_config() -> None:
    """Construction should retain valid settings for later migration phases."""
    config = OmegaConf.create(
        {"agents": {"tool_registry": {"enabled": True, "health_check_ttl_seconds": 15}}}
    )

    registry = build_tool_registry(config)

    assert registry.registry_enabled is True
    assert registry.health_check_ttl_seconds == 15


def test_missing_registry_section_preserves_disabled_defaults() -> None:
    """Older configurations should remain on static behavior during migration."""
    registry = build_tool_registry(OmegaConf.create({}))

    assert registry.registry_enabled is False
    assert registry.health_check_ttl_seconds == 60


@pytest.mark.parametrize("ttl_seconds", [0, -1, 1.5, "60", True, None])
def test_registry_rejects_invalid_health_check_ttl(ttl_seconds: object) -> None:
    """TTL values must be real integers of at least one second."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "enabled": False,
                    "health_check_ttl_seconds": ttl_seconds,
                }
            }
        }
    )

    with pytest.raises(ValueError, match="must be an integer of at least 1"):
        build_tool_registry(config)


@pytest.mark.parametrize("enabled", [0, 1, "false", None])
def test_registry_rejects_non_boolean_enabled_flag(enabled: object) -> None:
    """Ambiguous migration flags should fail instead of silently enabling behavior."""
    config = OmegaConf.create(
        {"agents": {"tool_registry": {"enabled": enabled, "health_check_ttl_seconds": 60}}}
    )

    with pytest.raises(ValueError, match="enabled must be a boolean"):
        build_tool_registry(config)
