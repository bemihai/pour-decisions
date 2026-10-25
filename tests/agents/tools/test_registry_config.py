"""Tests for tool-registry readiness-cache configuration."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.agents.tools.catalog import build_tool_registry


def test_checked_in_registry_config_uses_reviewed_ttl() -> None:
    """Production configuration should retain the reviewed readiness TTL."""
    config_path = Path(__file__).resolve().parents[3] / "app_config.yml"

    registry = build_tool_registry(OmegaConf.load(config_path))

    assert registry.health_check_ttl_seconds == 60
    assert registry.web_search_tools_enabled is False


def test_registry_accepts_explicit_valid_config() -> None:
    """Construction should retain an explicit valid cache TTL."""
    config = OmegaConf.create({"agents": {"tool_registry": {"health_check_ttl_seconds": 15}}})

    registry = build_tool_registry(config)

    assert registry.health_check_ttl_seconds == 15
    assert registry.web_search_tools_enabled is False


def test_missing_registry_section_uses_default_ttl() -> None:
    """A missing registry section should use the reviewed cache TTL."""
    registry = build_tool_registry(OmegaConf.create({}))

    assert registry.health_check_ttl_seconds == 60
    assert registry.web_search_tools_enabled is False


@pytest.mark.parametrize("ttl_seconds", [0, -1, 1.5, "60", True, None])
def test_registry_rejects_invalid_health_check_ttl(ttl_seconds: object) -> None:
    """TTL values must be real integers of at least one second."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "health_check_ttl_seconds": ttl_seconds,
                }
            }
        }
    )

    with pytest.raises(ValueError, match="must be an integer of at least 1"):
        build_tool_registry(config)


@pytest.mark.parametrize("enabled", [0, 1, "true", None])
def test_registry_rejects_invalid_web_search_tool_switch(enabled: object) -> None:
    """The metered web-tool rollout setting must be an explicit boolean."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "health_check_ttl_seconds": 60,
                    "enable_web_search_tools": enabled,
                }
            }
        }
    )

    with pytest.raises(ValueError, match="enable_web_search_tools must be a boolean"):
        build_tool_registry(config)


def test_registry_accepts_enabled_web_search_tools() -> None:
    """The explicit switch should make metered web tools eligible for readiness checks."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "health_check_ttl_seconds": 60,
                    "enable_web_search_tools": True,
                }
            }
        }
    )

    assert build_tool_registry(config).web_search_tools_enabled is True
