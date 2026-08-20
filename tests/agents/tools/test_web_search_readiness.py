"""Tests for zero-cost web-search configuration readiness."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.agents.tools.registry import ToolPrerequisite, ToolRegistry


def _registry(config: dict[str, object] | None = None) -> ToolRegistry:
    """Build an empty registry with isolated application configuration."""
    return ToolRegistry((), config=OmegaConf.create(config) if config is not None else None)


def _web_config(
    *,
    provider: object = "tavily",
    key_env: object = "TEST_TAVILY_API_KEY",
) -> dict[str, object]:
    """Build the web-search portion of application configuration."""
    return {
        "web_search": {
            "provider": provider,
            "tavily": {"api_key_env": key_env},
        }
    }


def test_web_search_config_is_ready_when_configured_key_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supported provider with a non-empty configured key should be ready."""
    monkeypatch.setenv("TEST_TAVILY_API_KEY", "configured")

    result = _registry(_web_config())._check_web_search_config()

    assert result.prerequisite == ToolPrerequisite.WEB_SEARCH_CONFIG
    assert result.available is True
    assert result.reason_code is None
    assert result.reason is None


@pytest.mark.parametrize(
    ("config", "expected_reason"),
    [
        (None, "Web search configuration is missing."),
        ({}, "Web search provider is not configured."),
        (_web_config(provider=""), "Web search provider is not configured."),
        (_web_config(provider="unsupported"), "Configured web search provider is not supported."),
        (_web_config(key_env=""), "Web search credential configuration is missing."),
    ],
)
def test_web_search_config_reports_safe_missing_configuration(
    config: dict[str, object] | None,
    expected_reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing and unsupported settings should produce stable safe evidence."""
    result = _registry(config)._check_web_search_config()

    assert result.available is False
    assert result.reason_code == "missing_configuration"
    assert result.reason == expected_reason
    assert not caplog.records


@pytest.mark.parametrize("environment_value", [None, "", "   "])
def test_web_search_config_rejects_missing_or_blank_environment_value(
    monkeypatch: pytest.MonkeyPatch,
    environment_value: str | None,
) -> None:
    """Absent and blank credentials should be treated as missing."""
    if environment_value is None:
        monkeypatch.delenv("TEST_TAVILY_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TEST_TAVILY_API_KEY", environment_value)

    result = _registry(_web_config())._check_web_search_config()

    assert result.available is False
    assert result.reason_code == "missing_configuration"
    assert result.reason == "Web search credentials are not configured."
    assert "TEST_TAVILY_API_KEY" not in result.reason


def test_web_search_probe_does_not_construct_provider_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness should inspect local state without initializing web search."""
    monkeypatch.setenv("TEST_TAVILY_API_KEY", "configured")
    provider_client = MagicMock(side_effect=AssertionError("provider client must not be constructed"))
    monkeypatch.setattr("src.services.web_search.WineWebSearchEngine", provider_client)

    result = _registry(_web_config())._check_web_search_config()

    assert result.available is True
    provider_client.assert_not_called()


def test_web_search_probe_contains_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected configuration errors should become safe unavailable evidence."""
    registry = _registry(_web_config())
    monkeypatch.setattr(
        "src.agents.tools.registry.OmegaConf.select",
        MagicMock(side_effect=RuntimeError("configuration exploded")),
    )

    result = registry._check_web_search_config()

    assert result.available is False
    assert result.reason_code == "readiness_check_failed"
    assert result.reason == "Web search readiness check failed."
    assert "Unexpected failure while checking web search configuration" in caplog.text
    assert "configuration exploded" not in result.reason
