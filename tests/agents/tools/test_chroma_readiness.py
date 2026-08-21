"""Tests for lightweight Chroma collection readiness."""

from unittest.mock import MagicMock

import pytest
from chromadb.errors import NotFoundError
from omegaconf import OmegaConf

from src.agents.tools.registry import ToolPrerequisite, ToolRegistry


def _registry(config: dict[str, object] | None = None) -> ToolRegistry:
    """Build an empty registry with isolated Chroma configuration."""
    return ToolRegistry((), config=OmegaConf.create(config) if config is not None else None)


def _chroma_config(
    *,
    host: object = "chroma.local",
    port: object = 8100,
    collection_name: object = "wine_books",
) -> dict[str, object]:
    """Build the Chroma portion of application configuration."""
    return {
        "chroma": {
            "client": {"host": host, "port": port},
            "collections": [{"name": collection_name}, {"name": "unused"}],
        }
    }


def test_chroma_collection_is_ready_when_configured_collection_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe should reuse the client initializer and check the first collection."""
    client = MagicMock()
    initializer = MagicMock(return_value=client)
    monkeypatch.setattr("src.agents.tools.registry.initialize_chroma_client", initializer)

    result = _registry(_chroma_config())._check_chroma_collection()

    assert result.prerequisite == ToolPrerequisite.CHROMA_COLLECTION
    assert result.available is True
    assert result.reason_code is None
    initializer.assert_called_once_with("chroma.local", 8100)
    client.get_collection.assert_called_once_with("wine_books")


@pytest.mark.parametrize(
    ("config", "expected_reason"),
    [
        (None, "Chroma configuration is missing."),
        ({}, "Chroma host is not configured."),
        (_chroma_config(host=""), "Chroma host is not configured."),
        (_chroma_config(port=0), "Chroma port is not configured."),
        (_chroma_config(port="8100"), "Chroma port is not configured."),
        (_chroma_config(collection_name=""), "Chroma collection is not configured."),
    ],
)
def test_chroma_probe_reports_missing_configuration(
    config: dict[str, object] | None,
    expected_reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Incomplete local settings should return safe missing-configuration evidence."""
    result = _registry(config)._check_chroma_collection()

    assert result.available is False
    assert result.reason_code == "missing_configuration"
    assert result.reason == expected_reason
    assert not caplog.records


def test_chroma_connection_failure_is_dependency_unreachable(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Client initialization failures should be treated as an unavailable dependency."""
    monkeypatch.setattr(
        "src.agents.tools.registry.initialize_chroma_client",
        MagicMock(side_effect=ConnectionError("private endpoint details")),
    )

    result = _registry(_chroma_config())._check_chroma_collection()

    assert result.available is False
    assert result.reason_code == "dependency_unreachable"
    assert result.reason == "Chroma service is unavailable."
    assert "private endpoint details" not in result.reason
    assert not caplog.records


def test_missing_chroma_collection_has_stable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable service without the required collection should be distinguished."""
    client = MagicMock()
    client.get_collection.side_effect = NotFoundError("missing")
    monkeypatch.setattr(
        "src.agents.tools.registry.initialize_chroma_client",
        MagicMock(return_value=client),
    )

    result = _registry(_chroma_config())._check_chroma_collection()

    assert result.available is False
    assert result.reason_code == "collection_missing"
    assert result.reason == "Required Chroma collection is missing."


def test_chroma_collection_probe_contains_unexpected_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected collection errors should become safe unavailable evidence."""
    client = MagicMock()
    client.get_collection.side_effect = RuntimeError("collection internals")
    monkeypatch.setattr(
        "src.agents.tools.registry.initialize_chroma_client",
        MagicMock(return_value=client),
    )

    result = _registry(_chroma_config())._check_chroma_collection()

    assert result.available is False
    assert result.reason_code == "readiness_check_failed"
    assert result.reason == "Chroma readiness check failed."
    assert "Unexpected failure while checking Chroma collection" in caplog.text
    assert "collection internals" not in result.reason
