"""Contract and information-safety tests for the tools introspection API."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import (
    ToolDefinition,
    ToolReadiness,
    ToolRegistry,
    ToolSelectionSnapshot,
)
from src.api.schemas.tools import ToolsResponse


@pytest.fixture()
def client() -> TestClient:
    """Create a client with explicit empty tool-registry state."""
    from src.api.main import app

    app.state.tool_registry = None
    app.state.cloud_intelligent_agent = None
    return TestClient(app)


def _registry(enabled: bool) -> ToolRegistry:
    """Create a registry with deterministic rollout configuration."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "enabled": enabled,
                    "health_check_ttl_seconds": 60,
                }
            }
        }
    )
    return ToolRegistry(TOOL_DEFINITIONS, config=config)


def _ready_catalogue() -> tuple[ToolReadiness, ...]:
    """Return all catalogue tools as available in stable order."""
    return tuple(
        ToolReadiness(name=definition.metadata.name, available=True)
        for definition in TOOL_DEFINITIONS
    )


def _install_tool_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    *,
    registry_enabled: bool,
    readiness: tuple[ToolReadiness, ...],
    selected_definitions: tuple[ToolDefinition, ...] | None,
) -> tuple[ToolRegistry, MagicMock]:
    """Install deterministic registry and cloud-agent state for one request."""
    registry = _registry(registry_enabled)
    readiness_check = MagicMock(return_value=readiness)
    monkeypatch.setattr(registry, "check_readiness", readiness_check)
    client.app.state.tool_registry = registry
    if selected_definitions is None:
        client.app.state.cloud_intelligent_agent = None
    else:
        snapshot = ToolSelectionSnapshot(
            definitions=selected_definitions,
            readiness=(),
            registry_enabled=registry_enabled,
        )
        client.app.state.cloud_intelligent_agent = SimpleNamespace(
            tool_selection_snapshot=snapshot
        )
    return registry, readiness_check


def test_tools_endpoint_returns_complete_ordered_public_contract(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful response should match the schema and catalogue order exactly."""
    _, readiness_check = _install_tool_state(
        client,
        monkeypatch,
        registry_enabled=True,
        readiness=_ready_catalogue(),
        selected_definitions=TOOL_DEFINITIONS,
    )

    response = client.get("/api/tools")

    assert response.status_code == 200
    parsed = ToolsResponse.model_validate(response.json())
    assert parsed.total == 18
    assert parsed.available == 18
    assert parsed.unavailable == 0
    assert parsed.selected == 18
    assert parsed.registry_enabled is True
    assert [tool.name for tool in parsed.tools] == [
        definition.metadata.name for definition in TOOL_DEFINITIONS
    ]
    assert all(tool.reason_code is None for tool in parsed.tools)
    assert all(tool.unavailable_reason is None for tool in parsed.tools)
    readiness_check.assert_called_once_with()


def test_selected_tools_come_from_startup_snapshot_not_current_readiness(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A readiness refresh must not imply that the existing agent was rebound."""
    first_name = TOOL_DEFINITIONS[0].metadata.name
    readiness = (
        ToolReadiness(
            name=first_name,
            available=False,
            reason_code="dependency_unreachable",
            reason="Internal dependency changed after startup.",
        ),
        *(
            ToolReadiness(name=definition.metadata.name, available=True)
            for definition in TOOL_DEFINITIONS[1:]
        ),
    )
    _install_tool_state(
        client,
        monkeypatch,
        registry_enabled=True,
        readiness=readiness,
        selected_definitions=TOOL_DEFINITIONS[:2],
    )

    parsed = ToolsResponse.model_validate(client.get("/api/tools").json())

    assert parsed.available == 17
    assert parsed.unavailable == 1
    assert parsed.selected == 2
    assert parsed.tools[0].available is False
    assert parsed.tools[0].selected_for_agent is True
    assert parsed.tools[2].available is True
    assert parsed.tools[2].selected_for_agent is False


def test_disabled_registry_reports_readiness_and_static_agent_selection(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback mode should report live readiness without changing its static snapshot."""
    unavailable_name = TOOL_DEFINITIONS[-1].metadata.name
    readiness = (
        *(
            ToolReadiness(name=definition.metadata.name, available=True)
            for definition in TOOL_DEFINITIONS[:-1]
        ),
        ToolReadiness(
            name=unavailable_name,
            available=False,
            reason_code="missing_configuration",
            reason="Provider key is missing.",
        ),
    )
    _install_tool_state(
        client,
        monkeypatch,
        registry_enabled=False,
        readiness=readiness,
        selected_definitions=TOOL_DEFINITIONS,
    )

    parsed = ToolsResponse.model_validate(client.get("/api/tools").json())

    assert parsed.registry_enabled is False
    assert parsed.available == 17
    assert parsed.selected == 18
    assert [tool.name for tool in parsed.tools] == [
        definition.metadata.name for definition in TOOL_DEFINITIONS
    ]
    assert all(tool.selected_for_agent for tool in parsed.tools)
    assert parsed.tools[-1].available is False
    assert parsed.tools[-1].selected_for_agent is True
    assert parsed.tools[-1].unavailable_reason == "Web search is not configured."


def test_missing_cloud_agent_marks_every_tool_unselected(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing default agent should not manufacture startup selection state."""
    _install_tool_state(
        client,
        monkeypatch,
        registry_enabled=True,
        readiness=_ready_catalogue(),
        selected_definitions=None,
    )

    parsed = ToolsResponse.model_validate(client.get("/api/tools").json())

    assert parsed.selected == 0
    assert all(tool.selected_for_agent is False for tool in parsed.tools)


def test_missing_registry_state_returns_controlled_503(client: TestClient) -> None:
    """The endpoint should fail explicitly when startup registry state is absent."""
    response = client.get("/api/tools")

    assert response.status_code == 503
    assert response.json() == {"detail": "Tool registry is unavailable."}


def test_unavailable_tool_response_redacts_internal_reason_details(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw readiness reasons and unknown codes must not cross the API boundary."""
    internal_detail = "http://chroma.internal:8000 /private/cellar.db SECRET_VALUE"
    readiness = (
        ToolReadiness(
            name=TOOL_DEFINITIONS[0].metadata.name,
            available=False,
            reason_code=internal_detail,
            reason=internal_detail,
        ),
        *_ready_catalogue()[1:],
    )
    _install_tool_state(
        client,
        monkeypatch,
        registry_enabled=True,
        readiness=readiness,
        selected_definitions=(),
    )

    response = client.get("/api/tools")
    response_text = response.text
    first_tool = response.json()["tools"][0]

    assert response.status_code == 200
    assert first_tool["reason_code"] == "readiness_check_failed"
    assert first_tool["unavailable_reason"] == "Wine cellar service is unavailable."
    assert "chroma.internal" not in response_text
    assert "/private/cellar.db" not in response_text
    assert "SECRET_VALUE" not in response_text


def test_unexpected_registry_failure_returns_safe_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected registry errors should be logged but never returned verbatim."""
    registry = _registry(True)
    client.app.state.tool_registry = registry
    client.app.state.cloud_intelligent_agent = None
    internal_detail = "http://private-host:8100 /secret/path API_KEY_VALUE"
    monkeypatch.setattr(
        registry,
        "check_readiness",
        MagicMock(side_effect=RuntimeError(internal_detail)),
    )

    response = client.get("/api/tools")

    assert response.status_code == 503
    assert response.json() == {"detail": "Tool registry status is unavailable."}
    assert "private-host" not in response.text
    assert "/secret/path" not in response.text
    assert "API_KEY_VALUE" not in response.text


def test_openapi_includes_tools_endpoint(client: TestClient) -> None:
    """The registered public contract should appear in generated OpenAPI."""
    schema = client.get("/openapi.json").json()

    assert "/api/tools" in schema["paths"]
    operation = schema["paths"]["/api/tools"]["get"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema == {"$ref": "#/components/schemas/ToolsResponse"}
