"""Phase 5 degraded-mode rollout scenarios for the dynamic tool registry."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from omegaconf import OmegaConf

from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import (
    ToolPrerequisite,
    ToolRegistry,
    _PrerequisiteReadiness,
)
from src.api.schemas.tools import ToolsResponse


@dataclass(frozen=True)
class DegradedScenario:
    """One deterministic prerequisite state and its expected selection."""

    name: str
    unavailable: dict[ToolPrerequisite, str]
    expected_selected: int


SCENARIOS = (
    DegradedScenario("all_ready", {}, 18),
    DegradedScenario(
        "tavily_configuration_missing",
        {ToolPrerequisite.WEB_SEARCH_CONFIG: "missing_configuration"},
        15,
    ),
    DegradedScenario(
        "chroma_unreachable",
        {ToolPrerequisite.CHROMA_COLLECTION: "dependency_unreachable"},
        13,
    ),
    DegradedScenario(
        "chroma_collection_missing",
        {ToolPrerequisite.CHROMA_COLLECTION: "collection_missing"},
        13,
    ),
    DegradedScenario(
        "cellar_database_missing",
        {ToolPrerequisite.CELLAR_SCHEMA: "database_missing"},
        8,
    ),
    DegradedScenario("valid_empty_cellar_schema", {}, 18),
    DegradedScenario(
        "pairing_rules_table_missing",
        {ToolPrerequisite.PAIRING_RULES: "database_schema_incomplete"},
        17,
    ),
    DegradedScenario(
        "all_prerequisites_unavailable",
        {
            ToolPrerequisite.CELLAR_SCHEMA: "database_missing",
            ToolPrerequisite.PAIRING_RULES: "database_schema_incomplete",
            ToolPrerequisite.CHROMA_COLLECTION: "dependency_unreachable",
            ToolPrerequisite.WEB_SEARCH_CONFIG: "missing_configuration",
        },
        0,
    ),
)


def _enabled_registry(*, ttl_seconds: int = 60) -> ToolRegistry:
    """Build the production candidate with deterministic rollout settings."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "enabled": True,
                    "health_check_ttl_seconds": ttl_seconds,
                }
            }
        }
    )
    return ToolRegistry(TOOL_DEFINITIONS, config=config)


def _readiness(
    prerequisite: ToolPrerequisite,
    unavailable: dict[ToolPrerequisite, str],
) -> _PrerequisiteReadiness:
    """Return safe deterministic evidence for one scenario prerequisite."""
    reason_code = unavailable.get(prerequisite)
    return _PrerequisiteReadiness(
        prerequisite=prerequisite,
        available=reason_code is None,
        reason_code=reason_code,
        reason=None if reason_code is None else "Dependency unavailable for rollout test.",
    )


def _mock_llm(answer: str) -> MagicMock:
    """Create a no-network model that responds without requesting a tool."""
    llm = MagicMock()
    bound_model = MagicMock()
    bound_model.invoke.return_value = AIMessage(content=answer)
    llm.bind_tools.return_value = bound_model
    return llm


@pytest.fixture()
def client() -> TestClient:
    """Create a tools-endpoint client without running application startup."""
    from src.api.main import app

    previous_registry = getattr(app.state, "tool_registry", None)
    previous_cloud_agent = getattr(app.state, "cloud_intelligent_agent", None)
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.state.tool_registry = previous_registry
        app.state.cloud_intelligent_agent = previous_cloud_agent


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_enabled_degraded_scenario_keeps_selection_prompt_endpoint_and_chat_consistent(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    scenario: DegradedScenario,
) -> None:
    """Every reviewed dependency state should initialize and answer consistently."""
    from src.agents.intelligent.agent import WineAgent

    registry = _enabled_registry()
    readiness = MagicMock(
        side_effect=lambda prerequisite, **_kwargs: _readiness(
            prerequisite,
            scenario.unavailable,
        )
    )
    monkeypatch.setattr(registry, "_get_prerequisite_readiness", readiness)
    llm = _mock_llm(f"Graceful response for {scenario.name}.")
    agent = WineAgent(llm=llm, tool_registry=registry)
    selected_names = tuple(tool.name for tool in agent.tools)
    unavailable_names = {
        definition.metadata.name
        for definition in TOOL_DEFINITIONS
        if any(
            prerequisite in scenario.unavailable
            for prerequisite in definition.metadata.prerequisites
        )
    }

    assert len(selected_names) == scenario.expected_selected
    assert unavailable_names.isdisjoint(selected_names)
    assert all(name in agent.system_prompt for name in selected_names)
    assert all(name not in agent.system_prompt for name in unavailable_names)
    llm.bind_tools.assert_called_once_with(agent.tools)

    client.app.state.tool_registry = registry
    client.app.state.cloud_intelligent_agent = agent
    response = client.get("/api/tools")
    endpoint = ToolsResponse.model_validate(response.json())
    endpoint_selected = tuple(tool.name for tool in endpoint.tools if tool.selected_for_agent)

    assert response.status_code == 200
    assert endpoint.registry_enabled is True
    assert endpoint.total == 18
    assert endpoint.selected == scenario.expected_selected
    assert endpoint.available == scenario.expected_selected
    assert endpoint_selected == selected_names

    result = agent.invoke("Respond gracefully using only currently available capabilities.")

    assert result["final_answer"] == f"Graceful response for {scenario.name}."
    assert result["tools_used"] == []


def test_recovered_dependency_refreshes_endpoint_but_not_agent_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TTL recovery should update readiness without mutating a constructed agent."""
    from src.agents.intelligent.agent import WineAgent

    registry = _enabled_registry(ttl_seconds=10)
    clock = [100.0]
    web_probe_count = 0

    def probe(prerequisite: ToolPrerequisite) -> _PrerequisiteReadiness:
        """Recover web-search configuration on its second probe."""
        nonlocal web_probe_count
        if prerequisite == ToolPrerequisite.WEB_SEARCH_CONFIG:
            web_probe_count += 1
            if web_probe_count == 1:
                return _readiness(
                    prerequisite,
                    {prerequisite: "missing_configuration"},
                )
        return _readiness(prerequisite, {})

    monkeypatch.setattr("src.agents.tools.registry.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(registry, "_probe_prerequisite", probe)
    llm = _mock_llm("Graceful response after dependency recovery.")
    agent = WineAgent(llm=llm, tool_registry=registry)
    startup_names = tuple(tool.name for tool in agent.tools)
    web_names = {
        definition.metadata.name
        for definition in TOOL_DEFINITIONS
        if ToolPrerequisite.WEB_SEARCH_CONFIG in definition.metadata.prerequisites
    }

    assert len(startup_names) == 15
    assert web_names.isdisjoint(startup_names)
    assert all(name not in agent.system_prompt for name in web_names)

    clock[0] = 110.0
    client.app.state.tool_registry = registry
    client.app.state.cloud_intelligent_agent = agent
    response = client.get("/api/tools")
    endpoint = ToolsResponse.model_validate(response.json())

    assert response.status_code == 200
    assert endpoint.available == 18
    assert endpoint.selected == 15
    assert web_probe_count == 2
    assert all(tool.available for tool in endpoint.tools)
    assert all(
        not tool.selected_for_agent
        for tool in endpoint.tools
        if tool.name in web_names
    )
    assert tuple(tool.name for tool in agent.tools) == startup_names

    result = agent.invoke("Can you still answer after readiness changes?")

    assert result["final_answer"] == "Graceful response after dependency recovery."
    assert result["tools_used"] == []
