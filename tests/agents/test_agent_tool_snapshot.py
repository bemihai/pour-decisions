"""Tests for construction-time tool snapshots in WineAgent."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot


def _mock_llm() -> MagicMock:
    """Create the minimal language-model mock required by WineAgent."""
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()
    return llm


def _patch_prompt_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid prompt rendering in focused graph tests."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )


def test_agent_binds_exactly_one_registry_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent construction should retain and bind one immutable registry selection."""
    from src.agents.intelligent.agent import WineAgent
    from src.agents.tools.catalog import TOOL_DEFINITIONS

    _patch_prompt_loading(monkeypatch)
    snapshot = ToolSelectionSnapshot(
        definitions=TOOL_DEFINITIONS[:2],
        readiness=(),
        registry_enabled=True,
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.registry_enabled = True
    registry.select.return_value = snapshot
    llm = _mock_llm()

    agent = WineAgent(llm=llm, tool_registry=registry)

    assert agent.tool_selection_snapshot is snapshot
    assert agent.tools == [definition.tool for definition in snapshot.definitions]
    registry.select.assert_called_once_with(extended=True, available_only=True)
    llm.bind_tools.assert_called_once_with(agent.tools)


def test_disabled_registry_captures_static_eighteen_tool_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled mode should keep the complete static catalogue and graph."""
    from src.agents.intelligent.agent import WineAgent
    from src.agents.tools.catalog import TOOL_DEFINITIONS

    _patch_prompt_loading(monkeypatch)
    registry = ToolRegistry(TOOL_DEFINITIONS)

    agent = WineAgent(llm=_mock_llm(), tool_registry=registry)

    assert agent.tool_selection_snapshot.registry_enabled is False
    assert tuple(tool.name for tool in agent.tools) == tuple(
        definition.metadata.name for definition in TOOL_DEFINITIONS
    )
    assert len(agent.tools) == 18
    assert "tools" in agent.agent.get_graph().nodes


def test_empty_enabled_snapshot_builds_graph_without_tool_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-unavailable catalogue should still produce a valid agent graph."""
    from src.agents.intelligent.agent import WineAgent

    _patch_prompt_loading(monkeypatch)
    snapshot = ToolSelectionSnapshot(definitions=(), readiness=(), registry_enabled=True)
    registry = MagicMock(spec=ToolRegistry)
    registry.registry_enabled = True
    registry.select.return_value = snapshot
    llm = _mock_llm()
    llm.bind_tools.return_value.invoke.return_value = AIMessage(content="No tools are available.")

    agent = WineAgent(llm=llm, tool_registry=registry)
    result = agent.invoke("What can you do?")

    assert agent.tools == []
    assert "agent" in agent.agent.get_graph().nodes
    assert "tools" not in agent.agent.get_graph().nodes
    assert result["final_answer"] == "No tools are available."
