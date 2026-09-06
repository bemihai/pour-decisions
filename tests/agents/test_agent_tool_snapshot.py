"""Tests for construction-time tool snapshots in WineAgent."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.prompt_registry import RenderedPrompt, sha256_text
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot


def _mock_llm() -> MagicMock:
    """Create the minimal language-model mock required by WineAgent."""
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()
    return llm


def _patch_prompt_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid prompt rendering in focused graph tests."""
    content = "Test system prompt."
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: RenderedPrompt(
            name="intelligent_agent_system",
            content=content,
            source_hash=sha256_text("Test source."),
            rendered_hash=sha256_text(content),
            label="",
        ),
    )


def test_agent_binds_exactly_one_registry_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent construction should retain and bind one immutable registry selection."""
    from src.agents.intelligent.agent import WineAgent
    from src.agents.tools.catalog import TOOL_DEFINITIONS

    _patch_prompt_loading(monkeypatch)
    snapshot = ToolSelectionSnapshot(
        definitions=TOOL_DEFINITIONS[:2],
        readiness=(),
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = snapshot
    llm = _mock_llm()

    agent = WineAgent(llm=llm, tool_registry=registry)

    assert agent.tool_selection_snapshot is snapshot
    assert agent.tools == [definition.tool for definition in snapshot.definitions]
    registry.select.assert_called_once_with(extended=True)
    llm.bind_tools.assert_called_once_with(agent.tools)


def test_empty_snapshot_builds_graph_without_tool_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-unavailable catalogue should still produce a valid agent graph."""
    from src.agents.intelligent.agent import WineAgent

    _patch_prompt_loading(monkeypatch)
    snapshot = ToolSelectionSnapshot(definitions=(), readiness=())
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = snapshot
    llm = _mock_llm()
    llm.bind_tools.return_value.invoke.return_value = AIMessage(content="No tools are available.")

    agent = WineAgent(llm=llm, tool_registry=registry)
    result = agent.invoke("What can you do?")

    assert agent.tools == []
    assert "agent" in agent.agent.get_graph().nodes
    assert "tools" not in agent.agent.get_graph().nodes
    assert result["final_answer"] == "No tools are available."
