"""Tests for construction-time tool snapshots in WineAgent."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.prompt_registry import PromptRegistry, RenderedPrompt, get_prompt_registry, sha256_text
from src.agents.provenance import build_tool_contract_provenance
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

    assert agent.tool_selection_snapshot.readiness is snapshot.readiness
    assert [definition.metadata for definition in agent.tool_selection_snapshot.definitions] == [
        definition.metadata for definition in snapshot.definitions
    ]
    assert agent.tools == [
        definition.tool for definition in agent.tool_selection_snapshot.definitions
    ]
    assert [tool.description for tool in agent.tools] == [
        definition.metadata.capability for definition in snapshot.definitions
    ]
    assert [definition.tool.description for definition in snapshot.definitions] != [
        definition.metadata.capability for definition in snapshot.definitions
    ]
    registry.select.assert_called_once_with(extended=True)
    llm.bind_tools.assert_called_once_with(agent.tools)
    assert agent.execution_provenance.tools == build_tool_contract_provenance(
        agent.tool_selection_snapshot
    )


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


def test_explicit_prompt_registry_changes_only_prompt_identity() -> None:
    """A paired eval can inject an earlier source without changing tool selection."""
    from src.agents.intelligent.agent import WineAgent

    current = get_prompt_registry()
    records = {name: current.get(name) for name in current.get_source_version_map()}
    prior_source = "You are a wine assistant. Only use available tools."
    records["intelligent_agent_system"] = records["intelligent_agent_system"].model_copy(
        update={"source": prior_source, "source_hash": sha256_text(prior_source), "label": "prior"}
    )
    snapshot = ToolSelectionSnapshot(definitions=(), readiness=())
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = snapshot

    agent = WineAgent(
        llm=_mock_llm(),
        tool_registry=registry,
        prompt_registry=PromptRegistry(records),
    )

    assert agent.system_prompt == prior_source
    assert agent.rendered_system_prompt.source_hash == sha256_text(prior_source)
    assert agent.execution_provenance.prompts[0].source_hash == sha256_text(prior_source)
    assert agent.execution_provenance.tools is not None
    assert agent.execution_provenance.tools.selected_names == ()
