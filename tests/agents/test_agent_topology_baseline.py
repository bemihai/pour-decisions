"""M9A Gate 0 baselines for the current intelligent-agent graph topology."""

from unittest.mock import MagicMock

import pytest

from src.agents.intelligent.agent import WineAgent
from src.agents.prompt_registry import RenderedPrompt, sha256_text
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot


GraphEdge = tuple[str, str, bool]


def _mock_llm(name: str) -> MagicMock:
    """Create the minimal language-model mock required for graph construction."""
    llm = MagicMock()
    llm.__class__.__name__ = name
    llm.bind_tools.return_value = MagicMock()
    return llm


def _build_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_tools: bool,
) -> WineAgent:
    """Build an agent from a deterministic construction-time tool snapshot."""
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

    definitions = TOOL_DEFINITIONS[:1] if with_tools else ()
    snapshot = ToolSelectionSnapshot(definitions=definitions, readiness=())
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = snapshot

    llm = _mock_llm("GenerationModel")
    return WineAgent(llm=llm, tool_registry=registry)


def _graph_topology(agent: WineAgent) -> tuple[set[str], set[GraphEdge]]:
    """Return stable node and edge tuples from the compiled LangGraph graph."""
    graph = agent.agent.get_graph()
    nodes = set(graph.nodes)
    edges = {(edge.source, edge.target, edge.conditional) for edge in graph.edges}
    return nodes, edges


def test_standard_agent_topology_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the standard tool loop with its pre-model budget check."""
    agent = _build_agent(monkeypatch, with_tools=True)

    assert _graph_topology(agent) == (
        {
            "__start__",
            "check_relevance",
            "relevance_redirect",
            "check_agent_budget",
            "agent",
            "check_loop",
            "tools",
            "fail_soft",
            "__end__",
        },
        {
            ("__start__", "check_relevance", False),
            ("check_relevance", "check_agent_budget", True),
            ("check_relevance", "relevance_redirect", True),
            ("relevance_redirect", "__end__", False),
            ("check_agent_budget", "agent", True),
            ("check_agent_budget", "fail_soft", True),
            ("agent", "check_loop", True),
            ("agent", "__end__", True),
            ("check_loop", "tools", True),
            ("check_loop", "fail_soft", True),
            ("tools", "check_agent_budget", False),
            ("fail_soft", "__end__", False),
        },
    )




def test_standard_zero_tool_topology_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the current direct standard path when the tool snapshot is empty."""
    agent = _build_agent(monkeypatch, with_tools=False)

    assert agent.tools == []
    assert _graph_topology(agent) == (
        {
            "__start__",
            "check_relevance",
            "relevance_redirect",
            "check_agent_budget",
            "agent",
            "fail_soft",
            "__end__",
        },
        {
            ("__start__", "check_relevance", False),
            ("check_relevance", "check_agent_budget", True),
            ("check_relevance", "relevance_redirect", True),
            ("relevance_redirect", "__end__", False),
            ("check_agent_budget", "agent", True),
            ("check_agent_budget", "fail_soft", True),
            ("agent", "__end__", False),
            ("fail_soft", "__end__", False),
        },
    )
