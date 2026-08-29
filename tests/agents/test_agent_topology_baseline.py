"""M9A Gate 0 baselines for the current intelligent-agent graph topology."""

from unittest.mock import MagicMock

import pytest

from src.agents.intelligent.agent import WineAgent
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
    hybrid: bool,
    with_tools: bool,
) -> WineAgent:
    """Build an agent from a deterministic construction-time tool snapshot."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: "Test system prompt.",
    )

    definitions = TOOL_DEFINITIONS[:1] if with_tools else ()
    snapshot = ToolSelectionSnapshot(definitions=definitions, readiness=())
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = snapshot

    llm = _mock_llm("GenerationModel")
    tool_llm = _mock_llm("PlanningModel") if hybrid else None
    return WineAgent(llm=llm, tool_llm=tool_llm, tool_registry=registry)


def _graph_topology(agent: WineAgent) -> tuple[set[str], set[GraphEdge]]:
    """Return stable node and edge tuples from the compiled LangGraph graph."""
    graph = agent.agent.get_graph()
    nodes = set(graph.nodes)
    edges = {(edge.source, edge.target, edge.conditional) for edge in graph.edges}
    return nodes, edges


def test_standard_agent_topology_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the standard tool loop with its pre-model budget check."""
    agent = _build_agent(monkeypatch, hybrid=False, with_tools=True)

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


def test_hybrid_agent_topology_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the current hybrid planning, tools, and generation topology."""
    agent = _build_agent(monkeypatch, hybrid=True, with_tools=True)

    assert _graph_topology(agent) == (
        {
            "__start__",
            "check_relevance",
            "relevance_redirect",
            "check_agent_budget",
            "agent",
            "check_loop",
            "tools",
            "check_generation_budget",
            "generate",
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
            ("agent", "check_generation_budget", True),
            ("check_loop", "tools", True),
            ("check_loop", "fail_soft", True),
            ("tools", "check_generation_budget", False),
            ("check_generation_budget", "generate", True),
            ("check_generation_budget", "fail_soft", True),
            ("generate", "__end__", False),
            ("fail_soft", "__end__", False),
        },
    )


def test_standard_zero_tool_topology_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the current direct standard path when the tool snapshot is empty."""
    agent = _build_agent(monkeypatch, hybrid=False, with_tools=False)

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


def test_hybrid_zero_tool_topology_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze the current hybrid generation path when the tool snapshot is empty."""
    agent = _build_agent(monkeypatch, hybrid=True, with_tools=False)

    assert agent.tools == []
    assert _graph_topology(agent) == (
        {
            "__start__",
            "check_relevance",
            "relevance_redirect",
            "check_agent_budget",
            "agent",
            "check_generation_budget",
            "generate",
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
            ("agent", "check_generation_budget", False),
            ("check_generation_budget", "generate", True),
            ("check_generation_budget", "fail_soft", True),
            ("generate", "__end__", False),
            ("fail_soft", "__end__", False),
        },
    )
