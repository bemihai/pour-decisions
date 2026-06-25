"""Unit tests for Phase 4: Tool calling compatibility in WineAgent.

Tests cover:
- WineAgent accepts the optional ``tool_llm`` parameter.
- In normal mode (tool_llm is None / same as llm), tool binding uses self.llm.
- In hybrid mode (tool_llm != llm), planning uses tool_llm and generation uses llm.
- Graph structure differs between normal and hybrid modes.
- ``create_wine_agent`` factory correctly forwards ``tool_llm``.
- ``_load_agents`` in main.py passes ``tool_llm`` to the intelligent agent.
- Lifespan wires hybrid tool-calling based on ``cfg.model.hybrid_tool_calling``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from langchain_core.messages import AIMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(name: str = "MockLLM") -> MagicMock:
    """Return a mock BaseChatModel with bind_tools support."""
    mock = MagicMock()
    mock.__class__.__name__ = name
    mock.bind_tools.return_value = MagicMock()
    return mock


def _make_wine_agent(llm=None, tool_llm=None, verbose: bool = False):
    """Create a WineAgent with mocked dependencies."""
    from src.agents.intelligent.agent import WineAgent

    with patch("src.agents.intelligent.agent.get_tools", return_value=[]), \
         patch("src.agents.intelligent.agent.find_project_root", return_value="/tmp"), \
         patch("builtins.open", side_effect=FileNotFoundError):
        return WineAgent(llm=llm, tool_llm=tool_llm, verbose=verbose)


# ---------------------------------------------------------------------------
# WineAgent -- tool_llm storage and is_hybrid_mode property
# ---------------------------------------------------------------------------

class TestWineAgentToolLlmStorage:
    """WineAgent correctly stores llm and tool_llm references."""

    def test_default_tool_llm_equals_llm(self):
        """When tool_llm is not passed, self.tool_llm is the same object as self.llm."""
        llm = _make_mock_llm("GenerationModel")
        agent = _make_wine_agent(llm=llm)
        assert agent.tool_llm is agent.llm

    def test_explicit_tool_llm_stored(self):
        """When tool_llm is passed, it is stored as self.tool_llm."""
        llm = _make_mock_llm("GenerationModel")
        tool_llm = _make_mock_llm("PlanningModel")
        agent = _make_wine_agent(llm=llm, tool_llm=tool_llm)
        assert agent.tool_llm is tool_llm
        assert agent.llm is llm

    def test_is_hybrid_mode_false_when_same_model(self):
        """is_hybrid_mode is False when tool_llm equals llm."""
        llm = _make_mock_llm()
        agent = _make_wine_agent(llm=llm)
        assert agent.is_hybrid_mode is False

    def test_is_hybrid_mode_true_when_different_models(self):
        """is_hybrid_mode is True when tool_llm is a different instance from llm."""
        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")
        agent = _make_wine_agent(llm=llm, tool_llm=tool_llm)
        assert agent.is_hybrid_mode is True


# ---------------------------------------------------------------------------
# WineAgent -- bind_tools uses tool_llm
# ---------------------------------------------------------------------------

class TestWineAgentBindTools:
    """bind_tools is always called on tool_llm, not on llm."""

    def test_bind_tools_called_on_llm_in_normal_mode(self):
        """In normal mode (no separate tool_llm), bind_tools is called on self.llm."""
        llm = _make_mock_llm()
        agent = _make_wine_agent(llm=llm)
        llm.bind_tools.assert_called_once_with(agent.tools)

    def test_bind_tools_called_on_tool_llm_in_hybrid_mode(self):
        """In hybrid mode, bind_tools is called on tool_llm, not on llm."""
        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")
        agent = _make_wine_agent(llm=llm, tool_llm=tool_llm)
        tool_llm.bind_tools.assert_called_once_with(agent.tools)
        llm.bind_tools.assert_not_called()

    def test_llm_not_bound_to_tools_in_hybrid_mode(self):
        """In hybrid mode, self.llm is NOT bound to any tools (it generates freely)."""
        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")
        _make_wine_agent(llm=llm, tool_llm=tool_llm)
        llm.bind_tools.assert_not_called()


# ---------------------------------------------------------------------------
# WineAgent -- graph structure
# ---------------------------------------------------------------------------

class TestWineAgentGraphStructure:
    """Compiled graph nodes differ between normal and hybrid modes."""

    def _get_node_names(self, agent) -> set:
        """Extract node names from a compiled LangGraph agent."""
        graph_repr = agent.agent.get_graph()
        return set(graph_repr.nodes.keys())

    def test_normal_mode_no_generate_node(self):
        """Normal mode graph does NOT include a separate 'generate' node."""
        llm = _make_mock_llm()
        agent = _make_wine_agent(llm=llm)
        node_names = self._get_node_names(agent)
        assert "generate" not in node_names

    def test_normal_mode_has_agent_and_tools_nodes(self):
        """Normal mode graph includes 'agent' and 'tools' nodes."""
        llm = _make_mock_llm()
        agent = _make_wine_agent(llm=llm)
        node_names = self._get_node_names(agent)
        assert "agent" in node_names
        assert "tools" in node_names

    def test_hybrid_mode_has_generate_node(self):
        """Hybrid mode graph includes a separate 'generate' node for final answer."""
        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")
        agent = _make_wine_agent(llm=llm, tool_llm=tool_llm)
        node_names = self._get_node_names(agent)
        assert "generate" in node_names

    def test_hybrid_mode_has_agent_and_tools_nodes(self):
        """Hybrid mode graph still includes 'agent' and 'tools' nodes."""
        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")
        agent = _make_wine_agent(llm=llm, tool_llm=tool_llm)
        node_names = self._get_node_names(agent)
        assert "agent" in node_names
        assert "tools" in node_names

    def test_hybrid_mode_routes_no_tool_plan_to_generate_llm(self):
        """In hybrid mode, no-tool plans still run through local generation llm."""
        from src.agents.intelligent.agent import WineAgent

        llm = _make_mock_llm("LocalModel")
        llm.invoke.return_value = AIMessage(content="Final local answer")

        tool_llm = _make_mock_llm("CloudModel")
        planner = MagicMock()
        planner.invoke.return_value = AIMessage(content="No tools needed")
        tool_llm.bind_tools.return_value = planner

        with patch("src.agents.intelligent.agent.get_tools", return_value=[]), \
             patch("src.agents.intelligent.agent.find_project_root", return_value="/tmp"), \
             patch("builtins.open", side_effect=FileNotFoundError):
            agent = WineAgent(llm=llm, tool_llm=tool_llm, verbose=False)

        result = agent.invoke("Say hello")

        llm.invoke.assert_called_once()
        assert result["final_answer"] == "Final local answer"


# ---------------------------------------------------------------------------
# create_wine_agent factory
# ---------------------------------------------------------------------------

class TestCreateWineAgentFactory:
    """create_wine_agent factory correctly forwards tool_llm."""

    def test_forwards_tool_llm_to_wine_agent(self):
        """Factory passes tool_llm argument through to WineAgent constructor."""
        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")

        with patch("src.agents.intelligent.agent.WineAgent") as mock_cls:
            from src.agents.intelligent.agent import create_wine_agent
            create_wine_agent(llm=llm, tool_llm=tool_llm)
            mock_cls.assert_called_once_with(llm=llm, tool_llm=tool_llm, verbose=False)

    def test_tool_llm_defaults_to_none_in_factory(self):
        """When tool_llm is not passed, factory passes None to WineAgent."""
        llm = _make_mock_llm()

        with patch("src.agents.intelligent.agent.WineAgent") as mock_cls:
            from src.agents.intelligent.agent import create_wine_agent
            create_wine_agent(llm=llm)
            mock_cls.assert_called_once_with(llm=llm, tool_llm=None, verbose=False)

    def test_factory_returns_wine_agent(self):
        """Factory returns a WineAgent instance."""
        llm = _make_mock_llm()
        agent = _make_wine_agent(llm=llm)
        from src.agents.intelligent.agent import WineAgent
        assert isinstance(agent, WineAgent)

    def test_factory_hybrid_mode_active(self):
        """Agent returned by factory has is_hybrid_mode=True when tool_llm is different."""
        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")
        with patch("src.agents.intelligent.agent.get_tools", return_value=[]), \
             patch("src.agents.intelligent.agent.find_project_root", return_value="/tmp"), \
             patch("builtins.open", side_effect=FileNotFoundError):
            from src.agents.intelligent.agent import create_wine_agent
            agent = create_wine_agent(llm=llm, tool_llm=tool_llm)
        assert agent.is_hybrid_mode is True

    def test_factory_normal_mode_active(self):
        """Agent returned by factory has is_hybrid_mode=False when tool_llm not passed."""
        llm = _make_mock_llm()
        with patch("src.agents.intelligent.agent.get_tools", return_value=[]), \
             patch("src.agents.intelligent.agent.find_project_root", return_value="/tmp"), \
             patch("builtins.open", side_effect=FileNotFoundError):
            from src.agents.intelligent.agent import create_wine_agent
            agent = create_wine_agent(llm=llm)
        assert agent.is_hybrid_mode is False


# ---------------------------------------------------------------------------
# _load_agents in src/api/main.py
# ---------------------------------------------------------------------------

class TestLoadAgentsToolLlm:
    """_load_agents passes tool_llm to create_wine_agent."""

    def test_tool_llm_forwarded_to_intelligent_agent(self, mocker):
        """create_wine_agent is called with the tool_llm kwarg."""
        mock_create_wine = mocker.patch("src.agents.create_wine_agent", return_value=MagicMock())

        from src.api.main import _load_agents

        llm = _make_mock_llm("LocalModel")
        tool_llm = _make_mock_llm("CloudModel")
        _load_agents(llm=llm, tool_llm=tool_llm)

        mock_create_wine.assert_called_once_with(verbose=False, llm=llm, tool_llm=tool_llm)

    def test_no_tool_llm_by_default(self, mocker):
        """When tool_llm is omitted, create_wine_agent receives no tool_llm kwarg (None default)."""
        mock_create_wine = mocker.patch("src.agents.create_wine_agent", return_value=MagicMock())

        from src.api.main import _load_agents

        llm = _make_mock_llm()
        _load_agents(llm=llm)

        call_kwargs = mock_create_wine.call_args.kwargs
        assert call_kwargs.get("llm") is llm
        assert call_kwargs.get("verbose") is False
        assert call_kwargs.get("tool_llm") is None

    def test_returns_tuple_of_two(self, mocker):
        """Returns a (intelligent_agent, None) tuple."""
        mock_ia = MagicMock()
        mocker.patch("src.agents.create_wine_agent", return_value=mock_ia)

        from src.api.main import _load_agents

        result = _load_agents(llm=_make_mock_llm())
        assert result == (mock_ia, None)

    def test_intelligent_agent_none_on_failure(self, mocker):
        """Returns (None, None) when create_wine_agent raises."""
        mocker.patch("src.agents.create_wine_agent", side_effect=RuntimeError("boom"))

        from src.api.main import _load_agents

        ia, ka = _load_agents(llm=_make_mock_llm())
        assert ia is None
        assert ka is None


class TestLoadLocalModelConfigSelection:
    """_load_local_model uses the dedicated Ollama config slot."""

    def test_loads_ollama_model_when_primary_provider_is_cloud(self, mocker):
        """Local startup should not depend on the production model provider."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_local_model

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="google",
                name="gemini-2.5-flash",
                ollama=SimpleNamespace(
                    name="gemma3:4b",
                    base_url="http://localhost:11434",
                ),
            )
        )

        _load_local_model(cfg)
        mock_load.assert_called_once_with("ollama", "gemma3:4b", base_url="http://localhost:11434")

    def test_falls_back_to_model_name_for_legacy_config(self, mocker):
        """Older config shapes without model.ollama.name still load explicitly as Ollama."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_local_model

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="ollama",
                name="legacy-local-model",
                ollama=SimpleNamespace(base_url="http://remote:11434"),
            )
        )

        _load_local_model(cfg)
        mock_load.assert_called_once_with("ollama", "legacy-local-model", base_url="http://remote:11434")


class TestLoadCloudModelConfigSelection:
    """_load_cloud_model selects configured or fallback cloud model correctly."""

    def test_loads_configured_provider_when_primary_is_cloud(self, mocker):
        """When model.provider is cloud, load provider/name directly."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_cloud_model

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="google",
                name="gemini-2.5-pro",
                fallback_provider="google",
                fallback_name="gemini-2.5-flash",
            )
        )

        _load_cloud_model(cfg)
        mock_load.assert_called_once_with("google", "gemini-2.5-pro")

    def test_loads_fallback_when_primary_is_ollama(self, mocker):
        """When model.provider is ollama, load fallback_provider/fallback_name."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_cloud_model

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="ollama",
                name="gemma3:4b",
                fallback_provider="google",
                fallback_name="gemini-2.5-flash",
            )
        )

        _load_cloud_model(cfg)
        mock_load.assert_called_once_with("google", "gemini-2.5-flash")
