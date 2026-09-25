"""Single-model agent and Cloud configuration tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agents.prompt_registry import RenderedPrompt
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolRegistry, ToolSelectionSnapshot

def _make_mock_llm(name: str = "MockLLM") -> MagicMock:
    """Return a mock BaseChatModel with bind_tools support."""
    mock = MagicMock()
    mock.__class__.__name__ = name
    mock.bind_tools.return_value = MagicMock()
    return mock


def _make_wine_agent(llm=None, verbose: bool = False):
    """Create a WineAgent with mocked dependencies."""
    from src.agents.intelligent.agent import WineAgent

    registry = ToolRegistry(TOOL_DEFINITIONS)
    with patch(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        return_value=RenderedPrompt(
            name="intelligent_agent_system",
            content="Test system prompt.",
            source_hash="sha256:test-source",
            rendered_hash="sha256:test-rendered",
            label="",
        ),
    ), patch.object(
        registry,
        "select",
        return_value=ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=()),
    ):
        return WineAgent(
            llm=llm,
            tool_registry=registry,
            verbose=verbose,
        )


# ---------------------------------------------------------------------------
# WineAgent -- tool_llm storage and is_hybrid_mode property
# ---------------------------------------------------------------------------

class TestSingleModelAgent:
    """One model supplies planning, tools, and final answers."""

    def test_agent_binds_the_configured_model_once(self):
        llm = _make_mock_llm()
        agent = _make_wine_agent(llm=llm)
        assert agent.llm is llm
        llm.bind_tools.assert_called_once_with(agent.tools)
        assert "generate" not in agent.agent.get_graph().nodes

    def test_factory_forwards_only_one_model(self):
        llm = _make_mock_llm()
        registry = ToolRegistry(())
        with patch("src.agents.intelligent.agent.WineAgent") as mock_cls:
            from src.agents.intelligent.agent import create_wine_agent
            create_wine_agent(llm=llm, tool_registry=registry)
        assert mock_cls.call_args.kwargs["llm"] is llm
        assert "tool_llm" not in mock_cls.call_args.kwargs

    def test_api_loader_forwards_only_one_model(self, mocker):
        mock_create = mocker.patch("src.agents.create_wine_agent", return_value=MagicMock())
        from src.api.main import _load_agents
        llm = _make_mock_llm()
        intelligent, secondary = _load_agents(llm=llm)
        assert intelligent is mock_create.return_value
        assert secondary is None
        assert mock_create.call_args.kwargs["llm"] is llm
        assert "tool_llm" not in mock_create.call_args.kwargs

    def test_api_loader_fail_soft_on_agent_failure(self, mocker):
        mocker.patch("src.agents.create_wine_agent", side_effect=RuntimeError("failure"))
        from src.api.main import _load_agents
        assert _load_agents(llm=_make_mock_llm()) == (None, None)


class TestLoadCloudModelConfigSelection:
    """_load_cloud_model uses the one configured Cloud model."""

    def test_loads_configured_direct_cloud_model(self, mocker):
        """Application settings go to the shared Cloud loader."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_cloud_model

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="ollama",
                name="gemma4:31b",
                base_url="https://ollama.com",
                timeout_seconds=60,
            )
        )

        _load_cloud_model(cfg)
        mock_load.assert_called_once_with("ollama", "gemma4:31b", base_url="https://ollama.com", timeout=60.0)

    def test_rejects_legacy_fallback_before_model_construction(self, mocker):
        """Legacy fallback configuration cannot silently survive migration."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_cloud_model

        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="ollama",
                name="gemma4:31b",
                fallback_provider="unsupported",
                fallback_name="legacy-model",
                base_url="https://ollama.com",
                timeout_seconds=60,
            )
        )

        with pytest.raises(ValueError, match="Legacy fallback"):
            _load_cloud_model(cfg)
        mock_load.assert_not_called()

    @pytest.mark.parametrize("legacy_key", ["ollama", "hybrid_tool_calling"])
    def test_rejects_legacy_model_slots_even_when_disabled(self, mocker, legacy_key):
        """Old model slots must not survive as dormant configuration."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_cloud_model

        model = SimpleNamespace(provider="ollama", name="gemma4:31b")
        setattr(model, legacy_key, False)
        with pytest.raises(ValueError, match="Legacy fallback"):
            _load_cloud_model(SimpleNamespace(model=model))
        mock_load.assert_not_called()

    def test_rejects_legacy_local_startup_flag_even_when_disabled(self, mocker):
        """The removed API startup switch cannot be kept in deployment config."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        from src.api.main import _load_cloud_model

        cfg = SimpleNamespace(
            model=SimpleNamespace(provider="ollama", name="gemma4:31b"),
            api=SimpleNamespace(enable_local_model_startup=False),
        )
        with pytest.raises(ValueError, match="Local model startup"):
            _load_cloud_model(cfg)
        mock_load.assert_not_called()
