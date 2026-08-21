"""Tests for explicit tool-registry ownership across agent construction."""

from unittest.mock import MagicMock

import pytest

from src.agents.tools.registry import ToolRegistry


def _mock_llm() -> MagicMock:
    """Create the minimal language-model mock required by WineAgent."""
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()
    return llm


def _patch_agent_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove prompt I/O and static tool details from lifecycle tests."""
    monkeypatch.setattr("src.agents.intelligent.agent.find_project_root", lambda: "/missing")


def test_wine_agent_retains_explicit_registry_without_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct construction should retain an injected registry by identity."""
    from src.agents import intelligent

    _patch_agent_construction(monkeypatch)
    registry = ToolRegistry(())
    monkeypatch.setattr(
        intelligent.agent,
        "get_config",
        MagicMock(side_effect=AssertionError("explicit registry must not load config")),
    )

    agent = intelligent.agent.WineAgent(llm=_mock_llm(), tool_registry=registry)

    assert agent.tool_registry is registry


def test_wine_agent_builds_fresh_registry_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct construction should build a configured registry when none is injected."""
    from src.agents import intelligent

    _patch_agent_construction(monkeypatch)
    config = object()
    registry = ToolRegistry(())
    get_config = MagicMock(return_value=config)
    build_registry = MagicMock(return_value=registry)
    monkeypatch.setattr(intelligent.agent, "get_config", get_config)
    monkeypatch.setattr(intelligent.agent, "build_tool_registry", build_registry)

    agent = intelligent.agent.WineAgent(llm=_mock_llm())

    assert agent.tool_registry is registry
    get_config.assert_called_once_with()
    build_registry.assert_called_once_with(config)


def test_factory_builds_a_registry_for_each_omitted_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent factory calls should not share implicit registry state."""
    from src.agents import intelligent

    config = object()
    first_registry = ToolRegistry(())
    second_registry = ToolRegistry(())
    agent = MagicMock(is_hybrid_mode=False)
    wine_agent = MagicMock(return_value=agent)
    build_registry = MagicMock(side_effect=[first_registry, second_registry])
    monkeypatch.setattr(intelligent.agent, "get_config", lambda: config)
    monkeypatch.setattr(intelligent.agent, "build_tool_registry", build_registry)
    monkeypatch.setattr(intelligent.agent, "WineAgent", wine_agent)

    intelligent.agent.create_wine_agent(llm=_mock_llm())
    intelligent.agent.create_wine_agent(llm=_mock_llm())

    assert [call.kwargs["tool_registry"] for call in wine_agent.call_args_list] == [
        first_registry,
        second_registry,
    ]
    assert build_registry.call_args_list[0].args == (config,)
    assert build_registry.call_args_list[1].args == (config,)
