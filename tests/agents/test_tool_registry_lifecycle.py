"""Tests for explicit tool-registry ownership across agent construction."""

import asyncio

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.agents.guardrails import ToolExecutionConfig, ToolExecutionController
from src.agents.tools.registry import ToolRegistry


def _mock_llm() -> MagicMock:
    """Create the minimal language-model mock required by WineAgent."""
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()
    return llm


def _patch_agent_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove prompt rendering from lifecycle tests."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: MagicMock(content="Test system prompt."),
    )


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
    config = OmegaConf.create({})
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

    config = OmegaConf.create({})
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


def test_wine_agent_owns_or_retains_controller_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone agents should not share capacity unless a controller is injected."""
    from src.agents import intelligent

    _patch_agent_construction(monkeypatch)
    policy = ToolExecutionConfig(max_concurrent_calls=2)
    registry = ToolRegistry(())
    first = intelligent.agent.WineAgent(
        llm=_mock_llm(),
        tool_registry=registry,
        tool_execution=policy,
    )
    second = intelligent.agent.WineAgent(
        llm=_mock_llm(),
        tool_registry=registry,
        tool_execution=policy,
    )
    shared = ToolExecutionController(2)
    injected = intelligent.agent.WineAgent(
        llm=_mock_llm(),
        tool_registry=registry,
        tool_execution=policy,
        tool_execution_controller=shared,
    )

    assert first.tool_execution_controller is not second.tool_execution_controller
    assert first.tool_execution_controller.max_concurrent_calls == 2
    assert injected.tool_execution_controller is shared


@pytest.mark.asyncio
async def test_agents_with_injected_controller_share_one_admission_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two agent instances should contend for the injected controller capacity."""
    from src.agents import intelligent

    _patch_agent_construction(monkeypatch)
    policy = ToolExecutionConfig(max_concurrent_calls=1)
    controller = ToolExecutionController(1)
    registry = ToolRegistry(())
    agents = [
        intelligent.agent.WineAgent(
            llm=_mock_llm(),
            tool_registry=registry,
            tool_execution=policy,
            tool_execution_controller=controller,
        )
        for _ in range(2)
    ]
    release = asyncio.Event()
    first_entered = asyncio.Event()
    active = 0
    maximum_active = 0

    async def use_agent(agent: object) -> None:
        nonlocal active, maximum_active
        async with agent.tool_execution_controller.permit():  # type: ignore[attr-defined]
            active += 1
            maximum_active = max(maximum_active, active)
            first_entered.set()
            await release.wait()
            active -= 1

    first = asyncio.create_task(use_agent(agents[0]))
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second = asyncio.create_task(use_agent(agents[1]))
    await asyncio.sleep(0)
    assert active == 1

    release.set()
    await asyncio.gather(first, second)
    assert maximum_active == 1
