"""Tests for snapshot-scoped intelligent-agent capability prompts."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import (
    ToolPrerequisite,
    ToolRegistry,
    ToolSelectionSnapshot,
    _PrerequisiteReadiness,
)


def _mock_llm() -> MagicMock:
    """Create a language-model mock without executing model calls."""
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()
    return llm


def _readiness(
    prerequisite: ToolPrerequisite,
    available: bool,
) -> _PrerequisiteReadiness:
    """Build deterministic prerequisite evidence for prompt scenarios."""
    return _PrerequisiteReadiness(
        prerequisite=prerequisite,
        available=available,
        reason_code=None if available else "dependency_unreachable",
        reason=None if available else "Dependency unavailable.",
    )


def _registry(
    monkeypatch: pytest.MonkeyPatch,
    unavailable: set[ToolPrerequisite],
) -> ToolRegistry:
    """Build a registry with controlled prerequisite results."""
    config = OmegaConf.create(
        {
            "agents": {
                "tool_registry": {
                    "health_check_ttl_seconds": 60,
                }
            }
        }
    )
    registry = ToolRegistry(TOOL_DEFINITIONS, config=config)
    monkeypatch.setattr(
        registry,
        "_get_prerequisite_readiness",
        lambda prerequisite, **_kwargs: _readiness(
            prerequisite,
            prerequisite not in unavailable,
        ),
    )
    return registry


def _advertised_tool_names(prompt: str) -> tuple[str, ...]:
    """Extract callable names from the generated registry section."""
    return tuple(
        definition.metadata.name
        for definition in TOOL_DEFINITIONS
        if definition.metadata.name in prompt
    )


@pytest.mark.parametrize(
    ("unavailable", "excluded_names"),
    (
        (
            {ToolPrerequisite.WEB_SEARCH_CONFIG},
            {"search_web_for_wine", "search_wine_price", "search_wine_reviews"},
        ),
        (
            {ToolPrerequisite.CHROMA_COLLECTION},
            {
                "search_wine_knowledge",
                "search_wine_region_info",
                "search_grape_variety_info",
                "search_wine_term_definition",
                "search_wine_producer_info",
            },
        ),
        (
            {ToolPrerequisite.PAIRING_RULES},
            {"get_food_pairing_wines"},
        ),
    ),
)
def test_prompt_matches_degraded_bound_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    unavailable: set[ToolPrerequisite],
    excluded_names: set[str],
) -> None:
    """Known-unavailable tool families should disappear from bindings and guidance."""
    from src.agents.intelligent.agent import WineAgent

    llm = _mock_llm()
    agent = WineAgent(
        llm=llm,
        tool_registry=_registry(monkeypatch, unavailable),
    )
    bound_names = tuple(tool.name for tool in agent.tools)

    assert set(_advertised_tool_names(agent.system_prompt)) == set(bound_names)
    assert excluded_names.isdisjoint(bound_names)
    assert all(name not in agent.system_prompt for name in excluded_names)
    assert "**Critical Rules:**" in agent.system_prompt
    llm.invoke.assert_not_called()
    llm.bind_tools.return_value.invoke.assert_not_called()


def test_empty_snapshot_advertises_no_tools_and_keeps_grounding_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty selection should be explicit without removing stable safety guidance."""
    from src.agents.intelligent.agent import WineAgent

    registry = _registry(monkeypatch, set(ToolPrerequisite))
    agent = WineAgent(llm=_mock_llm(), tool_registry=registry)

    assert agent.tools == []
    assert _advertised_tool_names(agent.system_prompt) == ()
    assert "No tools are currently available." in agent.system_prompt
    assert "NEVER invent or fabricate" in agent.system_prompt
    assert "**Tool Selection Guidelines:**" not in agent.system_prompt


def test_prompt_output_is_stable_for_same_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent selections should produce byte-identical capability guidance."""
    from src.agents.intelligent.agent import WineAgent

    first = WineAgent(llm=_mock_llm(), tool_registry=_registry(monkeypatch, set()))
    second = WineAgent(llm=_mock_llm(), tool_registry=_registry(monkeypatch, set()))

    assert first.system_prompt == second.system_prompt
    assert set(_advertised_tool_names(first.system_prompt)) == {
        tool.name for tool in first.tools
    }


def test_all_ready_prompt_retains_reviewed_guidance() -> None:
    """The complete snapshot should advertise every tool and retain grounding rules."""
    from src.agents.intelligent.agent import WineAgent

    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = ToolSelectionSnapshot(
        definitions=TOOL_DEFINITIONS,
        readiness=(),
    )

    agent = WineAgent(llm=_mock_llm(), tool_registry=registry)

    assert set(_advertised_tool_names(agent.system_prompt)) == {
        definition.metadata.name for definition in TOOL_DEFINITIONS
    }
    assert "'my cellar' require get_cellar_wines before answering" in agent.system_prompt
    assert "NEVER invent or fabricate" in agent.system_prompt
