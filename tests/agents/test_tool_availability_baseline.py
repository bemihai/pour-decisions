"""Regression tests for optional tool dependency behavior."""

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool
from omegaconf import OmegaConf
from pytest_mock import MockerFixture

from src.agents.intelligent.agent import WineAgent
from src.agents.tools import rag_tools, web_search_tools
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import ToolPrerequisite, ToolRegistry, _PrerequisiteReadiness


def test_wine_agent_construction_does_not_initialize_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Tavily key should exclude web tools without constructing the engine."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(web_search_tools, "_engine", None)
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

    def readiness(prerequisite: ToolPrerequisite, **_kwargs: object) -> _PrerequisiteReadiness:
        available = prerequisite != ToolPrerequisite.WEB_SEARCH_CONFIG
        return _PrerequisiteReadiness(
            prerequisite=prerequisite,
            available=available,
            reason_code=None if available else "missing_configuration",
            reason=None if available else "Web search configuration is missing.",
        )

    monkeypatch.setattr(registry, "_get_prerequisite_readiness", readiness)
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()

    agent = WineAgent(llm=llm, tool_registry=registry)

    assert len(agent.tools) == 15
    assert {
        "search_web_for_wine",
        "search_wine_price",
        "search_wine_reviews",
    }.isdisjoint(tool.name for tool in agent.tools)
    assert web_search_tools._engine is None
    llm.bind_tools.assert_called_once_with(agent.tools)


@pytest.mark.parametrize(
    ("tool", "arguments"),
    (
        (web_search_tools.search_web_for_wine, {"query": "Barolo news"}),
        (web_search_tools.search_wine_price, {"wine_name": "Barolo", "vintage": 2019}),
        (web_search_tools.search_wine_reviews, {"wine_name": "Barolo", "vintage": 2019}),
    ),
)
def test_web_search_tools_raise_unexpected_initialization_failure(
    tool: BaseTool,
    arguments: dict[str, object],
    mocker: MockerFixture,
) -> None:
    """Every web-search wrapper should defer unexpected errors to the safe boundary."""
    mocker.patch.object(
        web_search_tools,
        "_get_engine",
        side_effect=ValueError("Tavily API key not found"),
    )

    with pytest.raises(ValueError, match="Tavily API key not found"):
        tool.invoke(arguments)


def test_rag_resource_initialization_failure_preserves_unavailable_message(mocker: MockerFixture) -> None:
    """Retriever setup failure should retain the established M3 user message."""
    mocker.patch.object(rag_tools, "get_config", return_value=object())
    mocker.patch.object(
        rag_tools,
        "build_retriever_from_config",
        side_effect=RuntimeError("embedding model unavailable"),
    )
    reranker_factory = mocker.patch.object(rag_tools, "build_reranker_from_config")

    output = rag_tools.search_wine_knowledge.invoke({"query": "What is Barolo?"})

    assert output == rag_tools.RAG_UNAVAILABLE_MESSAGE
    reranker_factory.assert_not_called()
