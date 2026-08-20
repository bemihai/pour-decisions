"""Regression tests for pre-M6 optional tool dependency behavior."""

from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool
from pytest_mock import MockerFixture

from src.agents.intelligent.agent import WineAgent
from src.agents.tools import rag_tools, web_search_tools


def test_wine_agent_construction_does_not_initialize_tavily(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Tavily key must not prevent the static agent from starting."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(web_search_tools, "_engine", None)
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()

    agent = WineAgent(llm=llm)

    assert len(agent.tools) == 18
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
def test_web_search_tools_preserve_unavailable_result(
    tool: BaseTool,
    arguments: dict[str, object],
    mocker: MockerFixture,
) -> None:
    """Every web-search wrapper should convert initialization errors to tool output."""
    mocker.patch.object(
        web_search_tools,
        "_get_engine",
        side_effect=ValueError("Tavily API key not found"),
    )

    output = tool.invoke(arguments)

    assert output == "Web search unavailable: Tavily API key not found"


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
