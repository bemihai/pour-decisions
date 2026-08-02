"""Contract tests for agent RAG tools using the shared production path."""

from src.agents.tools import rag_tools
from src.retrieval import RAGChunkArtifact, RAGExecutionResult


def _rag_result(*, context: str = "Retrieved wine context") -> RAGExecutionResult:
    """Build a minimal successful retrieval-only result."""
    chunk = RAGChunkArtifact(
        id="chunk-1",
        text="Barolo is made from Nebbiolo.",
        metadata={"source": "wine_book.pdf"},
    )
    return RAGExecutionResult(
        answer="",
        context=context,
        normalized_query="What is Barolo?",
        raw_retrieved_chunks=[chunk],
        context_chunks=[chunk],
    )


def test_execute_rag_query_uses_shared_factories_and_disables_generation(mocker) -> None:
    """Agent retrieval should use configured resources and never generate an answer."""
    config = object()
    retriever = object()
    reranker = object()
    expected_result = _rag_result()
    mocker.patch.object(rag_tools, "get_config", return_value=config)
    retriever_factory = mocker.patch.object(
        rag_tools,
        "build_retriever_from_config",
        return_value=retriever,
    )
    reranker_factory = mocker.patch.object(
        rag_tools,
        "build_reranker_from_config",
        return_value=reranker,
    )
    execute = mocker.patch.object(
        rag_tools,
        "execute_production_rag",
        return_value=expected_result,
    )

    result, available = rag_tools._execute_rag_query(
        "What is Barolo?",
        3,
        include_sources=False,
    )

    assert available is True
    assert result is expected_result
    retriever_factory.assert_called_once_with(config)
    reranker_factory.assert_called_once_with(config)
    execute.assert_called_once_with(
        prompt="What is Barolo?",
        config=config,
        model=None,
        retriever=retriever,
        reranker=reranker,
        message_history=[],
        n_results_override=3,
        generation_enabled=False,
        include_context_metadata=False,
    )


def test_search_wine_knowledge_preserves_context_and_source_option(mocker) -> None:
    """The general knowledge tool should return shared context without reformatting it."""
    expected_result = _rag_result(context="Context without source metadata")
    execute = mocker.patch.object(
        rag_tools,
        "_execute_rag_query",
        return_value=(expected_result, True),
    )

    output = rag_tools.search_wine_knowledge.invoke(
        {
            "query": "What is Barolo?",
            "max_results": 3,
            "include_sources": False,
        }
    )

    assert output == expected_result.context
    execute.assert_called_once_with(
        query="What is Barolo?",
        n_results=3,
        include_sources=False,
    )


def test_search_wine_knowledge_preserves_unavailable_message(mocker) -> None:
    """Resource initialization failure should retain the established user message."""
    mocker.patch.object(
        rag_tools,
        "_execute_rag_query",
        return_value=(None, False),
    )

    output = rag_tools.search_wine_knowledge.invoke({"query": "What is Barolo?"})

    assert output == rag_tools.RAG_UNAVAILABLE_MESSAGE
