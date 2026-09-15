"""Contracts for coroutine-backed RAG tools built from API resources."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.provenance import build_tool_contract_provenance
from src.agents.tools.rag_tools import (
    RAG_UNAVAILABLE_MESSAGE,
    TOOL_DEFINITIONS,
    build_async_rag_tool_definitions,
)
from src.agents.tools.registry import ToolSelectionSnapshot
from src.retrieval import AsyncRAGRuntimeResources


def _resources(*, retriever: object | None = None) -> AsyncRAGRuntimeResources:
    """Return a minimal injected resource bundle for tool tests."""
    return AsyncRAGRuntimeResources(
        config=object(),
        retriever=retriever,
        reranker=object(),
    )


def test_async_rag_definitions_preserve_every_public_contract_and_m5_hash() -> None:
    """Coroutine substitution must not change model-visible tool identity."""
    async_definitions = build_async_rag_tool_definitions(_resources(retriever=object()))

    assert tuple(definition.metadata.name for definition in async_definitions) == tuple(
        definition.metadata.name for definition in TOOL_DEFINITIONS
    )
    for sync_definition, async_definition in zip(
        TOOL_DEFINITIONS,
        async_definitions,
        strict=True,
    ):
        assert sync_definition.metadata is async_definition.metadata
        assert sync_definition.tool.name == async_definition.tool.name
        assert sync_definition.tool.description == async_definition.tool.description
        assert sync_definition.tool.tool_call_schema.model_json_schema() == (
            async_definition.tool.tool_call_schema.model_json_schema()
        )
        assert sync_definition.tool.coroutine is None
        assert async_definition.tool.coroutine is not None
        assert sync_definition.may_continue_in_worker_after_cancel is False
        assert async_definition.may_continue_in_worker_after_cancel is True

    sync_provenance = build_tool_contract_provenance(
        ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=())
    )
    async_provenance = build_tool_contract_provenance(
        ToolSelectionSnapshot(definitions=async_definitions, readiness=())
    )
    assert async_provenance == sync_provenance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_index", "arguments", "expected_query"),
    (
        (0, {"query": "Barolo", "max_results": 20, "include_sources": False}, "Barolo"),
        (1, {"region": "Burgundy"}, "Tell me about the Burgundy wine region:"),
        (2, {"varietal": "Nebbiolo"}, "Tell me about the Nebbiolo grape variety:"),
        (3, {"term": "terroir"}, "What is terroir?"),
        (4, {"producer": "Gaja"}, "Tell me about Gaja wine producer:"),
    ),
)
async def test_async_rag_tools_use_injected_resources_and_preserve_success_output(
    monkeypatch: pytest.MonkeyPatch,
    tool_index: int,
    arguments: dict[str, object],
    expected_query: str,
) -> None:
    """Each async tool should share the bundle and return production context."""
    from src.agents.tools import rag_tools

    resources = _resources(retriever=object())
    result = SimpleNamespace(
        context="Shared production context",
        context_chunks=[object()],
        retrieval_error=None,
    )
    execute_async = AsyncMock(return_value=result)
    monkeypatch.setattr(rag_tools, "execute_production_rag_async", execute_async)
    definition = build_async_rag_tool_definitions(resources)[tool_index]

    output = await definition.tool.ainvoke(arguments)

    assert output == "Shared production context"
    call = execute_async.await_args.kwargs
    assert expected_query in call["prompt"]
    assert call["config"] is resources.config
    assert call["retriever"] is resources.retriever
    assert call["reranker"] is resources.reranker
    assert call["generation_enabled"] is False
    assert call["n_results_override"] == (10 if tool_index == 0 else 5)
    assert call["include_context_metadata"] is (tool_index != 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_index", "arguments"),
    (
        (0, {"query": "Barolo"}),
        (1, {"region": "Burgundy"}),
        (2, {"varietal": "Nebbiolo"}),
        (3, {"term": "terroir"}),
        (4, {"producer": "Gaja"}),
    ),
)
async def test_async_rag_tools_preserve_unavailable_result(
    tool_index: int,
    arguments: dict[str, object],
) -> None:
    """Every injected tool should retain the existing unavailable response."""
    definition = build_async_rag_tool_definitions(_resources())[tool_index]

    assert await definition.tool.ainvoke(arguments) == RAG_UNAVAILABLE_MESSAGE


@pytest.mark.asyncio
async def test_async_rag_tool_preserves_retrieval_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fail-soft service retrieval error should remain an explicit tool failure."""
    from src.agents.tools import rag_tools

    resources = _resources(retriever=object())
    monkeypatch.setattr(
        rag_tools,
        "execute_production_rag_async",
        AsyncMock(
            return_value=SimpleNamespace(
                context="",
                context_chunks=[],
                retrieval_error="retrieval failed",
            )
        ),
    )
    definition = build_async_rag_tool_definitions(resources)[0]

    with pytest.raises(RuntimeError, match="retrieval failed"):
        await definition.tool.ainvoke({"query": "Barolo"})
