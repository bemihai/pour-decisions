"""Successful behavior tests for specialized local-RAG tool façades."""

from types import SimpleNamespace

import pytest

from src.agents.tools import rag_tools


@pytest.mark.parametrize(
    ("tool", "arguments", "query_fragment"),
    (
        (rag_tools.search_wine_region_info, {"region": "Bandol"}, "Bandol wine region"),
        (rag_tools.search_grape_variety_info, {"varietal": "Nebbiolo"}, "Nebbiolo grape variety"),
        (rag_tools.search_wine_term_definition, {"term": "terroir"}, "What is terroir?"),
        (rag_tools.search_wine_producer_info, {"producer": "Gaja"}, "Gaja wine producer"),
    ),
)
def test_specialized_rag_tool_formats_query_and_returns_context(
    monkeypatch: pytest.MonkeyPatch,
    tool: object,
    arguments: dict[str, str],
    query_fragment: str,
) -> None:
    """Each specialized façade should issue its focused query and return local evidence."""
    observed: dict[str, object] = {}

    def execute(*, query: str, n_results: int, include_sources: bool = True) -> tuple[object, bool]:
        observed.update(
            query=query,
            n_results=n_results,
            include_sources=include_sources,
        )
        return SimpleNamespace(context_chunks=["chunk"], context="grounded context"), True

    monkeypatch.setattr(rag_tools, "_execute_rag_query", execute)

    result = tool.invoke(arguments)  # type: ignore[attr-defined]

    assert result == "grounded context"
    assert query_fragment in str(observed["query"])
    assert observed["n_results"] == 5
