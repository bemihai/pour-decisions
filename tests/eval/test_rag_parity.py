"""Parity validation for API and eval production RAG execution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.eval.models import GoldenSample
from src.retrieval import RAGExecutionResult


class _DeterministicRetriever:
    """Return stable document copies for each production-path invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, n_results: int) -> list[dict]:
        """Return two deterministic source documents."""
        self.calls.append((query, n_results))
        return [
            {
                "id": "chunk-barolo",
                "document": "Barolo is made from Nebbiolo in Piedmont.",
                "metadata": {"source": "/books/wine_atlas.pdf", "page": 42},
                "similarity": 0.91,
            },
            {
                "id": "chunk-aging",
                "document": "Barolo requires an extended minimum aging period.",
                "metadata": {"source": "/books/wine_law.pdf", "page": 88},
                "similarity": 0.83,
            },
        ]


def _parity_config() -> SimpleNamespace:
    """Build the production RAG config fields used by the shared service."""
    return SimpleNamespace(
        chroma=SimpleNamespace(
            retrieval=SimpleNamespace(
                n_results=2,
                enable_metadata_boost=False,
                metadata_boost_factor=0.1,
                rerank_top_k=2,
                use_deduplication=False,
                deduplication_threshold=0.9,
                enable_compression=False,
                compression_max_chars=8000,
            ),
            chunking=SimpleNamespace(enable_small_to_big=False),
            settings=SimpleNamespace(embedder="test-embedder"),
        )
    )


def test_api_and_eval_produce_identical_structured_rag_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed question/config should produce identical API and eval RAG artifacts."""
    from src.api.routes import chat
    from src.eval import utils as eval_utils
    from src.retrieval import rag_service

    config = _parity_config()
    question = "What is Barolo?"
    sample = GoldenSample(
        id="rag_only_001",
        question=question,
        category="rag_only",
        difficulty="easy",
        expected_facts=["Nebbiolo", "Piedmont"],
        ground_truth="Barolo is a Nebbiolo wine from Piedmont.",
        ground_truth_chunk_ids=["chunk-barolo"],
        tags=["barolo", "italy"],
    )
    api_retriever = _DeterministicRetriever()
    eval_retriever = _DeterministicRetriever()
    model = object()
    api_results: list[RAGExecutionResult] = []

    def fake_process_user_prompt(
        _model: Any,
        _prompt: str,
        _context: str,
        _history: list[dict[str, Any]],
        _trace_context: dict[str, str] | None = None,
    ) -> str:
        """Return a deterministic answer with one source citation."""
        return "Barolo is from Piedmont [1]."

    monkeypatch.setattr(rag_service, "process_user_prompt", fake_process_user_prompt)
    real_execute = rag_service.execute_production_rag

    def capture_api_result(**kwargs: Any) -> RAGExecutionResult:
        """Capture the API adapter's structured service result."""
        result = real_execute(**kwargs)
        api_results.append(result)
        return result

    monkeypatch.setattr(chat, "execute_production_rag", capture_api_result)
    monkeypatch.setattr(eval_utils, "execute_production_rag", real_execute)

    api_answer, api_sources, api_web_sources = chat._invoke_rag_only(
        prompt=question,
        cfg=config,
        model=model,
        retriever=api_retriever,
        reranker=None,
        message_history=[],
        enable_rag=True,
        n_results_override=None,
    )
    eval_result = eval_utils.run_rag_sample_sync(
        sample=sample,
        config=config,
        retriever=eval_retriever,
        model=model,
        reranker=None,
    )

    assert len(api_results) == 1
    api_result = api_results[0]
    assert api_answer == eval_result.answer
    assert api_web_sources == []
    assert [source.name for source in api_sources] == [source.name for source in eval_result.sources]
    assert api_result.normalized_query == eval_result.normalized_query
    assert api_result.context == eval_result.context
    assert api_result.raw_retrieved_chunks == eval_result.raw_retrieved_chunks
    assert api_result.context_chunks == eval_result.context_chunks
    assert api_result.sources == eval_result.sources
    assert api_result.feature_usage == eval_result.feature_usage
    assert api_retriever.calls == eval_retriever.calls
