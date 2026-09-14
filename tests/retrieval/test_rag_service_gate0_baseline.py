"""Gate 0 characterization tests for the synchronous production RAG contract."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel

from src.agents.llm import ModelInternalError
from src.retrieval.rag_service import execute_production_rag
from src.retrieval.web_fallback import WebSearchFallback


class _RecordingRetriever:
    """Return isolated documents while recording the production query."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Return copies so downstream score mutation cannot alter fixtures."""
        self.calls.append((query, n_results))
        return [dict(document) for document in self.documents[:n_results]]


class _FailingRetriever:
    """Expose the current fail-soft retrieval boundary."""

    def retrieve(self, query: str, n_results: int) -> list[dict[str, Any]]:
        """Raise a stable failure after receiving the production arguments."""
        _ = query
        _ = n_results
        raise RuntimeError("gate0 retrieval unavailable")


class _ThresholdReranker:
    """Attach stable scores for fallback characterization."""

    def rerank_with_threshold(
        self,
        query: str,
        documents: list[dict[str, Any]],
        threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        """Return thresholded copies in input order."""
        _ = query
        scored = [
            {**document, "rerank_score": float(document["metadata"]["gate0_score"])}
            for document in documents
            if float(document["metadata"]["gate0_score"]) >= threshold
        ]
        return scored if top_k is None else scored[:top_k]


def _config(*, auto_fallback: bool = False) -> SimpleNamespace:
    """Return the production settings consumed by the Gate 0 fixtures."""
    return SimpleNamespace(
        web_search=SimpleNamespace(auto_fallback=auto_fallback),
        chroma=SimpleNamespace(
            retrieval=SimpleNamespace(
                n_results=2,
                enable_metadata_boost=False,
                rerank_top_k=2,
                rerank_threshold=0.0,
                min_retrieval_confidence=0.8,
                use_deduplication=False,
                enable_compression=False,
            ),
            chunking=SimpleNamespace(enable_small_to_big=False),
            settings=SimpleNamespace(embedder="gate0-embedder"),
        ),
    )


def _book_document(*, score: float = 2.0) -> dict[str, Any]:
    """Return one stable source-bearing book chunk."""
    return {
        "id": "book-1",
        "document": "Barolo is made from Nebbiolo.",
        "metadata": {
            "filename": "/books/wine-atlas.pdf",
            "page_number": 42,
            "gate0_score": score,
        },
        "similarity": 0.91,
    }


def test_gate0_freezes_success_result_generation_arguments_and_source_filtering(
    monkeypatch,
) -> None:
    """Freeze the semantic result while leaving elapsed diagnostics out of equality."""
    retriever = _RecordingRetriever([_book_document()])
    generation_calls: list[dict[str, Any]] = []

    def _record_generation(
        model: BaseChatModel,
        prompt: str,
        context: str,
        message_history: list[dict[str, Any]],
        trace_context: dict[str, str] | None,
    ) -> str:
        generation_calls.append(
            {
                "model": model,
                "prompt": prompt,
                "context": context,
                "message_history": message_history,
                "trace_context": trace_context,
            }
        )
        return "Barolo uses Nebbiolo [1]."

    monkeypatch.setattr("src.retrieval.rag_service.process_user_prompt", _record_generation)
    model = MagicMock(spec=BaseChatModel)
    history = [{"role": "human", "content": "Tell me about Piedmont."}]
    trace_context = {"request_id": "gate0", "agent_mode": "rag_only"}

    result = execute_production_rag(
        prompt="What is Barolo?",
        config=_config(),
        model=model,
        retriever=retriever,
        reranker=None,
        message_history=history,
        trace_context=trace_context,
    )

    assert retriever.calls == [("what is barolo?", 2)]
    assert len(generation_calls) == 1
    assert generation_calls[0] == {
        "model": model,
        "prompt": "What is Barolo?",
        "context": "[Source 1 - wine-atlas.pdf, Page 42]\nBarolo is made from Nebbiolo.",
        "message_history": history,
        "trace_context": trace_context,
    }
    assert asdict(result) == {
        "answer": "Barolo uses Nebbiolo [1].",
        "context": "[Source 1 - wine-atlas.pdf, Page 42]\nBarolo is made from Nebbiolo.",
        "normalized_query": "what is barolo?",
        "retrieval_query_plan": {
            "original_query": "What is Barolo?",
            "normalized_query": "what is barolo?",
            "semantic_query": "what is barolo?",
            "sparse_query": "barolo",
            "intent": "unknown",
            "entities": {
                "grapes": [],
                "regions": ["barolo"],
                "vintages": [],
                "classifications": [],
                "producers": [],
                "appellations": ["Barolo"],
            },
        },
        "raw_retrieved_chunks": [
            {
                "id": "book-1",
                "text": "Barolo is made from Nebbiolo.",
                "metadata": {
                    "filename": "/books/wine-atlas.pdf",
                    "page_number": 42,
                    "gate0_score": 2.0,
                },
                "similarity": 0.91,
                "rerank_score": None,
                "rrf_score": None,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_similarity": None,
                "bm25_score": None,
                "metadata_matches": None,
                "retrieval_channels": [],
                "retrieval_diagnostics": {},
            }
        ],
        "context_chunks": [
            {
                "id": "book-1",
                "text": "Barolo is made from Nebbiolo.",
                "metadata": {
                    "filename": "/books/wine-atlas.pdf",
                    "page_number": 42,
                    "gate0_score": 2.0,
                },
                "similarity": 0.91,
                "rerank_score": None,
                "rrf_score": None,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_similarity": None,
                "bm25_score": None,
                "metadata_matches": None,
                "retrieval_channels": [],
                "retrieval_diagnostics": {},
            }
        ],
        "sources": [
            {
                "name": "wine-atlas",
                "page": 42,
                "relevance": 0.91,
                "chunk_id": "book-1",
                "metadata": {
                    "filename": "/books/wine-atlas.pdf",
                    "page_number": 42,
                    "gate0_score": 2.0,
                },
            }
        ],
        "feature_usage": {
            "retrieval": True,
            "query_normalization": True,
            "query_analysis": True,
            "hybrid_retrieval": False,
            "metadata_filtering": False,
            "metadata_boosting": False,
            "reranking": False,
            "small_to_big": False,
            "deduplication": False,
            "compression": False,
            "source_attribution": True,
            "generation": True,
            "hyde_expansion": False,
            "rerank_thresholding": False,
            "web_fallback": False,
        },
        "retrieval_error": None,
        "retrieval_confidence": None,
        "low_confidence": False,
        "rerank_threshold": None,
    }


def test_gate0_freezes_generation_disabled_and_retrieval_failure_behavior(monkeypatch) -> None:
    """Retrieval failures stay typed and generation-disabled calls never invoke a model."""
    generation = MagicMock(side_effect=AssertionError("generation must remain disabled"))
    monkeypatch.setattr("src.retrieval.rag_service.process_user_prompt", generation)

    result = execute_production_rag(
        prompt="What is Barolo?",
        config=_config(),
        model=None,
        retriever=_FailingRetriever(),
        reranker=None,
        message_history=[],
        generation_enabled=False,
    )

    generation.assert_not_called()
    assert result.answer == ""
    assert result.context == ""
    assert result.raw_retrieved_chunks == []
    assert result.context_chunks == []
    assert result.sources == []
    assert result.retrieval_error == "gate0 retrieval unavailable"
    assert result.feature_usage.retrieval is False
    assert result.feature_usage.generation is False


def test_gate0_freezes_enabled_and_disabled_web_fallback_call_counts(monkeypatch) -> None:
    """Low confidence spends exactly one web call only when fallback is enabled."""
    calls: list[tuple[str, str]] = []

    def _search(
        query: str,
        search_type: str = "general",
        max_results: int | None = None,
    ) -> list[dict[str, str]]:
        _ = max_results
        calls.append((query, search_type))
        return [
            {
                "title": "Current Barolo report",
                "snippet": "Current external evidence.",
                "url": "https://example.test/barolo",
            }
        ]

    engine = SimpleNamespace(search=_search)
    monkeypatch.setattr(
        "src.retrieval.rag_service.build_web_fallback_from_config",
        lambda config: WebSearchFallback(enabled=config.web_search.auto_fallback, engine=engine),
    )

    disabled = execute_production_rag(
        prompt="What is Barolo?",
        config=_config(auto_fallback=False),
        model=None,
        retriever=_RecordingRetriever([_book_document(score=0.1)]),
        reranker=_ThresholdReranker(),
        message_history=[],
        generation_enabled=False,
    )
    enabled = execute_production_rag(
        prompt="What is Barolo?",
        config=_config(auto_fallback=True),
        model=None,
        retriever=_RecordingRetriever([_book_document(score=0.1)]),
        reranker=_ThresholdReranker(),
        message_history=[],
        generation_enabled=False,
    )

    assert calls == [("What is Barolo?", "general")]
    assert disabled.feature_usage.web_fallback is False
    assert [chunk.id for chunk in disabled.context_chunks] == ["book-1"]
    assert enabled.feature_usage.web_fallback is True
    assert enabled.context_chunks[0].id == "book-1"
    assert enabled.context_chunks[1].id.startswith("web_")
    assert enabled.context_chunks[1].metadata["source"] == "web"


def test_gate0_freezes_model_failure_answer(monkeypatch) -> None:
    """The generation adapter keeps its current fail-soft user answer."""
    failure = ModelInternalError("gate0 model unavailable")
    monkeypatch.setattr("src.agents.llm.invoke_llm", MagicMock(side_effect=failure))

    result = execute_production_rag(
        prompt="What is Barolo?",
        config=_config(),
        model=MagicMock(spec=BaseChatModel),
        retriever=_RecordingRetriever([]),
        reranker=None,
        message_history=[],
    )

    assert result.answer == failure.default_message
    assert result.context == ""
    assert result.retrieval_error is None
    assert result.feature_usage.generation is True
