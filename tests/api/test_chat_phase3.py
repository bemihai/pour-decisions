"""Phase 3 tests for chat helper trace-context propagation and retrieval span metadata."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _QueryAnalysis:
    """Minimal query analysis object used by RAG helper tests."""

    has_filters = False


class _StubRetriever:
    """Stub retriever returning one deterministic document."""

    def retrieve(self, prompt: str, n_results: int = 5) -> list[dict]:
        """Return one synthetic retrieval result.

        Args:
            prompt: User query.
            n_results: Requested number of chunks.

        Returns:
            List with one retrieved doc.
        """
        _ = prompt
        _ = n_results
        return [{"metadata": {"source": "wine_book.pdf", "page": 42}, "similarity": 0.91}]


def _build_cfg() -> SimpleNamespace:
    """Build minimal config object needed by _invoke_rag_only.

    Returns:
        Nested config namespace matching fields consumed by chat._invoke_rag_only.
    """
    retrieval_cfg = SimpleNamespace(
        n_results=5,
        enable_metadata_boost=False,
        metadata_boost_factor=0.1,
        rerank_top_k=5,
        use_deduplication=False,
        deduplication_threshold=0.9,
        enable_compression=False,
        compression_max_chars=8000,
    )
    chunking_cfg = SimpleNamespace(enable_small_to_big=False)
    settings_cfg = SimpleNamespace(embedder="sentence-transformers/all-MiniLM-L6-v2")
    chroma_cfg = SimpleNamespace(retrieval=retrieval_cfg, chunking=chunking_cfg, settings=settings_cfg)
    return SimpleNamespace(chroma=chroma_cfg)


def test_invoke_intelligent_agent_forwards_trace_context() -> None:
    """_invoke_intelligent_agent should forward trace_context into agent.invoke."""
    from src.api.routes.chat import _invoke_intelligent_agent

    agent = MagicMock()
    agent.invoke.return_value = {"final_answer": "ok", "messages": []}
    trace_context = {"request_id": "req-123", "agent_mode": "intelligent"}

    _invoke_intelligent_agent(agent, "question", [], trace_context=trace_context)

    agent.invoke.assert_called_once_with("question", message_history=[], trace_context=trace_context)



def test_invoke_rag_only_propagates_trace_context_and_sets_retrieval_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """_invoke_rag_only should pass trace_context to llm helper and emit retrieval span attributes."""
    from src.api.routes import chat
    import src.agents.llm as llm_module
    import src.retrieval as retrieval_module

    captured_llm_trace_context: dict[str, str] = {}
    captured_span_attributes: list[dict] = []

    def _fake_process_user_prompt(
        model,
        prompt: str,
        context: str,
        message_history: list,
        trace_context: dict[str, str] | None = None,
    ) -> str:
        _ = model
        _ = prompt
        _ = context
        _ = message_history
        if trace_context:
            captured_llm_trace_context.update(trace_context)
        return "Answer with citation [1]"

    monkeypatch.setattr(chat, "set_span_attributes", lambda _span, attrs: captured_span_attributes.append(attrs))
    monkeypatch.setattr(llm_module, "process_user_prompt", _fake_process_user_prompt)
    monkeypatch.setattr(retrieval_module, "analyze_query", lambda _query: _QueryAnalysis())
    monkeypatch.setattr(retrieval_module, "boost_by_metadata_match", lambda docs, *_args, **_kwargs: docs)
    monkeypatch.setattr(
        retrieval_module,
        "build_context_from_chunks",
        lambda _docs, **_kwargs: "context",
    )
    monkeypatch.setattr(retrieval_module, "build_semantic_context", lambda _docs, **_kwargs: "context")
    monkeypatch.setattr(retrieval_module, "compress_context", lambda text, **_kwargs: text)

    trace_context = {"request_id": "req-rag", "agent_mode": "rag_only"}
    answer, sources, web_sources = chat._invoke_rag_only(
        prompt="What is Barolo?",
        cfg=_build_cfg(),
        model=MagicMock(),
        retriever=_StubRetriever(),
        reranker=None,
        message_history=[],
        enable_rag=True,
        n_results_override=2,
        trace_context=trace_context,
    )

    assert answer == "Answer with citation [1]"
    assert len(sources) == 1
    assert web_sources == []
    assert captured_llm_trace_context == trace_context
    assert any("retriever_type" in attrs for attrs in captured_span_attributes)
    assert any("n_docs_retrieved" in attrs for attrs in captured_span_attributes)
