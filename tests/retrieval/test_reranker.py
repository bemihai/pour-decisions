"""Tests for contextual cross-encoder scoring with clean returned evidence."""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.retrieval.reranker import AsyncLazyDocumentReranker, DocumentReranker


def test_reranker_scores_contextual_text_but_returns_clean_document(mocker) -> None:
    """Pronoun-heavy evidence should be scored with its validated Nebbiolo parent."""
    model = mocker.Mock()
    model.predict.return_value = [2.5]
    mocker.patch("src.retrieval.reranker._get_reranker", return_value=model)
    reranker = DocumentReranker("test-model")
    body = "It is highly tannic and acidic, with aromas of roses and tar."
    documents = [
        {
            "id": "nebbiolo",
            "document": body,
            "metadata": {
                "document_title": "Grapes & Wines",
                "chapter": "NEBBIOLO",
                "section": "taste",
                "structural_role": "prose",
            },
        }
    ]

    results = reranker.rerank("nebbiolo flavour characteristics", documents, top_k=1)

    model.predict.assert_called_once_with(
        [("nebbiolo flavour characteristics", f"Grapes & Wines > NEBBIOLO > taste\n\n{body}")]
    )
    assert results[0]["document"] == body
    assert results[0]["rerank_score"] == 2.5


def test_threshold_reranking_excludes_corrupt_context(mocker) -> None:
    """The threshold path must use the same validated representation."""
    model = mocker.Mock()
    model.predict.return_value = [0.5]
    mocker.patch("src.retrieval.reranker._get_reranker", return_value=model)
    reranker = DocumentReranker("test-model")
    documents = [
        {
            "id": "clean",
            "document": "Readable evidence.",
            "metadata": {"chapter": "(cid:42) (cid:57)", "structural_role": "prose"},
        }
    ]

    results = reranker.rerank_with_threshold("wine question", documents, threshold=0.0, top_k=1)

    model.predict.assert_called_once_with([("wine question", "Readable evidence.")])
    assert [result["id"] for result in results] == ["clean"]


@pytest.mark.asyncio
async def test_lazy_reranker_initializes_once_for_concurrent_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent callers should await one shared cross-encoder construction."""
    reranker = object()
    constructor = MagicMock(return_value=reranker)
    monkeypatch.setattr("src.retrieval.reranker.DocumentReranker", constructor)
    lazy = AsyncLazyDocumentReranker("test-model")

    first, second = await asyncio.gather(lazy.resolve(), lazy.resolve())

    assert first is second is reranker
    constructor.assert_called_once_with(model_name="test-model")


@pytest.mark.asyncio
async def test_lazy_reranker_failure_is_cached_and_preserves_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable reranking should fail open and should not retry during the lifespan."""
    constructor = MagicMock(side_effect=RuntimeError("model unavailable"))
    monkeypatch.setattr("src.retrieval.reranker.DocumentReranker", constructor)
    lazy = AsyncLazyDocumentReranker("test-model")
    documents = [{"id": "one"}, {"id": "two"}]

    assert await lazy.arerank("Barolo", documents, top_k=1) == documents[:1]
    assert await lazy.resolve() is None
    constructor.assert_called_once_with(model_name="test-model")
