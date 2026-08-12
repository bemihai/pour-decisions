"""Tests for contextual cross-encoder scoring with clean returned evidence."""

from src.retrieval.reranker import DocumentReranker


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
