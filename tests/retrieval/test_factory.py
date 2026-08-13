"""Unit tests for shared production retrieval resource factories."""

from pathlib import Path
from types import SimpleNamespace

from src.retrieval.factory import build_reranker_from_config, build_retriever_from_config


def _config(*, enabled: bool = True, model_name: str = "test-reranker") -> SimpleNamespace:
    """Build the config fields required by the reranker factory."""
    return SimpleNamespace(
        chroma=SimpleNamespace(
            retrieval=SimpleNamespace(
                enable_reranking=enabled,
                reranker_model=model_name,
            )
        )
    )


def _retriever_config(*, validate_bm25_sync: bool) -> SimpleNamespace:
    """Build the config fields required by the shared retriever factory."""
    return SimpleNamespace(
        chroma=SimpleNamespace(
            client=SimpleNamespace(host="localhost", port=8100),
            collections=[SimpleNamespace(name="wine_books")],
            settings=SimpleNamespace(embedder="test-embedder", batch_size=100),
            indexing=SimpleNamespace(
                bm25=SimpleNamespace(sync_manifest_path="chroma-data/bm25_index.meta.json")
            ),
            retrieval=SimpleNamespace(
                n_results=5,
                similarity_threshold=0.3,
                enable_hybrid=True,
                semantic_candidate_pool=25,
                bm25_candidate_pool=25,
                reranker_input_limit=50,
                bm25_index_path="chroma-data/bm25_index.pkl",
                validate_bm25_sync=validate_bm25_sync,
            ),
        )
    )


def test_build_reranker_from_config_returns_none_when_disabled(mocker) -> None:
    """Disabled reranking should not initialize a cross-encoder."""
    reranker_class = mocker.patch("src.retrieval.factory.DocumentReranker")

    result = build_reranker_from_config(_config(enabled=False))

    assert result is None
    reranker_class.assert_not_called()


def test_build_reranker_from_config_uses_configured_model(mocker) -> None:
    """Enabled reranking should initialize the configured shared resource."""
    reranker = object()
    reranker_class = mocker.patch("src.retrieval.factory.DocumentReranker", return_value=reranker)

    result = build_reranker_from_config(_config(model_name="cross-encoder/test"))

    assert result is reranker
    reranker_class.assert_called_once_with(model_name="cross-encoder/test")


def test_build_reranker_from_config_fails_open_when_model_is_unavailable(mocker) -> None:
    """API and eval should both continue without reranking when initialization fails."""
    mocker.patch(
        "src.retrieval.factory.DocumentReranker",
        side_effect=RuntimeError("model unavailable"),
    )

    assert build_reranker_from_config(_config()) is None


def test_build_retriever_rejects_stale_bm25_manifest(mocker) -> None:
    """A synchronization mismatch should return the configured vector retriever."""
    config = _retriever_config(validate_bm25_sync=True)
    collection = object()
    vector_retriever = SimpleNamespace(collection=collection)
    mocker.patch("src.retrieval.factory.initialize_chroma_client", return_value=object())
    mocker.patch("src.retrieval.factory.ChromaRetriever", return_value=vector_retriever)
    bm25 = mocker.MagicMock()
    bm25.__len__.return_value = 3
    mocker.patch("src.retrieval.factory.BM25Index", return_value=bm25)
    validate = mocker.patch(
        "src.chroma.bm25_builder.validate_bm25_sync",
        return_value=(False, "manifest hash mismatch"),
    )
    hybrid_retriever = mocker.patch("src.retrieval.factory.HybridRetriever")

    result = build_retriever_from_config(config)

    assert result is vector_retriever
    validate.assert_called_once_with(
        collection=collection,
        collection_name="wine_books",
        bm25=bm25,
        index_path=Path("chroma-data/bm25_index.pkl"),
        manifest_path=Path("chroma-data/bm25_index.meta.json"),
        batch_size=100,
    )
    hybrid_retriever.assert_not_called()


def test_build_retriever_allows_legacy_index_when_validation_disabled(mocker) -> None:
    """The explicit compatibility flag should retain pre-manifest hybrid loading."""
    config = _retriever_config(validate_bm25_sync=False)
    vector_retriever = SimpleNamespace(collection=object())
    hybrid = object()
    mocker.patch("src.retrieval.factory.initialize_chroma_client", return_value=object())
    mocker.patch("src.retrieval.factory.ChromaRetriever", return_value=vector_retriever)
    bm25 = mocker.MagicMock()
    bm25.__len__.return_value = 3
    mocker.patch("src.retrieval.factory.BM25Index", return_value=bm25)
    validate = mocker.patch("src.chroma.bm25_builder.validate_bm25_sync")
    hybrid_retriever = mocker.patch("src.retrieval.factory.HybridRetriever", return_value=hybrid)

    result = build_retriever_from_config(config)

    assert result is hybrid
    validate.assert_not_called()
    hybrid_retriever.assert_called_once_with(
        vector_retriever=vector_retriever,
        bm25_index=bm25,
        semantic_candidate_pool=25,
        bm25_candidate_pool=25,
        reranker_input_limit=50,
    )
