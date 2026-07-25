"""Unit tests for shared production retrieval resource factories."""

from types import SimpleNamespace

from src.retrieval.factory import build_reranker_from_config


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
