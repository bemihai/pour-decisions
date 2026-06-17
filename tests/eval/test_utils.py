"""Unit tests for eval utility helpers."""

from __future__ import annotations

from types import SimpleNamespace

from src.eval.utils import extract_eval_config_snapshot, get_git_metadata


def _make_config() -> object:
    """Build a minimal config object for eval utility tests."""
    return SimpleNamespace(
        model=SimpleNamespace(
            provider="google",
            name="gemini-2.5-flash",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
        ),
        chroma=SimpleNamespace(
            settings=SimpleNamespace(embedder="text-embedding-3-small"),
            retrieval=SimpleNamespace(
                n_results=5,
                similarity_threshold=0.3,
                use_deduplication=True,
                deduplication_threshold=0.9,
                enable_hybrid=True,
                hybrid_vector_weight=0.7,
                hybrid_keyword_weight=0.3,
                enable_reranking=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                rerank_top_k=5,
                enable_compression=False,
                compression_max_chars=8000,
                enable_metadata_boost=True,
                metadata_boost_factor=0.1,
            ),
        ),
        eval=SimpleNamespace(
            execution_provider="ollama",
            execution_model="llama3.2:3b",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
            sample_timeout_seconds=30,
            skip_cellar_samples_if_empty=True,
            validate_tag_filters=True,
            ragas=SimpleNamespace(
                evaluator_provider="ollama",
                evaluator_model="gemma2:2b",
                metrics=["faithfulness", "context_precision"],
            ),
        ),
    )


def test_extract_eval_config_snapshot_includes_retrieval_and_eval_settings() -> None:
    """Snapshot should capture retrieval-affecting and eval-affecting settings."""
    snapshot = extract_eval_config_snapshot(_make_config())

    assert snapshot["model"] == "llama3.2:3b"
    assert snapshot["provider"] == "ollama"
    assert snapshot["eval_provider"] == "ollama"
    assert snapshot["eval_model"] == "gemma2:2b"
    assert snapshot["embedder"] == "text-embedding-3-small"
    assert snapshot["retrieval"]["n_results"] == 5
    assert snapshot["retrieval"]["similarity_threshold"] == 0.3
    assert snapshot["retrieval"]["enable_hybrid"] is True
    assert snapshot["retrieval"]["enable_reranking"] is True
    assert snapshot["retrieval"]["enable_metadata_boost"] is True
    assert snapshot["eval"]["ragas_metrics"] == ["faithfulness", "context_precision"]
    assert snapshot["eval"]["sample_timeout_seconds"] == 30.0
    assert snapshot["eval"]["skip_cellar_samples_if_empty"] is True
    assert snapshot["eval"]["validate_tag_filters"] is True


def test_get_git_metadata_returns_safe_fallbacks_when_git_unavailable(mocker) -> None:
    """Git metadata helper should fail safe when git commands are unavailable."""
    mocker.patch("src.eval.utils._run_git_command", return_value=None)

    metadata = get_git_metadata()

    assert metadata["sha"] == "unknown"
    assert metadata["branch"] == "unknown"
    assert metadata["is_dirty"] is None


def test_get_git_metadata_distinguishes_clean_and_dirty_states(mocker) -> None:
    """Git metadata helper should record branch and dirty working tree state."""
    values = {
        ("git", "rev-parse", "--short", "HEAD"): "abc123",
        ("git", "rev-parse", "--abbrev-ref", "HEAD"): "feature/eval",
        ("git", "status", "--porcelain"): " M src/eval/runner.py",
    }

    def fake_run_git_command(args: list[str]) -> str | None:
        return values.get(tuple(args))

    mocker.patch("src.eval.utils._run_git_command", side_effect=fake_run_git_command)

    metadata = get_git_metadata()

    assert metadata["sha"] == "abc123"
    assert metadata["branch"] == "feature/eval"
    assert metadata["is_dirty"] is True
