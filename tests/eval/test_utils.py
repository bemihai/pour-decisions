"""Unit tests for eval utility helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from src.agents.guardrails import EMPTY_FINAL_ANSWER_EVENT_CODE, EMPTY_FINAL_ANSWER_RETRY
from src.eval.models import GoldenSample
from src.eval.utils import (
    extract_eval_config_snapshot,
    get_git_metadata,
    resolve_eval_model_config,
    resolve_execution_model_config,
    run_agent_sample_sync,
)


def test_agent_sample_preserves_empty_terminal_outcome_and_calls() -> None:
    """Evaluation retains the failure signal and actual attempts beside safe text."""
    agent = Mock()
    agent.invoke.return_value = {
        "messages": [AIMessage(content="", usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16})],
        "final_answer": EMPTY_FINAL_ANSWER_RETRY,
        "llm_call_count": 2,
        "terminal_outcome": EMPTY_FINAL_ANSWER_EVENT_CODE,
        "guardrail_events": [{"code": EMPTY_FINAL_ANSWER_EVENT_CODE}],
    }
    sample = GoldenSample(
        id="rag_only_001",
        question="What is tannin?",
        category="rag_only",
        difficulty="easy",
        expected_facts=[],
        ground_truth="Tannin is a wine compound.",
        tags=["wine"],
    )

    result = run_agent_sample_sync(agent, sample)

    assert result.answer == EMPTY_FINAL_ANSWER_RETRY
    assert result.terminal_outcome == EMPTY_FINAL_ANSWER_EVENT_CODE
    assert result.llm_call_count == 2
    assert result.token_usage == {"input_tokens": 12, "output_tokens": 4}


def _make_config() -> SimpleNamespace:
    """Build a minimal config object for eval utility tests."""
    return SimpleNamespace(
        model=SimpleNamespace(
            provider="ollama",
            name="gemma4:31b",
        ),
        chroma=SimpleNamespace(
            settings=SimpleNamespace(embedder="text-embedding-3-small"),
            retrieval=SimpleNamespace(
                n_results=5,
                similarity_threshold=0.3,
                use_deduplication=True,
                deduplication_threshold=0.9,
                enable_hybrid=True,
                semantic_candidate_pool=25,
                bm25_candidate_pool=25,
                reranker_input_limit=50,
                enable_reranking=True,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                rerank_top_k=5,
                rerank_threshold=None,
                min_retrieval_confidence=0.3,
                enable_compression=False,
                compression_max_chars=8000,
                enable_metadata_boost=True,
                metadata_boost_factor=0.1,
            ),
        ),
        eval=SimpleNamespace(
            execution_provider="ollama",
            execution_model="gemma4:31b",
            ollama=SimpleNamespace(base_url="https://ollama.com"),
            execution_timeout_seconds=60,
            sample_timeout_seconds=30,
            skip_cellar_samples_if_empty=True,
            validate_tag_filters=True,
            ragas=SimpleNamespace(
                evaluator_provider="ollama",
                evaluator_model="gemma4:31b",
                temperature=0.0,
                reasoning=False,
                num_predict=2048,
                timeout_seconds=120,
                max_retries=1,
                max_workers=1,
                metrics=["faithfulness", "context_precision"],
            ),
        ),
    )


def test_extract_eval_config_snapshot_includes_retrieval_and_eval_settings() -> None:
    """Snapshot should capture retrieval-affecting and eval-affecting settings."""
    snapshot = extract_eval_config_snapshot(_make_config())

    assert snapshot["model"] == "gemma4:31b"
    assert snapshot["provider"] == "ollama"
    assert snapshot["eval_provider"] == "ollama"
    assert snapshot["eval_model"] == "gemma4:31b"
    assert snapshot["embedder"] == "text-embedding-3-small"
    assert snapshot["retrieval"]["n_results"] == 5
    assert snapshot["retrieval"]["similarity_threshold"] == 0.3
    assert snapshot["retrieval"]["enable_hybrid"] is True
    assert snapshot["retrieval"]["semantic_candidate_pool"] == 25
    assert snapshot["retrieval"]["bm25_candidate_pool"] == 25
    assert snapshot["retrieval"]["reranker_input_limit"] == 50
    assert snapshot["retrieval"]["enable_reranking"] is True
    assert snapshot["retrieval"]["rerank_threshold"] is None
    assert snapshot["retrieval"]["min_retrieval_confidence"] == 0.3
    assert snapshot["retrieval"]["enable_metadata_boost"] is True
    assert snapshot["eval"]["ragas_metrics"] == ["faithfulness", "context_precision"]
    assert snapshot["eval"]["ragas_temperature"] == 0.0
    assert snapshot["eval"]["ragas_reasoning"] is False
    assert snapshot["eval"]["ragas_num_predict"] == 2048
    assert snapshot["eval"]["ragas_timeout_seconds"] == 120.0
    assert snapshot["eval"]["ragas_max_retries"] == 1
    assert snapshot["eval"]["ragas_max_workers"] == 1
    assert snapshot["eval"]["retrieval_k_values"] == [3, 5]
    assert snapshot["eval"]["sample_timeout_seconds"] == 30.0
    assert snapshot["eval"]["execution_timeout_seconds"] == 60.0
    assert snapshot["eval"]["skip_cellar_samples_if_empty"] is True
    assert snapshot["eval"]["validate_tag_filters"] is True


def test_extract_eval_config_snapshot_preserves_numeric_zero_threshold() -> None:
    """A numeric zero threshold must remain distinct from the null default."""
    config = _make_config()
    config.chroma.retrieval.rerank_threshold = 0.0

    snapshot = extract_eval_config_snapshot(config)

    assert snapshot["retrieval"]["rerank_threshold"] == 0.0


def test_resolve_execution_model_config_includes_ollama_timeout() -> None:
    """Execution model config should use the direct Cloud call timeout."""
    provider, model_name, kwargs = resolve_execution_model_config(_make_config())

    assert provider == "ollama"
    assert model_name == "gemma4:31b"
    assert kwargs["base_url"] == "https://ollama.com"
    assert kwargs["timeout"] == 60.0


def test_resolve_eval_model_config_includes_ollama_timeout() -> None:
    """Evaluator model config should forward eval timeout into Ollama kwargs."""
    provider, model_name, kwargs = resolve_eval_model_config(_make_config())

    assert provider == "ollama"
    assert model_name == "gemma4:31b"
    assert kwargs["base_url"] == "https://ollama.com"
    assert kwargs["timeout"] == 120.0
    assert kwargs["temperature"] == 0.0
    assert kwargs["reasoning"] is False
    assert kwargs["num_predict"] == 2048


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
