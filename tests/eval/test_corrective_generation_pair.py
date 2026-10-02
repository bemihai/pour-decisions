"""Tests for the bounded M12 cloud generation and faithfulness gate."""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from omegaconf import OmegaConf
import pytest

from src.eval.scripts.corrective_generation_pair import (
    APPROVED_PROVIDER_ATTEMPT_CEILING,
    FROZEN_SAMPLE_IDS,
    ProviderAttemptBudget,
    ProviderUsageCallback,
    evaluate_cloud_gate,
    validate_cloud_contract,
)


def _result(sample_id: str, faithfulness: float) -> dict:
    return {
        "id": sample_id,
        "question": "Question?",
        "status": "passed",
        "answer": "Supported answer [1].",
        "scores": {"faithfulness": faithfulness},
        "metric_errors": {},
        "rag_feature_flags": {"web_fallback": False},
        "context_chunks": [
            {
                "id": f"{sample_id}-chunk",
                "text": "Evidence",
                "metadata": {"source": "book.pdf"},
            }
        ],
        "rag_sources": [
            {
                "name": "book",
                "page": 1,
                "relevance": 0.9,
                "chunk_id": f"{sample_id}-chunk",
                "metadata": {"source": "book.pdf"},
            }
        ],
        "correction": {
            "enabled": True,
            "eligible": False,
            "attempt_count": 0,
            "model_attempts": 0,
        },
    }


def _arm(faithfulness: float) -> dict:
    return {
        "execution_usage": {
            "completions": 6,
            "errors": 0,
            "token_usage_reports": 6,
            "input_tokens": 600,
            "output_tokens": 120,
            "total_tokens": 720,
        },
        "judge_usage": {
            "completions": 12,
            "errors": 0,
            "token_usage_reports": 12,
            "input_tokens": 1200,
            "output_tokens": 240,
            "total_tokens": 1440,
        },
        "results": [_result(sample_id, faithfulness) for sample_id in sorted(FROZEN_SAMPLE_IDS)],
    }


def test_provider_attempt_budget_fails_before_exceeding_ceiling() -> None:
    budget = ProviderAttemptBudget(maximum=2)

    budget.reserve("generation")
    budget.reserve("judge")

    with pytest.raises(RuntimeError, match="ceiling reached"):
        budget.reserve("judge")
    assert budget.snapshot()["attempts"] == 2


def test_usage_callback_records_provider_reported_tokens() -> None:
    budget = ProviderAttemptBudget()
    callback = ProviderUsageCallback("generation", budget)
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="answer",
                        usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
                    )
                )
            ]
        ]
    )

    callback.on_chat_model_start({}, [[]])
    callback.on_llm_end(response)

    assert budget.snapshot()["attempts"] == 1
    assert callback.snapshot() == {
        "completions": 1,
        "errors": 0,
        "token_usage_reports": 1,
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }


def test_cloud_contract_uses_approved_models_and_stays_below_ceiling() -> None:
    config = OmegaConf.create(
        {
            "eval": {
                "execution_provider": "ollama",
                "execution_model": "gemma4:cloud",
                "sample_timeout_seconds": 300,
                "ollama": {"base_url": "http://localhost:11434"},
                "ragas": {
                    "evaluator_provider": "ollama",
                    "evaluator_model": "gemma4:31b-cloud",
                    "temperature": 0.0,
                    "reasoning": False,
                    "num_predict": 2048,
                    "timeout_seconds": 120,
                },
            }
        }
    )

    contract = validate_cloud_contract(config)

    assert contract["estimated_retry_inclusive_maximum"] == 84
    assert contract["approved_ceiling"] == APPROVED_PROVIDER_ATTEMPT_CEILING


def test_cloud_gate_passes_common_faithfulness_with_one_point_regression() -> None:
    gate = evaluate_cloud_gate(
        _arm(0.80),
        _arm(0.79),
        {"attempts": 36, "attempts_by_role": {"generation": 12, "judge": 24}, "maximum": 108},
    )

    assert gate["decision"] == "pass"
    assert gate["coverage"]["common_faithfulness_ids"] == 6
    assert gate["faithfulness"]["enabled_minus_disabled"] == pytest.approx(-0.01)
    assert all(gate["checks"].values())


def test_cloud_gate_fails_when_enabled_arm_adds_blank_answer() -> None:
    disabled = _arm(0.8)
    enabled = _arm(0.8)
    enabled["results"][0]["answer"] = ""

    gate = evaluate_cloud_gate(
        disabled,
        enabled,
        {"attempts": 36, "attempts_by_role": {"generation": 12, "judge": 24}, "maximum": 108},
    )

    assert gate["decision"] == "fail"
    assert gate["checks"]["no_new_blank_answers"] is False
