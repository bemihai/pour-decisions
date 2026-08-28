"""Tests for synchronous intelligent-agent call budgets."""

import pytest
from omegaconf import OmegaConf

from src.agents.guardrails.budget import (
    CALL_BUDGET_EVENT_CODE,
    CallBudgetConfig,
    call_budget_triggered,
    load_call_budget_config,
    prepare_model_call,
)


def _config(*, enabled: object = True, max_llm_calls: object = 5, max_graph_steps: object = 30):
    """Build a focused call-budget configuration."""
    return OmegaConf.create(
        {
            "agents": {
                "guardrails": {
                    "call_budget": {
                        "enabled": enabled,
                        "max_llm_calls_per_query": max_llm_calls,
                        "max_graph_steps_per_query": max_graph_steps,
                    }
                }
            }
        }
    )


def test_call_budget_defaults_are_valid() -> None:
    """Missing configuration should resolve to the reviewed defaults."""
    assert load_call_budget_config(OmegaConf.create({})) == CallBudgetConfig()


@pytest.mark.parametrize("enabled", [0, 1, "true"])
def test_call_budget_enabled_requires_boolean(enabled: object) -> None:
    """The feature switch should not accept truthy non-booleans."""
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        load_call_budget_config(_config(enabled=enabled))


@pytest.mark.parametrize("max_llm_calls", [True, False, -1, 1.0, "5"])
def test_model_call_limit_requires_non_negative_integer(max_llm_calls: object) -> None:
    """Attempt limits should reject booleans, negatives, and non-integers."""
    with pytest.raises(ValueError, match="max_llm_calls_per_query"):
        load_call_budget_config(_config(max_llm_calls=max_llm_calls))


@pytest.mark.parametrize("max_graph_steps", [True, False, -1, 0, 1, 2, 3, 4.0, "30"])
def test_graph_step_limit_must_allow_a_terminal_path(max_graph_steps: object) -> None:
    """The graph backstop should permit relevance, budget, a direct answer, and END."""
    with pytest.raises(ValueError, match="max_graph_steps_per_query"):
        load_call_budget_config(_config(max_graph_steps=max_graph_steps))


def test_disabled_call_budget_still_validates_limits() -> None:
    """Disabled behavior should retain safe, ready-to-enable configuration."""
    with pytest.raises(ValueError, match="max_llm_calls_per_query"):
        load_call_budget_config(_config(enabled=False, max_llm_calls=-1))


@pytest.mark.parametrize(
    ("current_count", "limit", "expected_count", "triggered"),
    [
        (0, 0, 0, True),
        (3, 5, 4, False),
        (4, 5, 5, False),
        (5, 5, 5, True),
        (6, 5, 6, True),
    ],
)
def test_prepare_model_call_enforces_boundaries(
    current_count: int,
    limit: int,
    expected_count: int,
    triggered: bool,
) -> None:
    """The reservation step should enforce zero, N-1, N, and N+1 boundaries."""
    state = {"llm_call_count": current_count, "guardrail_events": []}
    result = prepare_model_call(state, CallBudgetConfig(max_llm_calls_per_query=limit))

    assert result["llm_call_count"] == expected_count
    assert call_budget_triggered(result) is triggered
    if triggered:
        assert result["guardrail_events"][-1]["code"] == CALL_BUDGET_EVENT_CODE


def test_disabled_budget_counts_without_blocking_attempts() -> None:
    """Disabling enforcement should retain attempted-call accounting."""
    result = prepare_model_call(
        {"llm_call_count": 8, "guardrail_events": []},
        CallBudgetConfig(enabled=False, max_llm_calls_per_query=0),
    )

    assert result == {"llm_call_count": 9, "guardrail_events": []}
