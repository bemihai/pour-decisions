"""Configuration and accounting helpers for synchronous agent call budgets."""

from dataclasses import dataclass
from typing import Mapping

from omegaconf import DictConfig, OmegaConf


DEFAULT_MAX_LLM_CALLS_PER_QUERY = 5
DEFAULT_MAX_GRAPH_STEPS_PER_QUERY = 30
MIN_GRAPH_STEPS_PER_QUERY = 2
CALL_BUDGET_EVENT_CODE = "call_budget_exhausted"


@dataclass(frozen=True)
class CallBudgetConfig:
    """Validated synchronous call-budget configuration."""

    enabled: bool = True
    max_llm_calls_per_query: int = DEFAULT_MAX_LLM_CALLS_PER_QUERY
    max_graph_steps_per_query: int = DEFAULT_MAX_GRAPH_STEPS_PER_QUERY


def prepare_model_call(
    state: Mapping[str, object],
    config: CallBudgetConfig,
) -> dict[str, object]:
    """Reserve one attempted model call or record budget exhaustion.

    This helper is intended to run in its own graph node immediately before a
    model node. Persisting the increment first keeps failed provider attempts
    visible to checkpointed graph state.

    Args:
        state: Current agent graph state.
        config: Validated call-budget settings.

    Returns:
        Complete replacement values for the call count and guardrail events.
    """
    call_count = state.get("llm_call_count", 0)
    if type(call_count) is not int or call_count < 0:
        call_count = 0

    raw_events = state.get("guardrail_events", [])
    events = list(raw_events) if isinstance(raw_events, list) else []
    if config.enabled and call_count >= config.max_llm_calls_per_query:
        events.append(
            {
                "code": CALL_BUDGET_EVENT_CODE,
                "llm_call_count": call_count,
                "limit": config.max_llm_calls_per_query,
            }
        )
        return {"llm_call_count": call_count, "guardrail_events": events}

    return {"llm_call_count": call_count + 1, "guardrail_events": events}


def call_budget_triggered(state: Mapping[str, object]) -> bool:
    """Return whether the latest guardrail event exhausted the call budget."""
    events = state.get("guardrail_events", [])
    if not isinstance(events, list) or not events:
        return False
    latest_event = events[-1]
    return isinstance(latest_event, dict) and latest_event.get("code") == CALL_BUDGET_EVENT_CODE


def load_call_budget_config(config: DictConfig | None) -> CallBudgetConfig:
    """Resolve and validate the intelligent-agent call-budget configuration.

    Args:
        config: Application configuration. Defaults are used when it is absent.

    Returns:
        Validated call-budget settings.

    Raises:
        ValueError: If a setting has an invalid type or range.
    """
    enabled = (
        OmegaConf.select(config, "agents.guardrails.call_budget.enabled", default=True)
        if config
        else True
    )
    max_llm_calls = (
        OmegaConf.select(
            config,
            "agents.guardrails.call_budget.max_llm_calls_per_query",
            default=DEFAULT_MAX_LLM_CALLS_PER_QUERY,
        )
        if config
        else DEFAULT_MAX_LLM_CALLS_PER_QUERY
    )
    max_graph_steps = (
        OmegaConf.select(
            config,
            "agents.guardrails.call_budget.max_graph_steps_per_query",
            default=DEFAULT_MAX_GRAPH_STEPS_PER_QUERY,
        )
        if config
        else DEFAULT_MAX_GRAPH_STEPS_PER_QUERY
    )

    if type(enabled) is not bool:
        raise ValueError("agents.guardrails.call_budget.enabled must be a boolean")
    if type(max_llm_calls) is not int or max_llm_calls < 0:
        raise ValueError(
            "agents.guardrails.call_budget.max_llm_calls_per_query must be a non-negative integer"
        )
    if type(max_graph_steps) is not int or max_graph_steps < MIN_GRAPH_STEPS_PER_QUERY:
        raise ValueError(
            "agents.guardrails.call_budget.max_graph_steps_per_query must be an integer of at least "
            f"{MIN_GRAPH_STEPS_PER_QUERY}"
        )

    return CallBudgetConfig(
        enabled=enabled,
        max_llm_calls_per_query=max_llm_calls,
        max_graph_steps_per_query=max_graph_steps,
    )
