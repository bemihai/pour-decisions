"""Tests for validated asynchronous tool-execution policy configuration."""

from dataclasses import FrozenInstanceError
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import pytest

from src.agents.guardrails import (
    ToolExecutionConfig,
    ToolRetryConfig,
    ToolTimeoutConfig,
    load_tool_execution_config,
)
from src.agents.tools.registry import CostClass, LatencyClass


def _valid_config() -> DictConfig:
    """Build the complete reviewed policy configuration for mutation in tests."""
    return OmegaConf.create(
        {
            "agents": {
                "guardrails": {
                    "tool_execution": {
                        "enabled": True,
                        "max_concurrent_calls": 4,
                        "timeout_seconds": {"fast": 10, "slow": 30},
                        "retry": {
                            "enabled": True,
                            "max_attempts": 2,
                            "delay_seconds": 0.1,
                            "min_remaining_seconds": 1.0,
                            "allowed_cost_classes": ["free"],
                        },
                    }
                }
            }
        }
    )


def _config_with(path: str, value: object) -> DictConfig:
    """Return valid configuration with one tool-execution value replaced."""
    config = _valid_config()
    OmegaConf.update(config, f"agents.guardrails.tool_execution.{path}", value, merge=False)
    return config


def test_absent_configuration_uses_frozen_reviewed_defaults() -> None:
    """Missing application policy should resolve to the approved rollout values."""
    expected = ToolExecutionConfig()

    assert load_tool_execution_config(None) == expected
    assert load_tool_execution_config(OmegaConf.create({})) == expected
    assert expected.timeout_seconds == ToolTimeoutConfig(fast=10.0, slow=30.0)
    assert expected.retry == ToolRetryConfig(
        enabled=True,
        max_attempts=2,
        delay_seconds=0.1,
        min_remaining_seconds=1.0,
        allowed_cost_classes=frozenset({CostClass.FREE}),
    )

    with pytest.raises(FrozenInstanceError):
        expected.max_concurrent_calls = 5


def test_project_configuration_matches_reviewed_defaults() -> None:
    """The checked-in rollout block should load to the same typed policy."""
    config = OmegaConf.load(Path(__file__).parents[3] / "app_config.yml")

    assert load_tool_execution_config(config) == ToolExecutionConfig()


def test_deadline_lookup_uses_explicit_latency_enum() -> None:
    """Deadline lookup should cover exactly the registry latency classes."""
    deadlines = ToolTimeoutConfig(fast=2.0, slow=7.0)

    assert deadlines.for_latency_class(LatencyClass.FAST) == 2.0
    assert deadlines.for_latency_class(LatencyClass.SLOW) == 7.0

    with pytest.raises(ValueError, match="Unsupported tool latency class"):
        deadlines.for_latency_class("fast")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        ("enabled", 1, "enabled must be a boolean"),
        ("retry.enabled", "true", "retry.enabled must be a boolean"),
        ("max_concurrent_calls", True, "integer of at least 1"),
        ("max_concurrent_calls", 0, "integer of at least 1"),
        ("max_concurrent_calls", 1.0, "integer of at least 1"),
        ("retry.max_attempts", True, "integer from 1 to 2"),
        ("retry.max_attempts", 0, "integer from 1 to 2"),
        ("retry.max_attempts", 3, "integer from 1 to 2"),
        ("retry.delay_seconds", True, "must be a number"),
        ("retry.delay_seconds", -0.1, "finite non-negative number"),
        ("retry.min_remaining_seconds", False, "must be a number"),
        ("retry.min_remaining_seconds", 0, "finite positive number"),
    ),
)
def test_scalar_policy_values_are_strictly_validated(
    path: str,
    value: object,
    message: str,
) -> None:
    """Boolean substitutes and unsupported numeric ranges must fail closed."""
    with pytest.raises(ValueError, match=message):
        load_tool_execution_config(_config_with(path, value))


@pytest.mark.parametrize(
    "timeout_seconds",
    (
        {"fast": 10},
        {"fast": 10, "slow": 30, "medium": 20},
        [10, 30],
    ),
)
def test_deadline_mapping_requires_exact_latency_keys(timeout_seconds: object) -> None:
    """Partial, extended, or non-mapping deadline declarations must fail."""
    with pytest.raises(ValueError, match="fast and slow|keys: fast, slow"):
        load_tool_execution_config(_config_with("timeout_seconds", timeout_seconds))


@pytest.mark.parametrize("value", (True, 0, -1, float("inf"), float("nan"), "10"))
@pytest.mark.parametrize("latency", ("fast", "slow"))
def test_deadlines_require_finite_positive_numbers(latency: str, value: object) -> None:
    """Each latency-class deadline should reject invalid numeric input."""
    with pytest.raises(ValueError, match=rf"timeout_seconds.{latency}"):
        load_tool_execution_config(_config_with(f"timeout_seconds.{latency}", value))


@pytest.mark.parametrize(
    "allowed_cost_classes",
    (
        "free",
        None,
        ["unknown"],
        [1],
    ),
)
def test_retry_cost_allowlist_rejects_invalid_values(allowed_cost_classes: object) -> None:
    """Retry authorization must resolve only reviewed cost-class names."""
    with pytest.raises(ValueError, match="allowed_cost_classes"):
        load_tool_execution_config(
            _config_with("retry.allowed_cost_classes", allowed_cost_classes)
        )


def test_retry_cost_allowlist_resolves_to_immutable_enums() -> None:
    """Configured cost classes should be deduplicated into a frozen enum set."""
    config = _config_with("retry.allowed_cost_classes", ["free", "cheap", "free"])

    policy = load_tool_execution_config(config)

    assert policy.retry.allowed_cost_classes == frozenset({CostClass.FREE, CostClass.CHEAP})


@pytest.mark.parametrize(
    ("delay", "minimum", "fast", "slow"),
    (
        (1.0, 1.0, 2.0, 30.0),
        (1.0, 2.0, 10.0, 3.0),
    ),
)
def test_retry_budget_must_fit_below_both_deadlines(
    delay: float,
    minimum: float,
    fast: float,
    slow: float,
) -> None:
    """Retry delay and useful remaining time must fit inside every deadline."""
    config = _valid_config()
    OmegaConf.update(config, "agents.guardrails.tool_execution.retry.delay_seconds", delay)
    OmegaConf.update(
        config,
        "agents.guardrails.tool_execution.retry.min_remaining_seconds",
        minimum,
    )
    OmegaConf.update(config, "agents.guardrails.tool_execution.timeout_seconds.fast", fast)
    OmegaConf.update(config, "agents.guardrails.tool_execution.timeout_seconds.slow", slow)

    with pytest.raises(ValueError, match="must be below both deadlines"):
        load_tool_execution_config(config)


def test_disabled_policy_still_validates_and_preserves_m9a_independence() -> None:
    """Disabling M9B should be explicit without making its data mutable or malformed."""
    policy = load_tool_execution_config(_config_with("enabled", False))

    assert policy.enabled is False
    assert policy.retry.enabled is True
    assert policy.max_concurrent_calls == 4
