"""Validated configuration for asynchronous tool-execution policy."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from omegaconf import DictConfig, OmegaConf

from src.agents.tools.registry import CostClass, LatencyClass


DEFAULT_MAX_CONCURRENT_TOOL_CALLS = 4
DEFAULT_FAST_TOOL_TIMEOUT_SECONDS = 10.0
DEFAULT_SLOW_TOOL_TIMEOUT_SECONDS = 30.0
DEFAULT_TOOL_MAX_ATTEMPTS = 2
DEFAULT_TOOL_RETRY_DELAY_SECONDS = 0.1
DEFAULT_TOOL_RETRY_MIN_REMAINING_SECONDS = 1.0


@dataclass(frozen=True)
class ToolTimeoutConfig:
    """Validated response deadlines for each supported latency class."""

    fast: float = DEFAULT_FAST_TOOL_TIMEOUT_SECONDS
    slow: float = DEFAULT_SLOW_TOOL_TIMEOUT_SECONDS

    def for_latency_class(self, latency_class: LatencyClass) -> float:
        """Return the deadline assigned to one explicit latency class."""
        if latency_class is LatencyClass.FAST:
            return self.fast
        if latency_class is LatencyClass.SLOW:
            return self.slow
        raise ValueError(f"Unsupported tool latency class: {latency_class!r}")


@dataclass(frozen=True)
class ToolRetryConfig:
    """Validated eligibility and timing policy for one optional retry."""

    enabled: bool = True
    max_attempts: int = DEFAULT_TOOL_MAX_ATTEMPTS
    delay_seconds: float = DEFAULT_TOOL_RETRY_DELAY_SECONDS
    min_remaining_seconds: float = DEFAULT_TOOL_RETRY_MIN_REMAINING_SECONDS
    allowed_cost_classes: frozenset[CostClass] = frozenset({CostClass.FREE})


@dataclass(frozen=True)
class ToolExecutionConfig:
    """Validated asynchronous tool-execution rollout policy."""

    enabled: bool = True
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_TOOL_CALLS
    timeout_seconds: ToolTimeoutConfig = ToolTimeoutConfig()
    retry: ToolRetryConfig = ToolRetryConfig()


def _select(config: DictConfig | None, path: str, default: object) -> object:
    """Select one resolved OmegaConf value or return its reviewed default."""
    if config is None:
        return default
    return OmegaConf.select(config, path, default=default)


def _validate_boolean(value: object, path: str) -> bool:
    """Reject truthy substitutes for an explicit boolean setting."""
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _validate_number(
    value: object,
    path: str,
    *,
    allow_zero: bool = False,
) -> float:
    """Return one finite numeric value after applying its lower bound."""
    if type(value) not in {int, float}:
        raise ValueError(f"{path} must be a number")
    normalized = float(value)
    minimum_is_valid = normalized >= 0 if allow_zero else normalized > 0
    if not math.isfinite(normalized) or not minimum_is_valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path} must be a finite {qualifier} number")
    return normalized


def _load_timeout_config(config: DictConfig | None) -> ToolTimeoutConfig:
    """Load an exact fast/slow deadline mapping."""
    path = "agents.guardrails.tool_execution.timeout_seconds"
    raw = _select(config, path, None)
    if raw is None:
        return ToolTimeoutConfig()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must be a mapping with exactly fast and slow keys")

    keys = set(raw.keys())
    expected_keys = {latency_class.value for latency_class in LatencyClass}
    if keys != expected_keys:
        raise ValueError(f"{path} must contain exactly the keys: fast, slow")

    return ToolTimeoutConfig(
        fast=_validate_number(raw[LatencyClass.FAST.value], f"{path}.fast"),
        slow=_validate_number(raw[LatencyClass.SLOW.value], f"{path}.slow"),
    )


def _load_allowed_cost_classes(config: DictConfig | None) -> frozenset[CostClass]:
    """Resolve the explicit retry cost allowlist to registry enums."""
    path = "agents.guardrails.tool_execution.retry.allowed_cost_classes"
    raw = _select(config, path, (CostClass.FREE.value,))
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{path} must be a sequence of cost-class names")

    allowed: set[CostClass] = set()
    for value in raw:
        if not isinstance(value, str):
            raise ValueError(f"{path} must contain only cost-class names")
        try:
            allowed.add(CostClass(value))
        except ValueError as exc:
            raise ValueError(f"{path} contains unknown cost class: {value!r}") from exc
    return frozenset(allowed)


def load_tool_execution_config(config: DictConfig | None) -> ToolExecutionConfig:
    """Resolve and validate asynchronous tool-execution configuration.

    Args:
        config: Application configuration. Reviewed defaults are used when absent.

    Returns:
        Frozen, typed tool-execution policy data.

    Raises:
        ValueError: If any setting has an invalid type, value, or combination.
    """
    root = "agents.guardrails.tool_execution"
    enabled = _validate_boolean(_select(config, f"{root}.enabled", True), f"{root}.enabled")

    max_concurrent_calls = _select(
        config,
        f"{root}.max_concurrent_calls",
        DEFAULT_MAX_CONCURRENT_TOOL_CALLS,
    )
    if type(max_concurrent_calls) is not int or max_concurrent_calls < 1:
        raise ValueError(f"{root}.max_concurrent_calls must be an integer of at least 1")

    timeout_seconds = _load_timeout_config(config)
    retry_enabled = _validate_boolean(
        _select(config, f"{root}.retry.enabled", True),
        f"{root}.retry.enabled",
    )
    max_attempts = _select(config, f"{root}.retry.max_attempts", DEFAULT_TOOL_MAX_ATTEMPTS)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 2:
        raise ValueError(f"{root}.retry.max_attempts must be an integer from 1 to 2")

    delay_seconds = _validate_number(
        _select(config, f"{root}.retry.delay_seconds", DEFAULT_TOOL_RETRY_DELAY_SECONDS),
        f"{root}.retry.delay_seconds",
        allow_zero=True,
    )
    min_remaining_seconds = _validate_number(
        _select(
            config,
            f"{root}.retry.min_remaining_seconds",
            DEFAULT_TOOL_RETRY_MIN_REMAINING_SECONDS,
        ),
        f"{root}.retry.min_remaining_seconds",
    )
    if delay_seconds + min_remaining_seconds >= min(timeout_seconds.fast, timeout_seconds.slow):
        raise ValueError(
            f"{root}.retry delay_seconds plus min_remaining_seconds must be below both deadlines"
        )

    return ToolExecutionConfig(
        enabled=enabled,
        max_concurrent_calls=max_concurrent_calls,
        timeout_seconds=timeout_seconds,
        retry=ToolRetryConfig(
            enabled=retry_enabled,
            max_attempts=max_attempts,
            delay_seconds=delay_seconds,
            min_remaining_seconds=min_remaining_seconds,
            allowed_cost_classes=_load_allowed_cost_classes(config),
        ),
    )
