"""Validated configuration for asynchronous tool-execution policy."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import math
from threading import Lock

from omegaconf import DictConfig, OmegaConf

from src.agents.tools.registry import CostClass, LatencyClass


DEFAULT_MAX_CONCURRENT_TOOL_CALLS = 4
DEFAULT_FAST_TOOL_TIMEOUT_SECONDS = 10.0
DEFAULT_SLOW_TOOL_TIMEOUT_SECONDS = 30.0
DEFAULT_TOOL_MAX_ATTEMPTS = 2
DEFAULT_TOOL_RETRY_DELAY_SECONDS = 0.1
DEFAULT_TOOL_RETRY_MIN_REMAINING_SECONDS = 1.0
TOOL_EXECUTION_REPORT_CONFIG_KEY = "_pour_decisions_tool_execution_report"


class ToolExecutionEventCode(str, Enum):
    """Stable internal outcomes emitted by M9B tool execution policy."""

    DEADLINE_EXCEEDED = "tool_deadline_exceeded"
    SYNC_TIMEOUT = "tool_sync_timeout"
    RETRY_STARTED = "tool_retry_started"
    RETRY_SUCCEEDED = "tool_retry_succeeded"
    TERMINAL_FAILURE = "tool_terminal_failure"


class ToolTimeoutPhase(str, Enum):
    """Bounded phases where a total tool-call deadline may expire."""

    ADMISSION = "admission"
    EXECUTION = "execution"


class ToolFailureClassifierCode(str, Enum):
    """Reviewed transient failure classes available to later retry policy."""

    SQLITE_BUSY = "sqlite_busy"
    SQLITE_LOCKED = "sqlite_locked"


@dataclass(frozen=True)
class ToolExecutionEvent:
    """One bounded, non-disclosing internal tool-execution outcome."""

    code: ToolExecutionEventCode
    tool_name: str
    latency_class: LatencyClass | None = None
    cost_class: CostClass | None = None
    attempt_number: int | None = None
    classifier_code: ToolFailureClassifierCode | None = None
    timeout_phase: ToolTimeoutPhase | None = None
    sync_bridge: bool | None = None

    def __post_init__(self) -> None:
        """Reject unbounded or incorrectly typed event values."""
        if not isinstance(self.code, ToolExecutionEventCode):
            raise TypeError("code must be a ToolExecutionEventCode")
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-blank catalogue name")
        if len(self.tool_name) > 128:
            raise ValueError("tool_name must not exceed 128 characters")
        if self.latency_class is not None and not isinstance(self.latency_class, LatencyClass):
            raise TypeError("latency_class must be a LatencyClass")
        if self.cost_class is not None and not isinstance(self.cost_class, CostClass):
            raise TypeError("cost_class must be a CostClass")
        if self.attempt_number is not None and (
            type(self.attempt_number) is not int or not 1 <= self.attempt_number <= 2
        ):
            raise ValueError("attempt_number must be 1 or 2")
        if self.classifier_code is not None and not isinstance(
            self.classifier_code, ToolFailureClassifierCode
        ):
            raise TypeError("classifier_code must be a ToolFailureClassifierCode")
        if self.timeout_phase is not None and not isinstance(self.timeout_phase, ToolTimeoutPhase):
            raise TypeError("timeout_phase must be a ToolTimeoutPhase")
        if self.sync_bridge is not None and type(self.sync_bridge) is not bool:
            raise TypeError("sync_bridge must be a boolean")

    def as_guardrail_event(self) -> dict[str, str | int | bool]:
        """Return a detached state-safe representation with no raw data."""
        event: dict[str, str | int | bool] = {
            "code": self.code.value,
            "tool_name": self.tool_name,
        }
        optional_values = {
            "latency_class": self.latency_class,
            "cost_class": self.cost_class,
            "attempt_number": self.attempt_number,
            "classifier_code": self.classifier_code,
            "timeout_phase": self.timeout_phase,
            "sync_bridge": self.sync_bridge,
        }
        for name, value in optional_values.items():
            if isinstance(value, Enum):
                event[name] = value.value
            elif value is not None:
                event[name] = value
        return event


class ToolExecutionReport:
    """Concurrency-safe request-local collection of bounded events."""

    def __init__(self) -> None:
        """Initialize an empty report for one asynchronous agent request."""
        self._events: list[ToolExecutionEvent] = []
        self._lock = Lock()

    def append(self, event: ToolExecutionEvent) -> None:
        """Append one validated immutable event."""
        if not isinstance(event, ToolExecutionEvent):
            raise TypeError("event must be a ToolExecutionEvent")
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[dict[str, str | int | bool], ...]:
        """Return a stable detached snapshot in append order."""
        with self._lock:
            events = tuple(self._events)
        return tuple(event.as_guardrail_event() for event in events)


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


class ToolExecutionController:
    """Own one event-loop-scoped async admission pool."""

    def __init__(self, max_concurrent_calls: int) -> None:
        """Initialize an unbound controller with an explicit capacity."""
        if type(max_concurrent_calls) is not int or max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be an integer of at least 1")
        self.max_concurrent_calls = max_concurrent_calls
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def _bind_to_running_loop(self) -> None:
        """Bind on first async use and reject reuse from another event loop."""
        running_loop = asyncio.get_running_loop()
        if self._event_loop is None:
            self._event_loop = running_loop
        elif self._event_loop is not running_loop:
            raise RuntimeError("ToolExecutionController cannot be shared across event loops")

    @asynccontextmanager
    async def permit(self) -> AsyncIterator[None]:
        """Admit one async call and always release its permit on exit."""
        self._bind_to_running_loop()
        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


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
