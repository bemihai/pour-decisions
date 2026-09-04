"""Tests for validated asynchronous tool-execution policy configuration."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from omegaconf import DictConfig, OmegaConf
import pytest

from src.agents.guardrails import (
    TOOL_EXECUTION_REPORT_CONFIG_KEY,
    ToolExecutionConfig,
    ToolExecutionController,
    ToolExecutionEvent,
    ToolExecutionEventCode,
    ToolExecutionReport,
    ToolFailureClassifierCode,
    ToolRetryConfig,
    ToolTimeoutPhase,
    ToolTimeoutConfig,
    build_async_tool_execution_wrapper,
    classify_tool_failure,
    get_retry_eligibility_code,
    load_tool_execution_config,
)
from src.agents.tools.catalog import TOOL_DEFINITIONS
from src.agents.tools.registry import CostClass, LatencyClass, ToolDefinition, ToolSelectionSnapshot


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


def _snapshot() -> ToolSelectionSnapshot:
    """Return a stable selected-tool snapshot for wrapper tests."""
    return ToolSelectionSnapshot(definitions=TOOL_DEFINITIONS, readiness=())


def _short_policy(*, enabled: bool = True) -> ToolExecutionConfig:
    """Return deterministic sub-second deadlines for wrapper tests."""
    return ToolExecutionConfig(
        enabled=enabled,
        max_concurrent_calls=1,
        timeout_seconds=ToolTimeoutConfig(fast=0.02, slow=0.04),
    )


def _retry_policy(
    *,
    timeout_seconds: float = 0.2,
    delay_seconds: float = 0.01,
    min_remaining_seconds: float = 0.01,
) -> ToolExecutionConfig:
    """Return a deterministic policy that leaves useful retry budget."""
    return ToolExecutionConfig(
        max_concurrent_calls=1,
        timeout_seconds=ToolTimeoutConfig(
            fast=timeout_seconds,
            slow=timeout_seconds,
        ),
        retry=ToolRetryConfig(
            delay_seconds=delay_seconds,
            min_remaining_seconds=min_remaining_seconds,
        ),
    )


def _request(
    definition: ToolDefinition,
    call_id: str,
    report: ToolExecutionReport | None = None,
) -> SimpleNamespace:
    """Build the minimal ToolCallRequest shape consumed by the wrapper."""
    configurable = (
        {"configurable": {TOOL_EXECUTION_REPORT_CONFIG_KEY: report}}
        if report is not None
        else {}
    )
    return SimpleNamespace(
        tool_call={"name": definition.metadata.name, "id": call_id},
        tool=definition.tool,
        runtime=SimpleNamespace(config=configurable),
    )


def _sqlite_operational_error(error_code: int | None) -> sqlite3.OperationalError:
    """Build a deterministic SQLite error with optional structured code."""
    exception = sqlite3.OperationalError("synthetic private SQLite failure")
    if error_code is not None:
        exception.sqlite_errorcode = error_code
    return exception


@pytest.mark.parametrize(
    ("error_code", "expected"),
    (
        (sqlite3.SQLITE_BUSY, ToolFailureClassifierCode.SQLITE_BUSY),
        (sqlite3.SQLITE_BUSY | (1 << 8), ToolFailureClassifierCode.SQLITE_BUSY),
        (sqlite3.SQLITE_LOCKED, ToolFailureClassifierCode.SQLITE_LOCKED),
        (sqlite3.SQLITE_LOCKED | (1 << 8), ToolFailureClassifierCode.SQLITE_LOCKED),
    ),
)
def test_classifier_normalizes_structured_sqlite_contention_codes(
    error_code: int,
    expected: ToolFailureClassifierCode,
) -> None:
    """Primary and extended contention codes should map to the reviewed cohort."""
    assert classify_tool_failure(_sqlite_operational_error(error_code)) is expected


@pytest.mark.parametrize(
    "exception",
    (
        _sqlite_operational_error(None),
        _sqlite_operational_error(sqlite3.SQLITE_CONSTRAINT),
        sqlite3.OperationalError("database is locked"),
        RuntimeError("database is locked"),
        TimeoutError("database is busy"),
    ),
)
def test_classifier_rejects_unstructured_and_unreviewed_failures(
    exception: BaseException,
) -> None:
    """Exception text and unsupported structured codes must not authorize retry."""
    assert classify_tool_failure(exception) is None


def test_retry_eligibility_returns_reviewed_code_when_every_gate_passes() -> None:
    """A free idempotent busy failure with useful budget should authorize retry."""
    metadata = TOOL_DEFINITIONS[0].metadata

    assert (
        get_retry_eligibility_code(
            _sqlite_operational_error(sqlite3.SQLITE_BUSY),
            metadata,
            ToolRetryConfig(),
            attempts_started=1,
            remaining_seconds=1.101,
        )
        is ToolFailureClassifierCode.SQLITE_BUSY
    )


@pytest.mark.parametrize(
    ("exception", "metadata_updates", "retry_updates", "attempts_started", "remaining_seconds"),
    (
        (_sqlite_operational_error(sqlite3.SQLITE_BUSY), {}, {"enabled": False}, 1, 10.0),
        (_sqlite_operational_error(sqlite3.SQLITE_BUSY), {}, {"max_attempts": 1}, 1, 10.0),
        (
            _sqlite_operational_error(sqlite3.SQLITE_BUSY),
            {"idempotent": False},
            {},
            1,
            10.0,
        ),
        (
            _sqlite_operational_error(sqlite3.SQLITE_BUSY),
            {"cost_class": CostClass.CHEAP},
            {},
            1,
            10.0,
        ),
        (
            _sqlite_operational_error(sqlite3.SQLITE_BUSY),
            {"cost_class": CostClass.EXPENSIVE},
            {},
            1,
            10.0,
        ),
        (RuntimeError("generic failure"), {}, {}, 1, 10.0),
        (TimeoutError("upstream timeout"), {}, {}, 1, 10.0),
        (_sqlite_operational_error(None), {}, {}, 1, 10.0),
        (_sqlite_operational_error(sqlite3.SQLITE_BUSY), {}, {}, 1, 1.1),
        (_sqlite_operational_error(sqlite3.SQLITE_BUSY), {}, {}, 1, 1.0),
    ),
)
def test_retry_eligibility_denies_when_any_gate_fails(
    exception: BaseException,
    metadata_updates: dict[str, object],
    retry_updates: dict[str, object],
    attempts_started: int,
    remaining_seconds: float,
) -> None:
    """Every conservative policy gate must independently deny retry."""
    metadata = TOOL_DEFINITIONS[0].metadata.model_copy(update=metadata_updates)
    retry_policy = replace(ToolRetryConfig(), **retry_updates)

    assert (
        get_retry_eligibility_code(
            exception,
            metadata,
            retry_policy,
            attempts_started=attempts_started,
            remaining_seconds=remaining_seconds,
        )
        is None
    )


@pytest.mark.asyncio
async def test_execution_wrapper_preserves_known_success_result_identity() -> None:
    """Composition must return the exact delegated result for a known tool."""
    definition = TOOL_DEFINITIONS[0]
    request = SimpleNamespace(
        tool_call={"name": definition.metadata.name, "id": "known-call"}
    )
    expected = ToolMessage(content="ok", tool_call_id="known-call")

    async def execute(_request: object) -> ToolMessage:
        return expected

    result = await build_async_tool_execution_wrapper(
        _snapshot(), ToolExecutionConfig(), ToolExecutionController(1)
    )(request, execute)

    assert result is expected


@pytest.mark.asyncio
async def test_execution_wrapper_preserves_disabled_m9a_behavior() -> None:
    """Disabled policy must retain the existing category-safe async boundary."""
    definition = TOOL_DEFINITIONS[0]
    request = SimpleNamespace(
        tool_call={"name": definition.metadata.name, "id": "disabled-call"}
    )

    async def execute(_request: object) -> ToolMessage:
        raise RuntimeError("private disabled failure")

    result = await build_async_tool_execution_wrapper(
        _snapshot(),
        ToolExecutionConfig(enabled=False),
        ToolExecutionController(1),
    )(request, execute)

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "disabled-call"
    assert result.status == "error"
    assert "private disabled failure" not in str(result.content)


@pytest.mark.asyncio
async def test_execution_wrapper_leaves_unknown_calls_on_m9a_boundary() -> None:
    """Unknown names must receive no metadata-driven policy or fallback lookup."""
    request = SimpleNamespace(tool_call={"name": "unknown_tool", "id": "unknown-call"})
    expected = ToolMessage(content="framework invalid", tool_call_id="unknown-call", status="error")

    async def execute(_request: object) -> ToolMessage:
        return expected

    result = await build_async_tool_execution_wrapper(
        _snapshot(), ToolExecutionConfig(), ToolExecutionController(1)
    )(request, execute)

    assert result is expected


@pytest.mark.asyncio
async def test_execution_deadline_includes_admission_without_invoking_handler() -> None:
    """A queued call must expire safely before its handler begins."""
    definition = TOOL_DEFINITIONS[0]
    controller = ToolExecutionController(1)
    report = ToolExecutionReport()
    handler_called = False

    async def execute(_request: object) -> ToolMessage:
        nonlocal handler_called
        handler_called = True
        return ToolMessage(content="unexpected", tool_call_id="queued-call")

    wrapper = build_async_tool_execution_wrapper(_snapshot(), _short_policy(), controller)
    async with controller.permit():
        result = await wrapper(_request(definition, "queued-call", report), execute)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "queued-call"
    assert not handler_called
    assert report.snapshot() == (
        {
            "code": "tool_deadline_exceeded",
            "tool_name": definition.metadata.name,
            "latency_class": definition.metadata.latency_class.value,
            "cost_class": definition.metadata.cost_class.value,
            "attempt_number": 1,
            "timeout_phase": "admission",
            "sync_bridge": True,
        },
    )


@pytest.mark.asyncio
async def test_execution_deadline_cancels_cooperative_coroutine_and_releases_permit() -> None:
    """A cooperative overrun must return safely and release admission promptly."""
    definition = TOOL_DEFINITIONS[0]
    controller = ToolExecutionController(1)
    report = ToolExecutionReport()
    cancelled = asyncio.Event()

    async def execute(_request: object) -> ToolMessage:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    wrapper = build_async_tool_execution_wrapper(_snapshot(), _short_policy(), controller)
    result = await wrapper(_request(definition, "deadline-call", report), execute)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "deadline-call"
    assert cancelled.is_set()
    assert report.snapshot()[0]["timeout_phase"] == "execution"
    async with asyncio.timeout(0.1):
        async with controller.permit():
            pass


@pytest.mark.asyncio
async def test_execution_deadline_discards_late_cancellation_suppressed_result() -> None:
    """An expired timeout context must reject a coroutine's late success."""
    definition = TOOL_DEFINITIONS[0]
    report = ToolExecutionReport()
    late = ToolMessage(content="late", tool_call_id="late-call")

    async def execute(_request: object) -> ToolMessage:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return late

    result = await build_async_tool_execution_wrapper(
        _snapshot(), _short_policy(), ToolExecutionController(1)
    )(_request(definition, "late-call", report), execute)

    assert result is not late
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert report.snapshot()[0]["code"] == "tool_deadline_exceeded"


@pytest.mark.asyncio
async def test_execution_wrapper_does_not_misclassify_upstream_timeout() -> None:
    """A handler-owned TimeoutError remains an ordinary M9A-safe failure."""
    definition = TOOL_DEFINITIONS[0]
    report = ToolExecutionReport()

    async def execute(_request: object) -> ToolMessage:
        raise TimeoutError("private upstream timeout")

    result = await build_async_tool_execution_wrapper(
        _snapshot(), _short_policy(), ToolExecutionController(1)
    )(_request(definition, "upstream-call", report), execute)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert [event["code"] for event in report.snapshot()] == ["tool_terminal_failure"]
    assert report.snapshot()[0].get("classifier_code") is None
    assert "private upstream timeout" not in str(result.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (asyncio.CancelledError, GraphBubbleUp))
async def test_execution_wrapper_propagates_framework_control_flow(failure: type[BaseException]) -> None:
    """Caller cancellation and LangGraph control flow must never become tool output."""
    definition = TOOL_DEFINITIONS[0]

    async def execute(_request: object) -> ToolMessage:
        raise failure()

    with pytest.raises(failure):
        await build_async_tool_execution_wrapper(
            _snapshot(), _short_policy(), ToolExecutionController(1)
        )(_request(definition, "control-call"), execute)


@pytest.mark.asyncio
async def test_execution_wrapper_retries_once_and_preserves_success_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eligible contention failure should delay once and return attempt 2 unchanged."""
    definition = TOOL_DEFINITIONS[0]
    report = ToolExecutionReport()
    expected = ToolMessage(content="recovered", tool_call_id="retry-success-call")
    attempts = 0
    sleep = AsyncMock()
    monkeypatch.setattr("src.agents.guardrails.tool_execution.asyncio.sleep", sleep)

    async def execute(_request: object) -> ToolMessage:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _sqlite_operational_error(sqlite3.SQLITE_BUSY)
        return expected

    result = await build_async_tool_execution_wrapper(
        _snapshot(), _retry_policy(), ToolExecutionController(1)
    )(_request(definition, "retry-success-call", report), execute)

    assert result is expected
    assert attempts == 2
    sleep.assert_awaited_once_with(0.01)
    assert [event["code"] for event in report.snapshot()] == [
        "tool_retry_started",
        "tool_retry_succeeded",
    ]
    assert [event["attempt_number"] for event in report.snapshot()] == [2, 2]
    assert all(
        event["classifier_code"] == "sqlite_busy" for event in report.snapshot()
    )


@pytest.mark.asyncio
async def test_execution_wrapper_retry_exhaustion_returns_existing_safe_error() -> None:
    """A second structured failure should stop after two attempts and remain non-disclosing."""
    definition = TOOL_DEFINITIONS[0]
    report = ToolExecutionReport()
    attempts = 0

    async def execute(_request: object) -> ToolMessage:
        nonlocal attempts
        attempts += 1
        raise _sqlite_operational_error(sqlite3.SQLITE_LOCKED)

    result = await build_async_tool_execution_wrapper(
        _snapshot(), _retry_policy(delay_seconds=0.0), ToolExecutionController(1)
    )(_request(definition, "retry-exhausted-call", report), execute)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "retry-exhausted-call"
    assert "synthetic private SQLite failure" not in str(result.content)
    assert attempts == 2
    assert [event["code"] for event in report.snapshot()] == [
        "tool_retry_started",
        "tool_terminal_failure",
    ]
    assert report.snapshot()[-1]["attempt_number"] == 2
    assert report.snapshot()[-1]["classifier_code"] == "sqlite_locked"


@pytest.mark.asyncio
async def test_execution_wrapper_attempt_two_timeout_uses_original_deadline() -> None:
    """A retry overrun should become a Phase 3 deadline with no third attempt."""
    definition = TOOL_DEFINITIONS[0]
    report = ToolExecutionReport()
    attempts = 0

    async def execute(_request: object) -> ToolMessage:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _sqlite_operational_error(sqlite3.SQLITE_BUSY)
        await asyncio.Event().wait()
        return ToolMessage(content="unreachable", tool_call_id="retry-timeout-call")

    result = await build_async_tool_execution_wrapper(
        _snapshot(),
        _retry_policy(timeout_seconds=0.02, delay_seconds=0.0, min_remaining_seconds=0.001),
        ToolExecutionController(1),
    )(_request(definition, "retry-timeout-call", report), execute)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert attempts == 2
    assert [event["code"] for event in report.snapshot()] == [
        "tool_retry_started",
        "tool_deadline_exceeded",
        "tool_sync_timeout",
    ]
    assert report.snapshot()[1]["attempt_number"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deny_case",
    (
        "retry_disabled",
        "one_attempt",
        "non_idempotent",
        "cheap",
        "expensive",
        "upstream_timeout",
        "generic_failure",
        "missing_sqlite_code",
        "unsupported_sqlite_code",
        "insufficient_budget",
    ),
)
async def test_execution_wrapper_deny_cases_never_start_attempt_two(deny_case: str) -> None:
    """Every denied failure should make one attempt and return the M9A-safe result."""
    definition = TOOL_DEFINITIONS[0]
    metadata_updates: dict[str, object] = {}
    retry_updates: dict[str, object] = {}
    exception: BaseException = _sqlite_operational_error(sqlite3.SQLITE_BUSY)
    policy = _retry_policy(timeout_seconds=0.05, delay_seconds=0.001, min_remaining_seconds=0.001)

    if deny_case == "retry_disabled":
        retry_updates["enabled"] = False
    elif deny_case == "one_attempt":
        retry_updates["max_attempts"] = 1
    elif deny_case == "non_idempotent":
        metadata_updates["idempotent"] = False
    elif deny_case == "cheap":
        metadata_updates["cost_class"] = CostClass.CHEAP
    elif deny_case == "expensive":
        metadata_updates["cost_class"] = CostClass.EXPENSIVE
    elif deny_case == "upstream_timeout":
        exception = TimeoutError("private upstream timeout")
    elif deny_case == "generic_failure":
        exception = RuntimeError("private generic failure")
    elif deny_case == "missing_sqlite_code":
        exception = _sqlite_operational_error(None)
    elif deny_case == "unsupported_sqlite_code":
        exception = _sqlite_operational_error(sqlite3.SQLITE_CONSTRAINT)
    elif deny_case == "insufficient_budget":
        policy = _retry_policy(
            timeout_seconds=0.02,
            delay_seconds=0.015,
            min_remaining_seconds=0.01,
        )

    if retry_updates:
        policy = replace(policy, retry=replace(policy.retry, **retry_updates))
    if metadata_updates:
        definition = replace(
            definition,
            metadata=definition.metadata.model_copy(update=metadata_updates),
        )
    snapshot = ToolSelectionSnapshot(definitions=(definition,), readiness=())
    report = ToolExecutionReport()
    attempts = 0

    async def execute(_request: object) -> ToolMessage:
        nonlocal attempts
        attempts += 1
        raise exception

    result = await build_async_tool_execution_wrapper(
        snapshot,
        policy,
        ToolExecutionController(1),
    )(_request(definition, f"{deny_case}-call", report), execute)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert attempts == 1
    assert [event["code"] for event in report.snapshot()] == ["tool_terminal_failure"]
    assert all("private" not in str(value) for event in report.snapshot() for value in event.values())


@pytest.mark.asyncio
async def test_validation_error_result_is_not_retried_or_reported_as_failure() -> None:
    """A framework validation ToolMessage is a completed handler result, not an exception."""
    definition = TOOL_DEFINITIONS[0]
    report = ToolExecutionReport()
    expected = ToolMessage(
        content="framework validation error",
        tool_call_id="validation-call",
        status="error",
    )
    attempts = 0

    async def execute(_request: object) -> ToolMessage:
        nonlocal attempts
        attempts += 1
        return expected

    result = await build_async_tool_execution_wrapper(
        _snapshot(), _retry_policy(), ToolExecutionController(1)
    )(_request(definition, "validation-call", report), execute)

    assert result is expected
    assert attempts == 1
    assert report.snapshot() == ()


@pytest.mark.asyncio
async def test_timeout_during_retry_delay_does_not_claim_sync_worker_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delay overrun has no active sync worker and must not emit sync-timeout evidence."""
    definition = TOOL_DEFINITIONS[0]
    report = ToolExecutionReport()
    attempts = 0

    async def block_during_delay(_delay: float) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "src.agents.guardrails.tool_execution.asyncio.sleep",
        block_during_delay,
    )

    async def execute(_request: object) -> ToolMessage:
        nonlocal attempts
        attempts += 1
        raise _sqlite_operational_error(sqlite3.SQLITE_BUSY)

    result = await build_async_tool_execution_wrapper(
        _snapshot(),
        _retry_policy(timeout_seconds=0.02, delay_seconds=0.001, min_remaining_seconds=0.001),
        ToolExecutionController(1),
    )(_request(definition, "retry-delay-timeout", report), execute)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert attempts == 1
    assert [event["code"] for event in report.snapshot()] == [
        "tool_deadline_exceeded"
    ]
    assert report.snapshot()[0]["sync_bridge"] is True


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


@pytest.mark.parametrize("value", (True, 0, -1, 1.0))
def test_controller_requires_a_positive_integer_capacity(value: object) -> None:
    """Direct controller construction should fail on ambiguous capacities."""
    with pytest.raises(ValueError, match="integer of at least 1"):
        ToolExecutionController(value)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_controller_bounds_same_loop_admission() -> None:
    """Concurrent callers should never exceed the configured async capacity."""
    controller = ToolExecutionController(1)
    release = asyncio.Event()
    first_entered = asyncio.Event()
    active = 0
    maximum_active = 0

    async def worker() -> None:
        nonlocal active, maximum_active
        async with controller.permit():
            active += 1
            maximum_active = max(maximum_active, active)
            first_entered.set()
            await release.wait()
            active -= 1

    first = asyncio.create_task(worker())
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second = asyncio.create_task(worker())
    await asyncio.sleep(0)

    assert active == 1
    assert maximum_active == 1

    release.set()
    await asyncio.gather(first, second)
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_controller_releases_permit_after_failure_and_cancellation() -> None:
    """Failure and caller cancellation must not leak admission capacity."""
    controller = ToolExecutionController(1)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        async with controller.permit():
            raise RuntimeError("synthetic failure")

    entered = asyncio.Event()

    async def wait_for_cancellation() -> None:
        async with controller.permit():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(wait_for_cancellation())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with asyncio.timeout(1):
        async with controller.permit():
            pass


def test_controller_rejects_cross_loop_reuse() -> None:
    """A controller should belong to the first event loop that uses it."""
    controller = ToolExecutionController(1)

    async def use_once() -> None:
        async with controller.permit():
            pass

    asyncio.run(use_once())
    with pytest.raises(RuntimeError, match="cannot be shared across event loops"):
        asyncio.run(use_once())


def test_execution_event_serializes_only_bounded_policy_fields() -> None:
    """Typed events should detach enum values without raw execution data."""
    event = ToolExecutionEvent(
        code=ToolExecutionEventCode.TERMINAL_FAILURE,
        tool_name="get_cellar_wines",
        latency_class=LatencyClass.FAST,
        cost_class=CostClass.FREE,
        attempt_number=2,
        classifier_code=ToolFailureClassifierCode.SQLITE_BUSY,
        timeout_phase=ToolTimeoutPhase.EXECUTION,
        sync_bridge=True,
    )

    assert event.as_guardrail_event() == {
        "code": "tool_terminal_failure",
        "tool_name": "get_cellar_wines",
        "latency_class": "fast",
        "cost_class": "free",
        "attempt_number": 2,
        "classifier_code": "sqlite_busy",
        "timeout_phase": "execution",
        "sync_bridge": True,
    }


@pytest.mark.parametrize(
    "event",
    (
        lambda: ToolExecutionEvent(code="tool_terminal_failure", tool_name="known"),
        lambda: ToolExecutionEvent(
            code=ToolExecutionEventCode.TERMINAL_FAILURE,
            tool_name=" ",
        ),
        lambda: ToolExecutionEvent(
            code=ToolExecutionEventCode.RETRY_STARTED,
            tool_name="known",
            attempt_number=3,
        ),
        lambda: ToolExecutionEvent(
            code=ToolExecutionEventCode.DEADLINE_EXCEEDED,
            tool_name="known",
            sync_bridge="yes",
        ),
    ),
)
def test_execution_event_rejects_unbounded_values(event: object) -> None:
    """Raw strings and unsupported policy values should fail at construction."""
    with pytest.raises((TypeError, ValueError)):
        event()  # type: ignore[operator]


def test_execution_report_returns_ordered_detached_snapshots() -> None:
    """Callers may mutate a snapshot without changing stored report events."""
    report = ToolExecutionReport()
    report.append(
        ToolExecutionEvent(
            code=ToolExecutionEventCode.RETRY_STARTED,
            tool_name="get_cellar_wines",
            attempt_number=2,
        )
    )
    report.append(
        ToolExecutionEvent(
            code=ToolExecutionEventCode.RETRY_SUCCEEDED,
            tool_name="get_cellar_wines",
            attempt_number=2,
        )
    )

    snapshot = report.snapshot()
    snapshot[0]["tool_name"] = "changed"

    assert [event["code"] for event in report.snapshot()] == [
        "tool_retry_started",
        "tool_retry_succeeded",
    ]
    assert report.snapshot()[0]["tool_name"] == "get_cellar_wines"


def test_execution_report_accepts_only_typed_events() -> None:
    """The report boundary should reject arbitrary mappings and exceptions."""
    report = ToolExecutionReport()

    with pytest.raises(TypeError, match="ToolExecutionEvent"):
        report.append({"code": "raw"})  # type: ignore[arg-type]


def test_execution_report_preserves_all_concurrent_typed_appends() -> None:
    """Concurrent producers should not lose or corrupt bounded report events."""
    report = ToolExecutionReport()

    def append_event(index: int) -> None:
        report.append(
            ToolExecutionEvent(
                code=ToolExecutionEventCode.TERMINAL_FAILURE,
                tool_name=f"synthetic_tool_{index}",
            )
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append_event, range(100)))

    snapshot = report.snapshot()
    assert len(snapshot) == 100
    assert {event["tool_name"] for event in snapshot} == {
        f"synthetic_tool_{index}" for index in range(100)
    }
