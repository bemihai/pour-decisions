"""Tests for the M07 deterministic streaming closeout measurement."""

from pathlib import Path

import pytest

from src.eval.scripts.streaming_closeout import (
    SampleMeasurement,
    TokenUsage,
    measure_streaming_closeout,
    summarize_measurement,
)


def _sample(
    scenario: str,
    *,
    progress_delay_ms: float | None,
    progress_before_completion: bool | None,
    overhead_ms: float = 5.0,
) -> SampleMeasurement:
    """Build one passing aggregate-only sample."""
    return SampleMeasurement(
        scenario=scenario,  # type: ignore[arg-type]
        pair_index=0,
        blocking_total_ms=20.0,
        streaming_final_ms=25.0,
        streaming_total_ms=26.0,
        added_final_delivery_overhead_ms=overhead_ms,
        time_to_first_useful_progress_ms=10.0 if progress_delay_ms is not None else None,
        progress_observation_delay_ms=progress_delay_ms,
        progress_before_tool_completion=progress_before_completion,
        event_count=3 if progress_delay_ms is not None else 1,
        event_bytes=200,
        peak_progress_buffer=1,
        dropped_progress_events=0,
        final_response_equal=True,
        tool_trajectory_equal=True,
        model_attempts_equal=True,
        tool_attempts_equal=True,
        committed_thread_state_equal=True,
        blocking_model_attempts=2 if progress_delay_ms is not None else 1,
        streaming_model_attempts=2 if progress_delay_ms is not None else 1,
        blocking_tool_attempts=1 if progress_delay_ms is not None else 0,
        streaming_tool_attempts=1 if progress_delay_ms is not None else 0,
        blocking_token_usage=TokenUsage(10, 5, 15),
        streaming_token_usage=TokenUsage(10, 5, 15),
    )


def test_summary_evaluates_every_approved_gate() -> None:
    """The aggregate should expose a passing complete cohort and recommendation."""
    samples = [
        _sample("zero_tool", progress_delay_ms=None, progress_before_completion=None),
        _sample("one_tool", progress_delay_ms=5.0, progress_before_completion=True),
        _sample("multi_tool", progress_delay_ms=8.0, progress_before_completion=True),
    ]

    result = summarize_measurement(
        samples,
        repetitions=1,
        tool_delay_seconds=0.01,
        request_owned_task_leaks=0,
    )

    assert result.gate.passed is True
    assert result.recommendation == "eligible_for_separate_enablement_decision"
    assert result.scenario_counts == {"zero_tool": 1, "one_tool": 1, "multi_tool": 1}
    assert result.progress_applicable_samples == 2
    assert result.blocking_token_usage == result.streaming_token_usage == TokenUsage(30, 15, 45)


def test_summary_keeps_streaming_disabled_when_a_threshold_fails() -> None:
    """A failed provisional threshold must fail the combined rollout gate."""
    samples = [
        _sample("zero_tool", progress_delay_ms=None, progress_before_completion=None),
        _sample("one_tool", progress_delay_ms=251.0, progress_before_completion=True),
        _sample("multi_tool", progress_delay_ms=8.0, progress_before_completion=True),
    ]

    result = summarize_measurement(
        samples,
        repetitions=1,
        tool_delay_seconds=0.01,
        request_owned_task_leaks=0,
    )

    assert result.gate.progress_p95_within_limit is False
    assert result.gate.passed is False
    assert result.recommendation == "keep_streaming_disabled"


@pytest.mark.asyncio
async def test_reduced_socket_cohort_proves_pairing_and_cleanup(tmp_path: Path) -> None:
    """A fast three-pair run should exercise the complete production route boundary."""
    result = await measure_streaming_closeout(
        repetitions=1,
        tool_delay_seconds=0.02,
        work_directory=tmp_path,
    )

    assert result.cohort_size == 3
    assert result.gate.passed is True
    assert result.progress_applicable_samples == 2
    assert result.progress_before_completion_samples == 2
    assert result.total_event_count == 9
    assert result.blocking_model_attempts == result.streaming_model_attempts == 5
    assert result.blocking_tool_attempts == result.streaming_tool_attempts == 3
    assert result.blocking_token_usage == result.streaming_token_usage
    assert result.peak_progress_buffer <= 16
    assert result.dropped_progress_events == 0
    assert result.request_owned_task_leaks == 0


@pytest.mark.asyncio
async def test_measurement_rejects_invalid_cohort_configuration(tmp_path: Path) -> None:
    """Invalid measurement bounds must fail before resources are created."""
    with pytest.raises(ValueError, match="repetitions"):
        await measure_streaming_closeout(repetitions=0, work_directory=tmp_path)
    with pytest.raises(ValueError, match="tool_delay_seconds"):
        await measure_streaming_closeout(tool_delay_seconds=0, work_directory=tmp_path)
