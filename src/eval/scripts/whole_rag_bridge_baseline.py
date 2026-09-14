"""Measure the temporary whole-RAG worker bridge on a fixed live cohort.

This developer command loads the same preloaded resources as the API and runs
the synchronous RAG-only route helper through one ``asyncio.to_thread()`` call
per question. Resource startup is excluded from the reported request metrics.

Usage::

    python -m src.eval.scripts.whole_rag_bridge_baseline
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import resource
import statistics
import threading
import time
import tracemalloc
from typing import Any

from src.agents.llm import load_base_model
from src.api.routes.chat import _invoke_rag_only
from src.retrieval import build_reranker_from_config, build_retriever_from_config
from src.utils import get_config, logger


_COHORT = (
    "What grape is used in Barolo?",
    "How does acidity affect a wine's ability to age?",
    "What distinguishes Chablis from other Chardonnay wines?",
)
_HEARTBEAT_INTERVAL_SECONDS = 0.01


@dataclass(frozen=True)
class _SampleMeasurement:
    """Measurements for one fixed-cohort request."""

    question: str
    latency_ms: float
    worker_thread_id: int
    answer_chars: int
    source_count: int


@dataclass(frozen=True)
class _BridgeBaseline:
    """Machine-readable current-bridge measurements."""

    cohort_size: int
    model_provider: str
    model_name: str
    auto_web_fallback: bool
    wall_time_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    whole_rag_worker_busy_ms: float
    whole_rag_worker_threads: int
    event_loop_heartbeat_samples: int
    event_loop_lag_median_ms: float
    event_loop_lag_p95_ms: float
    python_traced_peak_bytes: int
    process_peak_rss_before_bytes: int
    process_peak_rss_after_bytes: int
    logical_llm_calls: int
    external_web_calls: int
    samples: tuple[_SampleMeasurement, ...]


def _peak_rss_bytes() -> int:
    """Return peak resident memory in bytes on the supported macOS runtime."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile without adding a statistics dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int((percentile * len(ordered)) + 0.999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


async def _measure_event_loop(stop: asyncio.Event, lags_ms: list[float]) -> None:
    """Sample scheduling delay while the whole pipeline occupies a worker."""
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        expected = loop.time() + _HEARTBEAT_INTERVAL_SECONDS
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        lags_ms.append(max(0.0, (loop.time() - expected) * 1000))


async def measure_whole_rag_bridge() -> _BridgeBaseline:
    """Run the fixed live cohort through the current API bridge boundary."""
    config = get_config()
    model = load_base_model(str(config.model.provider), str(config.model.name))
    retriever = build_retriever_from_config(config)
    reranker = build_reranker_from_config(config)
    auto_web_fallback = bool(config.web_search.auto_fallback)
    if auto_web_fallback:
        raise RuntimeError("Gate 0's exact web-call baseline requires automatic web fallback to remain disabled")

    loop_lags_ms: list[float] = []
    worker_busy_ms = 0.0
    sample_measurements: list[_SampleMeasurement] = []
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_measure_event_loop(stop, loop_lags_ms))
    rss_before = _peak_rss_bytes()
    tracemalloc.start()
    wall_started = time.perf_counter()

    try:
        for question in _COHORT:
            def _invoke() -> tuple[tuple[str, list[Any], list[Any]], float, int]:
                worker_started = time.perf_counter()
                result = _invoke_rag_only(
                    prompt=question,
                    cfg=config,
                    model=model,
                    retriever=retriever,
                    reranker=reranker,
                    message_history=[],
                    enable_rag=True,
                    n_results_override=None,
                )
                worker_elapsed_ms = (time.perf_counter() - worker_started) * 1000
                return result, worker_elapsed_ms, threading.get_ident()

            request_started = time.perf_counter()
            (answer, sources, _web_sources), busy_ms, thread_id = await asyncio.to_thread(_invoke)
            latency_ms = (time.perf_counter() - request_started) * 1000
            worker_busy_ms += busy_ms
            sample_measurements.append(
                _SampleMeasurement(
                    question=question,
                    latency_ms=round(latency_ms, 3),
                    worker_thread_id=thread_id,
                    answer_chars=len(answer),
                    source_count=len(sources),
                )
            )
    finally:
        wall_time_ms = (time.perf_counter() - wall_started) * 1000
        stop.set()
        await heartbeat
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    latencies = [sample.latency_ms for sample in sample_measurements]
    return _BridgeBaseline(
        cohort_size=len(sample_measurements),
        model_provider=str(config.model.provider),
        model_name=str(config.model.name),
        auto_web_fallback=auto_web_fallback,
        wall_time_ms=round(wall_time_ms, 3),
        latency_median_ms=round(statistics.median(latencies), 3),
        latency_p95_ms=round(_percentile_nearest_rank(latencies, 0.95), 3),
        whole_rag_worker_busy_ms=round(worker_busy_ms, 3),
        whole_rag_worker_threads=len({sample.worker_thread_id for sample in sample_measurements}),
        event_loop_heartbeat_samples=len(loop_lags_ms),
        event_loop_lag_median_ms=round(statistics.median(loop_lags_ms), 3),
        event_loop_lag_p95_ms=round(_percentile_nearest_rank(loop_lags_ms, 0.95), 3),
        python_traced_peak_bytes=traced_peak,
        process_peak_rss_before_bytes=rss_before,
        process_peak_rss_after_bytes=_peak_rss_bytes(),
        logical_llm_calls=len(sample_measurements),
        external_web_calls=0,
        samples=tuple(sample_measurements),
    )


def main() -> int:
    """Run the live baseline command and log one JSON result."""
    try:
        baseline = asyncio.run(measure_whole_rag_bridge())
    except Exception as exc:
        logger.error("Whole-RAG bridge baseline failed: %s", exc)
        return 1
    logger.info("M6B_GATE0_BASELINE=%s", json.dumps(asdict(baseline), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
