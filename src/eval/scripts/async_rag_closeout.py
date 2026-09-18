"""Measure the completed asynchronous RAG runtime on the Gate 0 cohort.

This developer command uses the same lifespan-owned resources and direct async
RAG helper as the API. Resource startup and shutdown are excluded from request
metrics. Run it from the repository root with configured Chroma and model
services::

    python -m src.eval.scripts.async_rag_closeout
"""

from __future__ import annotations

import asyncio
import json
import resource
import statistics
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from src.agents.llm import load_base_model
from src.api.routes.chat import _ainvoke_rag_only
from src.retrieval import build_async_rag_runtime
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
    answer_chars: int
    source_count: int


@dataclass(frozen=True)
class _Gate0Reference:
    """Recorded live whole-RAG bridge evidence from Gate 0 PR #110."""

    cohort_size: int = 3
    latency_median_ms: float = 6243.737
    latency_p95_ms: float = 9599.928
    worker_busy_ms: float = 20723.511
    event_loop_lag_median_ms: float = 0.898
    event_loop_lag_p95_ms: float = 1.446
    python_traced_peak_bytes: int = 5840341
    process_peak_rss_before_bytes: int = 914636800
    process_peak_rss_after_bytes: int = 1190211584
    logical_llm_calls: int = 3
    external_web_calls: int = 0


@dataclass(frozen=True)
class _Gate0Delta:
    """Signed closeout-minus-Gate-0 differences for comparable measurements."""

    latency_median_ms: float
    latency_p95_ms: float
    worker_busy_ms: float
    event_loop_lag_median_ms: float
    event_loop_lag_p95_ms: float
    python_traced_peak_bytes: int
    process_peak_rss_after_bytes: int
    logical_llm_calls: int
    external_web_calls: int


@dataclass(frozen=True)
class _CloseoutMeasurement:
    """Machine-readable measurements for the completed async runtime."""

    cohort_size: int
    model_provider: str
    model_name: str
    auto_web_fallback: bool
    wall_time_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    worker_bridge_busy_ms: float
    worker_bridge_threads: int
    event_loop_heartbeat_samples: int
    event_loop_lag_median_ms: float
    event_loop_lag_p95_ms: float
    python_traced_peak_bytes: int
    process_peak_rss_before_bytes: int
    process_peak_rss_after_bytes: int
    logical_llm_calls: int
    external_web_calls: int
    gate0: _Gate0Reference
    delta_from_gate0: _Gate0Delta
    samples: tuple[_SampleMeasurement, ...]


@dataclass
class _WorkerBridgeMeter:
    """Measure explicit ``asyncio.to_thread`` work during cohort requests."""

    original_to_thread: Callable[..., Any]
    busy_ms: float = 0.0
    thread_ids: set[int] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    async def to_thread(self, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run one bridge through the default executor and record its occupancy."""

        def _measured() -> Any:
            started = time.perf_counter()
            thread_id = threading.get_ident()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                with self._lock:
                    self.busy_ms += elapsed_ms
                    self.thread_ids.add(thread_id)

        return await self.original_to_thread(_measured)


def _peak_rss_bytes() -> int:
    """Return peak resident memory in bytes on the supported macOS runtime."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile without another dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int((percentile * len(ordered)) + 0.999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


async def _measure_event_loop(stop: asyncio.Event, lags_ms: list[float]) -> None:
    """Sample scheduling delay while async RAG requests execute."""
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        expected = loop.time() + _HEARTBEAT_INTERVAL_SECONDS
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        lags_ms.append(max(0.0, (loop.time() - expected) * 1000))


async def measure_async_rag_closeout() -> _CloseoutMeasurement:
    """Run the Gate 0 cohort through the completed API async boundary."""
    config = get_config()
    model = load_base_model(str(config.model.provider), str(config.model.name))
    resources = await build_async_rag_runtime(config)
    if resources.retriever is None:
        await resources.close()
        raise RuntimeError("Async RAG retriever is unavailable")

    auto_web_fallback = bool(config.web_search.auto_fallback)
    if auto_web_fallback:
        await resources.close()
        raise RuntimeError("The Gate 0 comparison requires automatic web fallback to remain disabled")

    original_to_thread = asyncio.to_thread
    worker_meter = _WorkerBridgeMeter(original_to_thread)
    loop_lags_ms: list[float] = []
    samples: list[_SampleMeasurement] = []
    stop = asyncio.Event()
    heartbeat = asyncio.create_task(_measure_event_loop(stop, loop_lags_ms))
    rss_before = _peak_rss_bytes()
    tracemalloc.start()
    wall_started = time.perf_counter()

    try:
        asyncio.to_thread = worker_meter.to_thread
        for question in _COHORT:
            request_started = time.perf_counter()
            answer, sources, _web_sources = await _ainvoke_rag_only(
                prompt=question,
                cfg=config,
                model=model,
                retriever=resources.retriever,
                reranker=resources.reranker,
                message_history=[],
                enable_rag=True,
                n_results_override=None,
            )
            samples.append(
                _SampleMeasurement(
                    question=question,
                    latency_ms=round((time.perf_counter() - request_started) * 1000, 3),
                    answer_chars=len(answer),
                    source_count=len(sources),
                )
            )
    finally:
        asyncio.to_thread = original_to_thread
        wall_time_ms = (time.perf_counter() - wall_started) * 1000
        stop.set()
        await heartbeat
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        await resources.close()

    latencies = [sample.latency_ms for sample in samples]
    gate0 = _Gate0Reference()
    latency_median_ms = round(statistics.median(latencies), 3)
    latency_p95_ms = round(_percentile_nearest_rank(latencies, 0.95), 3)
    worker_busy_ms = round(worker_meter.busy_ms, 3)
    event_loop_lag_median_ms = round(statistics.median(loop_lags_ms), 3)
    event_loop_lag_p95_ms = round(_percentile_nearest_rank(loop_lags_ms, 0.95), 3)
    process_peak_rss_after_bytes = _peak_rss_bytes()
    logical_llm_calls = len(samples)
    external_web_calls = 0
    return _CloseoutMeasurement(
        cohort_size=len(samples),
        model_provider=str(config.model.provider),
        model_name=str(config.model.name),
        auto_web_fallback=auto_web_fallback,
        wall_time_ms=round(wall_time_ms, 3),
        latency_median_ms=latency_median_ms,
        latency_p95_ms=latency_p95_ms,
        worker_bridge_busy_ms=worker_busy_ms,
        worker_bridge_threads=len(worker_meter.thread_ids),
        event_loop_heartbeat_samples=len(loop_lags_ms),
        event_loop_lag_median_ms=event_loop_lag_median_ms,
        event_loop_lag_p95_ms=event_loop_lag_p95_ms,
        python_traced_peak_bytes=traced_peak,
        process_peak_rss_before_bytes=rss_before,
        process_peak_rss_after_bytes=process_peak_rss_after_bytes,
        logical_llm_calls=logical_llm_calls,
        external_web_calls=external_web_calls,
        gate0=gate0,
        delta_from_gate0=_Gate0Delta(
            latency_median_ms=round(latency_median_ms - gate0.latency_median_ms, 3),
            latency_p95_ms=round(latency_p95_ms - gate0.latency_p95_ms, 3),
            worker_busy_ms=round(worker_busy_ms - gate0.worker_busy_ms, 3),
            event_loop_lag_median_ms=round(
                event_loop_lag_median_ms - gate0.event_loop_lag_median_ms,
                3,
            ),
            event_loop_lag_p95_ms=round(
                event_loop_lag_p95_ms - gate0.event_loop_lag_p95_ms,
                3,
            ),
            python_traced_peak_bytes=traced_peak - gate0.python_traced_peak_bytes,
            process_peak_rss_after_bytes=(
                process_peak_rss_after_bytes - gate0.process_peak_rss_after_bytes
            ),
            logical_llm_calls=logical_llm_calls - gate0.logical_llm_calls,
            external_web_calls=external_web_calls - gate0.external_web_calls,
        ),
        samples=tuple(samples),
    )


def main() -> int:
    """Run the live closeout command and log one JSON result."""
    try:
        measurement = asyncio.run(measure_async_rag_closeout())
    except Exception as exc:
        logger.error("Async RAG closeout measurement failed: %s", exc)
        return 1
    logger.info("M6B_CLOSEOUT_MEASUREMENT=%s", json.dumps(asdict(measurement), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
