"""Tests for the M6B async RAG closeout measurement."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.eval.scripts import async_rag_closeout


@pytest.mark.asyncio
async def test_closeout_measurement_uses_async_api_resources_and_closes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed cohort should use one owned bundle and report comparable fields."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            provider="ollama", name="gemma4:31b",
            base_url="https://ollama.com", timeout_seconds=60,
        ),
        web_search=SimpleNamespace(auto_fallback=False),
    )
    resources = SimpleNamespace(
        retriever=object(),
        reranker=object(),
        close=AsyncMock(),
    )
    invoke = AsyncMock()

    async def _invoke(**_kwargs: object) -> tuple[str, list[object], list[object]]:
        await asyncio.sleep(0.012)
        return "answer", [object()], []

    invoke.side_effect = _invoke
    monkeypatch.setattr(async_rag_closeout, "get_config", lambda: config)
    monkeypatch.setattr(async_rag_closeout, "load_base_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        async_rag_closeout,
        "build_async_rag_runtime",
        AsyncMock(return_value=resources),
    )
    monkeypatch.setattr(async_rag_closeout, "_ainvoke_rag_only", invoke)

    result = await async_rag_closeout.measure_async_rag_closeout()

    assert result.cohort_size == 3
    assert result.logical_llm_calls == 3
    assert result.external_web_calls == 0
    assert result.gate0.external_web_calls == 0
    assert result.delta_from_gate0.logical_llm_calls == 0
    assert result.delta_from_gate0.external_web_calls == 0
    assert result.samples[0].question == "What grape is used in Barolo?"
    assert all(sample.answer_chars == 6 for sample in result.samples)
    assert all(sample.source_count == 1 for sample in result.samples)
    assert invoke.await_count == 3
    resources.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_worker_bridge_meter_records_default_executor_occupancy() -> None:
    """Explicit stage bridges should contribute busy time and worker identity."""
    meter = async_rag_closeout._WorkerBridgeMeter(asyncio.to_thread)

    result = await meter.to_thread(lambda: "done")

    assert result == "done"
    assert meter.busy_ms >= 0
    assert len(meter.thread_ids) == 1


def test_closeout_main_reports_unavailable_live_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The developer command should fail clearly when live resources are absent."""
    monkeypatch.setattr(
        async_rag_closeout,
        "measure_async_rag_closeout",
        AsyncMock(side_effect=RuntimeError("unavailable")),
    )
    logger = MagicMock()
    monkeypatch.setattr(async_rag_closeout, "logger", logger)

    assert async_rag_closeout.main() == 1
    logger.error.assert_called_once()
