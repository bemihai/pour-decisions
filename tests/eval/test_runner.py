"""Unit tests for eval runner (Phase 4).

All external dependencies are mocked; tests do not require ChromaDB, LLM keys,
or a populated cellar DB.
"""

from __future__ import annotations

import asyncio

import pytest
from omegaconf import DictConfig

from src.eval.models import GoldenSample, SampleResult
from src.eval.runner import EvalRunner


@pytest.fixture()
def sample_rag() -> GoldenSample:
    """Return a representative rag_only sample."""
    return GoldenSample(
        id="rag_only_001",
        question="What is the minimum aging for Barolo DOCG?",
        category="rag_only",
        difficulty="easy",
        expected_facts=["38 months", "18 months oak"],
        ground_truth="Barolo requires at least 38 months aging from harvest, with at least 18 months in oak.",
        ground_truth_chunk_ids=["chunk-1"],
        tags=["barolo", "aging"],
    )


@pytest.fixture()
def sample_cellar_skip() -> GoldenSample:
    """Return a cellar sample with skip note for empty DB."""
    return GoldenSample(
        id="cellar_001",
        question="Do I have any Nebbiolo wines not yet ready to drink?",
        category="cellar",
        difficulty="easy",
        expected_facts=["name", "drink_from_year"],
        expected_tool_calls=["get_cellar_inventory"],
        ground_truth="Answer must identify whether cellar has Nebbiolo wines and include drink-from year.",
        ground_truth_chunk_ids=[],
        tags=["cellar", "not_ready", "tool_required"],
        notes="Requires live cellar DB; skip if DB is empty",
    )


@pytest.fixture()
def runner_config() -> DictConfig:
    """Load project config for runner initialization."""
    from src.utils import get_config

    return get_config()


@pytest.mark.asyncio
async def test_run_sample_rag_returns_structured_result(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """run_sample() returns a fully populated SampleResult for rag backend."""
    runner = EvalRunner(backend="rag", config=runner_config)

    fake_docs = [
        {"id": "chunk-a", "document": "Barolo is aged 38 months.", "metadata": {"source": "book.pdf"}},
        {"id": "chunk-b", "document": "Riserva requires 62 months.", "metadata": {"source": "book.pdf"}},
    ]

    retriever_mock = mocker.Mock()
    retriever_mock.retrieve.return_value = fake_docs
    runner._retriever = retriever_mock
    runner._model = object()

    mocker.patch("src.eval.runner.invoke_llm", return_value="Barolo requires 38 months aging.")

    result = await runner.run_sample(sample_rag)

    assert isinstance(result, SampleResult)
    assert result.id == sample_rag.id
    assert result.question == sample_rag.question
    assert result.answer == "Barolo requires 38 months aging."
    assert result.contexts == ["Barolo is aged 38 months.", "Riserva requires 62 months."]
    assert result.retrieved_chunk_ids == ["chunk-a", "chunk-b"]
    assert result.tool_calls_made == []
    assert result.error is None
    assert result.latency_ms >= 0.0

    retriever_mock.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_run_sample_catches_errors_and_sets_error_field(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """Exceptions are captured into SampleResult.error and not raised."""
    runner = EvalRunner(backend="rag", config=runner_config)

    mocker.patch.object(runner, "_run_rag_sync", side_effect=RuntimeError("retrieval failure"))

    result = await runner.run_sample(sample_rag)

    assert result.id == sample_rag.id
    assert result.error is not None
    assert "retrieval failure" in result.error
    assert result.answer == ""
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_run_respects_max_concurrency(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """run() enforces concurrency limit with asyncio.Semaphore."""
    runner = EvalRunner(backend="rag", config=runner_config)

    samples = [
        sample_rag.model_copy(update={"id": f"rag_only_{i:03d}"})
        for i in range(1, 7)
    ]

    state = {"active": 0, "max_seen": 0}

    async def fake_run_sample(sample: GoldenSample) -> SampleResult:
        state["active"] += 1
        state["max_seen"] = max(state["max_seen"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return SampleResult(id=sample.id, question=sample.question, answer="ok", latency_ms=1.0)

    mocker.patch.object(runner, "run_sample", side_effect=fake_run_sample)
    mocker.patch.object(runner, "_is_cellar_db_empty", return_value=False)
    mocker.patch.object(runner, "_ensure_rag_resources")

    results = await runner.run(samples=samples, mode="retrieval", max_concurrency=2)

    assert len(results) == 6
    assert state["max_seen"] == 2


@pytest.mark.asyncio
async def test_run_skips_cellar_samples_when_db_is_empty(
    mocker,
    runner_config: object,
    sample_cellar_skip: GoldenSample,
) -> None:
    """Cellar samples with skip note are skipped when DB has no inventory."""
    runner = EvalRunner(backend="rag", config=runner_config)

    class _Conn:
        """Minimal fake DB connection context manager for tests."""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, _query: str):
            class _Cursor:
                @staticmethod
                def fetchone():
                    return [0]

            return _Cursor()

    mocker.patch("src.eval.runner.get_db_connection", return_value=_Conn())

    run_rag_mock = mocker.patch.object(runner, "_run_rag_sync")
    mocker.patch.object(runner, "_ensure_rag_resources")

    results = await runner.run(samples=[sample_cellar_skip], mode="retrieval", max_concurrency=1)

    assert len(results) == 1
    assert results[0].id == sample_cellar_skip.id
    assert results[0].error == "skipped: cellar DB is empty"
    run_rag_mock.assert_not_called()

