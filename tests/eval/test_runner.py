"""Unit tests for eval runner (Phase 4).

All external dependencies are mocked; tests do not require ChromaDB, LLM keys,
or a populated cellar DB.
"""

from __future__ import annotations

import asyncio

import pytest
from omegaconf import DictConfig

from src.eval.models import AgentToolOutput, GoldenSample, SampleResult
from src.eval.runner import EvalRunner
from src.eval.utils import AgentExecutionResult
from src.retrieval import RAGChunkArtifact, RAGExecutionResult, RAGFeatureUsage


def _rag_execution_result(
    *,
    answer: str = "",
    contexts: list[str] | None = None,
    chunk_ids: list[str] | None = None,
    rerank_scores: list[float] | None = None,
    retrieval_confidence: float | None = None,
    low_confidence: bool = False,
    rerank_threshold: float | None = None,
) -> RAGExecutionResult:
    """Build a structured RAG result for runner orchestration tests."""
    context_values = contexts or []
    id_values = chunk_ids or []
    score_values: list[float | None] = rerank_scores or [None] * len(context_values)
    chunks = [
        RAGChunkArtifact(id=chunk_id, text=context, rerank_score=rerank_score)
        for chunk_id, context, rerank_score in zip(id_values, context_values, score_values)
    ]
    return RAGExecutionResult(
        answer=answer,
        context="\n\n".join(context_values),
        normalized_query="normalized question",
        raw_retrieved_chunks=chunks,
        context_chunks=chunks,
        feature_usage=RAGFeatureUsage(
            retrieval=True,
            reranking=retrieval_confidence is not None,
            rerank_thresholding=rerank_threshold is not None,
        ),
        retrieval_confidence=retrieval_confidence,
        low_confidence=low_confidence,
        rerank_threshold=rerank_threshold,
    )


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

    mocker.patch(
        "src.retrieval.rag_service.process_user_prompt",
        return_value="Barolo requires 38 months aging.",
    )
    mocker.patch("src.retrieval.rag_service.deduplicate_chunks", side_effect=lambda docs, **_kwargs: docs)

    result = await runner.run_sample(sample_rag)

    assert isinstance(result, SampleResult)
    assert result.id == sample_rag.id
    assert result.question == sample_rag.question
    assert result.answer == "Barolo requires 38 months aging."
    assert result.contexts == ["Barolo is aged 38 months.", "Riserva requires 62 months."]
    assert result.retrieved_chunk_ids == ["chunk-a", "chunk-b"]
    assert result.normalized_query == sample_rag.question.lower()
    assert "Barolo is aged 38 months." in result.context_text
    assert [chunk.id for chunk in result.raw_retrieved_chunks] == ["chunk-a", "chunk-b"]
    assert [chunk.id for chunk in result.context_chunks] == ["chunk-a", "chunk-b"]
    assert result.rag_sources[0].metadata["source"] == "book.pdf"
    assert result.rag_feature_flags["retrieval"] is True
    assert result.rag_feature_flags["generation"] is True
    assert result.tool_calls_made == []
    assert result.status == "passed"
    assert result.error is None
    assert result.latency_ms >= 0.0

    retriever_mock.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_run_sample_rag_retrieval_only_does_not_require_model(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """Retrieval-only RAG eval should retrieve chunks without loading an LLM."""
    runner = EvalRunner(backend="rag", config=runner_config, generation_enabled=False)

    fake_docs = [
        {"id": "chunk-a", "document": "Barolo is aged 38 months.", "metadata": {"source": "book.pdf"}},
        {"id": "chunk-b", "document": "Riserva requires 62 months.", "metadata": {"source": "book.pdf"}},
    ]

    retriever_mock = mocker.Mock()
    retriever_mock.retrieve.return_value = fake_docs
    runner._retriever = retriever_mock

    load_model_mock = mocker.patch("src.eval.runner.load_execution_model")
    generation_mock = mocker.patch("src.retrieval.rag_service.process_user_prompt")
    mocker.patch("src.retrieval.rag_service.deduplicate_chunks", side_effect=lambda docs, **_kwargs: docs)

    result = await runner.run_sample(sample_rag)

    assert result.status == "passed"
    assert result.answer == ""
    assert result.contexts == ["Barolo is aged 38 months.", "Riserva requires 62 months."]
    assert result.retrieved_chunk_ids == ["chunk-a", "chunk-b"]
    assert result.context_text
    assert result.rag_feature_flags["retrieval"] is True
    assert result.rag_feature_flags["generation"] is False
    load_model_mock.assert_not_called()
    generation_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_sample_persists_retrieval_confidence_artifacts(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """RAG eval samples should retain confidence, threshold, and final rerank scores."""
    runner = EvalRunner(backend="rag", config=runner_config, generation_enabled=False)
    runner._retriever = mocker.Mock()
    runner._reranker = mocker.Mock()
    expected = _rag_execution_result(
        contexts=["Thresholded Barolo context."],
        chunk_ids=["chunk-thresholded"],
        rerank_scores=[1.25],
        retrieval_confidence=0.7772998611746911,
        low_confidence=False,
        rerank_threshold=0.0,
    )
    run_rag = mocker.patch(
        "src.eval.runner.run_rag_retrieval_only_sync",
        return_value=expected,
    )
    load_model = mocker.patch("src.eval.runner.load_execution_model")

    result = await runner.run_sample(sample_rag)

    assert result.retrieval_confidence == pytest.approx(0.7772998611746911)
    assert result.low_confidence is False
    assert result.rerank_threshold == 0.0
    assert result.context_chunks[0].rerank_score == 1.25
    assert result.rag_feature_flags["reranking"] is True
    assert result.rag_feature_flags["rerank_thresholding"] is True
    run_rag.assert_called_once_with(
        sample_rag,
        runner_config,
        runner._retriever,
        runner._reranker,
    )
    load_model.assert_not_called()


@pytest.mark.asyncio
async def test_retriever_backend_runs_isolated_benchmark(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """Retriever backend should bypass production RAG post-processing and generation."""
    runner = EvalRunner(backend="retriever", config=runner_config, generation_enabled=False)
    runner._retriever = mocker.Mock()
    benchmark_mock = mocker.patch(
        "src.eval.runner.run_retriever_benchmark_sync",
        return_value=_rag_execution_result(
            contexts=["raw context"],
            chunk_ids=["raw-chunk"],
        ),
    )
    production_rag_mock = mocker.patch("src.eval.runner.run_rag_retrieval_only_sync")
    load_model_mock = mocker.patch("src.eval.runner.load_execution_model")
    build_reranker_mock = mocker.patch("src.eval.runner.build_reranker_from_config")

    result = await runner.run_sample(sample_rag)

    assert result.status == "passed"
    assert result.contexts == ["raw context"]
    assert result.retrieved_chunk_ids == ["raw-chunk"]
    assert result.context_text == "raw context"
    assert result.rag_feature_flags["retrieval"] is True
    assert result.rag_feature_flags["query_analysis"] is False
    benchmark_mock.assert_called_once_with(
        sample_rag,
        runner._retriever,
        int(runner_config.chroma.retrieval.n_results),
    )
    production_rag_mock.assert_not_called()
    load_model_mock.assert_not_called()
    build_reranker_mock.assert_not_called()


@pytest.mark.asyncio
async def test_agent_backend_preserves_typed_outputs_and_scores_required_tools(
    mocker,
    runner_config: object,
) -> None:
    """Agent samples retain typed evidence and deterministic trajectory scores."""
    sample = GoldenSample(
        id="multi_hop_001",
        question="Which Barolo in my cellar is ready, and what should I know about it?",
        category="multi_hop",
        difficulty="hard",
        expected_facts=["wine name", "Barolo context"],
        expected_tool_calls=["get_cellar_wines", "search_wine_knowledge"],
        ground_truth="Identify a ready Barolo and explain it.",
        tags=["cellar", "tool_required"],
    )
    runner = EvalRunner(backend="agent", config=runner_config)
    runner._agent = mocker.Mock()
    runner._cellar_db_is_empty = False
    mocker.patch(
        "src.eval.runner.run_agent_sample_sync",
        return_value=AgentExecutionResult(
            answer="Your Barolo is ready. Barolo is made from Nebbiolo.",
            rag_contexts=["Barolo is made from Nebbiolo."],
            tool_calls=["get_cellar_wines", "search_wine_knowledge"],
            tool_outputs=[
                AgentToolOutput(
                    tool_name="get_cellar_wines",
                    output_type="cellar_result",
                    content='{"wine": "Barolo"}',
                ),
                AgentToolOutput(
                    tool_name="search_wine_knowledge",
                    output_type="rag_context",
                    content="Barolo is made from Nebbiolo.",
                ),
            ],
        ),
    )

    result = await runner.run_sample(sample)

    assert result.contexts == ["Barolo is made from Nebbiolo."]
    assert [output.output_type for output in result.tool_outputs] == [
        "cellar_result",
        "rag_context",
    ]
    assert result.expected_facts == sample.expected_facts
    assert result.expected_tool_calls == sample.expected_tool_calls
    assert result.tool_calls_made == sample.expected_tool_calls
    assert result.scores["tool_recall"] == 1.0
    assert result.scores["tool_precision"] == 1.0
    assert result.scores["tool_exact_match"] == 1.0
    assert result.scores["tool_ordered_match"] == 1.0


@pytest.mark.asyncio
async def test_run_sample_catches_errors_and_sets_error_field(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """Exceptions are captured into SampleResult.error and not raised."""
    runner = EvalRunner(backend="rag", config=runner_config)
    runner._retriever = mocker.Mock()
    runner._model = object()

    mocker.patch("src.eval.runner.run_rag_sample_sync", side_effect=RuntimeError("retrieval failure"))

    result = await runner.run_sample(sample_rag)

    assert result.id == sample_rag.id
    assert result.status == "failed"
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
    stats_repo_mock = mocker.Mock()
    stats_repo_mock.is_cellar_empty.return_value = False
    mocker.patch("src.eval.runner.StatsRepository", return_value=stats_repo_mock)
    mocker.patch.object(runner, "_ensure_rag_resources")

    results = await runner.run(samples=samples, max_concurrency=2)

    assert len(results) == 6
    assert state["max_seen"] == 2


@pytest.mark.asyncio
async def test_run_sample_initializes_rag_resources_once_under_concurrency(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """Concurrent direct sample runs should only initialize shared RAG resources once."""
    runner = EvalRunner(backend="rag", config=runner_config)
    init_calls = {"count": 0}

    def fake_ensure_rag_resources() -> None:
        init_calls["count"] += 1
        runner._model = object()
        runner._retriever = mocker.Mock()

    mocker.patch.object(runner, "_ensure_rag_resources", side_effect=fake_ensure_rag_resources)
    mocker.patch(
        "src.eval.runner.run_rag_sample_sync",
        return_value=_rag_execution_result(
            answer="answer",
            contexts=["ctx"],
            chunk_ids=["chunk-1"],
        ),
    )
    stats_repo_mock = mocker.Mock()
    stats_repo_mock.is_cellar_empty.return_value = False
    mocker.patch("src.eval.runner.StatsRepository", return_value=stats_repo_mock)

    samples = [
        sample_rag.model_copy(update={"id": f"rag_only_{index:03d}"})
        for index in range(1, 4)
    ]
    results = await asyncio.gather(*(runner.run_sample(sample) for sample in samples))

    assert len(results) == 3
    assert all(result.status == "passed" for result in results)
    assert init_calls["count"] == 1


@pytest.mark.asyncio
async def test_prepare_backend_resources_preflights_prompts_before_backend_setup(
    mocker,
    runner_config: object,
) -> None:
    """Eval preparation should validate prompts before constructing backend resources."""
    runner = EvalRunner(backend="rag", config=runner_config)
    calls: list[str] = []
    preflight = mocker.patch(
        "src.eval.runner.get_prompt_registry",
        side_effect=lambda: calls.append("prompt_registry") or object(),
    )

    def _ensure_resources() -> None:
        calls.append("rag_resources")
        runner._retriever = mocker.Mock()
        runner._model = object()

    mocker.patch.object(runner, "_ensure_rag_resources", side_effect=_ensure_resources)

    await runner._prepare_backend_resources()
    await runner._prepare_backend_resources()

    assert calls == ["prompt_registry", "rag_resources"]
    preflight.assert_called_once_with()


@pytest.mark.asyncio
async def test_prompt_preflight_failure_stops_eval_resource_setup(
    mocker,
    runner_config: object,
) -> None:
    """Invalid prompt assets should fail before any eval backend construction."""
    runner = EvalRunner(backend="agent", config=runner_config)
    mocker.patch(
        "src.eval.runner.get_prompt_registry",
        side_effect=FileNotFoundError("missing prompt manifest"),
    )
    ensure_agent = mocker.patch.object(runner, "_ensure_agent_resources")

    with pytest.raises(FileNotFoundError, match="missing prompt manifest"):
        await runner._prepare_backend_resources()

    assert runner._prompt_registry_preflight_complete is False
    ensure_agent.assert_not_called()


@pytest.mark.asyncio
async def test_run_marks_sample_as_timeout_when_budget_is_exceeded(
    mocker,
    runner_config: object,
    sample_rag: GoldenSample,
) -> None:
    """Backend timeout exceptions are mapped to timeout sample results."""
    runner_config.eval.sample_timeout_seconds = 30
    runner = EvalRunner(backend="rag", config=runner_config)
    stats_repo_mock = mocker.Mock()
    stats_repo_mock.is_cellar_empty.return_value = False
    mocker.patch("src.eval.runner.StatsRepository", return_value=stats_repo_mock)
    runner._retriever = mocker.Mock()
    runner._model = object()
    mocker.patch("src.eval.runner.run_rag_sample_sync", side_effect=TimeoutError("request exceeded model timeout"))

    results = await runner.run(samples=[sample_rag], max_concurrency=1)

    assert len(results) == 1
    assert results[0].id == sample_rag.id
    assert results[0].status == "timeout"
    assert results[0].error is not None
    assert results[0].error.startswith("timeout:")
    assert results[0].latency_ms >= 0.0


@pytest.mark.asyncio
async def test_run_skips_cellar_samples_when_db_is_empty(
    mocker,
    runner_config: object,
    sample_cellar_skip: GoldenSample,
) -> None:
    """Cellar samples with skip note are skipped when DB has no inventory."""
    runner = EvalRunner(backend="rag", config=runner_config)

    stats_repo_mock = mocker.Mock()
    stats_repo_mock.is_cellar_empty.return_value = True
    mocker.patch("src.eval.runner.StatsRepository", return_value=stats_repo_mock)

    run_rag_mock = mocker.patch("src.eval.runner.run_rag_sample_sync")
    mocker.patch.object(runner, "_ensure_rag_resources")

    results = await runner.run(samples=[sample_cellar_skip], max_concurrency=1)

    assert len(results) == 1
    assert results[0].id == sample_cellar_skip.id
    assert results[0].status == "skipped"
    assert results[0].error == "skipped: cellar DB is empty"
    run_rag_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_does_not_skip_cellar_samples_when_skip_flag_disabled(
    mocker,
    runner_config: object,
    sample_cellar_skip: GoldenSample,
) -> None:
    """Cellar samples are not skipped when skip_cellar_samples_if_empty is false."""
    runner_config.eval.skip_cellar_samples_if_empty = False
    runner = EvalRunner(backend="rag", config=runner_config)

    stats_repo_mock = mocker.Mock()
    stats_repo_mock.is_cellar_empty.return_value = True
    mocker.patch("src.eval.runner.StatsRepository", return_value=stats_repo_mock)
    mocker.patch.object(runner, "_ensure_rag_resources")
    runner._retriever = mocker.Mock()
    runner._model = object()
    run_rag_mock = mocker.patch(
        "src.eval.runner.run_rag_sample_sync",
        return_value=_rag_execution_result(answer="answer"),
    )

    results = await runner.run(samples=[sample_cellar_skip], max_concurrency=1)

    assert len(results) == 1
    assert results[0].status == "passed"
    assert results[0].error is None
    run_rag_mock.assert_called_once()


def test_eval_runner_uses_main_model_config(mocker, runner_config: object) -> None:
    """Eval execution model should come from model.provider/model.name."""
    runner_config.model.provider = "google"
    runner_config.model.name = "gemini-2.5-flash"
    runner_config.model.ollama.base_url = "http://localhost:11434"
    runner_config.eval.ragas.evaluator_provider = "ollama"
    runner_config.eval.ragas.evaluator_model = "gemma2:2b"

    runner = EvalRunner(backend="rag", config=runner_config)
    load_model_mock = mocker.patch("src.eval.runner.load_execution_model", return_value=object())
    mocker.patch("src.eval.runner.build_retriever_from_config", return_value=mocker.Mock())
    build_reranker_mock = mocker.patch("src.eval.runner.build_reranker_from_config", return_value=mocker.Mock())

    runner._ensure_rag_resources()

    load_model_mock.assert_called_once_with(runner_config)
    build_reranker_mock.assert_called_once_with(runner_config)
