"""Eval runner for executing golden dataset samples.

This module executes golden dataset samples against either the RAG backend or the
full intelligent agent backend, collects raw outputs, and records per-sample latency
and errors.
"""

import asyncio
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig

from src.agents.intelligent.agent import WineAgent
from src.database.repository import StatsRepository
from src.eval.models import GoldenSample, SampleResult
from src.eval.utils import (
    extract_eval_config_snapshot,
    get_git_metadata,
    load_execution_model,
    run_agent_sample_sync,
    run_rag_retrieval_only_sync,
    run_rag_sample_sync,
    run_retriever_benchmark_sync,
)
from src.retrieval import (
    ChromaRetriever,
    DocumentReranker,
    HybridRetriever,
    build_reranker_from_config,
    build_retriever_from_config,
)
from src.utils import get_config, logger

_TIMEOUT_ERROR_TYPES = (TimeoutError, asyncio.TimeoutError, FutureTimeoutError)


class EvalRunner:
    """Run raw evaluation samples against the configured backend.

    This runner only executes backend calls and returns per-sample outputs used
    by retrieval evals and by full eval post-processing.

    The runner supports three backends:

    - ``rag``: shared production RAG stages, with optional generation
    - ``retriever``: isolated low-level retriever benchmark
    - ``agent``: full intelligent agent invocation

    Attributes:
        backend: Backend under test (``rag``, ``retriever``, or ``agent``).
        config: App configuration object.
        config_snapshot: Stable subset of config captured at runner creation.
        git_sha: Short git SHA captured at runner creation.
    """

    def __init__(
        self,
        backend: str = "rag",
        config: DictConfig | None = None,
        generation_enabled: bool = True,
    ):
        """Initialize the eval runner.

        Args:
            backend: Backend under test. Must be ``rag``, ``retriever``, or ``agent``.
            config: Optional app config override. If omitted, loaded from
                ``app_config.yml`` using :func:`src.utils.get_config`.
            generation_enabled: Whether the RAG backend should run answer
                generation after retrieval. Ignored for the agent backend.

        Raises:
            ValueError: If backend is unsupported.
        """
        if backend not in {"rag", "retriever", "agent"}:
            raise ValueError(f"backend must be 'rag', 'retriever', or 'agent', got {backend!r}")

        self.backend = backend
        self.generation_enabled = bool(generation_enabled)
        self.config = config or get_config()
        self.config_snapshot = extract_eval_config_snapshot(self.config)
        self.git_metadata = get_git_metadata()
        self.git_sha = str(self.git_metadata["sha"])

        self._model: BaseChatModel | None = None
        self._retriever: HybridRetriever | ChromaRetriever | None = None
        self._reranker: DocumentReranker | None = None
        self._reranker_initialized = False
        self._agent: WineAgent | None = None
        self._cellar_db_is_empty: bool | None = None
        self._resource_init_lock = asyncio.Lock()

    async def _prepare_backend_resources(self) -> None:
        """Initialize backend resources once, guarding against concurrent init races."""
        async with self._resource_init_lock:
            if self.backend in {"rag", "retriever"}:
                model_needed = self.backend == "rag" and self.generation_enabled
                if self._retriever is None or (model_needed and self._model is None):
                    self._ensure_rag_resources()
            else:
                if self._agent is None:
                    self._ensure_agent_resources()

    async def run_sample(self, sample: GoldenSample) -> SampleResult:
        """Execute one sample and return the captured result.

        Args:
            sample: Golden sample to execute.

        Returns:
            SampleResult containing answer, contexts, tool usage, latency, and error.
        """
        if self._cellar_db_is_empty is None:
            try:
                self._cellar_db_is_empty = StatsRepository().is_cellar_empty()
            except Exception as exc:
                logger.warning("Failed to check cellar DB emptiness (%s); assuming non-empty", exc)
                self._cellar_db_is_empty = False

        if self._should_skip_sample(sample):
            return SampleResult(
                id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                status="skipped",
                error="skipped: cellar DB is empty",
                latency_ms=0.0,
            )

        start_time = time.perf_counter()
        try:
            normalized_query: str | None = None
            context_text = ""
            raw_retrieved_chunks: list[dict[str, object]] = []
            context_chunks: list[dict[str, object]] = []
            rag_sources: list[dict[str, object]] = []
            rag_feature_flags: dict[str, bool] = {}

            if self.backend in {"rag", "retriever"}:
                model_needed = self.backend == "rag" and self.generation_enabled
                if self._retriever is None or (model_needed and self._model is None):
                    await self._prepare_backend_resources()
                if self._retriever is None:
                    raise RuntimeError("RAG resources are not initialized")
                if self.backend == "retriever":
                    rag_result = await asyncio.to_thread(
                        run_retriever_benchmark_sync,
                        sample,
                        self._retriever,
                        int(self.config.chroma.retrieval.n_results),
                    )
                elif self.generation_enabled:
                    if self._model is None:
                        raise RuntimeError("RAG generation model is not initialized")
                    rag_result = await asyncio.to_thread(
                        run_rag_sample_sync,
                        sample,
                        self.config,
                        self._retriever,
                        self._model,
                        self._reranker,
                    )
                else:
                    rag_result = await asyncio.to_thread(
                        run_rag_retrieval_only_sync,
                        sample,
                        self.config,
                        self._retriever,
                        self._reranker,
                    )
                if rag_result.retrieval_error:
                    raise RuntimeError(rag_result.retrieval_error)
                answer = rag_result.answer
                contexts = [chunk.text for chunk in rag_result.context_chunks if chunk.text]
                retrieved_chunk_ids = [chunk.id for chunk in rag_result.context_chunks if chunk.id]
                tool_calls = []
                normalized_query = rag_result.normalized_query
                context_text = rag_result.context
                raw_retrieved_chunks = [chunk.to_dict() for chunk in rag_result.raw_retrieved_chunks]
                context_chunks = [chunk.to_dict() for chunk in rag_result.context_chunks]
                rag_sources = [source.to_dict() for source in rag_result.sources]
                rag_feature_flags = rag_result.feature_usage.to_dict()
            else:
                if self._agent is None:
                    await self._prepare_backend_resources()
                if self._agent is None:
                    raise RuntimeError("Agent resources are not initialized")
                
                answer, contexts, retrieved_chunk_ids, tool_calls = await asyncio.to_thread(
                    run_agent_sample_sync,
                    self._agent,
                    sample,
                )
            latency_ms = (time.perf_counter() - start_time) * 1000

            return SampleResult(
                id=sample.id,
                question=sample.question,
                answer=answer,
                ground_truth=sample.ground_truth,
                contexts=contexts,
                retrieved_chunk_ids=retrieved_chunk_ids,
                normalized_query=normalized_query,
                context_text=context_text,
                raw_retrieved_chunks=raw_retrieved_chunks,
                context_chunks=context_chunks,
                rag_sources=rag_sources,
                rag_feature_flags=rag_feature_flags,
                tool_calls_made=tool_calls,
                latency_ms=latency_ms,
                status="passed",
                error=None,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Eval sample failed for %s: %s", sample.id, exc)
            if isinstance(exc, _TIMEOUT_ERROR_TYPES):
                return SampleResult(
                    id=sample.id,
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    latency_ms=latency_ms,
                    status="timeout",
                    error=f"timeout: {exc}",
                )
            return SampleResult(
                id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                latency_ms=latency_ms,
                status="failed",
                error=str(exc),
            )

    async def run_sample_with_timeout(
        self,
        sample: GoldenSample,
        timeout_seconds: float | None,
    ) -> SampleResult:
        """Execute one sample with an optional backend timeout budget.

        Args:
            sample: Golden sample to execute.
            timeout_seconds: Timeout budget in seconds applied at model-load level.
                ``None`` disables timeout enforcement.

        Returns:
            SampleResult for the sample.
        """
        return await self.run_sample(sample)

    async def run(
        self,
        samples: list[GoldenSample],
        max_concurrency: int = 1,
    ) -> list[SampleResult]:
        """Execute multiple samples with bounded concurrency.

        Args:
            samples: The list of samples to run.
            max_concurrency: Maximum number of in-flight sample executions.

        Returns:
            List of SampleResult objects, preserving input order.
        """
        max_concurrency = max(1, max_concurrency)
        timeout_seconds = getattr(self.config.eval, "sample_timeout_seconds", None)
        if timeout_seconds is not None:
            timeout_seconds = max(0.0, float(timeout_seconds))
            if timeout_seconds == 0.0:
                timeout_seconds = None

        try:
            self._cellar_db_is_empty = StatsRepository().is_cellar_empty()
        except Exception as exc:
            logger.warning("Failed to check cellar DB emptiness (%s); assuming non-empty", exc)
            self._cellar_db_is_empty = False

        logger.info(
            "Starting eval run: backend=%s samples=%d max_concurrency=%d model_timeout=%s cellar_empty=%s",
            self.backend,
            len(samples),
            max_concurrency,
            timeout_seconds,
            self._cellar_db_is_empty,
        )

        # Pre-warm resources once before concurrent threads start to avoid race conditions
        await self._prepare_backend_resources()

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_sample(sample: GoldenSample) -> SampleResult:
            async with semaphore:
                return await self.run_sample_with_timeout(sample, timeout_seconds)

        return list(await asyncio.gather(*[_run_sample(sample) for sample in samples]))
    
    def _ensure_rag_resources(self) -> None:
        """Lazily initialize RAG resources (model + retriever)."""
        if self.backend == "rag" and self.generation_enabled and self._model is None:
            self._model = load_execution_model(self.config)

        if self._retriever is not None:
            return

        self._retriever = build_retriever_from_config(
            self.config,
            enable_cache=True,
            enable_query_expansion=False,
        )
        if self.backend == "rag" and not self._reranker_initialized:
            self._reranker = build_reranker_from_config(self.config)
            self._reranker_initialized = True

    def _ensure_agent_resources(self) -> None:
        """Lazily initialize intelligent agent resources."""
        if self._agent is None:
            if self._model is None:
                self._model = load_execution_model(self.config)
            self._agent = WineAgent(verbose=False, llm=self._model)

    def _should_skip_sample(self, sample: GoldenSample) -> bool:
        """Return True if sample should be skipped for empty cellar DB.

        Args:
            sample: Sample to evaluate.

        Returns:
            Whether the sample should be skipped.
        """
        skip_when_empty = bool(getattr(self.config.eval, "skip_cellar_samples_if_empty", True))
        if not skip_when_empty:
            return False

        if self._cellar_db_is_empty is not True:
            return False

        notes = (sample.notes or "").lower()
        return "skip if db is empty" in notes
