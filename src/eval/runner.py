"""Eval runner for executing golden dataset samples.

This module executes golden dataset samples against either the RAG-only backend or
full intelligent agent backend, collects raw outputs, and records per-sample latency
and errors.
"""

import asyncio
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig

from src.agents.intelligent.agent import WineAgent
from src.database.db import get_db_connection
from src.eval.models import GoldenSample, SampleResult
from src.eval.utils import (
    extract_eval_config_snapshot,
    get_git_sha,
    load_eval_model,
    run_agent_sample_sync,
    run_rag_sample_sync,
)
from src.retrieval import ChromaRetriever, HybridRetriever, build_retriever_from_config
from src.utils import get_config, logger


class EvalRunner:
    """Run evaluation samples against the configured backend.

    The runner supports two backends:

    - ``rag``: vector/hybrid retrieval + single LLM generation
    - ``agent``: full intelligent agent invocation

    Attributes:
        backend: Backend under test (``rag`` or ``agent``).
        config: App configuration object.
        config_snapshot: Stable subset of config captured at runner creation.
        git_sha: Short git SHA captured at runner creation.
    """

    def __init__(self, backend: str = "rag", config: DictConfig | None = None):
        """Initialize the eval runner.

        Args:
            backend: Backend under test. Must be ``rag`` or ``agent``.
            config: Optional app config override. If omitted, loaded from
                ``app_config.yml`` using :func:`src.utils.get_config`.

        Raises:
            ValueError: If backend is not ``rag`` or ``agent``.
        """
        if backend not in {"rag", "agent"}:
            raise ValueError(f"backend must be 'rag' or 'agent', got {backend!r}")

        self.backend = backend
        self.config = config or get_config()
        self.config_snapshot = extract_eval_config_snapshot(self.config)
        self.git_sha = get_git_sha()

        self._model: BaseChatModel | None = None
        self._retriever: HybridRetriever | ChromaRetriever | None = None
        self._agent: WineAgent | None = None
        self._cellar_db_is_empty: bool | None = None

    async def run_sample(self, sample: GoldenSample) -> SampleResult:
        """Execute one sample and return the captured result.

        Args:
            sample: Golden sample to execute.

        Returns:
            SampleResult containing answer, contexts, tool usage, latency, and error.
        """
        if self._should_skip_sample(sample):
            return SampleResult(
                id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                error="skipped: cellar DB is empty",
                latency_ms=0.0,
            )

        start_time = time.perf_counter()
        try:
            if self.backend == "rag":
                self._ensure_rag_resources()
                if self._retriever is None or self._model is None:
                    raise RuntimeError("RAG resources are not initialized")
                
                answer, contexts, retrieved_chunk_ids, tool_calls = await asyncio.to_thread(
                    run_rag_sample_sync,
                    sample,
                    self._retriever,
                    self._model,
                    int(self.config.chroma.retrieval.n_results),
                )
            else:
                self._ensure_agent_resources()
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
                tool_calls_made=tool_calls,
                latency_ms=latency_ms,
                error=None,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Eval sample failed for %s: %s", sample.id, exc)
            return SampleResult(
                id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def run(
        self,
        samples: list[GoldenSample],
        mode: str = "retrieval",
        max_concurrency: int = 3,
    ) -> list[SampleResult]:
        """Execute multiple samples with bounded concurrency.

        Args:
            samples: The list of samples to run.
            mode: Eval mode. 
            max_concurrency: Maximum number of in-flight sample executions.

        Returns:
            List of SampleResult objects, preserving input order.
        """
        max_concurrency = max(1, max_concurrency)
        
        self._cellar_db_is_empty = self._is_cellar_db_empty()
        logger.info(
            "Starting eval run: backend=%s mode=%s samples=%d max_concurrency=%d cellar_empty=%s",
            self.backend, mode, len(samples), max_concurrency, self._cellar_db_is_empty
            )

        # Pre-warm resources once before concurrent threads start to avoid race conditions
        if self.backend == "rag":
            self._ensure_rag_resources()
        else:
            self._ensure_agent_resources()

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_sample(sample: GoldenSample) -> SampleResult:
            async with semaphore:
                return await self.run_sample(sample)

        return list(await asyncio.gather(*[_run_sample(sample) for sample in samples]))
    
    def _ensure_rag_resources(self) -> None:
        """Lazily initialize RAG resources (model + retriever)."""
        if self._model is None:
            self._model = load_eval_model(self.config)

        if self._retriever is not None:
            return

        self._retriever = build_retriever_from_config(
            self.config,
            enable_cache=True,
            enable_query_expansion=False,
        )

    def _ensure_agent_resources(self) -> None:
        """Lazily initialize intelligent agent resources."""
        if self._agent is None:
            if self._model is None:
                self._model = load_eval_model(self.config)
            self._agent = WineAgent(verbose=False, llm=self._model)

    def _is_cellar_db_empty(self) -> bool:
        """Check whether the cellar DB has any wines in inventory.

        Returns:
            True when the cellar DB has no wines with positive quantity.
        """
        try:
            with get_db_connection() as conn:
                row = conn.execute("SELECT COUNT(*) FROM wines WHERE q_quantity > 0").fetchone()
                count = int(row[0]) if row else 0
                return count == 0
        except Exception as exc:
            logger.warning("Failed to check cellar DB emptiness (%s); assuming non-empty", exc)
            return False

    def _should_skip_sample(self, sample: GoldenSample) -> bool:
        """Return True if sample should be skipped for empty cellar DB.

        Args:
            sample: Sample to evaluate.

        Returns:
            Whether the sample should be skipped.
        """
        if self._cellar_db_is_empty is None:
            self._cellar_db_is_empty = self._is_cellar_db_empty()

        skip_when_empty = bool(getattr(self.config.eval, "skip_cellar_samples_if_empty", True))
        if not skip_when_empty:
            return False

        if not self._cellar_db_is_empty:
            return False

        notes = (sample.notes or "").lower()
        return "skip if db is empty" in notes
