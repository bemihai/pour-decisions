"""Eval runner for executing golden dataset samples.

This module executes golden dataset samples against either the RAG-only backend or
full intelligent agent backend, collects raw outputs, and records per-sample latency
and errors.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig

from src.agents.intelligent.agent import WineAgent
from src.agents.llm import invoke_llm, load_base_model
from src.database.db import get_db_connection
from src.eval.models import GoldenSample, SampleResult
from src.retrieval import BM25Index, ChromaRetriever, HybridRetriever
from src.utils import get_config, initialize_chroma_client, logger


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
        self.config_snapshot = self._extract_config_snapshot()
        self.git_sha = self._get_git_sha()

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
                answer, contexts, retrieved_chunk_ids, tool_calls = await asyncio.to_thread(
                    self._run_rag_sync,
                    sample,
                )
            else:
                answer, contexts, retrieved_chunk_ids, tool_calls = await asyncio.to_thread(
                    self._run_agent_sync,
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
            samples: Samples to run.
            mode: Eval mode. Accepted for interface compatibility; Phase 4 does not
                use it for scoring.
            max_concurrency: Maximum number of in-flight sample executions.

        Returns:
            List of SampleResult objects, preserving input order.

        Raises:
            ValueError: If max_concurrency is non-positive.
        """
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be > 0, got {max_concurrency}")

        self._cellar_db_is_empty = self._is_cellar_db_empty()
        logger.info(
            "Starting eval run: backend=%s mode=%s samples=%d max_concurrency=%d cellar_empty=%s",
            self.backend,
            mode,
            len(samples),
            max_concurrency,
            self._cellar_db_is_empty,
        )

        # Pre-warm resources once before concurrent threads start.  Without this
        # guard multiple threads in the first concurrency batch race to initialize
        # the embedding model, causing "meta tensor" errors under PyTorch lazy load.
        if self.backend == "rag":
            self._ensure_rag_resources()
        else:
            self._ensure_agent_resources()

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_with_limit(sample: GoldenSample) -> SampleResult:
            async with semaphore:
                return await self.run_sample(sample)

        return list(await asyncio.gather(*[_run_with_limit(sample) for sample in samples]))

    def _run_rag_sync(self, sample: GoldenSample) -> tuple[str, list[str], list[str], list[str]]:
        """Execute one sample against RAG backend (sync helper).

        Args:
            sample: Sample to execute.

        Returns:
            Tuple of (answer, contexts, retrieved_chunk_ids, tool_calls_made).
        """
        self._ensure_rag_resources()
        assert self._retriever is not None
        assert self._model is not None

        retrieval_count = int(self.config.chroma.retrieval.n_results)
        retrieved_docs = self._retriever.retrieve(sample.question, n_results=retrieval_count)
        contexts = [doc.get("document", "") for doc in retrieved_docs if doc.get("document")]
        retrieved_chunk_ids = self._get_retrieved_chunk_ids(retrieved_docs)

        context_text = "\n\n".join(contexts)
        answer = invoke_llm(
            question=sample.question,
            context=context_text,
            model=self._model,
            message_history=[],
        )

        return answer, contexts, retrieved_chunk_ids, []

    def _run_agent_sync(self, sample: GoldenSample) -> tuple[str, list[str], list[str], list[str]]:
        """Execute one sample against agent backend (sync helper).

        Args:
            sample: Sample to execute.

        Returns:
            Tuple of (answer, contexts, retrieved_chunk_ids, tool_calls_made).
        """
        self._ensure_agent_resources()
        assert self._agent is not None

        result = self._agent.invoke(sample.question)
        answer = str(result.get("final_answer", ""))
        tool_calls_made = [str(name) for name in result.get("tools_used", [])]
        contexts = self._extract_contexts_from_agent_messages(result.get("messages", []))

        return answer, contexts, [], tool_calls_made

    def _ensure_rag_resources(self) -> None:
        """Lazily initialize RAG resources (model + retriever)."""
        if self._model is None:
            self._model = load_base_model(self.config.model.provider, self.config.model.name)

        if self._retriever is not None:
            return

        chroma_client = initialize_chroma_client(
            host=self.config.chroma.client.host,
            port=int(self.config.chroma.client.port),
        )
        vector_retriever = ChromaRetriever(
            client=chroma_client,
            collection_name=self.config.chroma.collections[0].name,
            embedding_model=self.config.chroma.settings.embedder,
            n_results=int(self.config.chroma.retrieval.n_results),
            similarity_threshold=float(self.config.chroma.retrieval.similarity_threshold),
            enable_cache=True,
        )

        if bool(getattr(self.config.chroma.retrieval, "enable_hybrid", False)):
            try:
                bm25 = BM25Index(index_path=str(self.config.chroma.retrieval.bm25_index_path))
                if len(bm25) > 0:
                    self._retriever = HybridRetriever(
                        vector_retriever=vector_retriever,
                        bm25_index=bm25,
                        vector_weight=float(self.config.chroma.retrieval.hybrid_vector_weight),
                        keyword_weight=float(self.config.chroma.retrieval.hybrid_keyword_weight),
                    )
                    return
                logger.warning("BM25 index empty; falling back to vector-only retrieval")
            except Exception as exc:
                logger.warning("Failed to initialize hybrid retrieval (%s); falling back to vector-only", exc)

        self._retriever = vector_retriever

    def _ensure_agent_resources(self) -> None:
        """Lazily initialize intelligent agent resources."""
        if self._agent is None:
            self._agent = WineAgent(verbose=False)

    def _get_retrieved_chunk_ids(self, retrieved_docs: list[dict[str, Any]]) -> list[str]:
        """Extract retrieved chunk IDs from retriever documents.

        Args:
            retrieved_docs: Retrieved document dictionaries from retriever.

        Returns:
            List of non-empty chunk ID strings in retrieval order.
        """
        chunk_ids: list[str] = []
        for doc in retrieved_docs:
            chunk_id = doc.get("id")
            if isinstance(chunk_id, str) and chunk_id:
                chunk_ids.append(chunk_id)
        return chunk_ids

    def _extract_contexts_from_agent_messages(self, messages: list[Any]) -> list[str]:
        """Extract text contexts from agent tool messages.

        Args:
            messages: Message objects returned by `WineAgent.invoke`.

        Returns:
            Best-effort list of textual tool outputs.
        """
        contexts: list[str] = []
        for message in messages:
            message_type = getattr(message, "type", None)
            if message_type != "tool":
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                contexts.append(content)
            elif isinstance(content, list):
                parts = [str(item) for item in content if str(item).strip()]
                if parts:
                    contexts.append(" ".join(parts))
            elif content:
                contexts.append(str(content))
        return contexts

    def _extract_config_snapshot(self) -> dict[str, Any]:
        """Extract a stable config snapshot for reproducible eval runs."""
        return {
            "model": str(self.config.model.name),
            "provider": str(self.config.model.provider),
            "embedder": str(self.config.chroma.settings.embedder),
            "n_results": int(self.config.chroma.retrieval.n_results),
            "enable_reranking": bool(self.config.chroma.retrieval.enable_reranking),
            "enable_hybrid": bool(self.config.chroma.retrieval.enable_hybrid),
        }

    def _get_git_sha(self) -> str:
        """Get short git SHA for the current working tree.

        Returns:
            Short git SHA string, or ``unknown`` when unavailable.
        """
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .strip()
            )
        except Exception:
            return "unknown"

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

