"""Shared production RAG execution path.

The API and eval harness both call this module so retrieval-affecting changes
cannot silently drift between user traffic and quality measurement.
"""

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig
from opentelemetry import trace as otel_trace

from src.utils import logger, set_span_attributes

from .confidence import RetrievalResult, compute_confidence
from .context_builder import build_context_from_chunks, deduplicate_chunks, deduplicate_chunks_async
from .factory import build_web_fallback_from_config
from .hybrid_retriever import HybridRetriever
from .query_analyzer import RetrievalQueryPlan, build_retrieval_query_plan, boost_by_metadata_match
from .query_compression import compress_context

_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def process_user_prompt(
    model: BaseChatModel,
    prompt: str,
    context: str,
    message_history: list[dict[str, Any]],
    trace_context: dict[str, str] | None = None,
) -> str:
    """Load the generation adapter lazily to keep retrieval imports acyclic."""
    from src.agents.llm import process_user_prompt as invoke_user_prompt

    return invoke_user_prompt(model, prompt, context, message_history, trace_context)


async def process_user_prompt_async(
    model: BaseChatModel,
    prompt: str,
    context: str,
    message_history: list[dict[str, Any]],
    trace_context: dict[str, str] | None = None,
) -> str:
    """Load the async generation adapter lazily to keep imports acyclic."""
    from src.agents.llm import process_user_prompt_async as invoke_user_prompt_async

    return await invoke_user_prompt_async(model, prompt, context, message_history, trace_context)


@dataclass(frozen=True)
class RAGChunkArtifact:
    """Serializable snapshot of one retrieved chunk."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_similarity: float | None = None
    bm25_score: float | None = None
    metadata_matches: int | None = None
    retrieval_channels: list[str] = field(default_factory=list)
    retrieval_diagnostics: dict[str, int | float] = field(default_factory=dict)

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "RAGChunkArtifact":
        """Build an artifact from a retriever document dictionary."""
        return cls(
            id=str(document.get("id", "")),
            text=str(document.get("document", "")),
            metadata=dict(document.get("metadata", {}) or {}),
            similarity=_optional_float(document.get("similarity")),
            rerank_score=_optional_float(document.get("rerank_score")),
            rrf_score=_optional_float(document.get("rrf_score")),
            dense_rank=_optional_int(document.get("dense_rank")),
            sparse_rank=_optional_int(document.get("sparse_rank")),
            dense_similarity=_optional_float(document.get("dense_similarity")),
            bm25_score=_optional_float(document.get("bm25_score")),
            metadata_matches=_optional_int(document.get("metadata_matches")),
            retrieval_channels=list(document.get("retrieval_channels", []) or []),
            retrieval_diagnostics=dict(document.get("retrieval_diagnostics", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class RAGSourceArtifact:
    """Source attribution independent of the API response schema."""

    name: str
    page: int | None
    relevance: float | None
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class RAGFeatureUsage:
    """Record which production RAG features were actually used."""

    retrieval: bool = False
    query_normalization: bool = False
    query_analysis: bool = False
    hybrid_retrieval: bool = False
    metadata_filtering: bool = False
    metadata_boosting: bool = False
    reranking: bool = False
    small_to_big: bool = False
    deduplication: bool = False
    compression: bool = False
    source_attribution: bool = False
    generation: bool = False
    hyde_expansion: bool = False
    rerank_thresholding: bool = False
    web_fallback: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class RAGExecutionResult:
    """Structured output from one production RAG execution."""

    answer: str
    context: str
    normalized_query: str
    retrieval_query_plan: dict[str, Any] = field(default_factory=dict)
    raw_retrieved_chunks: list[RAGChunkArtifact] = field(default_factory=list)
    context_chunks: list[RAGChunkArtifact] = field(default_factory=list)
    sources: list[RAGSourceArtifact] = field(default_factory=list)
    feature_usage: RAGFeatureUsage = field(default_factory=RAGFeatureUsage)
    retrieval_error: str | None = None
    retrieval_confidence: float | None = None
    low_confidence: bool = False
    rerank_threshold: float | None = None


@dataclass
class _RAGExecutionDraft:
    """Mutable internal state shared by sync and async orchestration."""

    query_plan: RetrievalQueryPlan
    feature_values: dict[str, bool]
    raw_artifacts: list[RAGChunkArtifact] = field(default_factory=list)
    context_artifacts: list[RAGChunkArtifact] = field(default_factory=list)
    context: str = ""
    sources: list[RAGSourceArtifact] = field(default_factory=list)
    retrieval_error: str | None = None
    retrieval_confidence: float | None = None
    low_confidence: bool = False
    rerank_threshold: float | None = None


def execute_production_rag(
    *,
    prompt: str,
    config: DictConfig,
    model: BaseChatModel | None,
    retriever: Any,
    reranker: Any,
    message_history: list[dict[str, Any]],
    enable_retrieval: bool = True,
    n_results_override: int | None = None,
    generation_enabled: bool = True,
    include_context_metadata: bool = True,
    trace_context: dict[str, str] | None = None,
) -> RAGExecutionResult:
    """Execute the production RAG path with explicit stage artifacts.

    Args:
        prompt: User question.
        config: Application configuration.
        model: Generation model. Required when generation is enabled.
        retriever: Preloaded vector or hybrid retriever.
        reranker: Optional preloaded cross-encoder reranker.
        message_history: Previous conversation turns.
        enable_retrieval: Whether to run retrieval before generation.
        n_results_override: Optional final chunk-count override.
        generation_enabled: Whether to generate the final answer.
        include_context_metadata: Whether formatted context includes source metadata.
        trace_context: Optional request trace metadata.

    Returns:
        Structured production RAG result with intermediate artifacts.

    Raises:
        ValueError: If generation is enabled without a model.
    """
    if generation_enabled and model is None:
        raise ValueError("RAG generation requires a model")

    query_plan = build_retrieval_query_plan(prompt)
    draft = _new_execution_draft(query_plan, generation_enabled=generation_enabled)

    if enable_retrieval and retriever is not None:
        try:
            retrieval_cfg = config.chroma.retrieval
            n_results = int(n_results_override or retrieval_cfg.n_results)
            retrieve_count = n_results * 2 if reranker is not None else n_results
            draft.feature_values["query_normalization"] = True
            draft.feature_values["query_analysis"] = True
            draft.feature_values["hybrid_retrieval"] = isinstance(retriever, HybridRetriever)

            tracer = otel_trace.get_tracer(__name__)
            with tracer.start_as_current_span("retrieval") as retrieval_span:
                set_span_attributes(
                    retrieval_span,
                    {
                        "retriever_type": type(retriever).__name__,
                        "n_results_requested": retrieve_count,
                        "query_intent": query_plan.intent,
                    },
                )
                if isinstance(retriever, HybridRetriever):
                    retrieved_docs = retriever.retrieve(
                        query_plan.normalized_query,
                        n_results=retrieve_count,
                        query_plan=query_plan,
                        use_rrf_fallback=reranker is None,
                    )
                else:
                    retrieved_docs = retriever.retrieve(query_plan.semantic_query, n_results=retrieve_count)
                set_span_attributes(retrieval_span, {"n_docs_retrieved": len(retrieved_docs)})

            retrieved_docs = _accept_retrieved_documents(
                draft,
                retrieved_docs,
                retrieval_cfg,
            )

            if reranker is not None:
                if retrieved_docs:
                    rerank_top_k, configured_threshold = _get_rerank_parameters(
                        retrieval_cfg,
                        n_results=n_results,
                        n_results_override=n_results_override,
                    )
                    if configured_threshold is None:
                        retrieved_docs = reranker.rerank(
                            query_plan.normalized_query,
                            retrieved_docs,
                            top_k=rerank_top_k,
                        )
                    else:
                        active_threshold = float(configured_threshold)
                        retrieved_docs = reranker.rerank_with_threshold(
                            query_plan.normalized_query,
                            retrieved_docs,
                            threshold=active_threshold,
                            top_k=rerank_top_k,
                        )
                        draft.rerank_threshold = active_threshold
                        draft.feature_values["rerank_thresholding"] = True
                    draft.feature_values["reranking"] = True
                retrieved_docs = _apply_confidence(draft, retrieved_docs, retrieval_cfg)

            enable_small_to_big = bool(getattr(config.chroma.chunking, "enable_small_to_big", False))
            if enable_small_to_big and retrieved_docs:
                retrieved_docs = _expand_to_parent_context(retrieved_docs)
                draft.feature_values["small_to_big"] = True
                logger.debug("Expanded to parent context (small-to-big)")

            context_docs = retrieved_docs
            if bool(getattr(retrieval_cfg, "use_deduplication", False)) and retrieved_docs:
                context_docs = deduplicate_chunks(
                    retrieved_docs,
                    similarity_threshold=float(retrieval_cfg.deduplication_threshold),
                    embedding_model=str(config.chroma.settings.embedder),
                )
                draft.feature_values["deduplication"] = True

            if draft.retrieval_confidence is not None:
                book_context_result = RetrievalResult(
                    documents=context_docs,
                    confidence=draft.retrieval_confidence,
                    low_confidence=draft.low_confidence,
                )
                fallback = build_web_fallback_from_config(config)
                with tracer.start_as_current_span("web_fallback") as fallback_span:
                    set_span_attributes(
                        fallback_span,
                        {
                            "enabled": fallback.enabled,
                            "low_confidence": book_context_result.low_confidence,
                        },
                    )
                    fallback_result = fallback.fetch_and_merge(prompt, book_context_result)
                    set_span_attributes(
                        fallback_span,
                        {
                            "used": fallback_result.web_fallback_used,
                            "web_document_count": len(fallback_result.documents) - len(context_docs),
                        },
                    )
                context_docs = _accept_fallback_result(draft, fallback_result)

            _accept_context_documents(
                draft,
                context_docs,
                include_metadata=include_context_metadata,
            )

            enable_compression = bool(getattr(retrieval_cfg, "enable_compression", False))
            if enable_compression and draft.context:
                draft.context = compress_context(
                    draft.context,
                    max_chars=int(getattr(retrieval_cfg, "compression_max_chars", 8000)),
                )
                draft.feature_values["compression"] = True
        except Exception as exc:
            _record_retrieval_failure(draft, exc)

    answer = ""
    if generation_enabled:
        answer = process_user_prompt(
            model,
            prompt,
            draft.context,
            message_history,
            trace_context,
        )
        draft.sources = _filter_cited_sources(answer, draft.sources)

    return _build_execution_result(draft, answer=answer)


async def execute_production_rag_async(
    *,
    prompt: str,
    config: DictConfig,
    model: BaseChatModel | None,
    retriever: Any,
    reranker: Any,
    message_history: list[dict[str, Any]],
    enable_retrieval: bool = True,
    n_results_override: int | None = None,
    generation_enabled: bool = True,
    include_context_metadata: bool = True,
    trace_context: dict[str, str] | None = None,
) -> RAGExecutionResult:
    """Execute production RAG through native async APIs and explicit bridges.

    The synchronous and asynchronous entry points share every deterministic
    transformation and public result constructor. Blocking compatibility
    stages remain explicit worker bridges until they gain an approved native
    async implementation.

    Args:
        prompt: User question.
        config: Application configuration.
        model: Generation model. Required when generation is enabled.
        retriever: Preloaded async-capable vector or hybrid retriever.
        reranker: Optional preloaded async-capable cross-encoder reranker.
        message_history: Previous conversation turns.
        enable_retrieval: Whether to run retrieval before generation.
        n_results_override: Optional final chunk-count override.
        generation_enabled: Whether to generate the final answer.
        include_context_metadata: Whether formatted context includes source metadata.
        trace_context: Optional request trace metadata.

    Returns:
        Structured production RAG result with intermediate artifacts.

    Raises:
        ValueError: If generation is enabled without a model.
    """
    if generation_enabled and model is None:
        raise ValueError("RAG generation requires a model")

    query_plan = build_retrieval_query_plan(prompt)
    draft = _new_execution_draft(query_plan, generation_enabled=generation_enabled)

    if enable_retrieval and retriever is not None:
        try:
            retrieval_cfg = config.chroma.retrieval
            n_results = int(n_results_override or retrieval_cfg.n_results)
            retrieve_count = n_results * 2 if reranker is not None else n_results
            draft.feature_values["query_normalization"] = True
            draft.feature_values["query_analysis"] = True
            draft.feature_values["hybrid_retrieval"] = isinstance(retriever, HybridRetriever)

            tracer = otel_trace.get_tracer(__name__)
            with tracer.start_as_current_span("retrieval") as retrieval_span:
                set_span_attributes(
                    retrieval_span,
                    {
                        "retriever_type": type(retriever).__name__,
                        "n_results_requested": retrieve_count,
                        "query_intent": query_plan.intent,
                    },
                )
                if isinstance(retriever, HybridRetriever):
                    retrieved_docs = await retriever.aretrieve(
                        query_plan.normalized_query,
                        n_results=retrieve_count,
                        query_plan=query_plan,
                        use_rrf_fallback=reranker is None,
                    )
                else:
                    retrieved_docs = await retriever.aretrieve(
                        query_plan.semantic_query,
                        n_results=retrieve_count,
                    )
                set_span_attributes(retrieval_span, {"n_docs_retrieved": len(retrieved_docs)})

            retrieved_docs = _accept_retrieved_documents(
                draft,
                retrieved_docs,
                retrieval_cfg,
            )

            if reranker is not None:
                if retrieved_docs:
                    rerank_top_k, configured_threshold = _get_rerank_parameters(
                        retrieval_cfg,
                        n_results=n_results,
                        n_results_override=n_results_override,
                    )
                    if configured_threshold is None:
                        retrieved_docs = await reranker.arerank(
                            query_plan.normalized_query,
                            retrieved_docs,
                            top_k=rerank_top_k,
                        )
                    else:
                        active_threshold = float(configured_threshold)
                        retrieved_docs = await reranker.arerank_with_threshold(
                            query_plan.normalized_query,
                            retrieved_docs,
                            threshold=active_threshold,
                            top_k=rerank_top_k,
                        )
                        draft.rerank_threshold = active_threshold
                        draft.feature_values["rerank_thresholding"] = True
                    draft.feature_values["reranking"] = True
                retrieved_docs = _apply_confidence(draft, retrieved_docs, retrieval_cfg)

            enable_small_to_big = bool(getattr(config.chroma.chunking, "enable_small_to_big", False))
            if enable_small_to_big and retrieved_docs:
                retrieved_docs = await asyncio.to_thread(_expand_to_parent_context, retrieved_docs)
                draft.feature_values["small_to_big"] = True
                logger.debug("Expanded to parent context (small-to-big)")

            context_docs = retrieved_docs
            if bool(getattr(retrieval_cfg, "use_deduplication", False)) and retrieved_docs:
                context_docs = await deduplicate_chunks_async(
                    retrieved_docs,
                    similarity_threshold=float(retrieval_cfg.deduplication_threshold),
                    embedding_model=str(config.chroma.settings.embedder),
                )
                draft.feature_values["deduplication"] = True

            if draft.retrieval_confidence is not None:
                book_context_result = RetrievalResult(
                    documents=context_docs,
                    confidence=draft.retrieval_confidence,
                    low_confidence=draft.low_confidence,
                )
                fallback = build_web_fallback_from_config(config)
                with tracer.start_as_current_span("web_fallback") as fallback_span:
                    set_span_attributes(
                        fallback_span,
                        {
                            "enabled": fallback.enabled,
                            "low_confidence": book_context_result.low_confidence,
                        },
                    )
                    fallback_result = await asyncio.to_thread(
                        fallback.fetch_and_merge,
                        prompt,
                        book_context_result,
                    )
                    set_span_attributes(
                        fallback_span,
                        {
                            "used": fallback_result.web_fallback_used,
                            "web_document_count": len(fallback_result.documents) - len(context_docs),
                        },
                    )
                context_docs = _accept_fallback_result(draft, fallback_result)

            _accept_context_documents(
                draft,
                context_docs,
                include_metadata=include_context_metadata,
            )

            enable_compression = bool(getattr(retrieval_cfg, "enable_compression", False))
            if enable_compression and draft.context:
                draft.context = await asyncio.to_thread(
                    compress_context,
                    draft.context,
                    max_chars=int(getattr(retrieval_cfg, "compression_max_chars", 8000)),
                )
                draft.feature_values["compression"] = True
        except Exception as exc:
            _record_retrieval_failure(draft, exc)

    answer = ""
    if generation_enabled:
        answer = await process_user_prompt_async(
            model,
            prompt,
            draft.context,
            message_history,
            trace_context,
        )
        draft.sources = _filter_cited_sources(answer, draft.sources)

    return _build_execution_result(draft, answer=answer)


def _new_execution_draft(
    query_plan: RetrievalQueryPlan,
    *,
    generation_enabled: bool,
) -> _RAGExecutionDraft:
    """Initialize the exact feature defaults for either execution mode."""
    return _RAGExecutionDraft(
        query_plan=query_plan,
        feature_values={
            "retrieval": False,
            "query_normalization": False,
            "query_analysis": False,
            "hybrid_retrieval": False,
            "metadata_filtering": False,
            "metadata_boosting": False,
            "reranking": False,
            "small_to_big": False,
            "deduplication": False,
            "compression": False,
            "source_attribution": False,
            "generation": generation_enabled,
            "hyde_expansion": False,
            "rerank_thresholding": False,
            "web_fallback": False,
        },
    )


def _accept_retrieved_documents(
    draft: _RAGExecutionDraft,
    retrieved_docs: list[dict[str, Any]],
    retrieval_cfg: Any,
) -> list[dict[str, Any]]:
    """Record raw retrieval and apply the shared metadata boost."""
    draft.feature_values["retrieval"] = True
    draft.raw_artifacts = [RAGChunkArtifact.from_document(doc) for doc in retrieved_docs]
    query_analysis = draft.query_plan.to_analysis()
    enable_metadata_boost = bool(getattr(retrieval_cfg, "enable_metadata_boost", True))
    if not (enable_metadata_boost and query_analysis.has_filters and retrieved_docs):
        return retrieved_docs

    boost_factor = float(getattr(retrieval_cfg, "metadata_boost_factor", 0.1))
    boosted_docs = boost_by_metadata_match(
        retrieved_docs,
        query_analysis,
        boost_factor=boost_factor,
    )
    draft.feature_values["metadata_boosting"] = True
    logger.debug("Applied metadata boosting for: %s", query_analysis.get_boost_terms())
    return boosted_docs


def _get_rerank_parameters(
    retrieval_cfg: Any,
    *,
    n_results: int,
    n_results_override: int | None,
) -> tuple[int, Any]:
    """Resolve the shared top-k and threshold settings for reranking."""
    top_k = (
        n_results
        if n_results_override is not None
        else int(getattr(retrieval_cfg, "rerank_top_k", n_results))
    )
    return top_k, getattr(retrieval_cfg, "rerank_threshold", None)


def _apply_confidence(
    draft: _RAGExecutionDraft,
    retrieved_docs: list[dict[str, Any]],
    retrieval_cfg: Any,
) -> list[dict[str, Any]]:
    """Apply the shared confidence calculation and record its artifacts."""
    confidence_result = compute_confidence(
        retrieved_docs,
        min_confidence=float(getattr(retrieval_cfg, "min_retrieval_confidence", 0.3)),
    )
    draft.retrieval_confidence = confidence_result.confidence
    draft.low_confidence = confidence_result.low_confidence
    logger.debug("Retrieval confidence %.4f", draft.retrieval_confidence)
    return confidence_result.documents


def _accept_fallback_result(
    draft: _RAGExecutionDraft,
    fallback_result: RetrievalResult,
) -> list[dict[str, Any]]:
    """Record web-fallback usage and return the selected context documents."""
    draft.feature_values["web_fallback"] = fallback_result.web_fallback_used
    return fallback_result.documents


def _accept_context_documents(
    draft: _RAGExecutionDraft,
    context_docs: list[dict[str, Any]],
    *,
    include_metadata: bool,
) -> None:
    """Build shared context, source, and chunk artifacts before compression."""
    draft.context_artifacts = [RAGChunkArtifact.from_document(doc) for doc in context_docs]
    draft.context = build_context_from_chunks(
        context_docs,
        include_metadata=include_metadata,
        include_similarity=False,
        max_chunks=None,
    )
    if context_docs:
        draft.sources = [_source_from_document(doc) for doc in context_docs]
        draft.feature_values["source_attribution"] = True


def _expand_to_parent_context(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load the optional contextual expansion lazily for either execution mode."""
    from src.chroma.hierarchical_chunks import expand_to_parent_context

    return expand_to_parent_context(retrieved_docs)


def _record_retrieval_failure(draft: _RAGExecutionDraft, exc: Exception) -> None:
    """Apply the established fail-soft retrieval state."""
    draft.retrieval_error = str(exc)
    draft.raw_artifacts = []
    draft.context_artifacts = []
    draft.context = ""
    draft.sources = []
    logger.error("Error during document retrieval: %s", exc)


def _build_execution_result(
    draft: _RAGExecutionDraft,
    *,
    answer: str,
) -> RAGExecutionResult:
    """Construct the public result identically for sync and async execution."""
    return RAGExecutionResult(
        answer=answer,
        context=draft.context,
        normalized_query=draft.query_plan.normalized_query,
        retrieval_query_plan=draft.query_plan.to_dict(),
        raw_retrieved_chunks=draft.raw_artifacts,
        context_chunks=draft.context_artifacts,
        sources=draft.sources,
        feature_usage=RAGFeatureUsage(**draft.feature_values),
        retrieval_error=draft.retrieval_error,
        retrieval_confidence=draft.retrieval_confidence,
        low_confidence=draft.low_confidence,
        rerank_threshold=draft.rerank_threshold,
    )


def _source_from_document(document: dict[str, Any]) -> RAGSourceArtifact:
    """Build a source artifact from a final context document."""
    metadata = dict(document.get("metadata", {}) or {})
    raw_source = str(metadata.get("source", metadata.get("filename", "Unknown")) or "Unknown")
    name = Path(raw_source).stem
    page = _optional_int(metadata.get("page", metadata.get("page_number")))
    if page is not None and page <= 0:
        page = None
    return RAGSourceArtifact(
        name=name,
        page=page,
        relevance=_optional_float(document.get("similarity")),
        chunk_id=str(document.get("id", "")),
        metadata=metadata,
    )


def _filter_cited_sources(answer: str, sources: list[RAGSourceArtifact]) -> list[RAGSourceArtifact]:
    """Keep only source artifacts cited by bracketed source number."""
    matches = _CITATION_PATTERN.findall(answer)
    if not matches:
        return sources

    cited_numbers: set[int] = set()
    for match in matches:
        cited_numbers.update(int(number.strip()) for number in match.split(","))

    cited = [sources[number - 1] for number in sorted(cited_numbers) if 1 <= number <= len(sources)]
    return cited or sources


def _optional_float(value: Any) -> float | None:
    """Convert a numeric value to float while preserving missing values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    """Convert an integer-like value while preserving missing values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
