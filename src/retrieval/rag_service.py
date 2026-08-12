"""Shared production RAG execution path.

The API and eval harness both call this module so retrieval-affecting changes
cannot silently drift between user traffic and quality measurement.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig
from opentelemetry import trace as otel_trace

from src.utils import logger, set_span_attributes

from .confidence import compute_confidence
from .context_builder import build_context_from_chunks, deduplicate_chunks
from .hybrid_retriever import HybridRetriever
from .query_analyzer import analyze_query, build_retrieval_query_plan, boost_by_metadata_match
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
    normalized_query = query_plan.normalized_query
    raw_artifacts: list[RAGChunkArtifact] = []
    context_artifacts: list[RAGChunkArtifact] = []
    context = ""
    sources: list[RAGSourceArtifact] = []
    retrieval_error: str | None = None
    retrieval_confidence: float | None = None
    low_confidence = False
    rerank_threshold: float | None = None
    feature_values: dict[str, bool] = {
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
    }

    if enable_retrieval and retriever is not None:
        try:
            retrieval_cfg = config.chroma.retrieval
            n_results = int(n_results_override or retrieval_cfg.n_results)
            retrieve_count = n_results * 2 if reranker is not None else n_results
            query_analysis = query_plan.to_analysis()
            feature_values["query_normalization"] = True
            feature_values["query_analysis"] = True
            feature_values["hybrid_retrieval"] = isinstance(retriever, HybridRetriever)

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
                        normalized_query,
                        n_results=retrieve_count,
                        query_plan=query_plan,
                        use_rrf_fallback=reranker is None,
                    )
                else:
                    retrieved_docs = retriever.retrieve(query_plan.semantic_query, n_results=retrieve_count)
                set_span_attributes(retrieval_span, {"n_docs_retrieved": len(retrieved_docs)})

            feature_values["retrieval"] = True
            raw_artifacts = [RAGChunkArtifact.from_document(doc) for doc in retrieved_docs]

            enable_metadata_boost = bool(getattr(retrieval_cfg, "enable_metadata_boost", True))
            if enable_metadata_boost and query_analysis.has_filters and retrieved_docs:
                boost_factor = float(getattr(retrieval_cfg, "metadata_boost_factor", 0.1))
                retrieved_docs = boost_by_metadata_match(
                    retrieved_docs,
                    query_analysis,
                    boost_factor=boost_factor,
                )
                feature_values["metadata_boosting"] = True
                logger.debug("Applied metadata boosting for: %s", query_analysis.get_boost_terms())

            if reranker is not None and retrieved_docs:
                rerank_top_k = (
                    n_results
                    if n_results_override is not None
                    else int(getattr(retrieval_cfg, "rerank_top_k", n_results))
                )
                configured_threshold = getattr(retrieval_cfg, "rerank_threshold", None)
                if configured_threshold is None:
                    retrieved_docs = reranker.rerank(normalized_query, retrieved_docs, top_k=rerank_top_k)
                else:
                    active_threshold = float(configured_threshold)
                    retrieved_docs = reranker.rerank_with_threshold(
                        normalized_query,
                        retrieved_docs,
                        threshold=active_threshold,
                        top_k=rerank_top_k,
                    )
                    rerank_threshold = active_threshold
                    feature_values["rerank_thresholding"] = True
                feature_values["reranking"] = True
                confidence_result = compute_confidence(
                    retrieved_docs,
                    min_confidence=float(getattr(retrieval_cfg, "min_retrieval_confidence", 0.3)),
                )
                retrieved_docs = confidence_result.documents
                retrieval_confidence = confidence_result.confidence
                low_confidence = confidence_result.low_confidence
                logger.debug(
                    "Reranked to top %d documents with confidence %.4f",
                    rerank_top_k,
                    retrieval_confidence,
                )

            enable_small_to_big = bool(getattr(config.chroma.chunking, "enable_small_to_big", False))
            if enable_small_to_big and retrieved_docs:
                from src.chroma.hierarchical_chunks import expand_to_parent_context

                retrieved_docs = expand_to_parent_context(retrieved_docs)
                feature_values["small_to_big"] = True
                logger.debug("Expanded to parent context (small-to-big)")

            context_docs = retrieved_docs
            if bool(getattr(retrieval_cfg, "use_deduplication", False)) and retrieved_docs:
                context_docs = deduplicate_chunks(
                    retrieved_docs,
                    similarity_threshold=float(retrieval_cfg.deduplication_threshold),
                    embedding_model=str(config.chroma.settings.embedder),
                )
                feature_values["deduplication"] = True

            context_artifacts = [RAGChunkArtifact.from_document(doc) for doc in context_docs]
            context = build_context_from_chunks(
                context_docs,
                include_metadata=include_context_metadata,
                include_similarity=False,
                max_chunks=None,
            )

            enable_compression = bool(getattr(retrieval_cfg, "enable_compression", False))
            if enable_compression and context:
                context = compress_context(
                    context,
                    max_chars=int(getattr(retrieval_cfg, "compression_max_chars", 8000)),
                )
                feature_values["compression"] = True

            if context_docs:
                sources = [_source_from_document(doc) for doc in context_docs]
                feature_values["source_attribution"] = True
        except Exception as exc:
            retrieval_error = str(exc)
            raw_artifacts = []
            context_artifacts = []
            context = ""
            sources = []
            logger.error("Error during document retrieval: %s", exc)

    answer = ""
    if generation_enabled:
        answer = process_user_prompt(
            model,
            prompt,
            context,
            message_history,
            trace_context,
        )
        sources = _filter_cited_sources(answer, sources)

    return RAGExecutionResult(
        answer=answer,
        context=context,
        normalized_query=normalized_query,
        retrieval_query_plan=query_plan.to_dict(),
        raw_retrieved_chunks=raw_artifacts,
        context_chunks=context_artifacts,
        sources=sources,
        feature_usage=RAGFeatureUsage(**feature_values),
        retrieval_error=retrieval_error,
        retrieval_confidence=retrieval_confidence,
        low_confidence=low_confidence,
        rerank_threshold=rerank_threshold,
    )


def _source_from_document(document: dict[str, Any]) -> RAGSourceArtifact:
    """Build a source artifact from a final context document."""
    metadata = dict(document.get("metadata", {}) or {})
    raw_source = str(metadata.get("source", metadata.get("filename", "Unknown")) or "Unknown")
    name = Path(raw_source).stem
    page_value = metadata.get("page", metadata.get("page_number"))
    try:
        page = int(page_value) if page_value is not None else None
    except (TypeError, ValueError):
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
