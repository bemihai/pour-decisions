"""
Service for generating and managing LLM-generated descriptions for wines and producers.

This service integrates with the RAG pipeline and an optional web fallback to provide
descriptions grounded in evidence that identifies the specific wine or producer.

For wines, one function-calling response is validated before description and
drinking-window values are persisted.
"""

import asyncio
import os
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field

from src.agents.llm import load_base_model
from src.agents.prompt_registry import PromptRegistry, get_prompt_registry
from src.agents.provenance import ExecutionProvenance, build_description_execution_provenance
from src.database.models import Grape, Producer, Wine
from src.database.repository import ProducerRepository, WineRepository
from src.database.repository.grape import GrapeRepository, extract_explicit_grapes
from src.retrieval import ChromaRetriever, DocumentReranker, HybridRetriever
from src.utils import (
    get_config,
    logger,
    set_execution_provenance_attributes,
    set_span_attributes,
)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve a value from a config object regardless of its type.

    Supports plain dicts, SimpleNamespace objects, and OmegaConf DictConfig.

    Args:
        obj: Config object (dict, SimpleNamespace, or OmegaConf DictConfig).
        key: Key to retrieve.
        default: Default value when the key is absent.

    Returns:
        The value for ``key``, or ``default`` if not found.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg_has(obj: Any, key: str) -> bool:
    """Check whether a setting is present, including an explicit null value."""
    if obj is None:
        return False
    if isinstance(obj, dict):
        return key in obj
    return hasattr(obj, key)


class WineAnalysis(BaseModel):
    """Structured output returned by the LLM for wine analysis.

    Used with model.with_structured_output(WineAnalysis) so the LLM response is
    automatically validated and parsed -- no JSON handling needed in the service.
    """

    description: str = Field(
        min_length=1,
        description="Evidence-grounded wine-guide paragraph describing the specific wine",
    )
    drink_from_year: int | None = Field(None, description="Year the wine begins drinking well (absolute year)")
    drink_to_year: int | None = Field(None, description="Year the wine is past its peak (absolute year)")


class DescriptionService:
    """
    Service for generating and caching LLM descriptions for wines and producers.

    Features:
    - Lazy generation (only when description is NULL)
    - RAG-enhanced descriptions using wine book context
    - Database persistence (no repeated LLM calls)
    - One cached web-search fallback when qualifying RAG context is unavailable

    Example:
        >>> service = DescriptionService()
        >>> wine = wine_repo.get_by_id(123)
        >>> description = service.get_wine_description(wine)
        >>> print(description)
        "This Chianti Classico showcases Sangiovese from Tuscany..."
    """

    def __init__(
        self,
        model: BaseChatModel | None = None,
        retriever: HybridRetriever | ChromaRetriever | None = None,
        reranker: DocumentReranker | None = None,
        use_rag_context: bool = True,
        use_web_search: bool = False,
        config: dict | None = None,
        wine_repo: WineRepository | None = None,
        producer_repo: ProducerRepository | None = None,
        prompt_registry: PromptRegistry | None = None,
        grape_repo: GrapeRepository | None = None,
    ):
        """
        Initialize the description service.

        Args:
            model: Pre-loaded direct Cloud model. When ``None``, load the
                application model from config.
            retriever: HybridRetriever for context retrieval (optional)
            reranker: DocumentReranker for result refinement (optional)
            use_rag_context: Whether to use RAG for context enrichment
            use_web_search: Whether to inject web search snippets into context.
                            Requires TAVILY_API_KEY. Defaults to False.
            config: Configuration object (loads from app_config.yml if None)
            wine_repo: Wine repository dependency.
            producer_repo: Producer repository dependency.
            grape_repo: Repository required when generating grape descriptions.
            prompt_registry: Validated prompt registry. The process-cached
                production registry is used when omitted.
        """
        self.config = config or get_config()
        self.use_rag_context = use_rag_context
        self.use_web_search = use_web_search
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_registry = prompt_registry or get_prompt_registry()
        self.wine_prompt_record = self.prompt_registry.get("wine_description")
        self.producer_prompt_record = self.prompt_registry.get("producer_description")
        self.grape_prompt_record = self.prompt_registry.get("grape_description")
        self.grape_repo = grape_repo
        self._wine_prompt_template = self.wine_prompt_record.source
        self._producer_prompt_template = self.producer_prompt_record.source

        desc_cfg = _cfg_get(self.config, "description_generation")
        self.enable_web_search = bool(_cfg_get(desc_cfg, "enable_web_search", True))
        web_search_cfg = _cfg_get(self.config, "web_search")
        tavily_cfg = _cfg_get(web_search_cfg, "tavily")
        self.web_search_api_key_env = str(_cfg_get(tavily_cfg, "api_key_env", "TAVILY_API_KEY"))

        # Load LLM model
        if model is None:
            model_cfg = _cfg_get(self.config, "model")
            desc_cfg = _cfg_get(self.config, "description_generation")
            if model_cfg is None:
                raise ValueError("Cloud model configuration is required for descriptions")
            if any(
                _cfg_has(model_cfg, key)
                for key in ("fallback_provider", "fallback_name", "ollama", "hybrid_tool_calling")
            ):
                raise ValueError("Legacy fallback or local model settings are unsupported")
            if _cfg_has(desc_cfg, "use_cloud_model"):
                raise ValueError("Legacy description model selection is unsupported")
            provider = str(_cfg_get(model_cfg, "provider", ""))
            model_name = str(_cfg_get(model_cfg, "name", ""))
            self.model = load_base_model(
                provider,
                model_name,
                base_url=str(_cfg_get(model_cfg, "base_url", "https://ollama.com")),
                timeout=float(_cfg_get(model_cfg, "timeout_seconds", 60)),
            )
            logger.info("Loaded direct Cloud model for description generation: %s", model_name)
        else:
            self.model = model
            logger.info(f"Using provided model for description generation: {type(model).__name__}")

        # Structured-output model for wine analysis (description + drinking window)
        self._structured_model = self.model.with_structured_output(WineAnalysis, method="function_calling")
        self.wine_execution_provenance = build_description_execution_provenance(
            entity_type="wine",
            prompt=self.wine_prompt_record,
            model=self.model,
        )
        self.producer_execution_provenance = build_description_execution_provenance(
            entity_type="producer",
            prompt=self.producer_prompt_record,
            model=self.model,
        )

        self.grape_execution_provenance = build_description_execution_provenance(
            entity_type="grape",
            prompt=self.grape_prompt_record,
            model=self.model,
        )

        # Web search engine (lazy init inside method to avoid import cost when disabled)
        self._web_search_engine = None
        self._web_search_available = True
        if self.use_web_search and not self.enable_web_search:
            self.use_web_search = False
            self._web_search_available = False
            logger.info(
                "Web search disabled for description generation: " "description_generation.enable_web_search is false"
            )
        elif self.use_web_search and not os.getenv(self.web_search_api_key_env, "").strip():
            self.use_web_search = False
            self._web_search_available = False
            logger.info(
                "Web search disabled for description generation: " f"{self.web_search_api_key_env} is not configured"
            )

        logger.info(
            "DescriptionService web-search status: requested=%s enabled_by_config=%s "
            "effective_use_web_search=%s api_key_env=%s api_key_present=%s",
            use_web_search,
            self.enable_web_search,
            self.use_web_search,
            self.web_search_api_key_env,
            bool(os.getenv(self.web_search_api_key_env, "").strip()),
        )

        if wine_repo is None or producer_repo is None:
            raise ValueError("DescriptionService requires explicit wine_repo and producer_repo dependencies")
        self.wine_repo = wine_repo
        self.producer_repo = producer_repo

        # RAG context configuration
        self.max_context_chunks = _cfg_get(desc_cfg, "max_context_chunks", 3)
        self.min_relevance_score = _cfg_get(desc_cfg, "min_relevance_score", 0.4)

        logger.info(
            f"DescriptionService initialized (RAG: {use_rag_context}, "
            f"web_search: {use_web_search}, max_chunks: {self.max_context_chunks})"
        )

    @contextmanager
    def _start_description_span(self, entity_type: str, entity_id: int | None):
        """Create a description-generation span tagged for Phoenix filtering.

        Args:
            entity_type: Entity type being described (``"wine"``, ``"producer"``, or ``"grape"``).
            entity_id: Entity ID when available.

        Yields:
            Active span for the description-generation operation.
        """
        tracer = otel_trace.get_tracer(__name__)
        span_name = f"description_generation.{entity_type}"
        with tracer.start_as_current_span(span_name) as span:
            set_span_attributes(
                span,
                {
                    "feature": "description_generation",
                    "agent_mode": "background",
                    "entity_type": entity_type,
                    "entity_id": str(entity_id) if entity_id is not None else None,
                },
            )
            provenance = getattr(
                self,
                f"{entity_type}_execution_provenance",
                None,
            )
            if isinstance(provenance, ExecutionProvenance):
                set_execution_provenance_attributes(
                    span,
                    provenance.to_trace_attributes(),
                )
            yield span

    def get_wine_description(self, wine: Wine, force_regenerate: bool = False) -> str | None:
        """Get description for a wine, generating if not cached.

        Uses LLM structured output (WineAnalysis) to extract both the description
        and a drinking window estimate in a single call. The drinking window is
        persisted with source='llm' only when no higher-priority source exists.
        The description is persisted as before.

        Args:
            wine: Wine model with all relevant fields.
            force_regenerate: When ``True``, skip the cached description and
                regenerate unconditionally. Defaults to ``False``.

        Returns:
            Description string or None if generation fails.

        Example:
            >>> wine = Wine(id=1, wine_name="Tignanello", producer_name="Antinori",
            ...             vintage=2019, wine_type="Red", varietal="Sangiovese")
            >>> description = service.get_wine_description(wine)
        """
        if wine.description and not force_regenerate:
            logger.debug(f"Using cached description for wine ID {wine.id}")
            return wine.description

        logger.info(f"Generating description for wine: {wine.wine_name} ({wine.vintage})")

        try:
            with self._start_description_span("wine", wine.id):
                query = self._build_wine_search_query(wine)
                context_section = self._build_context_section(query, wine)
                if not context_section:
                    logger.warning("No qualifying evidence for wine ID %s", wine.id)
                    return None

                prompt = self._format_wine_prompt(wine, context_section)

                analysis = self._invoke_structured(prompt)
                if analysis is None:
                    return None

                return self._persist_wine_analysis(wine, analysis)

        except Exception as e:
            logger.error("Error generating wine description: %s", type(e).__name__)
            return None

    async def aget_wine_description(self, wine: Wine, force_regenerate: bool = False) -> str | None:
        """Get or generate a wine description without blocking the event loop.

        Args:
            wine: Wine model with all relevant fields.
            force_regenerate: When ``True``, regenerate even when cached.

        Returns:
            Description string or None if generation fails.
        """
        if wine.description and not force_regenerate:
            logger.debug(f"Using cached description for wine ID {wine.id}")
            return wine.description

        logger.info(f"Generating description for wine: {wine.wine_name} ({wine.vintage})")

        try:
            with self._start_description_span("wine", wine.id):
                query = self._build_wine_search_query(wine)
                context_section = await self._abuild_context_section(query, wine)
                if not context_section:
                    logger.warning("No qualifying evidence for wine ID %s", wine.id)
                    return None
                prompt = self._format_wine_prompt(wine, context_section)

                analysis = await self._ainvoke_structured(prompt)
                if analysis is None:
                    return None

                return await asyncio.to_thread(self._persist_wine_analysis, wine, analysis)

        except Exception as e:
            logger.error("Error generating wine description: %s", type(e).__name__)
            return None

    def get_producer_description(self, producer: Producer) -> str | None:
        """
        Get description for a producer, generating if not cached.

        The description is generated on first call and persisted to database.
        Subsequent calls return the cached value.

        Args:
            producer: Producer model with id, name, country, region

        Returns:
            Description string or None if generation fails

        Example:
            >>> producer = Producer(
            ...     id=1,
            ...     name="Antinori",
            ...     country="Italy",
            ...     region="Tuscany"
            ... )
            >>> description = service.get_producer_description(producer)
        """
        # Return cached description if exists
        if producer.description:
            logger.debug(f"Using cached description for producer ID {producer.id}")
            return producer.description

        # Generate new description
        logger.info(f"Generating description for producer: {producer.name}")

        try:
            with self._start_description_span("producer", producer.id):
                query = self._build_producer_search_query(producer)
                context_section = self._build_producer_context_section(query, producer)
                if not context_section:
                    logger.warning("No qualifying evidence for producer ID %s", producer.id)
                    return None

                prompt = self._format_producer_prompt(producer, context_section)

                description = self._generate_with_llm(prompt)

                if description:
                    return self._persist_producer_description(producer, description)

                logger.warning(f"Failed to generate description for producer ID {producer.id}")
                return None

        except Exception as e:
            logger.error("Error generating producer description: %s", type(e).__name__)
            return None

    async def aget_producer_description(self, producer: Producer) -> str | None:
        """Get or generate a producer description without blocking the event loop.

        Args:
            producer: Producer model with identifying fields.

        Returns:
            Description string or None if generation fails.
        """
        if producer.description:
            logger.debug(f"Using cached description for producer ID {producer.id}")
            return producer.description

        logger.info(f"Generating description for producer: {producer.name}")

        try:
            with self._start_description_span("producer", producer.id):
                query = self._build_producer_search_query(producer)
                context_section = await self._abuild_producer_context_section(query, producer)
                if not context_section:
                    logger.warning("No qualifying evidence for producer ID %s", producer.id)
                    return None

                prompt = self._format_producer_prompt(producer, context_section)
                description = await self._agenerate_with_llm(prompt)

                if description:
                    return await asyncio.to_thread(self._persist_producer_description, producer, description)

                logger.warning(f"Failed to generate description for producer ID {producer.id}")
                return None

        except Exception as e:
            logger.error("Error generating producer description: %s", type(e).__name__)
            return None

    async def aget_grape_description(self, grape: Grape, force_regenerate: bool = False) -> str | None:
        """Generate on request, preserving any cached description on failure.

        Args:
            grape: Persisted canonical grape to describe.
            force_regenerate: Ignore the cache only for an explicit regeneration.

        Returns:
            Cached or newly persisted description, or None on failure.
        """
        if grape.description and not force_regenerate:
            return grape.description
        if self.grape_repo is None:
            raise ValueError("Grape descriptions require an explicit grape_repo")
        try:
            with self._start_description_span("grape", grape.id):
                query = f"{grape.name} wine grape typical character style growing regions"
                context = ""
                if self.use_rag_context and self.retriever:
                    chunks = await self._aretrieve_context(query, self.max_context_chunks)
                    qualifying = [
                        chunk
                        for chunk in (chunks or [])
                        if grape.name in extract_explicit_grapes(str(chunk.get("document", chunk.get("content", ""))))
                    ]
                    context = self._format_context_section(qualifying)
                if not context and self.use_web_search:
                    context = await asyncio.to_thread(self._get_grape_web_search_context, grape, query)
                if not context:
                    logger.warning("No qualifying evidence for grape ID %s", grape.id)
                    return None
                prompt = self.grape_prompt_record.source.format(grape_name=grape.name, context_section=context)
                description = await self._agenerate_with_llm(prompt)
                if not description:
                    return None
                saved = await asyncio.to_thread(self.grape_repo.update_description, grape.id, description)
                if not saved:
                    raise RuntimeError("Grape description update failed")
                grape.description = description
                return description
        except Exception as error:
            logger.error("Grape description generation failed: %s", type(error).__name__)
            return None

    def _get_grape_web_search_context(self, grape: Grape, query: str) -> str:
        """Fetch one cached web fallback with evidence naming the grape."""
        if not self._web_search_available:
            return ""
        try:
            if self._web_search_engine is None:
                from src.services.web_search import WineWebSearchEngine

                self._web_search_engine = WineWebSearchEngine()
            results = self._web_search_engine.search(query, search_type="general", max_results=3)
            evidence = []
            for result in results:
                text = ": ".join(str(result.get(key, "")).strip() for key in ("title", "snippet"))
                if grape.name in extract_explicit_grapes(text):
                    evidence.append(text)
            return "Web Search Context:\n" + "\n".join(evidence) if evidence else ""
        except Exception as error:
            logger.warning("Grape web search failed: %s", type(error).__name__)
            return ""

    def generate_batch(
        self, producers: list[Producer] | None = None, wines: list[Wine] | None = None
    ) -> dict[str, int]:
        """
        Batch generate descriptions for multiple items.

        Useful for pre-generating descriptions for newly imported items
        or regenerating descriptions for items with NULL values.

        Args:
            producers: List of producers to generate descriptions for
            wines: List of wines to generate descriptions for

        Returns:
            Dict with counts: {"producers_generated": N, "wines_generated": M}

        Example:
            >>> producers = producer_repo.get_all()
            >>> wines = wine_repo.get_all()
            >>> stats = service.generate_batch(producers, wines)
            >>> print(f"Generated {stats['wines_generated']} wine descriptions")
        """
        stats = {"producers_generated": 0, "wines_generated": 0}

        # Generate producer descriptions
        if producers:
            logger.info(f"Batch generating descriptions for {len(producers)} producers")
            for producer in producers:
                if not producer.description:  # Only generate if NULL
                    if self.get_producer_description(producer):
                        stats["producers_generated"] += 1

        # Generate wine descriptions
        if wines:
            logger.info(f"Batch generating descriptions for {len(wines)} wines")
            for wine in wines:
                if not wine.description:  # Only generate if NULL
                    if self.get_wine_description(wine):
                        stats["wines_generated"] += 1

        logger.info(
            f"Batch generation complete: {stats['producers_generated']} producers, " f"{stats['wines_generated']} wines"
        )
        return stats

    def _retrieve_context(self, query: str, max_chunks: int = 5) -> list[dict[str, Any]] | None:
        """
        Retrieve relevant context from wine books using RAG pipeline.

        Args:
            query: Search query built from wine/producer data
            max_chunks: Maximum number of chunks to return

        Returns:
            List of context chunks or None if no relevant results
        """
        if not self.retriever:
            return None

        try:
            # Retrieve with hybrid search (vector + BM25)
            results = self.retriever.retrieve(query, n_results=max_chunks * 2)

            # Rerank if available
            if self.reranker and results:
                results = self.reranker.rerank(query, results, top_k=max_chunks)

            return self._select_relevant_context(results, max_chunks)

        except Exception as e:
            logger.warning("Error retrieving context: %s", type(e).__name__)
            return None

    async def _aretrieve_context(self, query: str, max_chunks: int = 5) -> list[dict[str, Any]] | None:
        """Retrieve and rerank context through the async retrieval boundary."""
        if not self.retriever:
            return None

        try:
            results = await self.retriever.aretrieve(query, n_results=max_chunks * 2)
            if self.reranker and results:
                results = await self.reranker.arerank(query, results, top_k=max_chunks)
            return self._select_relevant_context(results, max_chunks)
        except Exception as e:
            logger.warning("Error retrieving context: %s", type(e).__name__)
            return None

    def _select_relevant_context(
        self,
        results: list[dict[str, Any]],
        max_chunks: int,
    ) -> list[dict[str, Any]] | None:
        """Apply the shared relevance threshold and context limit."""
        relevant = [result for result in results if result.get("distance", 1.0) <= (1.0 - self.min_relevance_score)]
        return relevant[:max_chunks] or None

    def _format_context_section(self, chunks: list[dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context section for prompt.

        Args:
            chunks: List of retrieved document chunks

        Returns:
            Formatted context section string
        """
        if not chunks:
            return ""

        formatted_chunks = []
        for chunk in chunks:
            # Extract content and metadata
            content = chunk.get("document", chunk.get("content", ""))
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown")
            page = metadata.get("page_number", metadata.get("page", ""))

            # Format chunk with source attribution
            chunk_text = f"{content.strip()}"
            if source or page:
                chunk_text += f"\n[Source: {source}"
                if page:
                    chunk_text += f", Page {page}"
                chunk_text += "]"

            formatted_chunks.append(chunk_text)

        # Build context section
        context_section = (
            "Reference Context:\n"
            "The following excerpts from wine reference books may contain relevant information:\n\n"
            + "\n---\n".join(formatted_chunks)
        )

        return context_section

    def _build_wine_search_query(self, wine: Wine) -> str:
        """
        Build an effective search query from wine attributes.

        Prioritizes specific identifiers (appellation, varietal) over generic fields.

        Args:
            wine: Wine model

        Returns:
            Search query string

        Example:
            "Tignanello Sangiovese Super Tuscan Antinori Tuscany"
        """
        fields = (
            wine.wine_name,
            wine.producer_name,
            wine.vintage,
            wine.varietal,
            getattr(wine, "designation", None),
            getattr(wine, "vineyard", None),
            wine.appellation,
            wine.region_name,
            wine.country,
        )
        return " ".join(str(field).strip() for field in fields if field is not None and str(field).strip())

    def _build_producer_search_query(self, producer: Producer) -> str:
        """
        Build an effective search query from producer attributes.

        Args:
            producer: Producer model

        Returns:
            Search query string

        Example:
            "Antinori Tuscany Italy winery"
        """
        query_parts = []

        # Producer name (most important)
        if producer.name:
            query_parts.append(producer.name)

        # Region (for context)
        if producer.region:
            query_parts.append(producer.region)

        # Country
        if producer.country:
            query_parts.append(producer.country)

        # Add "winery" or "producer" to help find relevant passages
        query_parts.append("winery")

        return " ".join(query_parts)

    @staticmethod
    def _normalize_evidence_text(value: str) -> str:
        """Normalize text for strict evidence identity matching."""
        decomposed = unicodedata.normalize("NFKD", value.casefold())
        without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
        without_punctuation = "".join(
            character if character.isalnum() or character.isspace() else " " for character in without_accents
        )
        return " ".join(without_punctuation.split())

    @classmethod
    def _contains_normalized_phrase(cls, value: str, phrase: str) -> bool:
        """Return whether evidence contains a complete normalized identity phrase."""
        normalized_phrase = cls._normalize_evidence_text(phrase)
        if not normalized_phrase:
            return False
        normalized_value = cls._normalize_evidence_text(value)
        return f" {normalized_phrase} " in f" {normalized_value} "

    @classmethod
    def _contains_all_normalized_tokens(cls, value: str, phrase: str) -> bool:
        """Return whether every identity token occurs within a small text window."""
        identity_tokens = cls._normalize_evidence_text(phrase).split()
        if not identity_tokens:
            return False
        evidence_tokens = cls._normalize_evidence_text(value).split()
        window_size = len(identity_tokens) + 4
        for start in range(len(evidence_tokens)):
            window_tokens = evidence_tokens[start : start + window_size]
            if all(window_tokens.count(token) >= identity_tokens.count(token) for token in set(identity_tokens)):
                return True
        return False

    @classmethod
    def _contains_unquantified_normalized_phrase(cls, value: str, phrase: str) -> bool:
        """Return whether a phrase occurs without an adjacent percentage.

        A grape mentioned only as a percentage in a blend does not identify the
        wine's primary varietal. Preserve percent markers while applying the
        established text normalization so that both prefix and suffix forms are
        recognized.
        """
        phrase_tokens = cls._normalize_evidence_text(phrase).split()
        if not phrase_tokens:
            return False
        decomposed = unicodedata.normalize("NFKD", value.casefold())
        without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
        quantity_aware_value = "".join(
            character if character.isalnum() or character.isspace() or character == "%" else " "
            for character in without_accents
        )
        evidence_tokens = quantity_aware_value.split()
        phrase_length = len(phrase_tokens)
        for start in range(len(evidence_tokens) - phrase_length + 1):
            if evidence_tokens[start : start + phrase_length] != phrase_tokens:
                continue
            previous_token = evidence_tokens[start - 1] if start > 0 else ""
            following_tokens = evidence_tokens[start + phrase_length : start + phrase_length + 2]
            followed_by_percentage = bool(following_tokens) and (
                following_tokens[0].endswith("%") or (len(following_tokens) == 2 and following_tokens[1] == "%")
            )
            if not previous_token.endswith("%") and not followed_by_percentage:
                return True
        return False

    @classmethod
    def _contains_normalized_producer_name(cls, value: str, producer_name: str) -> bool:
        """Return whether evidence explicitly contains the full normalized producer name."""
        return cls._contains_normalized_phrase(value, producer_name)

    @classmethod
    def _is_wine_specific_evidence(cls, value: str, wine: Wine) -> bool:
        """Require the producer and one complete wine identity in one text value.

        Exact normalized phrases remain preferred. The complete wine name may also
        match when all its tokens occur with a different order or intervening words.
        Shorter designation or vineyard matches additionally require the stored
        varietal. This supports source subtitles and reordered classifications
        without accepting partial cuvee names.
        """
        if not cls._contains_normalized_phrase(value, wine.producer_name or ""):
            return False
        wine_name = wine.wine_name or ""
        if cls._contains_normalized_phrase(value, wine_name):
            return True

        varietal = wine.varietal or ""
        if cls._contains_all_normalized_tokens(value, wine_name):
            wine_name_contains_varietal = cls._contains_normalized_phrase(wine_name, varietal)
            if not wine_name_contains_varietal or cls._contains_unquantified_normalized_phrase(value, varietal):
                return True

        if not cls._contains_unquantified_normalized_phrase(value, varietal):
            return False
        shorter_identities = [getattr(wine, "designation", None), getattr(wine, "vineyard", None)]
        return any(
            cls._contains_all_normalized_tokens(value, f"{varietal} {identity}")
            for identity in shorter_identities
            if identity
        )

    @classmethod
    def _select_wine_evidence(cls, chunks: list[dict[str, Any]] | None, wine: Wine) -> list[dict[str, Any]]:
        """Return RAG chunks that explicitly identify both producer and wine."""
        if not chunks:
            return []
        return [
            chunk
            for chunk in chunks
            if cls._is_wine_specific_evidence(str(chunk.get("document", chunk.get("content", ""))), wine)
        ]

    def _select_producer_evidence(
        self,
        chunks: list[dict[str, Any]] | None,
        producer_name: str,
    ) -> list[dict[str, Any]]:
        """Keep only RAG chunks whose content explicitly names the producer."""
        if not chunks:
            return []
        return [
            chunk
            for chunk in chunks
            if self._contains_normalized_producer_name(
                str(chunk.get("document", chunk.get("content", ""))),
                producer_name,
            )
        ]

    def _build_producer_context_section(self, query: str, producer: Producer) -> str:
        """Return qualifying RAG evidence or one producer web-search fallback."""
        if self.use_rag_context and self.retriever:
            context = self._retrieve_context(query, self.max_context_chunks)
            qualifying_context = self._select_producer_evidence(context, producer.name or "")
            if qualifying_context:
                logger.debug("Retrieved %s qualifying context chunks for producer", len(qualifying_context))
                return self._format_context_section(qualifying_context)

        if self.use_web_search:
            return self._get_producer_web_search_context(producer)
        return ""

    async def _abuild_producer_context_section(self, query: str, producer: Producer) -> str:
        """Build producer context using async retrieval and worker-backed web search."""
        if self.use_rag_context and self.retriever:
            context = await self._aretrieve_context(query, self.max_context_chunks)
            qualifying_context = self._select_producer_evidence(context, producer.name or "")
            if qualifying_context:
                logger.debug("Retrieved %s qualifying context chunks for producer", len(qualifying_context))
                return self._format_context_section(qualifying_context)

        if self.use_web_search:
            return await asyncio.to_thread(self._get_producer_web_search_context, producer)
        return ""

    def _format_wine_prompt(self, wine: Wine, context_section: str) -> str:
        """Format the registered wine prompt for sync and async generation."""
        return self._wine_prompt_template.format(
            wine_name=wine.wine_name or "Unknown",
            producer_name=wine.producer_name or "Unknown",
            vintage=wine.vintage or "NV",
            wine_type=wine.wine_type or "Unknown",
            varietal=wine.varietal or "Unknown",
            region=wine.region_name or "Unknown",
            country=wine.country or "Unknown",
            appellation=wine.appellation or "N/A",
            designation=getattr(wine, "designation", None) or "N/A",
            vineyard=getattr(wine, "vineyard", None) or "N/A",
            context_section=context_section,
        )

    def _format_producer_prompt(self, producer: Producer, context_section: str) -> str:
        """Format the registered producer prompt for sync and async generation."""
        return self._producer_prompt_template.format(
            producer_name=producer.name or "Unknown",
            country=producer.country or "Unknown",
            region=producer.region or "Unknown",
            context_section=context_section,
        )

    @staticmethod
    def _validate_wine_analysis(result: Any) -> "WineAnalysis | None":
        """Validate a structured wine response without issuing another model call."""
        if not isinstance(result, WineAnalysis):
            logger.warning("Structured wine analysis returned an invalid result")
            return None
        validated = WineAnalysis.model_validate(result.model_dump())
        if not validated.description.strip():
            logger.warning("Structured wine analysis returned an empty description")
            return None
        return validated

    def _invoke_structured(self, prompt: str) -> "WineAnalysis | None":
        """Return validated structured output or fail without a second model call.

        Args:
            prompt: Formatted prompt string.

        Returns:
            WineAnalysis instance or None on hard failure.
        """
        try:
            result = self._structured_model.invoke(prompt)
            return self._validate_wine_analysis(result)
        except Exception as error:
            logger.warning("Structured wine analysis failed: %s", type(error).__name__)
            return None

    async def _ainvoke_structured(self, prompt: str) -> "WineAnalysis | None":
        """Await one structured model call and validate its result."""
        try:
            result = await self._structured_model.ainvoke(prompt)
            return self._validate_wine_analysis(result)
        except Exception as error:
            logger.warning("Structured wine analysis failed: %s", type(error).__name__)
            return None

    def _persist_wine_analysis(self, wine: Wine, analysis: WineAnalysis) -> str:
        """Persist validated wine output with established precedence and rollback behavior."""
        previous_description = wine.description
        wine.description = analysis.description
        try:
            if not self.wine_repo.update(wine):
                raise RuntimeError("Wine description update failed")
        except Exception:
            wine.description = previous_description
            raise
        logger.info(f"Saved description for wine ID {wine.id}")

        if analysis.drink_from_year and analysis.drink_to_year and wine.id:
            self._persist_llm_drinking_window(wine, analysis.drink_from_year, analysis.drink_to_year)

        return analysis.description

    def _persist_llm_drinking_window(self, wine: Wine, from_year: int, to_year: int) -> None:
        """Persist an LLM-estimated drinking window if no higher-priority source exists.

        Args:
            wine: Wine model (needs wine.id and wine.drink_window_source).
            from_year: Estimated window start.
            to_year: Estimated window end.
        """
        from src.agents.drinking_window_service import compute_drink_index

        _MIN_YEAR = 1900
        _MAX_FUTURE_YEARS = 50
        current_year = datetime.now().year
        if from_year >= to_year:
            logger.debug(
                f"Skipping LLM drinking window for wine {wine.id}: "
                f"from_year must be less than to_year ({from_year}-{to_year})"
            )
            return
        if not (
            _MIN_YEAR <= from_year <= current_year + _MAX_FUTURE_YEARS
            and _MIN_YEAR <= to_year <= current_year + _MAX_FUTURE_YEARS
        ):
            logger.debug(f"Skipping LLM drinking window for wine {wine.id}: unrealistic years ({from_year}-{to_year})")
            return

        _priority = {"manual": 1, "cellar_tracker": 2, "llm": 3, "heuristic": 4}
        existing_priority = _priority.get(wine.drink_window_source or "", 99)
        if _priority["llm"] > existing_priority:
            logger.debug(
                f"Skipping LLM drinking window for wine {wine.id}: "
                f"existing source '{wine.drink_window_source}' has higher priority"
            )
            return

        index = compute_drink_index(from_year, to_year)
        updated = self.wine_repo.update_drinking_window(
            wine.id, from_year, to_year, index, "llm"
        )  # ty:ignore[invalid-argument-type]
        if updated:
            logger.info(f"Saved LLM drinking window for wine {wine.id}: {from_year}-{to_year}")

    def _build_context_section(self, query: str, wine: Wine) -> str:
        """Assemble the context section from RAG chunks and optional web search snippets.

        Args:
            query: Search query derived from wine attributes.
            wine: Wine model (used to build a focused web search query).

        Returns:
            Formatted context string to inject into the prompt, or empty string.
        """
        if self.use_rag_context and self.retriever:
            context = self._retrieve_context(query, self.max_context_chunks)
            qualifying_context = self._select_wine_evidence(context, wine)
            if qualifying_context:
                logger.debug("Retrieved %s qualifying RAG chunks for wine", len(qualifying_context))
                return self._format_context_section(context or qualifying_context)

        if self.use_web_search:
            logger.info(
                "DescriptionService web search triggered: wine_id=%s query=%s",
                wine.id,
                query,
            )
            snippets = self._get_web_search_snippets(wine)
            if snippets:
                logger.info("DescriptionService web search used: wine_id=%s snippets_added=true", wine.id)
                return snippets
            else:
                logger.info("DescriptionService web search used: wine_id=%s snippets_added=false", wine.id)

        return ""

    async def _abuild_context_section(self, query: str, wine: Wine) -> str:
        """Assemble wine context using async retrieval and worker-backed web search."""
        if self.use_rag_context and self.retriever:
            context = await self._aretrieve_context(query, self.max_context_chunks)
            qualifying_context = self._select_wine_evidence(context, wine)
            if qualifying_context:
                logger.debug("Retrieved %s qualifying RAG chunks for wine", len(qualifying_context))
                return self._format_context_section(context or qualifying_context)

        if self.use_web_search:
            logger.info(
                "DescriptionService web search triggered: wine_id=%s query=%s",
                wine.id,
                query,
            )
            snippets = await asyncio.to_thread(self._get_web_search_snippets, wine)
            if snippets:
                logger.info("DescriptionService web search used: wine_id=%s snippets_added=true", wine.id)
                return snippets
            else:
                logger.info("DescriptionService web search used: wine_id=%s snippets_added=false", wine.id)

        return ""

    def _get_web_search_snippets(self, wine: Wine) -> str:
        """Fetch and format web search results for a wine's aging potential.

        Results are cached (7-day TTL) by the WineWebSearchEngine, so repeated
        calls for the same wine are free after the first hit.

        Args:
            wine: Wine model.

        Returns:
            Formatted snippet string or empty string if unavailable.
        """
        if not self._web_search_available:
            logger.info("DescriptionService web search skipped: wine_id=%s reason=not_available", wine.id)
            return ""

        if self._web_search_engine is None:
            try:
                from src.services.web_search import WineWebSearchEngine

                self._web_search_engine = WineWebSearchEngine()
            except Exception as e:
                self._web_search_available = False
                logger.warning("Could not initialise web search engine: %s", type(e).__name__)
                return ""

        query = f"{self._build_wine_search_query(wine)} tasting review drinking window"

        try:
            results = self._web_search_engine.search(query, search_type="review", max_results=3)
            if not results:
                logger.info("DescriptionService web search completed: wine_id=%s results=0", wine.id)
                return ""
            qualifying_results = []
            for result in results:
                title = str(result.get("title", "")).strip()
                snippet = str(result.get("snippet", "")).strip()
                if self._is_wine_specific_evidence(title, wine) or self._is_wine_specific_evidence(snippet, wine):
                    qualifying_results.append((title, snippet))
            logger.info(
                "DescriptionService web search completed: wine_id=%s results=%s qualifying=%s",
                wine.id,
                len(results),
                len(qualifying_results),
            )
            lines = ["Web Search Context:"]
            for title, snippet in qualifying_results:
                if snippet:
                    lines.append(f"- {title}: {snippet}" if title else f"- {snippet}")
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as e:
            logger.warning("Web search failed for wine context: %s", type(e).__name__)
            return ""

    def _get_producer_web_search_context(self, producer: Producer) -> str:
        """Return web results whose title or snippet explicitly names the producer."""
        if not self._web_search_available:
            logger.info("Producer web search skipped: producer_id=%s reason=not_available", producer.id)
            return ""

        if self._web_search_engine is None:
            try:
                from src.services.web_search import WineWebSearchEngine

                self._web_search_engine = WineWebSearchEngine()
            except Exception as error:
                self._web_search_available = False
                logger.warning("Could not initialise producer web search: %s", type(error).__name__)
                return ""

        query = self._build_producer_search_query(producer)
        try:
            results = self._web_search_engine.search(query, search_type="producer", max_results=3)
        except Exception as error:
            logger.warning("Web search failed for producer context: %s", type(error).__name__)
            return ""

        qualifying_results = []
        for result in results:
            title = str(result.get("title", "")).strip()
            snippet = str(result.get("snippet", "")).strip()
            if self._contains_normalized_producer_name(
                title, producer.name or ""
            ) or self._contains_normalized_producer_name(snippet, producer.name or ""):
                qualifying_results.append((title, snippet))

        logger.info(
            "Producer web search completed: producer_id=%s results=%s qualifying=%s",
            producer.id,
            len(results),
            len(qualifying_results),
        )
        if not qualifying_results:
            return ""

        lines = ["Web Search Context:"]
        for title, snippet in qualifying_results:
            evidence = ": ".join(part for part in (title, snippet) if part)
            lines.append(f"- {evidence}")
        return "\n".join(lines)

    def _generate_with_llm(self, prompt: str) -> str | None:
        """Generate a plain-text description using the LLM (used for producers).

        Args:
            prompt: Formatted prompt string.

        Returns:
            Generated description string or None if failed.
        """
        try:
            response = self.model.invoke(prompt)
            return self._normalize_generated_description(response)

        except Exception as e:
            logger.error("LLM generation failed: %s", type(e).__name__)
            return None

    async def _agenerate_with_llm(self, prompt: str) -> str | None:
        """Await one plain-text model call and normalize its response."""
        try:
            response = await self.model.ainvoke(prompt)
            return self._normalize_generated_description(response)
        except Exception as e:
            logger.error("LLM generation failed: %s", type(e).__name__)
            return None

    @staticmethod
    def _normalize_generated_description(response: Any) -> str | None:
        """Normalize a producer response without truncating its supported narrative."""
        description = (response.content if hasattr(response, "content") else str(response)).strip()

        if len(description) < 20:
            logger.warning(f"Generated description too short: {len(description)} chars")
            return None

        return description

    def _persist_producer_description(self, producer: Producer, description: str) -> str:
        """Persist a producer description with supplied-object rollback on failure."""
        previous_description = producer.description
        producer.description = description
        try:
            if not self.producer_repo.update(producer):
                raise RuntimeError("Producer description update failed")
        except Exception:
            producer.description = previous_description
            raise
        logger.info(f"Generated and saved description for producer ID {producer.id}")
        return description


# Singleton instance for easy access across the application
_service_instance: DescriptionService | None = None
_service_config: dict[str, Any] | None = None


def get_description_service(
    retriever: HybridRetriever | ChromaRetriever | None = None,
    reranker: DocumentReranker | None = None,
    use_rag_context: bool = True,
    use_web_search: bool = False,
    model: BaseChatModel | None = None,
) -> DescriptionService:
    """Get or create the singleton DescriptionService instance.

    The instance is recreated when use_rag_context or use_web_search changes.

    When ``model`` is ``None``, the service loads the configured direct Cloud model.

    Args:
        retriever: Optional retriever (used when creating/recreating instance).
        reranker: Optional reranker (used when creating/recreating instance).
        use_rag_context: Whether to use RAG for context enrichment.
        use_web_search: Whether to inject web search snippets. Requires TAVILY_API_KEY.
        model: Optional pre-loaded LLM to override automatic model selection.

    Returns:
        DescriptionService instance.

    Example:
        >>> from src.agents.description_service import get_description_service
        >>> service = get_description_service(use_rag_context=True, use_web_search=False)
        >>> description = service.get_wine_description(wine)
    """
    global _service_instance, _service_config

    needs_recreation = (
        _service_instance is None
        or _service_config is None
        or _service_config.get("use_rag_context") != use_rag_context
        or _service_config.get("use_web_search") != use_web_search
    )

    if needs_recreation:
        is_first_creation = _service_instance is None
        _service_instance = DescriptionService(
            model=model,
            retriever=retriever,
            reranker=reranker,
            use_rag_context=use_rag_context,
            use_web_search=use_web_search,
            wine_repo=WineRepository(),
            producer_repo=ProducerRepository(),
        )
        _service_config = {"use_rag_context": use_rag_context, "use_web_search": use_web_search}
        logger.info(
            f"{'Created' if is_first_creation else 'Recreated'} "
            f"DescriptionService (RAG: {use_rag_context}, web_search: {use_web_search})"
        )

    return _service_instance  # ty:ignore[invalid-return-type]
