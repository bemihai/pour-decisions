"""
Service for generating and managing LLM-generated descriptions for wines and producers.

This service integrates with the RAG pipeline to provide context-aware descriptions
that are grounded in wine book knowledge when available, falling back to LLM general
knowledge when no relevant context is found.

For wines, the LLM call uses structured output (Pydantic) to extract both a text
description and a drinking window estimate in a single call at no extra cost.
"""

from datetime import datetime
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field

from src.agents.llm import load_base_model
from src.database.models import Wine, Producer
from src.database.repository import WineRepository, ProducerRepository
from src.retrieval import HybridRetriever, DocumentReranker
from src.utils import get_config, logger, set_span_attributes


class WineAnalysis(BaseModel):
    """Structured output returned by the LLM for wine analysis.

    Used with model.with_structured_output(WineAnalysis) so the LLM response is
    automatically validated and parsed -- no JSON handling needed in the service.
    """

    description: str = Field(description="2-3 sentence wine description focusing on flavor profile and style")
    drink_from_year: int | None = Field(None, description="Year the wine begins drinking well (absolute year)")
    drink_to_year: int | None = Field(None, description="Year the wine is past its peak (absolute year)")

class DescriptionService:
    """
    Service for generating and caching LLM descriptions for wines and producers.

    Features:
    - Lazy generation (only when description is NULL)
    - RAG-enhanced descriptions using wine book context
    - Database persistence (no repeated LLM calls)
    - Graceful fallback when RAG context not available

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
        retriever: HybridRetriever | None = None,
        reranker: DocumentReranker | None = None,
        use_rag_context: bool = True,
        use_web_search: bool = False,
        config: dict | None = None,
    ):
        """
        Initialize the description service.

        Args:
            model: Pre-loaded LLM model. When ``None``, the service selects a model
                automatically based on ``description_generation.use_cloud_model`` in
                ``app_config.yml`` (default ``True``):

                * ``True``  -- loads the fallback/cloud model (e.g. Gemini) for
                  reliable structured output. Recommended: local CPU inference is
                  ~93 s per call, while cloud is < 5 s.
                * ``False`` -- loads the primary model from config (e.g. Ollama/Gemma 4).

                Pass an explicit model to override the config entirely.
            retriever: HybridRetriever for context retrieval (optional)
            reranker: DocumentReranker for result refinement (optional)
            use_rag_context: Whether to use RAG for context enrichment
            use_web_search: Whether to inject web search snippets into context.
                            Requires TAVILY_API_KEY. Defaults to False.
            config: Configuration object (loads from app_config.yml if None)
        """
        self.config = config or get_config()
        self.use_rag_context = use_rag_context
        self.use_web_search = use_web_search
        self.retriever = retriever
        self.reranker = reranker

        desc_config = self.config.get("description_generation", {})
        self.enable_web_search = bool(desc_config.get("enable_web_search", True))
        web_search_config = self.config.get("web_search", {})
        tavily_config = web_search_config.get("tavily", {})
        self.web_search_api_key_env = str(tavily_config.get("api_key_env", "TAVILY_API_KEY"))

        # Load LLM model
        if model is None:
            model_cfg = getattr(self.config, "model", None)
            desc_cfg = getattr(self.config, "description_generation", None)
            use_cloud = getattr(desc_cfg, "use_cloud_model", True) if desc_cfg is not None else True

            if use_cloud:
                # Prefer cloud/fallback model: structured output is unreliable / very slow on CPU.
                # Descriptions are persisted in SQLite, so this is a one-time cost per wine.
                provider = str(getattr(model_cfg, "fallback_provider", "google")) if model_cfg else "google"
                model_name = str(getattr(model_cfg, "fallback_name", "gemini-2.5-flash")) if model_cfg else "gemini-2.5-flash"
                logger.info(
                    f"Loading cloud/fallback model for description generation: {provider}/{model_name} "
                    f"(override with description_generation.use_cloud_model: false)"
                )
            else:
                # Use the primary model from config (may be local/Ollama)
                provider = str(getattr(model_cfg, "provider", "google")) if model_cfg else "google"
                model_name = str(getattr(model_cfg, "name", "gemini-2.5-flash")) if model_cfg else "gemini-2.5-flash"
                logger.info(f"Loading primary model for description generation: {provider}/{model_name}")

            self.model = load_base_model(provider, model_name)
        else:
            self.model = model
            logger.info(f"Using provided model for description generation: {type(model).__name__}")

        # Structured-output model for wine analysis (description + drinking window)
        self._structured_model = self.model.with_structured_output(WineAnalysis)

        # Web search engine (lazy init inside method to avoid import cost when disabled)
        self._web_search_engine = None
        self._web_search_available = True
        if self.use_web_search and not self.enable_web_search:
            self.use_web_search = False
            self._web_search_available = False
            logger.info(
                "Web search disabled for description generation: "
                "description_generation.enable_web_search is false"
            )
        elif self.use_web_search and not os.getenv(self.web_search_api_key_env, "").strip():
            self.use_web_search = False
            self._web_search_available = False
            logger.info(
                "Web search disabled for description generation: "
                f"{self.web_search_api_key_env} is not configured"
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

        # Initialize repositories
        self.wine_repo = WineRepository()
        self.producer_repo = ProducerRepository()

        # Load prompt templates
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._wine_prompt_template = self._load_prompt("wine_description_prompt.md")
        self._producer_prompt_template = self._load_prompt("producer_description_prompt.md")

        # RAG context configuration
        self.max_context_chunks = desc_config.get("max_context_chunks", 3)
        self.min_relevance_score = desc_config.get("min_relevance_score", 0.4)

        logger.info(
            f"DescriptionService initialized (RAG: {use_rag_context}, "
            f"web_search: {use_web_search}, max_chunks: {self.max_context_chunks})"
        )

    @contextmanager
    def _start_description_span(self, entity_type: str, entity_id: int | None):
        """Create a description-generation span tagged for Phoenix filtering.

        Args:
            entity_type: Entity type being described (``"wine"`` or ``"producer"``).
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

                prompt = self._wine_prompt_template.format(
                    wine_name=wine.wine_name or "Unknown",
                    producer_name=wine.producer_name or "Unknown",
                    vintage=wine.vintage or "NV",
                    wine_type=wine.wine_type or "Unknown",
                    varietal=wine.varietal or "Unknown",
                    region=wine.region_name or "Unknown",
                    country=wine.country or "Unknown",
                    appellation=wine.appellation or "N/A",
                    context_section=context_section,
                )

                analysis = self._invoke_structured(prompt)
                if analysis is None:
                    return None

                # Persist description
                wine.description = analysis.description
                self.wine_repo.update(wine)
                logger.info(f"Saved description for wine ID {wine.id}")

                # Persist drinking window if LLM provided one and no better source exists
                if analysis.drink_from_year and analysis.drink_to_year and wine.id:
                    self._persist_llm_drinking_window(wine, analysis.drink_from_year, analysis.drink_to_year)

                return analysis.description

        except Exception as e:
            logger.error(f"Error generating wine description: {e}", exc_info=True)
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
                context_section = ""
                if self.use_rag_context and self.retriever:
                    context = self._retrieve_context(query, self.max_context_chunks)
                    if context:
                        context_section = self._format_context_section(context)
                        logger.debug(f"Retrieved {len(context)} context chunks for producer")

                prompt = self._producer_prompt_template.format(
                    producer_name=producer.name or "Unknown",
                    country=producer.country or "Unknown",
                    region=producer.region or "Unknown",
                    context_section=context_section,
                )

                description = self._generate_with_llm(prompt)

                if description:
                    producer.description = description
                    self.producer_repo.update(producer)
                    logger.info(f"Generated and saved description for producer ID {producer.id}")
                    return description

                logger.warning(f"Failed to generate description for producer ID {producer.id}")
                return None

        except Exception as e:
            logger.error(f"Error generating producer description: {e}", exc_info=True)
            return None

    def generate_batch(
        self,
        producers: list[Producer] | None = None,
        wines: list[Wine] | None = None
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
            f"Batch generation complete: {stats['producers_generated']} producers, "
            f"{stats['wines_generated']} wines"
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

            # Filter by minimum relevance score
            relevant = [
                r for r in results
                if r.get('distance', 1.0) <= (1.0 - self.min_relevance_score)  # distance is inverted
            ]

            if not relevant:
                return None

            return relevant[:max_chunks]

        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")
            return None

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
            content = chunk.get('document', chunk.get('content', ''))
            metadata = chunk.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page_number', metadata.get('page', ''))

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
        query_parts = []

        # Wine name (often contains key info)
        if wine.wine_name:
            query_parts.append(wine.wine_name)

        # Varietal (important for style)
        if wine.varietal:
            query_parts.append(wine.varietal)

        # Appellation (specific region designation)
        if wine.appellation:
            query_parts.append(wine.appellation)

        # Producer (for context)
        if wine.producer_name:
            query_parts.append(wine.producer_name)

        # Region (broader geographic context)
        if wine.region_name and wine.region_name not in str(wine.appellation):
            query_parts.append(wine.region_name)

        return " ".join(query_parts)

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

    def _invoke_structured(self, prompt: str) -> "WineAnalysis | None":
        """Invoke the structured-output model and return a WineAnalysis instance.

        Falls back gracefully: if structured output fails (e.g. OutputParserException),
        attempts a plain invoke and returns a WineAnalysis with only the description
        populated, so the caller always has a usable object.

        Args:
            prompt: Formatted prompt string.

        Returns:
            WineAnalysis instance or None on hard failure.
        """
        try:
            result = self._structured_model.invoke(prompt)
            return result  # type: ignore[return-value]
        except Exception as structured_error:
            logger.warning(
                f"Structured output failed ({structured_error}), falling back to plain invoke"
            )
            try:
                response = self.model.invoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)
                text = text.strip()
                if len(text) < 20:
                    return None
                return WineAnalysis(description=text[:500], drink_from_year=None, drink_to_year=None)
            except Exception as e:
                logger.error(f"Plain LLM invocation also failed: {e}", exc_info=True)
                return None

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
        if not (_MIN_YEAR <= from_year <= current_year + _MAX_FUTURE_YEARS and
                _MIN_YEAR <= to_year <= current_year + _MAX_FUTURE_YEARS):
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
        updated = self.wine_repo.update_drinking_window(wine.id, from_year, to_year, index, "llm")
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
        parts: list[str] = []

        if self.use_rag_context and self.retriever:
            context = self._retrieve_context(query, self.max_context_chunks)
            if context:
                parts.append(self._format_context_section(context))
                logger.debug(f"Retrieved {len(context)} RAG chunks for wine")

        if self.use_web_search:
            logger.info(
                "DescriptionService web search triggered: wine_id=%s query=%s",
                wine.id,
                query,
            )
            snippets = self._get_web_search_snippets(wine)
            if snippets:
                parts.append(snippets)
                logger.info("DescriptionService web search used: wine_id=%s snippets_added=true", wine.id)
            else:
                logger.info("DescriptionService web search used: wine_id=%s snippets_added=false", wine.id)

        return "\n\n".join(parts)

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
                from src.agents.tools.web_search_tools import WineWebSearchEngine
                self._web_search_engine = WineWebSearchEngine()
            except Exception as e:
                self._web_search_available = False
                logger.warning(f"Could not initialise web search engine: {e}")
                return ""

        vintage_str = str(wine.vintage) if wine.vintage else ""
        query = f"{wine.wine_name} {vintage_str} aging potential drinking window".strip()

        try:
            results = self._web_search_engine.search(query, search_type="review", max_results=3)
            if not results:
                logger.info("DescriptionService web search completed: wine_id=%s results=0", wine.id)
                return ""
            logger.info("DescriptionService web search completed: wine_id=%s results=%s", wine.id, len(results))
            lines = ["Web Search Context:"]
            for r in results:
                snippet = r.get("snippet", "").strip()
                if snippet:
                    lines.append(f"- {snippet}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Web search failed for wine context: {e}")
            return ""

    def _generate_with_llm(self, prompt: str) -> str | None:
        """Generate a plain-text description using the LLM (used for producers).

        Args:
            prompt: Formatted prompt string.

        Returns:
            Generated description string or None if failed.
        """
        try:
            response = self.model.invoke(prompt)
            description = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()

            if len(description) < 20:
                logger.warning(f"Generated description too short: {len(description)} chars")
                return None

            if len(description) > 500:
                sentences = description.split(". ")
                description = ". ".join(sentences[:3])
                if not description.endswith("."):
                    description += "."

            return description

        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return None

    def _load_prompt(self, filename: str) -> str:
        """
        Load prompt template from markdown file.

        Args:
            filename: Name of the prompt file

        Returns:
            Prompt template string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompt_path = self.prompts_dir / filename

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        logger.debug(f"Loaded prompt template: {filename}")
        return template


# Singleton instance for easy access across the application
_service_instance: DescriptionService | None = None
_service_config: dict[str, Any] | None = None


def get_description_service(
    retriever: HybridRetriever | None = None,
    reranker: DocumentReranker | None = None,
    use_rag_context: bool = True,
    use_web_search: bool = False,
    model: BaseChatModel | None = None,
) -> DescriptionService:
    """Get or create the singleton DescriptionService instance.

    The instance is recreated when use_rag_context or use_web_search changes.

    When ``model`` is ``None`` (default), the service selects a model based on
    ``description_generation.use_cloud_model`` from ``app_config.yml`` -- the cloud
    model is used by default (see ``DescriptionService.__init__`` docstring).

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
        )
        _service_config = {"use_rag_context": use_rag_context, "use_web_search": use_web_search}
        logger.info(
            f"{'Created' if is_first_creation else 'Recreated'} "
            f"DescriptionService (RAG: {use_rag_context}, web_search: {use_web_search})"
        )

    return _service_instance
