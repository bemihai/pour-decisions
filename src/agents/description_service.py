"""
Service for generating and managing LLM-generated descriptions for wines and producers.

This service integrates with the RAG pipeline to provide context-aware descriptions
that are grounded in wine book knowledge when available, falling back to LLM general
knowledge when no relevant context is found.
"""

from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.llm import load_base_model
from src.database.models import Wine, Producer
from src.database.repository import WineRepository, ProducerRepository
from src.retrieval import HybridRetriever, DocumentReranker
from src.utils import logger, get_config


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
        config: dict | None = None,
    ):
        """
        Initialize the description service.

        Args:
            model: Pre-loaded LLM model (loads from config if None)
            retriever: HybridRetriever for context retrieval (optional)
            reranker: DocumentReranker for result refinement (optional)
            use_rag_context: Whether to use RAG for context enrichment
            config: Configuration dict (loads from app_config.yml if None)
        """
        self.config = config or get_config()
        self.use_rag_context = use_rag_context
        self.retriever = retriever
        self.reranker = reranker

        # Load LLM model
        if model is None:
            model_config = self.config.get("model", {})
            provider = model_config.get("provider", "google")
            model_name = model_config.get("name", "gemini-2.5-flash")
            self.model = load_base_model(provider, model_name)
            logger.info(f"Loaded LLM model: {provider}/{model_name}")
        else:
            self.model = model

        # Initialize repositories
        self.wine_repo = WineRepository()
        self.producer_repo = ProducerRepository()

        # Load prompt templates
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._wine_prompt_template = self._load_prompt("wine_description_prompt.md")
        self._producer_prompt_template = self._load_prompt("producer_description_prompt.md")

        # RAG context configuration
        desc_config = self.config.get("description_generation", {})
        self.max_context_chunks = desc_config.get("max_context_chunks", 2)
        self.min_relevance_score = desc_config.get("min_relevance_score", 0.4)

        logger.info(
            f"DescriptionService initialized (RAG: {use_rag_context}, "
            f"max_chunks: {self.max_context_chunks})"
        )

    def get_wine_description(self, wine: Wine) -> str | None:
        """
        Get description for a wine, generating if not cached.

        The description is generated on first call and persisted to database.
        Subsequent calls return the cached value.

        Args:
            wine: Wine model with all relevant fields

        Returns:
            Description string or None if generation fails

        Example:
            >>> wine = Wine(
            ...     id=1,
            ...     wine_name="Tignanello",
            ...     producer_name="Antinori",
            ...     vintage=2019,
            ...     wine_type="Red",
            ...     varietal="Sangiovese, Cabernet Sauvignon"
            ... )
            >>> description = service.get_wine_description(wine)
        """
        # Return cached description if exists
        if wine.description:
            logger.debug(f"Using cached description for wine ID {wine.id}")
            return wine.description

        # Generate new description
        logger.info(f"Generating description for wine: {wine.wine_name} ({wine.vintage})")

        try:
            # Build search query for RAG
            query = self._build_wine_search_query(wine)

            # Retrieve context if RAG is enabled
            context_section = ""
            if self.use_rag_context and self.retriever:
                context = self._retrieve_context(query, self.max_context_chunks)
                if context:
                    context_section = self._format_context_section(context)
                    logger.debug(f"Retrieved {len(context)} context chunks for wine")
                else:
                    logger.debug("No relevant context found, using LLM general knowledge")

            # Format prompt with wine data
            prompt = self._wine_prompt_template.format(
                wine_name=wine.wine_name or "Unknown",
                producer_name=wine.producer_name or "Unknown",
                vintage=wine.vintage or "NV",
                wine_type=wine.wine_type or "Unknown",
                varietal=wine.varietal or "Unknown",
                region=wine.region_name or "Unknown",
                country=wine.country or "Unknown",
                appellation=wine.appellation or "N/A",
                context_section=context_section
            )

            # Generate description with LLM
            description = self._generate_with_llm(prompt)

            if description:
                # Persist to database
                wine.description = description
                self.wine_repo.update(wine)
                logger.info(f"Generated and saved description for wine ID {wine.id}")
                return description
            else:
                logger.warning(f"Failed to generate description for wine ID {wine.id}")
                return None

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
            # Build search query for RAG
            query = self._build_producer_search_query(producer)

            # Retrieve context if RAG is enabled
            context_section = ""
            if self.use_rag_context and self.retriever:
                context = self._retrieve_context(query, self.max_context_chunks)
                if context:
                    context_section = self._format_context_section(context)
                    logger.debug(f"Retrieved {len(context)} context chunks for producer")
                else:
                    logger.debug("No relevant context found, using LLM general knowledge")

            # Format prompt with producer data
            prompt = self._producer_prompt_template.format(
                producer_name=producer.name or "Unknown",
                country=producer.country or "Unknown",
                region=producer.region or "Unknown",
                context_section=context_section
            )

            # Generate description with LLM
            description = self._generate_with_llm(prompt)

            if description:
                # Persist to database
                producer.description = description
                self.producer_repo.update(producer)
                logger.info(f"Generated and saved description for producer ID {producer.id}")
                return description
            else:
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

    def _retrieve_context(self, query: str, max_chunks: int = 2) -> list[dict[str, Any]] | None:
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

    def _generate_with_llm(self, prompt: str) -> str | None:
        """
        Generate description using LLM.

        Args:
            prompt: Formatted prompt string

        Returns:
            Generated description or None if failed
        """
        try:
            response = self.model.invoke(prompt)

            # Extract text from response
            if hasattr(response, 'content'):
                description = response.content
            elif isinstance(response, dict) and 'content' in response:
                description = response['content']
            else:
                description = str(response)

            # Clean up the description
            description = description.strip()

            # Validate length (should be 2-3 sentences, roughly 50-300 chars)
            if len(description) < 20:
                logger.warning(f"Generated description too short: {len(description)} chars")
                return None

            if len(description) > 500:
                logger.warning(f"Generated description too long: {len(description)} chars, truncating")
                # Truncate to roughly 3 sentences
                sentences = description.split('. ')
                description = '. '.join(sentences[:3])
                if not description.endswith('.'):
                    description += '.'

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
    use_rag_context: bool = True
) -> DescriptionService:
    """
    Get or create the singleton DescriptionService instance.

    This provides a convenient way to access the service across the application
    without repeatedly initializing the LLM model. The instance is recreated if
    the use_rag_context configuration changes.

    Args:
        retriever: Optional retriever (used when creating/recreating instance)
        reranker: Optional reranker (used when creating/recreating instance)
        use_rag_context: Whether to use RAG for context enrichment

    Returns:
        DescriptionService instance

    Example:
        >>> from src.agents.description_service import get_description_service
        >>> service = get_description_service(use_rag_context=True)
        >>> description = service.get_wine_description(wine)
    """
    global _service_instance, _service_config

    # Current configuration
    current_config = {
        'use_rag_context': use_rag_context,
        'has_retriever': retriever is not None,
        'has_reranker': reranker is not None
    }

    # Check if we need to create or recreate the instance
    needs_recreation = (
        _service_instance is None or
        _service_config is None or
        _service_config['use_rag_context'] != use_rag_context
    )

    if needs_recreation:
        _service_instance = DescriptionService(
            retriever=retriever,
            reranker=reranker,
            use_rag_context=use_rag_context
        )
        _service_config = current_config
        logger.info(
            f"{'Created' if _service_config is None else 'Recreated'} "
            f"DescriptionService instance (RAG: {use_rag_context})"
        )

    return _service_instance
