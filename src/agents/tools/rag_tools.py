"""
Wine agent tools for RAG-based wine knowledge retrieval.

This module provides tools for querying the wine knowledge base
using the existing RAG pipeline (ChromaDB + LangChain).
"""

from typing import Any

from langchain_core.tools import tool

from src.agents.tools.registry import (
    LatencyClass,
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolPrerequisite,
    ToolTier,
)
from src.retrieval import (
    RAGExecutionResult,
    build_reranker_from_config,
    build_retriever_from_config,
    execute_production_rag,
)
from src.utils import get_config, logger


RAG_UNAVAILABLE_MESSAGE = (
    "Wine knowledge search is temporarily unavailable. "
    "Please verify local embedding model setup and try again."
)


def _get_rag_resources() -> tuple[Any, Any, Any] | None:
    """Build the configured shared RAG resources for an agent tool call."""
    try:
        cfg = get_config()
        retriever = build_retriever_from_config(cfg)
        reranker = build_reranker_from_config(cfg)
        return cfg, retriever, reranker
    except Exception:
        logger.exception("Failed to initialize retrieval")
        return None


def _execute_rag_query(
    query: str,
    n_results: int,
    *,
    include_sources: bool = True,
) -> tuple[RAGExecutionResult | None, bool]:
    """Run one agent query through the shared production RAG path."""
    resources = _get_rag_resources()
    if resources is None:
        logger.warning("RAG retriever unavailable; returning empty result set")
        return None, False

    cfg, retriever, reranker = resources
    result = execute_production_rag(
        prompt=query,
        config=cfg,
        model=None,
        retriever=retriever,
        reranker=reranker,
        message_history=[],
        n_results_override=n_results,
        generation_enabled=False,
        include_context_metadata=include_sources,
    )
    if result.retrieval_error:
        raise RuntimeError(result.retrieval_error)
    return result, True


@tool
def search_wine_knowledge(
    query: str,
    max_results: int = 5,
    include_sources: bool = True,
) -> str:
    """Search wine knowledge base for general wine information.

    Retrieves relevant information from wine books and documents stored
    in the ChromaDB vector store.
    Use this for general wine education, wine region information, winemaking techniques, grape varieties,
    wine styles, buying guides, aging potential, food pairings, tasting notes, vintages, wine producers,
    wine producting countries, etc.

    Args:
        query: Question or topic to search for. Examples:
              - "What makes Barolo special?"
              - "Difference between Burgundy and Bordeaux"
              - "How is Champagne made?"
              - "What are the best producers in Napa Valley?"
              - "Aging potential of Rioja Reserva"
        max_results: Maximum number of source chunks to retrieve (1-10).
                    More results = more context but longer response. Default is 5.
        include_sources: Whether to include source citations in response.
                        If True, shows which wine books the information came from.

    Returns:
        String containing relevant information from wine knowledge base
        with source citations if requested.

    Example:
        >>> info = search_wine_knowledge("What is malolactic fermentation?")
        >>> info = search_wine_knowledge("Barolo aging requirements", max_results=3)

    Notes:
        - Does NOT query user's personal cellar or taste cellar-data
        - Returns general wine knowledge, not personalized information
    """
    try:
        max_results = min(max(max_results, 1), 10)

        result, retriever_available = _execute_rag_query(
            query=query,
            n_results=max_results,
            include_sources=include_sources,
        )

        if result is None or not result.context_chunks:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return "No relevant information found in the wine knowledge base for this query."

        logger.info("Retrieved %d documents for wine knowledge query", len(result.context_chunks))
        return result.context

    except Exception:
        logger.exception("Unexpected failure while searching wine knowledge")
        raise


@tool
def search_wine_region_info(region: str) -> str:
    """Search for detailed information about a specific wine region.

    Specialized search focused on wine region characteristics, history,
    terroir, climate, and notable wines.

    Args:
        region: Wine region name. Examples:
               - "Bordeaux"
               - "Burgundy"
               - "Barolo"
               - "Napa Valley"
               - "Rioja"
               - "Champagne"
               Can be broad (country/region) or specific (appellation).

    Returns:
        String containing detailed region information with source citations.

    Example:
        >>> info = search_wine_region_info("Burgundy")
        >>> info = search_wine_region_info("Barolo")

    Notes:
        - Optimized for region-specific queries
        - Uses semantic search with region-focused prompting
    """
    try:
        formatted_query = (
            f"Tell me about the {region} wine region: "
            f"climate, terroir, grape varieties, wine styles, characteristics, "
            f"sub-regions, and notable producers"
        )

        result, retriever_available = _execute_rag_query(query=formatted_query, n_results=5)

        if result is None or not result.context_chunks:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No information found about the {region} wine region."

        logger.info("Retrieved %d documents for region: %s", len(result.context_chunks), region)
        return result.context

    except Exception:
        logger.exception("Unexpected failure while searching region information")
        raise


@tool
def search_grape_variety_info(varietal: str) -> str:
    """Search for detailed information about a grape variety.

    Specialized search focused on grape variety characteristics,
    growing regions, wine styles, and tasting notes.

    Args:
        varietal: Grape variety name. Examples:
                 - "Pinot Noir"
                 - "Cabernet Sauvignon"
                 - "Chardonnay"
                 - "Nebbiolo"
                 - "Tempranillo"
                 - "Riesling"
                 Can include synonyms (e.g., "Syrah" or "Shiraz")

    Returns:
        String containing grape variety information with source citations.

    Example:
        >>> info = search_grape_variety_info("Nebbiolo")
        >>> info = search_grape_variety_info("Pinot Noir")

    Notes:
        - Optimized for grape variety queries
        - May include historical information if available
    """
    try:
        formatted_query = (
            f"Tell me about the {varietal} grape variety: "
            f"characteristics, growing regions, climate preferences, "
            f"typical flavors, aging potential, winemaking techniques, "
            f"and notable wines"
        )

        result, retriever_available = _execute_rag_query(query=formatted_query, n_results=5)

        if result is None or not result.context_chunks:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No information found about the {varietal} grape variety."

        logger.info("Retrieved %d documents for varietal: %s", len(result.context_chunks), varietal)
        return result.context

    except Exception:
        logger.exception("Unexpected failure while searching varietal information")
        raise


@tool
def search_wine_term_definition(term: str) -> str:
    """Search for definition and explanation of wine terminology.

    Look up wine-specific terminology, concepts, and jargon in the knowledge base.

    Args:
        term: Wine term to define. Examples:
             - "terroir"
             - "malolactic fermentation"
             - "sur lie aging"
             - "Grand Cru"
             - "batonnage"
             - "tannins"
             - "botrytis"
             - "skin contact"
             - "carbonic maceration"

    Returns:
        String containing definition and explanation with source citations.

    Example:
        >>> definition = search_wine_term_definition("terroir")
        >>> definition = search_wine_term_definition("malolactic fermentation")

    Notes:
        - Provides wine-specific definitions, not generic dictionary definitions
        - Includes context and examples
        - Completely free operation (local vector search)
        - Explains both traditional and modern winemaking terminology
    """
    try:
        formatted_query = (
            f"What is {term}? Define and explain {term} in the context of wine, "
            f"including how it affects wine character and examples"
        )

        result, retriever_available = _execute_rag_query(query=formatted_query, n_results=5)

        if result is None or not result.context_chunks:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No definition found for '{term}' in the wine knowledge base."

        logger.info("Retrieved %d documents for term: %s", len(result.context_chunks), term)
        return result.context

    except Exception:
        logger.exception("Unexpected failure while searching term definition")
        raise


@tool
def search_wine_producer_info(producer: str) -> str:
    """Search for detailed information about a wine producer/winery.

    Specialized search focused on producer history, philosophy, vineyard holdings,
    winemaking style, and notable wines.

    Args:
        producer: Producer/winery name. Examples:
                 - "Domaine de la Romanée-Conti"
                 - "Opus One"
                 - "Château Margaux"
                 - "Antinori"
                 - "Ridge Vineyards"
                 - "Gaja"
                 Can include estate, château, or domaine prefix.

    Returns:
        String containing detailed producer information with source citations:
        - Producer history and founding
        - Vineyard locations and holdings
        - Winemaking philosophy and techniques
        - Notable wines and flagship bottlings
        - Key vintages and ratings
        - Regional significance

    Example:
        >>> info = search_wine_producer_info("Domaine Leflaive")
        >>> info = search_wine_producer_info("Château Margaux")

    Notes:
        - Optimized for producer-specific queries
        - Includes both historical and current information
    """
    try:
        formatted_query = (
            f"Tell me about {producer} wine producer: "
            f"history, vineyard holdings, winemaking philosophy and techniques, "
            f"notable wines, key vintages, and significance in the region"
        )

        result, retriever_available = _execute_rag_query(query=formatted_query, n_results=5)

        if result is None or not result.context_chunks:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No information found about {producer} wine producer."

        logger.info("Retrieved %d documents for producer: %s", len(result.context_chunks), producer)
        return result.context

    except Exception:
        logger.exception("Unexpected failure while searching producer information")
        raise


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        tool=search_wine_knowledge,
        metadata=ToolMetadata(
            name="search_wine_knowledge",
            category=ToolCategory.RAG,
            tier=ToolTier.CORE,
            prerequisites=(ToolPrerequisite.CHROMA_COLLECTION,),
            latency_class=LatencyClass.SLOW,
            capability="Search the local wine-book knowledge base for general wine information.",
        ),
    ),
    ToolDefinition(
        tool=search_wine_region_info,
        metadata=ToolMetadata(
            name="search_wine_region_info",
            category=ToolCategory.RAG,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CHROMA_COLLECTION,),
            latency_class=LatencyClass.SLOW,
            capability="Search the local knowledge base for wine-region information.",
        ),
    ),
    ToolDefinition(
        tool=search_grape_variety_info,
        metadata=ToolMetadata(
            name="search_grape_variety_info",
            category=ToolCategory.RAG,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CHROMA_COLLECTION,),
            latency_class=LatencyClass.SLOW,
            capability="Search the local knowledge base for grape-variety information.",
        ),
    ),
    ToolDefinition(
        tool=search_wine_term_definition,
        metadata=ToolMetadata(
            name="search_wine_term_definition",
            category=ToolCategory.RAG,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CHROMA_COLLECTION,),
            latency_class=LatencyClass.SLOW,
            capability="Define wine terminology using the local knowledge base.",
        ),
    ),
    ToolDefinition(
        tool=search_wine_producer_info,
        metadata=ToolMetadata(
            name="search_wine_producer_info",
            category=ToolCategory.RAG,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CHROMA_COLLECTION,),
            latency_class=LatencyClass.SLOW,
            capability="Search the local knowledge base for wine-producer information.",
        ),
    ),
)
