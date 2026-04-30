"""
Wine agent tools for RAG-based wine knowledge retrieval.

This module provides tools for querying the wine knowledge base
using the existing RAG pipeline (ChromaDB + LangChain).
"""

from langchain_core.tools import tool

from src.retrieval import ChromaRetriever, build_context_from_chunks
from src.utils import initialize_chroma_client, get_config, logger


RAG_UNAVAILABLE_MESSAGE = (
    "Wine knowledge search is temporarily unavailable. "
    "Please verify local embedding model setup and try again."
)


def _get_rag_retriever(n_results: int | None = None) -> ChromaRetriever | None:
    """Helper function to initialize RAG retrieval."""
    try:
        cfg = get_config()
        chroma_cfg = cfg.chroma
        collection_name = chroma_cfg.collections[0].name

        client = initialize_chroma_client(
            host=chroma_cfg.client.host,
            port=chroma_cfg.client.port
        )

        return ChromaRetriever(
            client=client,
            collection_name=collection_name,
            embedding_model=chroma_cfg.settings.embedder,
            n_results=n_results or chroma_cfg.retrieval.n_results,
            similarity_threshold=chroma_cfg.retrieval.similarity_threshold,
        )
    except Exception as e:
        logger.error(f"Failed to initialize retrieval: {e}")
        return None


def _retrieve_docs(query: str, n_results: int) -> tuple[list[dict], bool]:
    """Run retrieval and report whether the retriever was available."""
    retriever = _get_rag_retriever(n_results=n_results)
    if retriever is None:
        logger.warning("RAG retriever unavailable; returning empty result set")
        return [], False
    return retriever.retrieve(query), True


@tool
def search_wine_knowledge(
    query: str,
    max_results: int = 5,
    include_sources: bool = True
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

        retrieved_docs, retriever_available = _retrieve_docs(query=query, n_results=max_results)

        if not retrieved_docs:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return "No relevant information found in the wine knowledge base for this query."

        context = build_context_from_chunks(
            retrieved_docs,
            include_metadata=include_sources,
            include_similarity=False,
            max_chunks=max_results
        )

        logger.info(f"Retrieved {len(retrieved_docs)} documents for wine knowledge query")
        return context

    except Exception as e:
        logger.error(f"Error searching wine knowledge: {e}")
        return f"Error retrieving wine knowledge: {str(e)}"


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

        retrieved_docs, retriever_available = _retrieve_docs(query=formatted_query, n_results=5)

        if not retrieved_docs:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No information found about the {region} wine region."

        context = build_context_from_chunks(
            retrieved_docs,
            include_metadata=True,
            include_similarity=False,
            max_chunks=5
        )

        logger.info(f"Retrieved {len(retrieved_docs)} documents for region: {region}")
        return context

    except Exception as e:
        logger.error(f"Error searching region info: {e}")
        return f"Error retrieving region information: {str(e)}"


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

        retrieved_docs, retriever_available = _retrieve_docs(query=formatted_query, n_results=5)

        if not retrieved_docs:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No information found about the {varietal} grape variety."

        # Build context with sources
        context = build_context_from_chunks(
            retrieved_docs,
            include_metadata=True,
            include_similarity=False,
            max_chunks=5
        )

        logger.info(f"Retrieved {len(retrieved_docs)} documents for varietal: {varietal}")
        return context

    except Exception as e:
        logger.error(f"Error searching varietal info: {e}")
        return f"Error retrieving varietal information: {str(e)}"


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

        retrieved_docs, retriever_available = _retrieve_docs(query=formatted_query, n_results=5)

        if not retrieved_docs:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No definition found for '{term}' in the wine knowledge base."

        context = build_context_from_chunks(
            retrieved_docs,
            include_metadata=True,
            include_similarity=False,
            max_chunks=5
        )

        logger.info(f"Retrieved {len(retrieved_docs)} documents for term: {term}")
        return context

    except Exception as e:
        logger.error(f"Error searching term definition: {e}")
        return f"Error retrieving term definition: {str(e)}"


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

        retrieved_docs, retriever_available = _retrieve_docs(query=formatted_query, n_results=5)

        if not retrieved_docs:
            if not retriever_available:
                return RAG_UNAVAILABLE_MESSAGE
            return f"No information found about {producer} wine producer."

        context = build_context_from_chunks(
            retrieved_docs,
            include_metadata=True,
            include_similarity=False,
            max_chunks=5
        )

        logger.info(f"Retrieved {len(retrieved_docs)} documents for producer: {producer}")
        return context

    except Exception as e:
        logger.error(f"Error searching producer info: {e}")
        return f"Error retrieving producer information: {str(e)}"
