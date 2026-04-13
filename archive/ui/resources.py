"""Cached resources for the Streamlit app."""
import streamlit as st

from src.agents import create_wine_agent, create_keyword_agent
from src.agents.llm import load_base_model
from src.retrieval import ChromaRetriever, BM25Index, HybridRetriever, DocumentReranker
from src.utils import get_config, logger
from src.utils import initialize_chroma_client


@st.cache_resource
def load_llm():
    """Load and cache the LLM instance from config (provider + model name)."""
    cfg = get_config()
    return load_base_model(cfg.model.provider, cfg.model.name)


@st.cache_resource
def load_intelligent_agent():
    """Load the intelligent wine agent with caching."""
    try:

        agent = create_wine_agent(verbose=False)
        logger.info("Intelligent wine agent loaded successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to load intelligent agent: {e}")
        st.error(f"Could not load intelligent agent: {str(e)}")
        return None


@st.cache_resource
def load_keyword_agent():
    """Load the keyword wine agent with caching."""
    try:
        agent = create_keyword_agent(verbose=False)
        logger.info("Keyword wine agent loaded successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to load keyword agent: {e}")
        st.error(f"Could not load keyword agent: {str(e)}")
        return None


@st.cache_resource
def load_chroma_client():
    """Initialize ChromaDB client with caching to avoid reconnection."""
    cfg = get_config()
    chroma_cfg = cfg.chroma

    try:
        client = initialize_chroma_client(
            host=chroma_cfg.client.host,
            port=chroma_cfg.client.port
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        st.error(
            f"Unable to connect to ChromaDB at {chroma_cfg.client.host}:{chroma_cfg.client.port}. "
            "RAG features will be disabled. Using LLM general knowledge only."
        )
        return None


@st.cache_resource
def load_vector_retriever():
    """Initialize base ChromaRetriever with caching."""
    try:
        cfg = get_config()
        chroma_cfg = cfg.chroma
        client = load_chroma_client()

        if client is None:
            logger.warning("ChromaDB client is unavailable. Retriever not initialized.")
            return None

        collection_name = chroma_cfg.collections[0].name
        retrieval_cfg = chroma_cfg.retrieval

        return ChromaRetriever(
            client=client,
            collection_name=collection_name,
            embedding_model=chroma_cfg.settings.embedder,
            n_results=retrieval_cfg.n_results,
            similarity_threshold=retrieval_cfg.similarity_threshold,
            enable_cache=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector retrieval: {e}")
        return None


@st.cache_resource
def load_bm25_index():
    """Load or build BM25 index for keyword search."""
    try:
        cfg = get_config()
        retrieval_cfg = cfg.chroma.retrieval

        if not getattr(retrieval_cfg, 'enable_hybrid', False):
            logger.info("Hybrid search disabled, skipping BM25 index")
            return None

        index_path = getattr(retrieval_cfg, 'bm25_index_path', 'chroma-data/bm25_index.pkl')
        bm25 = BM25Index(index_path=index_path)

        if len(bm25) > 0:
            logger.info(f"Loaded BM25 index with {len(bm25)} documents")
            return bm25

        # Build index from ChromaDB if not exists
        vector_retriever = load_vector_retriever()
        if vector_retriever is None:
            return None

        # Fetch all documents from collection to build BM25 index
        collection = vector_retriever.collection
        all_docs = collection.get(include=["documents", "metadatas"])

        if not all_docs or not all_docs['ids']:
            logger.warning("No documents in collection to build BM25 index")
            return None

        documents = []
        for i, doc_id in enumerate(all_docs['ids']):
            documents.append({
                'id': doc_id,
                'document': all_docs['documents'][i] if all_docs['documents'] else '',
                'metadata': all_docs['metadatas'][i] if all_docs['metadatas'] else {},
            })

        bm25.build_index(documents)
        bm25.save()
        logger.info(f"Built and saved BM25 index with {len(bm25)} documents")
        return bm25

    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        return None


@st.cache_resource
def load_reranker():
    """Load cross-encoder reranker."""
    try:
        cfg = get_config()
        retrieval_cfg = cfg.chroma.retrieval

        if not getattr(retrieval_cfg, 'enable_reranking', False):
            logger.info("Reranking disabled")
            return None

        model_name = getattr(retrieval_cfg, 'reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        reranker = DocumentReranker(model_name=model_name)
        logger.info(f"Loaded reranker: {model_name}")
        return reranker

    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        return None


@st.cache_resource
def load_retriever():
    """
    Initialize retrieval with optional hybrid search and reranking.

    Returns the best available retrieval based on configuration:
    - HybridRetriever if hybrid search is enabled and BM25 index available
    - ChromaRetriever as fallback

    Reranker is loaded separately and applied in the retrieval flow.
    """
    try:
        cfg = get_config()
        retrieval_cfg = cfg.chroma.retrieval

        vector_retriever = load_vector_retriever()
        if vector_retriever is None:
            logger.warning("Vector retrieval not available")
            st.warning(
                "Could not initialize the retrieval system. "
                "RAG features are unavailable."
            )
            return None

        # Check if hybrid search is enabled
        enable_hybrid = getattr(retrieval_cfg, 'enable_hybrid', False)

        if enable_hybrid:
            bm25_index = load_bm25_index()
            if bm25_index is not None and len(bm25_index) > 0:
                vector_weight = getattr(retrieval_cfg, 'hybrid_vector_weight', 0.7)
                keyword_weight = getattr(retrieval_cfg, 'hybrid_keyword_weight', 0.3)

                hybrid_retriever = HybridRetriever(
                    vector_retriever=vector_retriever,
                    bm25_index=bm25_index,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                )
                logger.info(f"Using HybridRetriever (vector={vector_weight}, keyword={keyword_weight})")
                return hybrid_retriever
            else:
                logger.warning("BM25 index not available, falling back to vector-only retrieval")

        logger.info("Using ChromaRetriever (vector-only)")
        return vector_retriever

    except Exception as e:
        logger.error(f"Failed to initialize retrieval: {e}")
        st.warning(
            "Could not initialize the retrieval system. "
            "RAG features are unavailable. Answers will be based on general knowledge only."
        )
        return None
