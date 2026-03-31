"""FastAPI application for Pour Decisions backend API.

Start with::
    PYTHONPATH=. uvicorn src.api.main:app --reload --port 8080
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import cellar, chat, taste_profile, wines
from src.utils import get_config, logger


def _load_model(cfg):
    """Load the LLM from config (provider + model name).

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A LangChain ``BaseChatModel`` instance.
    """
    from src.agents.llm import load_base_model

    return load_base_model(cfg.model.provider, cfg.model.name)


def _load_agents():
    """Load intelligent and keyword agents.

    Returns:
        Tuple of (intelligent_agent, keyword_agent). Either may be None on failure.
    """
    from src.agents import create_wine_agent, create_keyword_agent

    intelligent_agent = None
    keyword_agent = None

    try:
        intelligent_agent = create_wine_agent(verbose=False)
        logger.info("Intelligent wine agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load intelligent agent: {e}")

    try:
        keyword_agent = create_keyword_agent(verbose=False)
        logger.info("Keyword wine agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load keyword agent: {e}")

    return intelligent_agent, keyword_agent


def _load_retriever(cfg):
    """Load the retriever stack (vector, BM25, hybrid) from config.

    Mirrors the logic in ``src/ui/resources.py`` (load_vector_retriever,
    load_bm25_index, load_retriever) without Streamlit dependencies.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A ``HybridRetriever``, ``ChromaRetriever``, or None on failure.
    """
    from src.retrieval import BM25Index, ChromaRetriever, HybridRetriever
    from src.utils import initialize_chroma_client

    chroma_cfg = cfg.chroma
    retrieval_cfg = chroma_cfg.retrieval

    # 1. Connect to ChromaDB
    try:
        chroma_client = initialize_chroma_client(
            host=chroma_cfg.client.host,
            port=chroma_cfg.client.port,
        )
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return None

    # 2. Build vector retriever
    try:
        vector_retriever = ChromaRetriever(
            client=chroma_client,
            collection_name=chroma_cfg.collections[0].name,
            embedding_model=chroma_cfg.settings.embedder,
            n_results=retrieval_cfg.n_results,
            similarity_threshold=retrieval_cfg.similarity_threshold,
            enable_cache=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialise vector retriever: {e}")
        return None

    # 3. Optionally wrap with BM25 for hybrid search
    enable_hybrid = getattr(retrieval_cfg, "enable_hybrid", False)
    if enable_hybrid:
        try:
            index_path = getattr(retrieval_cfg, "bm25_index_path", "chroma-data/bm25_index.pkl")
            bm25 = BM25Index(index_path=index_path)

            if len(bm25) > 0:
                vector_weight = getattr(retrieval_cfg, "hybrid_vector_weight", 0.7)
                keyword_weight = getattr(retrieval_cfg, "hybrid_keyword_weight", 0.3)
                hybrid = HybridRetriever(
                    vector_retriever=vector_retriever,
                    bm25_index=bm25,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                )
                logger.info(
                    f"Using HybridRetriever (vector={vector_weight}, keyword={keyword_weight})"
                )
                return hybrid

            logger.warning("BM25 index empty, falling back to vector-only retrieval")
        except Exception as e:
            logger.warning(f"BM25 index unavailable ({e}), falling back to vector-only retrieval")

    logger.info("Using ChromaRetriever (vector-only)")
    return vector_retriever


def _load_reranker(cfg):
    """Load the cross-encoder reranker if enabled in config.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A ``DocumentReranker`` instance, or None if disabled / on failure.
    """
    from src.retrieval import DocumentReranker

    retrieval_cfg = cfg.chroma.retrieval

    if not getattr(retrieval_cfg, "enable_reranking", False):
        logger.info("Reranking disabled in config")
        return None

    try:
        model_name = getattr(retrieval_cfg, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = DocumentReranker(model_name=model_name)
        logger.info(f"Loaded reranker: {model_name}")
        return reranker
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load expensive resources once at startup, release on shutdown."""
    cfg = get_config()

    # LLM
    try:
        app.state.model = _load_model(cfg)
        logger.info("LLM loaded")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        app.state.model = None

    # Agents
    app.state.intelligent_agent, app.state.keyword_agent = _load_agents()

    # Retriever (vector / hybrid)
    app.state.retriever = _load_retriever(cfg)

    # Reranker
    app.state.reranker = _load_reranker(cfg)

    logger.info("API startup complete")
    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="Pour Decisions API",
    version="1.0.0",
    description="REST API for the Pour Decisions wine RAG chatbot and cellar management system.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",  # Next.js alt port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(cellar.router)
app.include_router(taste_profile.router)
app.include_router(wines.router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Return basic health status and loaded resource availability."""
    return {
        "status": "ok",
        "resources": {
            "model": app.state.model is not None,
            "intelligent_agent": app.state.intelligent_agent is not None,
            "keyword_agent": app.state.keyword_agent is not None,
            "retriever": app.state.retriever is not None,
            "reranker": app.state.reranker is not None,
        },
    }
