"""FastAPI application for Pour Decisions backend API.

Start with::
    PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.language_models import BaseChatModel

from src.api.routes import cellar, chat, taste_profile, wines
from src.utils import get_config, logger

if TYPE_CHECKING:
    from src.retrieval import ChromaRetriever, DocumentReranker, HybridRetriever


def _load_local_model(cfg: Any) -> BaseChatModel | None:
    """Load the local Ollama/Gemma model from config.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A ``ChatOllama`` instance, or ``None`` when the configured provider
        is not Ollama.

    Raises:
        Exception: If Ollama is unreachable or the model is not pulled.
    """
    from src.agents.llm import load_base_model

    if str(cfg.model.provider).lower() != "ollama":
        return None

    base_url = str(getattr(getattr(cfg.model, "ollama", None), "base_url", "http://localhost:11434"))
    return load_base_model("ollama", cfg.model.name, base_url=base_url)


def _load_cloud_model(cfg: Any) -> BaseChatModel:
    """Load the cloud model from config.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A cloud ``BaseChatModel`` instance.

    Raises:
        Exception: If the API key is missing or invalid.
    """
    from src.agents.llm import load_base_model

    cloud_provider, cloud_name = _resolve_cloud_model_config(cfg)
    return load_base_model(cloud_provider, cloud_name)


def _resolve_cloud_model_config(cfg: Any) -> tuple[str, str]:
    """Resolve which cloud provider/model should be loaded.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        Tuple of (provider, model_name) for the cloud model slot.
    """
    configured_provider = str(getattr(cfg.model, "provider", "ollama")).lower()
    if configured_provider == "ollama":
        return (
            str(getattr(cfg.model, "fallback_provider", "google")),
            str(getattr(cfg.model, "fallback_name", "gemini-2.5-flash")),
        )
    return (
        str(getattr(cfg.model, "provider", "google")),
        str(getattr(cfg.model, "name", "gemini-2.5-flash")),
    )


def _load_agents(
    llm: BaseChatModel | None = None,
    tool_llm: BaseChatModel | None = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Load intelligent and keyword agents with the given LLM.

    Args:
        llm: Pre-loaded model for final answer generation. If ``None`` each agent
             will load its own model from config.
        tool_llm: Optional model for tool selection / planning (hybrid mode). When
             provided and different from ``llm``, the intelligent agent uses
             ``tool_llm`` for planning and ``llm`` for generation.
             The keyword agent is not affected (pattern-based routing, no tool calls).

    Returns:
        Tuple of (intelligent_agent, keyword_agent). Either may be None on failure.
    """
    from src.agents import create_wine_agent, create_keyword_agent

    intelligent_agent = None
    keyword_agent = None

    try:
        intelligent_agent = create_wine_agent(verbose=False, llm=llm, tool_llm=tool_llm)
        logger.info("Intelligent wine agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load intelligent agent: {e}")

    try:
        keyword_agent = create_keyword_agent(verbose=False, llm=llm)
        logger.info("Keyword wine agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load keyword agent: {e}")

    return intelligent_agent, keyword_agent


def _load_retriever(cfg: Any) -> "Optional[Union[HybridRetriever, ChromaRetriever]]":
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


def _load_reranker(cfg: Any) -> "Optional[DocumentReranker]":
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

    # --- Local model (Ollama) ---
    try:
        app.state.local_model = _load_local_model(cfg)
        if app.state.local_model is not None:
            logger.info(f"Local LLM loaded: ollama/{cfg.model.name}")
        else:
            logger.info(f"Local LLM disabled (configured provider: {cfg.model.provider})")
    except Exception as e:
        logger.warning(f"Local LLM not available ({cfg.model.provider}/{cfg.model.name}): {e}")
        app.state.local_model = None

    # --- Cloud model ---
    try:
        app.state.cloud_model = _load_cloud_model(cfg)
        cloud_provider, cloud_name = _resolve_cloud_model_config(cfg)
        logger.info(f"Cloud LLM loaded: {cloud_provider}/{cloud_name}")
    except Exception as e:
        logger.warning(f"Cloud LLM not available: {e}")
        app.state.cloud_model = None

    # Backward-compatible single model reference (local preferred)
    app.state.model = app.state.local_model or app.state.cloud_model

    # --- Agents: one set per model provider ---
    # Hybrid tool-calling: when enabled and both models are available, the local
    # intelligent agent uses cloud model for planning and local model for generation.
    use_hybrid = bool(getattr(cfg.model, "hybrid_tool_calling", False))
    local_tool_llm: Optional[BaseChatModel] = (
        app.state.cloud_model if use_hybrid and app.state.cloud_model is not None else None
    )
    if use_hybrid and local_tool_llm is not None:
        logger.info("Hybrid tool-calling enabled: local agent will use cloud model for tool selection")
    elif use_hybrid:
        logger.warning("Hybrid tool-calling requested but cloud model unavailable; using single-model mode")

    app.state.local_intelligent_agent, app.state.local_keyword_agent = _load_agents(
        app.state.local_model,
        tool_llm=local_tool_llm,
    )
    if app.state.cloud_model is not None:
        app.state.cloud_intelligent_agent, app.state.cloud_keyword_agent = _load_agents(
            app.state.cloud_model
        )
    else:
        app.state.cloud_intelligent_agent = None
        app.state.cloud_keyword_agent = None

    # Backward-compatible single agent references (local preferred, cloud fallback)
    app.state.intelligent_agent = (
        app.state.local_intelligent_agent or app.state.cloud_intelligent_agent
    )
    app.state.keyword_agent = (
        app.state.local_keyword_agent or app.state.cloud_keyword_agent
    )

    # Retriever (vector / hybrid) -- shared across all models
    app.state.retriever = _load_retriever(cfg)

    # Reranker -- shared across all models
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
            "local_model": app.state.local_model is not None,
            "cloud_model": app.state.cloud_model is not None,
            "local_intelligent_agent": app.state.local_intelligent_agent is not None,
            "local_keyword_agent": app.state.local_keyword_agent is not None,
            "cloud_intelligent_agent": app.state.cloud_intelligent_agent is not None,
            "cloud_keyword_agent": app.state.cloud_keyword_agent is not None,
            "retriever": app.state.retriever is not None,
            "reranker": app.state.reranker is not None,
        },
    }
