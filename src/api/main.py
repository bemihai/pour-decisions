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
from src.retrieval import HybridRetriever, build_reranker_from_config, build_retriever_from_config
from src.utils import get_config, init_observability, is_observability_active, logger

if TYPE_CHECKING:
    from src.retrieval import ChromaRetriever, DocumentReranker, HybridRetriever


def _load_local_model(cfg: Any) -> BaseChatModel:
    """Load the local Ollama model from the dedicated local model config.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A ``ChatOllama`` instance.

    Raises:
        Exception: If Ollama is unreachable or the model is not pulled.
    """
    from src.agents.llm import load_base_model

    ollama_cfg = getattr(cfg.model, "ollama", None)
    model_name = str(getattr(ollama_cfg, "name", getattr(cfg.model, "name", "gemma3:4b")))
    base_url = str(getattr(ollama_cfg, "base_url", "http://localhost:11434"))
    return load_base_model("ollama", model_name, base_url=base_url)


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


def _is_local_model_startup_enabled(cfg: Any) -> bool:
    """Return whether API startup should load local Ollama resources.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        ``True`` when the explicit API startup flag enables local model loading.
    """
    api_cfg = getattr(cfg, "api", None)
    return bool(getattr(api_cfg, "enable_local_model_startup", False))


def _is_hybrid_tool_calling_enabled(cfg: Any) -> bool:
    """Return whether local agents should use the cloud model for tool planning."""
    return bool(getattr(cfg.model, "hybrid_tool_calling", False))


def _load_agents(
    llm: BaseChatModel | None = None,
    tool_llm: BaseChatModel | None = None,
) -> Tuple[Optional[Any], None]:
    """Load the intelligent agent with the given LLM.

    Args:
        llm: Pre-loaded model for final answer generation. If ``None`` the agent
             will load its own model from config.
        tool_llm: Optional model for tool selection / planning (hybrid mode). When
             provided and different from ``llm``, the intelligent agent uses
             ``tool_llm`` for planning and ``llm`` for generation.

    Returns:
        Tuple of (intelligent_agent, None).
    """
    from src.agents import create_wine_agent

    intelligent_agent = None

    try:
        intelligent_agent = create_wine_agent(verbose=False, llm=llm, tool_llm=tool_llm)
        logger.info("Intelligent wine agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load intelligent agent: {e}")

    return intelligent_agent, None


def _load_retriever(cfg: Any) -> "Optional[Union[HybridRetriever, ChromaRetriever]]":
    """Load the retriever stack (vector, BM25, hybrid) from config.

    Mirrors the logic in ``src/ui/resources.py`` (load_vector_retriever,
    load_bm25_index, load_retriever) without Streamlit dependencies.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A ``HybridRetriever``, ``ChromaRetriever``, or None on failure.
    """
    try:
        retriever = build_retriever_from_config(
            cfg,
            enable_cache=True,
            enable_query_expansion=False,
        )
        if isinstance(retriever, HybridRetriever):
            logger.info(
                "Using HybridRetriever (semantic_pool=%s, bm25_pool=%s, union_limit=%s)",
                getattr(cfg.chroma.retrieval, "semantic_candidate_pool", 25),
                getattr(cfg.chroma.retrieval, "bm25_candidate_pool", 25),
                getattr(cfg.chroma.retrieval, "reranker_input_limit", 50),
            )
        else:
            logger.info("Using ChromaRetriever (vector-only)")
        return retriever
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        return None


def _load_reranker(cfg: Any) -> "Optional[DocumentReranker]":
    """Load the cross-encoder reranker if enabled in config.

    Args:
        cfg: Application OmegaConf config.

    Returns:
        A ``DocumentReranker`` instance, or None if disabled / on failure.
    """
    return build_reranker_from_config(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load expensive resources once at startup, release on shutdown."""
    cfg = get_config()
    app.state.config = cfg
    init_observability(cfg)
    if is_observability_active():
        logger.info("Observability: enabled (phoenix)")
    else:
        logger.info("Observability: disabled")

    # --- Cloud model (Gemini) ---
    try:
        app.state.cloud_model = _load_cloud_model(cfg)
        cloud_provider, cloud_name = _resolve_cloud_model_config(cfg)
        logger.info(f"Cloud LLM loaded: {cloud_provider}/{cloud_name}")
    except Exception as e:
        logger.warning(f"Cloud LLM not available: {e}")
        app.state.cloud_model = None

    # Backward-compatible single model reference keeps the production default explicit.
    app.state.model = app.state.cloud_model

    if app.state.cloud_model is not None:
        app.state.cloud_intelligent_agent, _ = _load_agents(app.state.cloud_model)
    else:
        app.state.cloud_intelligent_agent = None

    enable_local_startup = _is_local_model_startup_enabled(cfg)

    # --- Local model (Ollama) ---
    # Production stays cloud-first by default. Local API startup is available only
    # when the explicit config flag is enabled for deliberate experiments.
    app.state.local_model = None
    app.state.local_intelligent_agent = None
    if enable_local_startup:
        try:
            app.state.local_model = _load_local_model(cfg)
            tool_llm = None
            if _is_hybrid_tool_calling_enabled(cfg):
                tool_llm = app.state.cloud_model
                if tool_llm is None:
                    logger.warning("Hybrid tool calling requested, but no cloud model is available")
            app.state.local_intelligent_agent, _ = _load_agents(app.state.local_model, tool_llm=tool_llm)
            logger.info("Local LLM startup enabled: Ollama model loaded")
        except Exception as e:
            logger.warning(f"Local LLM startup enabled, but Ollama is not available: {e}")
            app.state.local_model = None
            app.state.local_intelligent_agent = None
    else:
        logger.info("Local LLM startup disabled by config; API remains cloud-first")

    # Backward-compatible single agent reference keeps the production default explicit.
    app.state.intelligent_agent = app.state.cloud_intelligent_agent

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
            "cloud_intelligent_agent": app.state.cloud_intelligent_agent is not None,
            "retriever": app.state.retriever is not None,
            "reranker": app.state.reranker is not None,
        },
    }
