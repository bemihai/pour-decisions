"""Shared FastAPI dependencies for request-scoped and app-scoped resources.

Provides dependency-injection functions that retrieve preloaded resources
(LLM, agents, retriever, reranker) from ``app.state``, which is populated
during the application lifespan startup in ``main.py``.

Usage in route handlers::

    @router.get("/example")
    async def example(retriever: HybridRetriever | ChromaRetriever | None = Depends(get_retriever)):
        ...
"""
from typing import Union

from fastapi import HTTPException, Request
from langchain_core.language_models import BaseChatModel

from src.retrieval import ChromaRetriever, HybridRetriever, DocumentReranker


def get_model(request: Request) -> BaseChatModel:
    """Retrieve the default preloaded LLM from application state.

    Returns the local model if available, otherwise the cloud model.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached LLM instance loaded at startup.

    Raises:
        HTTPException: 503 if no LLM was loaded during startup.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="LLM model not available. Check startup logs for loading errors.")
    return model


def get_optional_model(request: Request) -> BaseChatModel | None:
    """Retrieve the default preloaded LLM without raising on absence.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached LLM instance, or None if it was not loaded during startup.
    """
    return getattr(request.app.state, "model", None)


def get_local_model(request: Request) -> BaseChatModel | None:
    """Retrieve the local (Ollama/Gemma 4) model from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The local ``ChatOllama`` instance, or None if Ollama was unavailable at startup.
    """
    return getattr(request.app.state, "local_model", None)


def get_cloud_model(request: Request) -> BaseChatModel | None:
    """Retrieve the cloud (Gemini) model from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cloud ``ChatGoogleGenerativeAI`` instance, or None if unavailable at startup.
    """
    return getattr(request.app.state, "cloud_model", None)


def get_description_model(request: Request) -> BaseChatModel | None:
    """Retrieve the preferred model for AI description generation.

    Prefers the cloud model (Gemini) for structured-output calls because:
    - ``with_structured_output`` on a CPU-only local Gemma 4 takes ~93 s per wine.
    - Descriptions are persisted in SQLite after the first generation, so the
      cloud API cost is negligible (one call per wine, ever).

    Falls back to the local model or any available model when the cloud model
    is not loaded.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cloud model if available, otherwise the local or default model, or None.
    """
    cloud = getattr(request.app.state, "cloud_model", None)
    if cloud is not None:
        return cloud
    return getattr(request.app.state, "model", None)


def get_retriever(request: Request) -> Union[HybridRetriever, ChromaRetriever, None]:
    """Retrieve the preloaded retriever (hybrid or vector-only) from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached retriever instance, or None if retrieval is unavailable.
    """
    return getattr(request.app.state, "retriever", None)


def get_reranker(request: Request) -> DocumentReranker | None:
    """Retrieve the preloaded cross-encoder reranker from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached reranker instance, or None if reranking is disabled.
    """
    return getattr(request.app.state, "reranker", None)


def get_intelligent_agent(request: Request):
    """Retrieve the default preloaded intelligent agent from application state.

    Returns the local agent if available, otherwise the cloud agent.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached WineAgent instance, or None if loading failed.
    """
    return getattr(request.app.state, "intelligent_agent", None)


def get_keyword_agent(request: Request):
    """Retrieve the default preloaded keyword-routing agent from application state.

    Returns the local agent if available, otherwise the cloud agent.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached KeywordWineAgent instance, or None if loading failed.
    """
    return getattr(request.app.state, "keyword_agent", None)

