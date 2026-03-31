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
    """Retrieve the preloaded LLM from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached LLM instance loaded at startup.

    Raises:
        HTTPException: 503 if the LLM was not loaded during startup.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="LLM model not available. Check startup logs for loading errors.")
    return model


def get_optional_model(request: Request) -> BaseChatModel | None:
    """Retrieve the preloaded LLM from application state without raising on absence.

    Use this in endpoints where the model is only required for certain execution
    paths (e.g. ``rag_only`` mode), so the endpoint is not unconditionally blocked
    when the LLM failed to load.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached LLM instance, or None if it was not loaded during startup.
    """
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
    """Retrieve the preloaded intelligent (LangGraph ReAct) agent from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached WineAgent instance, or None if loading failed.
    """
    return getattr(request.app.state, "intelligent_agent", None)


def get_keyword_agent(request: Request):
    """Retrieve the preloaded keyword-routing agent from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached KeywordWineAgent instance, or None if loading failed.
    """
    return getattr(request.app.state, "keyword_agent", None)

