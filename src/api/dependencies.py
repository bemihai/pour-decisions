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

from src.agents.memory import ConversationMemoryManager
from src.retrieval import AsyncRAGRuntimeResources, ChromaRetriever, DocumentReranker, HybridRetriever


def get_model(request: Request) -> BaseChatModel:
    """Retrieve the direct Cloud model preloaded at startup.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached LLM instance loaded at startup.

    Raises:
        HTTPException: 503 if no LLM was loaded during startup.
    """
    model = getattr(request.app.state, "cloud_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="LLM model not available. Check startup logs for loading errors.")
    return model


def get_optional_model(request: Request) -> BaseChatModel | None:
    """Retrieve the direct Cloud model without raising on absence.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached LLM instance, or None if it was not loaded during startup.
    """
    return getattr(request.app.state, "cloud_model", None)


def get_description_model(request: Request) -> BaseChatModel | None:
    """Retrieve the direct Cloud model for AI description generation.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The Cloud model if loaded, otherwise None.
    """
    return getattr(request.app.state, "cloud_model", None)


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


def get_async_rag_runtime(request: Request) -> AsyncRAGRuntimeResources:
    """Retrieve the lifespan-owned resources for asynchronous RAG execution.

    Raises:
        HTTPException: 503 if application startup did not construct the bundle.
    """
    resources = getattr(request.app.state, "async_rag_runtime", None)
    if resources is None:
        raise HTTPException(status_code=503, detail="Async RAG runtime not available. Check startup logs for errors.")
    return resources


def get_intelligent_agent(request: Request):
    """Retrieve the preloaded Cloud intelligent agent from application state.

    Args:
        request: The incoming FastAPI request (injected automatically).

    Returns:
        The cached WineAgent instance, or None if loading failed.
    """
    return getattr(request.app.state, "intelligent_agent", None)


def get_conversation_memory_manager(request: Request) -> ConversationMemoryManager | None:
    """Retrieve the optional lifespan-owned conversation memory manager."""
    return getattr(request.app.state, "conversation_memory_manager", None)
