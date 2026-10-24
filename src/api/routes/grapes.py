"""Explicit, asynchronous generation of reusable grape descriptions."""

import asyncio
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.language_models import BaseChatModel

from src.agents.description_service import DescriptionService
from src.api.dependencies import get_description_model, get_reranker, get_retriever
from src.api.schemas.wines import DescriptionRequest, DescriptionResponse
from src.database.repository import ProducerRepository, WineRepository
from src.database.repository.grape import GrapeRepository
from src.retrieval import ChromaRetriever, DocumentReranker, HybridRetriever
from src.utils import logger

router = APIRouter(prefix="/api/grapes", tags=["grapes"])


@router.post("/{grape_id}/description", response_model=DescriptionResponse)
async def generate_grape_description(
    grape_id: int,
    body: DescriptionRequest | None = None,
    model: BaseChatModel | None = Depends(get_description_model),
    retriever: Union[HybridRetriever, ChromaRetriever, None] = Depends(get_retriever),
    reranker: DocumentReranker | None = Depends(get_reranker),
) -> DescriptionResponse:
    """Generate or regenerate on explicit request, retaining the cache on failure."""
    repository = GrapeRepository()
    grape = await asyncio.to_thread(repository.get_by_id, grape_id)
    if grape is None:
        raise HTTPException(status_code=404, detail=f"Grape {grape_id} not found")
    options = body or DescriptionRequest()
    try:
        service = DescriptionService(
            model=model,
            retriever=retriever,
            reranker=reranker,
            use_rag_context=options.use_rag_context,
            use_web_search=options.use_web_search,
            wine_repo=WineRepository(),
            producer_repo=ProducerRepository(),
            grape_repo=repository,
        )
        description = await service.aget_grape_description(grape, force_regenerate=True)
        if not description:
            raise HTTPException(
                status_code=502,
                detail="Could not generate a supported grape description. Check retrieval and model availability, "
                "then retry. Any previous description has been kept.",
            )
        return DescriptionResponse(success=True, description=description)
    except HTTPException:
        raise
    except Exception as error:
        logger.error("Grape description failed for %s: %s", grape_id, type(error).__name__)
        raise HTTPException(status_code=502, detail="Grape description could not be saved. Please retry.") from error
