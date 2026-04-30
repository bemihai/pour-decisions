"""Wine detail API endpoints.

Exposes single wine detail retrieval and AI description generation.

Note: all route handlers are synchronous (``def``). FastAPI runs them in a
thread-pool executor so the event loop remains unblocked.
"""
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.language_models import BaseChatModel

from src.api.dependencies import get_description_model, get_reranker, get_retriever
from src.api.schemas.wines import (
    BottleDetail,
    DescriptionRequest,
    DescriptionResponse,
    WineDetailResponse,
)
from src.database.models import Bottle, Wine
from src.database.repository import BottleRepository, ProducerRepository, WineRepository
from src.retrieval import ChromaRetriever, DocumentReranker, HybridRetriever
from src.utils import logger

router = APIRouter(prefix="/api/wines", tags=["wines"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wine_to_detail(wine: Wine, bottles: list[Bottle], owned_quantity: int, producer_repo: ProducerRepository) -> WineDetailResponse:
    """Convert a Wine model and its bottles to a WineDetailResponse.

    Args:
        wine: Wine model instance from WineRepository.
        bottles: List of Bottle model instances for this wine.
        owned_quantity: Total in-cellar bottle count.
        producer_repo: Shared ProducerRepository instance (avoids repeated instantiation).

    Returns:
        Fully populated WineDetailResponse.
    """
    bottle_details = [
        BottleDetail(
            id=b.id,
            quantity=b.quantity,
            status=b.status,
            location=b.location,
            bin=b.bin,
            purchase_date=str(b.purchase_date) if b.purchase_date else None,
            purchase_price=b.purchase_price,
            valuation_price=b.valuation_price,
            currency=b.currency,
            store_name=b.store_name,
            consumed_date=str(b.consumed_date) if b.consumed_date else None,
            bottle_note=b.bottle_note,
        )
        for b in bottles
    ]

    producer_description = None
    if wine.producer_id:
        producer = producer_repo.get_by_id(wine.producer_id)
        if producer:
            producer_description = producer.description
    
    region_description = None
    if wine.region_id:
        from src.database.repository import RegionRepository
        region_repo = RegionRepository()
        region = region_repo.get_by_id(wine.region_id)
        if region:
            region_description = region.description

    return WineDetailResponse(
        id=wine.id,
        source=wine.source,
        external_id=wine.external_id,
        wine_name=wine.wine_name,
        vintage=wine.vintage,
        wine_type=wine.wine_type,
        varietal=wine.varietal,
        designation=wine.designation,
        appellation=wine.appellation,
        vineyard=wine.vineyard,
        bottle_size=wine.bottle_size,
        drink_from_year=wine.drink_from_year,
        drink_to_year=wine.drink_to_year,
        drink_index=wine.drink_index,
        drink_window_source=wine.drink_window_source,
        description=wine.description,
        producer_description=producer_description,
        producer_id=wine.producer_id,
        producer_name=wine.producer_name,
        region_id=wine.region_id,
        region_name=wine.region_name,
        region_description=region_description,
        country=wine.country,
        personal_rating=wine.personal_rating,
        community_rating=wine.community_rating,
        do_like=bool(wine.do_like) if wine.do_like is not None else None,
        is_defective=bool(wine.is_defective) if wine.is_defective is not None else None,
        tasting_notes=wine.tasting_notes,
        last_tasted_date=str(wine.last_tasted_date) if wine.last_tasted_date else None,
        q_purchased=wine.q_purchased,
        q_quantity=wine.q_quantity,
        q_consumed=wine.q_consumed,
        bottles=bottle_details,
        owned_quantity=owned_quantity,
        created_at=str(wine.created_at) if wine.created_at else None,
        updated_at=str(wine.updated_at) if wine.updated_at else None,
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/{wine_id}", response_model=WineDetailResponse)
def get_wine_detail(wine_id: int) -> WineDetailResponse:
    """Return full detail for a single wine including bottles.

    Args:
        wine_id: Database ID of the wine.

    Raises:
        HTTPException: 404 if wine not found.
    """
    wine_repo = WineRepository()
    wine = wine_repo.get_by_id(wine_id)
    if wine is None:
        raise HTTPException(status_code=404, detail=f"Wine {wine_id} not found")

    bottle_repo = BottleRepository()
    producer_repo = ProducerRepository()
    bottles = bottle_repo.get_by_wine(wine_id)
    owned_quantity = bottle_repo.get_owned_quantity(wine_id)

    return _wine_to_detail(wine, bottles, owned_quantity, producer_repo)


@router.post("/{wine_id}/description", response_model=DescriptionResponse)
def generate_wine_description(
    wine_id: int,
    body: DescriptionRequest | None = None,
    model: BaseChatModel | None = Depends(get_description_model),
    retriever: Union[HybridRetriever, ChromaRetriever, None] = Depends(get_retriever),
    reranker: DocumentReranker | None = Depends(get_reranker),
) -> DescriptionResponse:
    """Trigger AI description generation for a wine.

    Uses the DescriptionService to generate a RAG-enhanced description
    and optionally estimate a drinking window. The result is persisted
    in the database so subsequent GET requests return it immediately.

    The cloud model (Gemini) is preferred for this endpoint: structured-output
    generation on a CPU-only local Gemma 4 takes ~93 s, while cloud is < 5 s.
    The DescriptionService will auto-select the cloud model if ``model`` is None.

    Args:
        wine_id: Database ID of the wine.
        body: Optional request body with RAG/web search flags.
        model: Injected model from app state (cloud preferred, see get_description_model).
        retriever: Injected retriever from app state.
        reranker: Injected reranker from app state.

    Raises:
        HTTPException: 404 if wine not found, 502 if the LLM call fails.
    """
    wine_repo = WineRepository()
    wine = wine_repo.get_by_id(wine_id)
    if wine is None:
        raise HTTPException(status_code=404, detail=f"Wine {wine_id} not found")

    use_rag = body.use_rag_context if body else True
    use_web = body.use_web_search if body else True

    logger.info(
        "Description request received: route=/api/wines/%s/description body_present=%s "
        "use_rag_context=%s use_web_search=%s",
        wine_id,
        body is not None,
        use_rag,
        use_web,
    )

    try:
        from src.agents.description_service import DescriptionService

        service = DescriptionService(
            model=model,
            retriever=retriever,
            reranker=reranker,
            use_rag_context=use_rag,
            use_web_search=use_web,
        )

        logger.info(
            "Description service initialized: route=/api/wines/%s/description effective_use_web_search=%s",
            wine_id,
            service.use_web_search,
        )

        description = service.get_wine_description(wine, force_regenerate=True)

        if not description:
            raise HTTPException(status_code=502, detail="LLM failed to generate a description")

        updated_wine = wine_repo.get_by_id(wine_id)
        return DescriptionResponse(
            success=True,
            description=description,
            drink_from_year=updated_wine.drink_from_year if updated_wine else None,
            drink_to_year=updated_wine.drink_to_year if updated_wine else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Description generation failed for wine {wine_id}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/{wine_id}/producer-description", response_model=DescriptionResponse)
def generate_producer_description(
    wine_id: int,
    body: DescriptionRequest | None = None,
    model: BaseChatModel | None = Depends(get_description_model),
    retriever: Union[HybridRetriever, ChromaRetriever, None] = Depends(get_retriever),
    reranker: DocumentReranker | None = Depends(get_reranker),
) -> DescriptionResponse:
    """Trigger AI description generation for the producer of a wine.

    Looks up the producer via the wine, then delegates to DescriptionService.
    The result is persisted so subsequent GET requests return it immediately.

    Args:
        wine_id: Database ID of the wine whose producer needs a description.
        body: Optional flags for RAG / web search context.
        model: Injected LLM from app state.
        retriever: Injected retriever from app state.
        reranker: Injected reranker from app state.

    Raises:
        HTTPException: 404 if wine or producer not found, 502 if LLM call fails.
    """
    wine_repo = WineRepository()
    wine = wine_repo.get_by_id(wine_id)
    if wine is None:
        raise HTTPException(status_code=404, detail=f"Wine {wine_id} not found")
    if not wine.producer_id:
        raise HTTPException(status_code=404, detail=f"Wine {wine_id} has no associated producer")

    producer_repo = ProducerRepository()
    producer = producer_repo.get_by_id(wine.producer_id)
    if producer is None:
        raise HTTPException(status_code=404, detail=f"Producer {wine.producer_id} not found")

    use_rag = body.use_rag_context if body else True
    use_web = body.use_web_search if body else True

    logger.info(
        "Description request received: route=/api/wines/%s/producer-description body_present=%s "
        "use_rag_context=%s use_web_search=%s producer_id=%s",
        wine_id,
        body is not None,
        use_rag,
        use_web,
        wine.producer_id,
    )

    try:
        from src.agents.description_service import DescriptionService

        service = DescriptionService(
            model=model,
            retriever=retriever,
            reranker=reranker,
            use_rag_context=use_rag,
            use_web_search=use_web,
        )

        logger.info(
            "Description service initialized: route=/api/wines/%s/producer-description effective_use_web_search=%s",
            wine_id,
            service.use_web_search,
        )

        # force_regenerate by clearing cached value so DescriptionService re-generates
        producer.description = None
        description = service.get_producer_description(producer)

        if not description:
            raise HTTPException(status_code=502, detail="LLM failed to generate a producer description")

        return DescriptionResponse(success=True, description=description)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Producer description generation failed for wine {wine_id}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=str(e))
