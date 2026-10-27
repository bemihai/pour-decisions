"""Synchronous Shopping List endpoints over the local SQLite repository."""

from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas.shopping_list import (
    CatalogueItemCreate,
    CatalogueOptionResponse,
    CatalogueOptionsResponse,
    ManualItemCreate,
    ShoppingListCreateRequest,
    ShoppingListDeleteResponse,
    ShoppingListItemResponse,
    ShoppingListPageResponse,
    ShoppingListPatchRequest,
)
from src.database.models import ShoppingListStatus
from src.database.repository import (
    ShoppingListDuplicateError,
    ShoppingListNotFoundError,
    ShoppingListRepository,
    ShoppingListStoreBusyError,
    ShoppingListStoreUnavailableError,
    ShoppingListValidationError,
    ShoppingListVersionConflictError,
)
from src.utils import logger

router = APIRouter(prefix="/api/shopping-list", tags=["shopping-list"])
ResultType = TypeVar("ResultType")


def _api_error(status_code: int, code: str, message: str, *, retry: bool = False, **details: object) -> HTTPException:
    """Build one stable, non-sensitive Shopping List error."""
    body: dict[str, object] = {"code": code, "message": message, **details}
    return HTTPException(
        status_code=status_code,
        detail=body,
        headers={"Retry-After": "1"} if retry else None,
    )


def _execute(operation: Callable[[], ResultType]) -> ResultType:
    """Execute repository work and translate bounded failures."""
    try:
        return operation()
    except ShoppingListNotFoundError as error:
        raise _api_error(
            404, "shopping_list_not_found", "Shopping List item or catalogue wine was not found."
        ) from error
    except ShoppingListDuplicateError as error:
        raise _api_error(
            409,
            "shopping_list_duplicate",
            "A Wanted item already uses this identity.",
            existing_item_id=error.existing_item_id,
        ) from error
    except ShoppingListVersionConflictError as error:
        raise _api_error(409, "shopping_list_version_conflict", "Shopping List item version changed.") from error
    except ShoppingListValidationError as error:
        raise _api_error(422, "shopping_list_mutation_invalid", "Shopping List mutation is invalid.") from error
    except ShoppingListStoreBusyError as error:
        raise _api_error(503, "shopping_list_store_busy", "Shopping List storage is busy.", retry=True) from error
    except ShoppingListStoreUnavailableError as error:
        raise _api_error(503, "shopping_list_schema_unavailable", "Shopping List schema is unavailable.") from error
    except Exception as error:
        logger.exception("Unexpected Shopping List repository failure")
        raise _api_error(500, "shopping_list_store_unavailable", "Shopping List storage is unavailable.") from error


@router.get("/items", response_model=ShoppingListPageResponse)
def list_items(
    status_filter: ShoppingListStatus = Query(ShoppingListStatus.WANTED, alias="status"),
    search: str | None = Query(default=None, max_length=300),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ShoppingListPageResponse:
    """Return one filtered, bounded lifecycle page."""
    items, total = _execute(
        lambda: ShoppingListRepository().list_items(status_filter, search=search, limit=limit, offset=offset)
    )
    return ShoppingListPageResponse(
        items=[ShoppingListItemResponse.from_item(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/items", response_model=ShoppingListItemResponse, status_code=status.HTTP_201_CREATED)
def create_item(request: ShoppingListCreateRequest) -> ShoppingListItemResponse:
    """Create one Wanted manual or catalogue item."""
    repository = ShoppingListRepository()
    common = request.model_dump(include={"quantity", "priority", "notes", "retailer_url"})
    if isinstance(request, CatalogueItemCreate):
        item = _execute(lambda: repository.create_catalogue(request.wine_id, **common))
    elif isinstance(request, ManualItemCreate):
        item = _execute(lambda: repository.create_manual(request.name, vintage=request.vintage, **common))
    else:  # pragma: no cover - discriminated validation makes this unreachable
        raise _api_error(422, "shopping_list_mutation_invalid", "Shopping List mutation is invalid.")
    return ShoppingListItemResponse.from_item(item)


@router.patch("/items/{item_id}", response_model=ShoppingListItemResponse)
def update_item(item_id: int, request: ShoppingListPatchRequest) -> ShoppingListItemResponse:
    """Update mutable fields using optimistic concurrency."""
    changes = request.model_dump(exclude={"expected_version"}, exclude_unset=True)
    item = _execute(lambda: ShoppingListRepository().update(item_id, request.expected_version, changes))
    return ShoppingListItemResponse.from_item(item)


@router.delete("/items/{item_id}", response_model=ShoppingListDeleteResponse)
def delete_item(item_id: int, expected_version: int = Query(..., gt=0)) -> ShoppingListDeleteResponse:
    """Delete one item using optimistic concurrency."""
    deleted_id, deleted_version = _execute(lambda: ShoppingListRepository().delete(item_id, expected_version))
    return ShoppingListDeleteResponse(id=deleted_id, deleted_version=deleted_version)


@router.get("/catalogue-options", response_model=CatalogueOptionsResponse)
def list_catalogue_options(
    search: str = Query(..., min_length=1, max_length=300),
    offset: int = Query(default=0, ge=0),
) -> CatalogueOptionsResponse:
    """Return a local catalogue page of at most twenty options."""
    limit = 20
    options, has_more = _execute(
        lambda: ShoppingListRepository().list_catalogue_options(search, limit=limit, offset=offset)
    )
    return CatalogueOptionsResponse(
        items=[CatalogueOptionResponse.from_option(option) for option in options],
        has_more=has_more,
        limit=limit,
        offset=offset,
    )
