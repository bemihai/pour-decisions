"""Lifecycle API for server-owned saved conversations."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from src.agents.conversation_service import ConversationService
from src.api.dependencies import get_conversation_service
from src.api.schemas.conversations import (
    ConversationCreate,
    ConversationMetadata,
    ConversationPage,
    ConversationRename,
    ConversationTurnPage,
)
from src.database.repository.conversation import ConversationConflict, ConversationNotFound
from src.utils import logger

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _metadata(session) -> ConversationMetadata:
    """Translate validated repository metadata into the public schema."""
    return ConversationMetadata.model_validate(session, from_attributes=True)


def _raise_conflict(error: ConversationConflict) -> None:
    """Expose a stable code and current metadata for refreshable conflicts."""
    current = _metadata(error.session).model_dump(mode="json") if error.session is not None else None
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"code": error.code, "current": current},
    ) from error


@router.post("", response_model=ConversationMetadata, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: ConversationCreate,
    response: Response,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationMetadata:
    """Idempotently create an empty session without executing a turn."""
    try:
        session, created = await service.create(str(request.id), request.agent_mode, request.first_message)
    except ConversationConflict as error:
        _raise_conflict(error)
    if not created:
        response.status_code = status.HTTP_200_OK
    return _metadata(session)


@router.get("", response_model=ConversationPage)
async def list_conversations(
    limit: int = Query(30, ge=1, le=100),
    cursor: str | None = None,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationPage:
    """Return one stable bounded metadata page."""
    try:
        page = await service.list(limit, cursor)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return ConversationPage(items=[_metadata(item) for item in page.items], next_cursor=page.next_cursor)


@router.get("/{conversation_id}", response_model=ConversationMetadata)
async def get_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationMetadata:
    """Read one active or deleting conversation."""
    try:
        return _metadata(await service.get(str(conversation_id)))
    except ConversationNotFound as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from error


@router.get("/{conversation_id}/turns", response_model=ConversationTurnPage)
async def list_conversation_turns(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=100),
    before_sequence: int | None = Query(None, gt=0),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationTurnPage:
    """Read the latest public transcript slice in ascending sequence."""
    try:
        page = await service.list_turns(str(conversation_id), limit, before_sequence)
    except ConversationNotFound as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from error
    return ConversationTurnPage.model_validate(page, from_attributes=True)


@router.patch("/{conversation_id}", response_model=ConversationMetadata)
async def rename_conversation(
    conversation_id: UUID,
    request: ConversationRename,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationMetadata:
    """Rename one idle active conversation."""
    try:
        return _metadata(await service.rename(str(conversation_id), request.title, request.expected_version))
    except ConversationNotFound as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from error
    except ConversationConflict as error:
        _raise_conflict(error)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    expected_version: int | None = Query(None, gt=0),
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    """Delete content through saver APIs and preserve a permanent marker."""
    try:
        await service.delete(str(conversation_id), expected_version)
    except ConversationConflict as error:
        _raise_conflict(error)
    except Exception as error:
        logger.error("Saved conversation deletion failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Saved conversation deletion failed",
        ) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
