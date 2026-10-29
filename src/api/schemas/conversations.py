"""Pydantic contracts for saved-conversation lifecycle endpoints."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.schemas.chat import AgentMode, ChatResponse, Source, WebSource


class ConversationCreate(BaseModel):
    """Create an empty saved session from first-message display text."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    agent_mode: AgentMode
    first_message: str = Field(min_length=1, max_length=32_000)

    @field_validator("first_message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        """Reject whitespace-only display text without changing accepted content."""
        if not value.strip():
            raise ValueError("first_message must contain non-whitespace text")
        return value


class ConversationRename(BaseModel):
    """Optimistically rename one saved session."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    expected_version: int = Field(gt=0)


class ConversationMetadata(BaseModel):
    """Public saved-session metadata."""

    id: UUID
    agent_mode: AgentMode
    title: str
    version: int = Field(gt=0)
    state: Literal["active", "deleting"]
    created_at: datetime
    updated_at: datetime


class ConversationPage(BaseModel):
    """Recent-first metadata page."""

    items: list[ConversationMetadata]
    next_cursor: str | None = None


class ConversationTurn(BaseModel):
    """One public transcript turn with accepted and displayed input."""

    sequence: int = Field(gt=0)
    message: str
    display_message: str
    answer: str | None
    sources: list[Source]
    web_sources: list[WebSource]
    status: Literal["running", "completed", "failed", "interrupted"]
    created_at: datetime
    updated_at: datetime


class ConversationTurnPage(BaseModel):
    """Ascending transcript slice plus coherent session version."""

    items: list[ConversationTurn]
    next_before_sequence: int | None = None
    conversation_version: int = Field(gt=0)


class ConversationOperation(BaseModel):
    """Durable state used to reconcile an accepted saved operation."""

    operation_id: UUID
    action: Literal["append", "replace_last"]
    turn_sequence: int = Field(gt=0)
    status: Literal["running", "completed", "failed", "interrupted"]
    response: ChatResponse | None
    error_code: str | None
    conversation_version: int = Field(gt=0)
