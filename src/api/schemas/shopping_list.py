"""Strict wire contracts for the Shopping List API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.database.models import (
    ShoppingListCatalogueOption,
    ShoppingListIdentityKind,
    ShoppingListItem,
    ShoppingListPriority,
    ShoppingListStatus,
)


class _StrictModel(BaseModel):
    """Reject fields that are outside an endpoint contract."""

    model_config = ConfigDict(extra="forbid")


class _CreateFields(_StrictModel):
    """Fields shared by both creation modes."""

    quantity: int = Field(default=1, ge=1, le=999)
    priority: ShoppingListPriority = ShoppingListPriority.NORMAL
    notes: str | None = Field(default=None, max_length=2000)
    retailer_url: str | None = Field(default=None, max_length=2048)


class CatalogueItemCreate(_CreateFields):
    """Create from one authoritative catalogue record."""

    identity_kind: Literal[ShoppingListIdentityKind.CATALOGUE]
    wine_id: int = Field(gt=0)


class ManualItemCreate(_CreateFields):
    """Create from an explicit manual identity."""

    identity_kind: Literal[ShoppingListIdentityKind.MANUAL]
    name: str = Field(min_length=1, max_length=300)
    vintage: int | None = Field(default=None, ge=1000, le=9999)


ShoppingListCreateRequest = Annotated[CatalogueItemCreate | ManualItemCreate, Field(discriminator="identity_kind")]


class ShoppingListPatchRequest(_StrictModel):
    """Optimistically update mutable item fields."""

    expected_version: int = Field(gt=0)
    name: str | None = Field(default=None, min_length=1, max_length=300)
    vintage: int | None = Field(default=None, ge=1000, le=9999)
    quantity: int | None = Field(default=None, ge=1, le=999)
    priority: ShoppingListPriority | None = None
    notes: str | None = Field(default=None, max_length=2000)
    retailer_url: str | None = Field(default=None, max_length=2048)
    status: ShoppingListStatus | None = None

    @model_validator(mode="after")
    def require_change(self) -> "ShoppingListPatchRequest":
        """Require at least one explicitly supplied mutable field."""
        if not (self.model_fields_set - {"expected_version"}):
            raise ValueError("At least one mutable field is required.")
        return self


class ShoppingListItemResponse(_StrictModel):
    """Authoritative Shopping List item representation."""

    id: int = Field(gt=0)
    wine_id: int | None = Field(default=None, gt=0)
    identity_kind: ShoppingListIdentityKind
    catalogue_identity_id: int | None = Field(default=None, gt=0)
    link_available: bool
    name: str
    vintage: int | None = None
    quantity: int
    priority: ShoppingListPriority
    notes: str | None = None
    retailer_url: str | None = None
    status: ShoppingListStatus
    version: int
    created_at: datetime
    updated_at: datetime
    purchased_at: datetime | None = None

    @classmethod
    def from_item(cls, item: ShoppingListItem) -> "ShoppingListItemResponse":
        """Build a response from a validated repository model."""
        return cls.model_validate(item.model_dump())


class ShoppingListPageResponse(_StrictModel):
    """One bounded Shopping List page and its filtered total."""

    items: list[ShoppingListItemResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class ShoppingListDeleteResponse(_StrictModel):
    """Identity and version of one deleted item."""

    id: int = Field(gt=0)
    deleted_version: int = Field(gt=0)


class CatalogueOptionResponse(_StrictModel):
    """One local catalogue choice."""

    id: int = Field(gt=0)
    name: str
    vintage: int | None = None
    producer: str | None = None
    bottle_size: str

    @classmethod
    def from_option(cls, option: ShoppingListCatalogueOption) -> "CatalogueOptionResponse":
        """Build a response from a validated catalogue option."""
        return cls.model_validate(option.model_dump())


class CatalogueOptionsResponse(_StrictModel):
    """One bounded catalogue-search page."""

    items: list[CatalogueOptionResponse]
    has_more: bool
    limit: int = Field(ge=1, le=20)
    offset: int = Field(ge=0)
