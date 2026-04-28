"""Pydantic response schemas for the wine detail API."""
from pydantic import BaseModel, Field


class BottleDetail(BaseModel):
    """Individual bottle information for a wine."""

    id: int
    quantity: int = 1
    status: str = "in_cellar"
    location: str | None = None
    bin: str | None = None
    purchase_date: str | None = None
    purchase_price: float | None = None
    valuation_price: float | None = None
    currency: str = "RON"
    store_name: str | None = None
    consumed_date: str | None = None
    bottle_note: str | None = None


class WineDetailResponse(BaseModel):
    """Full wine detail with producer, region, tasting, and bottle data."""

    # Core identifiers
    id: int
    source: str = "manual"
    external_id: str | None = None

    # Wine information
    wine_name: str = ""
    vintage: int | None = None
    wine_type: str = "Red"
    varietal: str | None = None
    designation: str | None = None
    appellation: str | None = None
    vineyard: str | None = None
    bottle_size: str = "750ml"

    # Drinking window
    drink_from_year: int | None = None
    drink_to_year: int | None = None
    drink_index: float | None = None
    drink_window_source: str | None = None

    # Descriptions
    description: str | None = None
    producer_description: str | None = None

    # Producer
    producer_id: int | None = None
    producer_name: str | None = None

    # Region
    region_id: int | None = None
    region_name: str | None = None
    region_description: str | None = None
    country: str | None = None

    # Tasting
    personal_rating: int | None = None
    community_rating: float | None = None
    do_like: bool | None = None
    is_defective: bool | None = None
    tasting_notes: str | None = None
    last_tasted_date: str | None = None

    # Community inventory
    q_purchased: int = 0
    q_quantity: int = 0
    q_consumed: int = 0

    # Bottles
    bottles: list[BottleDetail] = Field(default_factory=list)
    owned_quantity: int = 0

    created_at: str | None = None
    updated_at: str | None = None


class DescriptionRequest(BaseModel):
    """Request body for triggering AI description generation."""

    use_rag_context: bool = Field(True, description="Use RAG wine book context for enrichment")
    use_web_search: bool = Field(True, description="Use web search for additional context")


class DescriptionResponse(BaseModel):
    """Result of an AI description generation request."""

    success: bool
    description: str | None = None
    drink_from_year: int | None = None
    drink_to_year: int | None = None
    error: str | None = None

