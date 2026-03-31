"""Pydantic request/response schemas for the cellar API."""
from pydantic import BaseModel, Field


class FilterOptions(BaseModel):
    """Available filter values for cellar inventory dropdowns."""

    wine_types: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)
    min_vintage: int = 2000
    max_vintage: int = 2025


class InventoryItem(BaseModel):
    """Single wine in the inventory (bottles aggregated by wine_id)."""

    wine_id: int
    wine_name: str = ""
    producer_name: str | None = None
    vintage: int | None = None
    wine_type: str | None = None
    varietal: str | None = None
    country: str | None = None
    region_name: str | None = None
    quantity: int = 0
    personal_rating: int | None = None
    community_rating: float | None = None
    drink_index: float | None = None
    drink_from_year: int | None = None
    drink_to_year: int | None = None
    drink_window_source: str | None = None
    location: str | None = None
    bin: str | None = None
    purchase_price: float | None = None
    currency: str | None = None
    description: str | None = None
    producer_description: str | None = None
    producer_id: int | None = None
    bottle_note: str | None = None
    last_tasted_date: str | None = None
    like_votes: int | None = None
    like_percentage: float | None = None
    # Community cellar fields
    q_purchased: int = 0
    q_consumed: int = 0
    q_quantity: int = 0
    created_at: str | None = None


class InventoryResponse(BaseModel):
    """Paginated / filtered cellar inventory response."""

    items: list[InventoryItem]
    total_wines: int = Field(..., description="Number of unique wines matching filters")
    total_bottles: int = Field(..., description="Sum of bottle quantities matching filters")
    filter_options: FilterOptions


class CellarOverview(BaseModel):
    """High-level cellar metrics."""

    total_bottles: int = 0
    unique_wines: int = 0
    by_type: list[dict] = Field(default_factory=list)
    by_country: list[dict] = Field(default_factory=list)


class DrinkingWindowStats(BaseModel):
    """Drinking window status breakdown."""

    ready_to_drink: int = 0
    to_hold: int = 0
    unknown: int = 0


class CellarValueStats(BaseModel):
    """Cellar value information."""

    by_currency: list[dict] = Field(default_factory=list)
    bottles_without_price: int = 0


class CellarStatsResponse(BaseModel):
    """Combined cellar statistics for the overview section."""

    overview: CellarOverview
    drinking_stats: DrinkingWindowStats
    value_stats: CellarValueStats


class ChartDataResponse(BaseModel):
    """Pre-computed chart data for the Statistics & Charts tab.

    Each field contains the data needed to render one Plotly chart
    on the frontend (labels, values, colors).
    """

    wine_type_distribution: list[dict] = Field(default_factory=list)
    country_distribution: list[dict] = Field(default_factory=list)
    varietal_distribution: list[dict] = Field(default_factory=list)
    region_distribution: list[dict] = Field(default_factory=list)
    drinking_window_wines: dict = Field(default_factory=dict)
    cellar_size_over_time: list[dict] = Field(default_factory=list)
    top_rated: list[dict] = Field(default_factory=list)


class SyncRequest(BaseModel):
    """CellarTracker sync request (credentials come from server .env)."""

    pass  # No body needed; credentials are server-side env vars


class SyncResponse(BaseModel):
    """CellarTracker sync result summary."""

    success: bool
    wines_processed: int = 0
    wines_imported: int = 0
    bottles_processed: int = 0
    bottles_imported: int = 0
    producers_created: int = 0
    regions_created: int = 0
    errors: list[str] = Field(default_factory=list)
    error_message: str | None = None

