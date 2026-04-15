"""Pydantic response schemas for the taste profile API."""
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

class TasteOverviewResponse(BaseModel):
    """Key insight metrics for the taste profile page."""

    avg_rating: float | None = None
    wines_rated: int = 0
    favorite_type: str = "N/A"
    highly_rated_count: int = 0
    highly_rated_pct: float = 0.0


# ---------------------------------------------------------------------------
# Rating distribution (donut chart)
# ---------------------------------------------------------------------------

class RatingBucket(BaseModel):
    """One segment of the rating distribution donut chart."""

    range: str
    count: int


class RatingDistributionResponse(BaseModel):
    """Data for the rating distribution donut chart."""

    buckets: list[RatingBucket] = Field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# Wine types
# ---------------------------------------------------------------------------

class WineTypeStats(BaseModel):
    """Statistics for a single wine type."""

    wine_type: str
    wines_tasted: int = 0
    avg_rating: float | None = None
    highest_rating: float | None = None
    most_recent_date: str | None = None


class WineTypesResponse(BaseModel):
    """Combined distribution and performance data for wine types."""

    types: list[WineTypeStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Varietals
# ---------------------------------------------------------------------------

class VarietalStats(BaseModel):
    """Statistics for a single varietal."""

    varietal: str
    wines_tasted: int = 0
    avg_rating: float | None = None
    highest_rating: float | None = None


class VarietalsResponse(BaseModel):
    """Top varietal preferences."""

    varietals: list[VarietalStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------

class ProducerStats(BaseModel):
    """Statistics for a single producer."""

    producer_name: str
    country: str | None = None
    wines_tasted: int = 0
    avg_rating: float | None = None
    highest_rating: float | None = None
    best_wine_id: int | None = None


class ProducersResponse(BaseModel):
    """Top producer preferences."""

    producers: list[ProducerStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

class RegionStats(BaseModel):
    """Statistics for a single region."""

    region_name: str
    country: str | None = None
    wines_tasted: int = 0
    avg_rating: float | None = None
    highest_rating: float | None = None
    best_wine_id: int | None = None


class RegionsResponse(BaseModel):
    """Top region preferences."""

    regions: list[RegionStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

class CountryStats(BaseModel):
    """Statistics for a single country."""

    country: str
    wines_tasted: int = 0
    avg_rating: float | None = None
    highest_rating: float | None = None
    best_wine_id: int | None = None


class CountriesResponse(BaseModel):
    """Top country preferences."""

    countries: list[CountryStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Vintages
# ---------------------------------------------------------------------------

class VintageStats(BaseModel):
    """Statistics for a single vintage year."""

    vintage: int
    wines_tasted: int = 0
    avg_rating: float | None = None
    highest_rating: float | None = None
    best_wine_id: int | None = None


class VintagesResponse(BaseModel):
    """Top vintage preferences."""

    vintages: list[VintageStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Appellations
# ---------------------------------------------------------------------------

class AppellationStats(BaseModel):
    """Statistics for a single appellation."""

    appellation: str
    country: str | None = None
    wines_tasted: int = 0
    avg_rating: float | None = None
    highest_rating: float | None = None
    best_wine_id: int | None = None


class AppellationsResponse(BaseModel):
    """Top appellation preferences."""

    appellations: list[AppellationStats] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rating trends
# ---------------------------------------------------------------------------

class RatingTrendPoint(BaseModel):
    """One month data point for rating trends."""

    month: str
    avg_rating: float
    wines_count: int


class RatingTrendsResponse(BaseModel):
    """Rating trends over time."""

    points: list[RatingTrendPoint] = Field(default_factory=list)
    trend: str | None = None


# ---------------------------------------------------------------------------
# Consumed wines
# ---------------------------------------------------------------------------

class ConsumedWineItem(BaseModel):
    """A single consumed wine with details and rating."""

    wine_id: int | None = None
    wine_name: str = ""
    producer_name: str | None = None
    vintage: int | None = None
    wine_type: str | None = None
    varietal: str | None = None
    country: str | None = None
    region_name: str | None = None
    consumed_date: str | None = None
    personal_rating: float | None = None
    community_rating: float | None = None
    rating_description: str | None = None
    tasting_notes: str | None = None


class ConsumedFilterOptions(BaseModel):
    """Available filter values for consumed wines dropdowns."""

    wine_types: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)
    min_vintage: int = 2000
    max_vintage: int = 2025


class ConsumedWinesResponse(BaseModel):
    """Filtered consumed wines with filter options."""

    items: list[ConsumedWineItem] = Field(default_factory=list)
    total: int = 0
    filter_options: ConsumedFilterOptions = Field(default_factory=ConsumedFilterOptions)

