"""Pydantic schemas for the taste profile API."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from src.database.models import (
    DeclaredPreference,
    PreferenceCurrency,
    PreferenceStance,
    PreferenceSubjectKind,
    WineStyle,
)


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
    bottle_id: int | None = None
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


# ---------------------------------------------------------------------------
# Declared preferences
# ---------------------------------------------------------------------------

_PRICE_AMOUNT_PATTERN = re.compile(r"^(?:0|[1-9][0-9]{0,5})(?:\.[0-9]{1,2})?$")


def validate_price_amount(value: str) -> str:
    """Validate the exact decimal-string wire representation for a price."""
    if not _PRICE_AMOUNT_PATTERN.fullmatch(value):
        raise ValueError("Price amount must be a decimal string with at most two decimal places.")
    amount = Decimal(value)
    if amount < Decimal("0.01") or amount > Decimal("999999.99"):
        raise ValueError("Price amount is outside the supported range.")
    return value


def price_amount_to_minor_units(value: str) -> int:
    """Convert one validated price amount to exact integer minor units."""
    validate_price_amount(value)
    return int(Decimal(value) * 100)


class _PreferenceRequest(BaseModel):
    """Strict base for preference mutation requests."""

    model_config = ConfigDict(extra="forbid")


class GrapePreferenceCreate(_PreferenceRequest):
    """Create one declared grape preference."""

    subject_kind: Literal["grape"]
    stance: PreferenceStance
    value: str = Field(min_length=1, max_length=120)


class RegionPreferenceCreate(_PreferenceRequest):
    """Create one declared region preference."""

    subject_kind: Literal["region"]
    stance: PreferenceStance
    value: str = Field(min_length=1, max_length=120)


class ProducerPreferenceCreate(_PreferenceRequest):
    """Create one declared producer preference."""

    subject_kind: Literal["producer"]
    stance: PreferenceStance
    value: str = Field(min_length=1, max_length=120)


class WineStylePreferenceCreate(_PreferenceRequest):
    """Create one declared wine-style preference."""

    subject_kind: Literal["wine_style"]
    stance: PreferenceStance
    value: WineStyle


class PricePreferenceCreate(_PreferenceRequest):
    """Create the single declared price-ceiling preference."""

    subject_kind: Literal["price_ceiling"]
    price_amount: StrictStr
    currency: PreferenceCurrency

    _validate_price_amount = field_validator("price_amount")(validate_price_amount)


PreferenceCreateRequest = Annotated[
    GrapePreferenceCreate
    | RegionPreferenceCreate
    | ProducerPreferenceCreate
    | WineStylePreferenceCreate
    | PricePreferenceCreate,
    Field(discriminator="subject_kind"),
]


class PreferenceStancePatch(_PreferenceRequest):
    """Update the stance of one non-price preference."""

    expected_version: int = Field(gt=0)
    stance: PreferenceStance


class PreferencePricePatch(_PreferenceRequest):
    """Update the amount and currency of the price ceiling."""

    expected_version: int = Field(gt=0)
    price_amount: StrictStr
    currency: PreferenceCurrency

    _validate_price_amount = field_validator("price_amount")(validate_price_amount)


PreferencePatchRequest = PreferenceStancePatch | PreferencePricePatch


class _PreferenceResponse(BaseModel):
    """Strict base for declared-preference responses."""

    model_config = ConfigDict(extra="forbid")


class PreferenceResponse(_PreferenceResponse):
    """Persisted declared-preference wire representation."""

    id: int = Field(gt=0)
    subject_kind: PreferenceSubjectKind
    stance: PreferenceStance | None = None
    normalized_value: str
    display_value: str | None = None
    price_amount: str | None = None
    currency: PreferenceCurrency | None = None
    provenance: Literal["explicit_user"]
    version: int = Field(gt=0)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_preference(cls, preference: DeclaredPreference) -> "PreferenceResponse":
        """Build a response without exposing integer storage representation."""
        price_amount = None
        if preference.price_minor_units is not None:
            price_amount = f"{Decimal(preference.price_minor_units) / Decimal(100):.2f}"
        return cls(
            id=preference.id,
            subject_kind=preference.subject_kind,
            stance=preference.stance,
            normalized_value=preference.normalized_value,
            display_value=preference.display_value,
            price_amount=price_amount,
            currency=preference.currency,
            provenance=preference.provenance,
            version=preference.version,
            created_at=preference.created_at,
            updated_at=preference.updated_at,
        )


class PreferenceListResponse(_PreferenceResponse):
    """All current declared preferences for the single local profile."""

    items: list[PreferenceResponse] = Field(default_factory=list)
    total: int = 0
    max_items: int = 100


class PreferenceOptionsResponse(_PreferenceResponse):
    """Canonical server-supported values for preference forms."""

    subject_kinds: list[PreferenceSubjectKind]
    stances: list[PreferenceStance]
    grapes: list[str]
    regions: list[str]
    producers: list[str]
    wine_styles: list[WineStyle]
    currencies: list[PreferenceCurrency]
    max_items: int = 100


class PreferenceDeleteResponse(_PreferenceResponse):
    """Confirmed deletion result."""

    id: int = Field(gt=0)
    deleted_version: int = Field(gt=0)


class PreferenceResetRequest(_PreferenceRequest):
    """Explicit destructive reset confirmation."""

    confirm: Literal[True]
    expected_count: int = Field(ge=0, le=100)


class PreferenceResetResponse(_PreferenceResponse):
    """Confirmed reset result."""

    deleted_count: int = Field(ge=0, le=100)
