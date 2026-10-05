"""Taste profile API endpoints.

Exposes taste profile analytics, chart data, and consumed wine history.
All data access goes through the repository layer; this module handles
HTTP concerns and schema mapping only.

Note: all route handlers are synchronous (``def``). FastAPI runs them in a
thread-pool executor so the event loop remains unblocked. Migrating to async
I/O would require async database drivers and is tracked as a future improvement.
"""
from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas.taste_profile import (
    AppellationsResponse,
    AppellationStats,
    ConsumedFilterOptions,
    ConsumedWineItem,
    ConsumedWinesResponse,
    CountriesResponse,
    CountryStats,
    PricePreferenceCreate,
    PreferenceCreateRequest,
    PreferenceDeleteResponse,
    PreferenceListResponse,
    PreferenceOptionsResponse,
    PreferencePatchRequest,
    PreferencePricePatch,
    PreferenceResetRequest,
    PreferenceResetResponse,
    PreferenceResponse,
    ProducersResponse,
    ProducerStats,
    RatingBucket,
    RatingDistributionResponse,
    RatingTrendPoint,
    RatingTrendsResponse,
    RegionsResponse,
    RegionStats,
    TasteOverviewResponse,
    VarietalsResponse,
    VarietalStats,
    VintagesResponse,
    VintageStats,
    WineTypesResponse,
    WineTypeStats,
    price_amount_to_minor_units,
)
from src.database.models import PreferenceCurrency, PreferenceStance, PreferenceSubjectKind, WineStyle
from src.database.repository import (
    DeclaredPreferenceRepository,
    MAX_DECLARED_PREFERENCES,
    PreferenceCombinationError,
    PreferenceCountChangedError,
    PreferenceDuplicateError,
    PreferenceLimitReachedError,
    PreferenceNotFoundError,
    PreferenceStoreBusyError,
    PreferenceValueUnresolvedError,
    PreferenceVersionConflictError,
    StatsRepository,
)
from src.etl.utils import get_rating_description
from src.utils import logger

router = APIRouter(prefix="/api/taste-profile", tags=["taste-profile"])
ResultType = TypeVar("ResultType")


def _preference_error(status_code: int, code: str, message: str, *, retry: bool = False) -> HTTPException:
    """Build one bounded preference API error."""
    headers = {"Retry-After": "1"} if retry else None
    return HTTPException(status_code=status_code, detail={"code": code, "message": message}, headers=headers)


def _execute_preference_operation(operation: Callable[[], ResultType]) -> ResultType:
    """Execute repository work and translate failures to the public contract."""
    try:
        return operation()
    except PreferenceNotFoundError as error:
        raise _preference_error(404, "preference_not_found", "Preference was not found.") from error
    except PreferenceDuplicateError as error:
        raise _preference_error(409, "preference_duplicate", "Preference already exists.") from error
    except PreferenceVersionConflictError as error:
        raise _preference_error(409, "preference_version_conflict", "Preference version changed.") from error
    except PreferenceCountChangedError as error:
        raise _preference_error(409, "preference_count_changed", "Preference count changed.") from error
    except PreferenceLimitReachedError as error:
        raise _preference_error(409, "preference_limit_reached", "Preference limit reached.") from error
    except PreferenceValueUnresolvedError as error:
        raise _preference_error(
            422,
            "preference_value_unresolved",
            "Preference value could not be resolved.",
        ) from error
    except PreferenceCombinationError as error:
        raise _preference_error(422, "preference_combination_invalid", "Preference mutation is invalid.") from error
    except PreferenceStoreBusyError as error:
        raise _preference_error(
            503,
            "preference_store_busy",
            "Preference store is busy. Try again shortly.",
            retry=True,
        ) from error
    except Exception as error:
        logger.exception("Unexpected declared-preference repository failure")
        raise _preference_error(500, "preference_store_unavailable", "Preference store is unavailable.") from error


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/preferences", response_model=PreferenceListResponse)
def get_preferences() -> PreferenceListResponse:
    """Return the complete declared-preference profile."""
    preferences = _execute_preference_operation(DeclaredPreferenceRepository().list_all)
    return PreferenceListResponse(
        items=[PreferenceResponse.from_preference(preference) for preference in preferences],
        total=len(preferences),
        max_items=MAX_DECLARED_PREFERENCES,
    )


@router.get("/preferences/options", response_model=PreferenceOptionsResponse)
def get_preference_options() -> PreferenceOptionsResponse:
    """Return canonical values accepted by preference creation."""
    options = _execute_preference_operation(DeclaredPreferenceRepository().list_options)
    return PreferenceOptionsResponse(
        subject_kinds=list(PreferenceSubjectKind),
        stances=list(PreferenceStance),
        grapes=options["grapes"],
        regions=options["regions"],
        producers=options["producers"],
        wine_styles=list(WineStyle),
        currencies=list(PreferenceCurrency),
        max_items=MAX_DECLARED_PREFERENCES,
    )


@router.post(
    "/preferences",
    response_model=PreferenceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_preference(request: PreferenceCreateRequest) -> PreferenceResponse:
    """Create one validated declared preference."""
    repository = DeclaredPreferenceRepository()
    if isinstance(request, PricePreferenceCreate):
        preference = _execute_preference_operation(
            lambda: repository.create(
                request.subject_kind,
                price_minor_units=price_amount_to_minor_units(request.price_amount),
                currency=request.currency,
            )
        )
    else:
        preference = _execute_preference_operation(
            lambda: repository.create(
                request.subject_kind,
                stance=request.stance,
                value=request.value,
            )
        )
    return PreferenceResponse.from_preference(preference)


@router.patch("/preferences/{preference_id}", response_model=PreferenceResponse)
def update_preference(preference_id: int, request: PreferencePatchRequest) -> PreferenceResponse:
    """Update mutable fields using an expected persisted version."""
    repository = DeclaredPreferenceRepository()
    if isinstance(request, PreferencePricePatch):
        preference = _execute_preference_operation(
            lambda: repository.update_mutable(
                preference_id,
                request.expected_version,
                price_minor_units=price_amount_to_minor_units(request.price_amount),
                currency=request.currency,
            )
        )
    else:
        preference = _execute_preference_operation(
            lambda: repository.update_mutable(
                preference_id,
                request.expected_version,
                stance=request.stance,
            )
        )
    return PreferenceResponse.from_preference(preference)


@router.delete("/preferences/{preference_id}", response_model=PreferenceDeleteResponse)
def delete_preference(
    preference_id: int,
    expected_version: int = Query(..., gt=0),
) -> PreferenceDeleteResponse:
    """Delete one preference using an expected persisted version."""
    deleted_id, deleted_version = _execute_preference_operation(
        lambda: DeclaredPreferenceRepository().delete(preference_id, expected_version)
    )
    return PreferenceDeleteResponse(id=deleted_id, deleted_version=deleted_version)


@router.post("/preferences/reset", response_model=PreferenceResetResponse)
def reset_preferences(request: PreferenceResetRequest) -> PreferenceResetResponse:
    """Delete the complete profile after explicit count confirmation."""
    deleted_count = _execute_preference_operation(
        lambda: DeclaredPreferenceRepository().reset(expected_count=request.expected_count)
    )
    return PreferenceResetResponse(deleted_count=deleted_count)


@router.get("/overview", response_model=TasteOverviewResponse)
def get_overview() -> TasteOverviewResponse:
    """Return key insight metrics for the taste profile page.

    Wraps ``StatsRepository.get_rating_statistics()`` and
    ``get_wine_type_stats()`` to compute overview numbers.
    """
    stats_repo = StatsRepository()

    rating_stats = stats_repo.get_rating_statistics()
    overall = rating_stats.get("overall", {})
    distribution = rating_stats.get("distribution", [])
    wine_type_stats = stats_repo.get_wine_type_stats()

    avg_rating = overall.get("avg_rating", 0)
    wines_rated = overall.get("wines_rated", 0)

    favorite_type = "N/A"
    if wine_type_stats:
        favorite_type = wine_type_stats[0].get("wine_type", "N/A")

    highly_rated_count = 0
    for bucket in distribution:
        if bucket.get("rating_range") == "90-100":
            highly_rated_count = bucket.get("count", 0)
            break

    highly_rated_pct = (highly_rated_count / wines_rated * 100) if wines_rated > 0 else 0.0

    return TasteOverviewResponse(
        avg_rating=avg_rating,
        wines_rated=wines_rated,
        favorite_type=favorite_type,
        highly_rated_count=highly_rated_count,
        highly_rated_pct=round(highly_rated_pct, 1),
    )


@router.get("/rating-distribution", response_model=RatingDistributionResponse)
def get_rating_distribution() -> RatingDistributionResponse:
    """Return rating distribution data for a donut chart.

    Delegates bucketing logic to ``StatsRepository.get_rating_distribution()``.
    """
    stats_repo = StatsRepository()
    data = stats_repo.get_rating_distribution()
    buckets = [RatingBucket(range=b["range"], count=b["count"]) for b in data.get("buckets", [])]
    return RatingDistributionResponse(buckets=buckets, total=data.get("total", 0))


@router.get("/wine-types", response_model=WineTypesResponse)
def get_wine_types() -> WineTypesResponse:
    """Return wine type distribution and performance data.

    Wraps ``StatsRepository.get_wine_type_stats()``.
    """
    stats_repo = StatsRepository()
    raw = stats_repo.get_wine_type_stats()

    types = [
        WineTypeStats(
            wine_type=row.get("wine_type", "Unknown"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
            most_recent_date=row.get("most_recent_date"),
        )
        for row in raw
    ]

    return WineTypesResponse(types=types)


@router.get("/varietals", response_model=VarietalsResponse)
def get_varietals(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of varietals to return"),
) -> VarietalsResponse:
    """Return top varietal preferences.

    Wraps ``StatsRepository.get_varietal_preferences()``.
    """
    stats_repo = StatsRepository()
    raw = stats_repo.get_varietal_preferences(limit=limit)

    varietals = [
        VarietalStats(
            varietal=row.get("varietal", "Unknown"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
        )
        for row in raw
    ]

    return VarietalsResponse(varietals=varietals)


@router.get("/producers", response_model=ProducersResponse)
def get_producers(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of producers to return"),
) -> ProducersResponse:
    """Return top producer preferences.

    Wraps ``StatsRepository.get_producer_preferences()``.
    """
    stats_repo = StatsRepository()
    raw = stats_repo.get_producer_preferences(limit=limit)

    producers = [
        ProducerStats(
            producer_name=row.get("producer_name", "Unknown"),
            country=row.get("country"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
            best_wine_id=row.get("best_wine_id"),
        )
        for row in raw
    ]

    return ProducersResponse(producers=producers)


@router.get("/regions", response_model=RegionsResponse)
def get_regions(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of regions to return"),
) -> RegionsResponse:
    """Return top region preferences.

    Wraps ``StatsRepository.get_region_preferences()``.
    """
    stats_repo = StatsRepository()
    raw = stats_repo.get_region_preferences(limit=limit)

    regions = [
        RegionStats(
            region_name=row.get("region_name", "Unknown"),
            country=row.get("country"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
            best_wine_id=row.get("best_wine_id"),
        )
        for row in raw
    ]

    return RegionsResponse(regions=regions)


@router.get("/countries", response_model=CountriesResponse)
def get_countries(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of countries to return"),
) -> CountriesResponse:
    """Return top country preferences based on consumed wines.

    Wraps ``StatsRepository.get_country_stats()``.
    """
    stats_repo = StatsRepository()
    rows = stats_repo.get_country_stats(limit=limit)

    countries = [
        CountryStats(
            country=row.get("country", "Unknown"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
            best_wine_id=row.get("best_wine_id"),
        )
        for row in rows
    ]

    return CountriesResponse(countries=countries)


@router.get("/vintages", response_model=VintagesResponse)
def get_vintages(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of vintages to return"),
) -> VintagesResponse:
    """Return top vintage preferences based on consumed wines.

    Wraps ``StatsRepository.get_vintage_stats()``.
    """
    stats_repo = StatsRepository()
    rows = stats_repo.get_vintage_stats(limit=limit)

    vintages = [
        VintageStats(
            vintage=row.get("vintage", 0),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
            best_wine_id=row.get("best_wine_id"),
        )
        for row in rows
    ]

    return VintagesResponse(vintages=vintages)


@router.get("/appellations", response_model=AppellationsResponse)
def get_appellations(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of appellations to return"),
) -> AppellationsResponse:
    """Return top appellation preferences based on consumed wines.

    Wraps ``StatsRepository.get_appellation_stats()``.
    """
    stats_repo = StatsRepository()
    rows = stats_repo.get_appellation_stats(limit=limit)

    appellations = [
        AppellationStats(
            appellation=row.get("appellation", "Unknown"),
            country=row.get("country"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
            best_wine_id=row.get("best_wine_id"),
        )
        for row in rows
    ]

    return AppellationsResponse(appellations=appellations)


@router.get("/rating-trends", response_model=RatingTrendsResponse)
def get_rating_trends() -> RatingTrendsResponse:
    """Return rating trends over time (monthly).

    Wraps ``StatsRepository.get_rating_timeline()`` and computes
    the overall trend direction.
    """
    stats_repo = StatsRepository()
    timeline = stats_repo.get_rating_timeline()

    if not timeline or len(timeline) < 2:
        return RatingTrendsResponse()

    recent = timeline[-12:] if len(timeline) > 12 else timeline

    points = [
        RatingTrendPoint(
            month=t["month"],
            avg_rating=t["avg_rating"],
            wines_count=t["wines_count"],
        )
        for t in recent
    ]

    ratings = [p.avg_rating for p in points]
    if ratings[-1] > ratings[0]:
        trend = "improving"
    elif ratings[-1] < ratings[0]:
        trend = "declining"
    else:
        trend = "stable"

    return RatingTrendsResponse(points=points, trend=trend)


@router.get("/consumed", response_model=ConsumedWinesResponse)
def get_consumed(
    wine_type: str | None = Query(None, description="Filter by wine type"),
    country: str | None = Query(None, description="Filter by country"),
    producer: str | None = Query(None, description="Filter by producer name"),
    min_vintage: int | None = Query(None, description="Minimum vintage year (inclusive)"),
    max_vintage: int | None = Query(None, description="Maximum vintage year (inclusive)"),
    rating_filter: str | None = Query(
        None,
        description="Rating filter: rated, unrated, 90+, 80+, 70+",
    ),
    search: str | None = Query(None, description="Free-text search (wine name, producer, varietal)"),
    sort_by: str = Query("consumed_date_desc", description="Sort key"),
    limit: int = Query(20, ge=5, le=100, description="Max results to return"),
) -> ConsumedWinesResponse:
    """Return consumed wines with filtering, sorting, and filter options.

    Filtering and sorting are applied at the SQL level by the repository.
    Filter options are derived from lightweight ``SELECT DISTINCT`` queries.
    """
    stats_repo = StatsRepository()

    raw_opts = stats_repo.get_consumed_filter_options()
    filter_options = ConsumedFilterOptions(
        wine_types=raw_opts["wine_types"],
        countries=raw_opts["countries"],
        producers=raw_opts["producers"],
        min_vintage=raw_opts["min_vintage"],
        max_vintage=raw_opts["max_vintage"],
    )

    result = stats_repo.get_consumed_wines(
        wine_type=wine_type,
        country=country,
        producer=producer,
        min_vintage=min_vintage,
        max_vintage=max_vintage,
        rating_filter=rating_filter,
        search=search,
        sort_by=sort_by,
        limit=limit,
    )

    items = [
        ConsumedWineItem(
            wine_id=w.get("wine_id"),
            bottle_id=w.get("bottle_id"),
            wine_name=w.get("wine_name", ""),
            producer_name=w.get("producer_name"),
            vintage=w.get("vintage"),
            wine_type=w.get("wine_type"),
            varietal=w.get("varietal"),
            country=w.get("country"),
            region_name=w.get("region_name"),
            consumed_date=w.get("consumed_date"),
            personal_rating=w.get("personal_rating"),
            community_rating=w.get("community_rating"),
            rating_description=get_rating_description(w.get("personal_rating")),
            tasting_notes=w.get("tasting_notes"),
        )
        for w in result["items"]
    ]

    return ConsumedWinesResponse(
        items=items,
        total=result["total"],
        filter_options=filter_options,
    )
