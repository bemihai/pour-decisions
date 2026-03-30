"""Taste profile API endpoints.

Exposes taste profile analytics, chart data, and consumed wine history.
Business logic lives in the repository layer and direct SQL queries;
this module handles HTTP concerns and data transformation.
"""
from fastapi import APIRouter, Query

from src.api.schemas.taste_profile import (
    AppellationsResponse,
    AppellationStats,
    ConsumedFilterOptions,
    ConsumedWineItem,
    ConsumedWinesResponse,
    CountriesResponse,
    CountryStats,
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
)
from src.database import get_db_connection
from src.database.repository import StatsRepository
from src.etl.utils import get_rating_description

router = APIRouter(prefix="/api/taste-profile", tags=["taste-profile"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rating_color_gradient(index: int, total: int) -> str:
    """Compute a red-to-green gradient color for a rating bucket.

    Args:
        index: Zero-based position of this bucket (0 = lowest).
        total: Total number of buckets.

    Returns:
        CSS ``rgb(...)`` color string.
    """
    if total <= 1:
        return "rgb(76, 175, 80)"
    ratio = index / (total - 1)
    if ratio < 0.33:
        r = 244
        g = int(67 + (193 - 67) * (ratio / 0.33))
        b = int(54 + (7 - 54) * (ratio / 0.33))
    elif ratio < 0.67:
        r = int(255 - (255 - 139) * ((ratio - 0.33) / 0.34))
        g = int(193 + (195 - 193) * ((ratio - 0.33) / 0.34))
        b = int(7 + (74 - 7) * ((ratio - 0.33) / 0.34))
    else:
        r = int(139 - (139 - 76) * ((ratio - 0.67) / 0.33))
        g = int(195 + (175 - 195) * ((ratio - 0.67) / 0.33))
        b = int(74 + (80 - 74) * ((ratio - 0.67) / 0.33))
    return f"rgb({r}, {g}, {b})"


_CONSUMED_SORT_KEYS: dict[str, tuple] = {
    "consumed_date_desc": ("consumed_date", True, "0000-00-00"),
    "consumed_date_asc": ("consumed_date", False, "9999-99-99"),
    "rating_desc": ("personal_rating", True, 0),
    "rating_asc": ("personal_rating", False, 9999),
    "producer": ("producer_name", False, ""),
    "wine_name": ("wine_name", False, ""),
}


def _sort_consumed(items: list[dict], sort_by: str) -> list[dict]:
    """Sort consumed wine dicts by the requested key.

    Args:
        items: List of consumed wine dicts.
        sort_by: Key name from ``_CONSUMED_SORT_KEYS``.

    Returns:
        Sorted list (new list, does not mutate input).
    """
    field, reverse, default = _CONSUMED_SORT_KEYS.get(
        sort_by, _CONSUMED_SORT_KEYS["consumed_date_desc"]
    )
    if sort_by == "producer":
        return sorted(
            items,
            key=lambda w: (w.get("producer_name") or "", w.get("vintage") or 0),
            reverse=reverse,
        )
    return sorted(
        items,
        key=lambda w: w.get(field) or default,
        reverse=reverse,
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/overview", response_model=TasteOverviewResponse)
async def get_overview() -> TasteOverviewResponse:
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
async def get_rating_distribution() -> RatingDistributionResponse:
    """Return rating distribution data for a donut chart.

    Queries consumed wines with ratings and builds 5-point interval
    buckets with a red-to-green color gradient.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.personal_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND t.personal_rating IS NOT NULL
        """)
        ratings = [row["personal_rating"] for row in cursor.fetchall()]

    if not ratings:
        return RatingDistributionResponse()

    ranges: list[str] = []
    counts: list[int] = []

    # 0-49 bucket
    poor_count = sum(1 for r in ratings if r < 50)
    if poor_count > 0:
        ranges.append("0-49")
        counts.append(poor_count)

    # 50-94 in 5-point intervals
    for i in range(50, 95, 5):
        count = sum(1 for r in ratings if i <= r < i + 5)
        if count > 0:
            ranges.append(f"{i}-{i + 4}")
            counts.append(count)

    # 95-100 bucket
    excellent_count = sum(1 for r in ratings if r >= 95)
    if excellent_count > 0:
        ranges.append("95-100")
        counts.append(excellent_count)

    num_ranges = len(ranges)
    if num_ranges <= 3:
        palette = ["#F44336", "#FFC107", "#4CAF50"]
        colors = palette[:num_ranges]
    else:
        colors = [_rating_color_gradient(i, num_ranges) for i in range(num_ranges)]

    buckets = [
        RatingBucket(range=r, count=c, color=col)
        for r, c, col in zip(ranges, counts, colors)
    ]

    return RatingDistributionResponse(
        buckets=buckets,
        total=len(ratings),
    )


@router.get("/wine-types", response_model=WineTypesResponse)
async def get_wine_types() -> WineTypesResponse:
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
async def get_varietals(
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
async def get_producers(
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
        )
        for row in raw
    ]

    return ProducersResponse(producers=producers)


@router.get("/regions", response_model=RegionsResponse)
async def get_regions(
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
        )
        for row in raw
    ]

    return RegionsResponse(regions=regions)


@router.get("/countries", response_model=CountriesResponse)
async def get_countries(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of countries to return"),
) -> CountriesResponse:
    """Return top country preferences based on consumed wines.

    Queries directly against the database to get country-level aggregations.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                r.country,
                COUNT(DISTINCT b.id) as wines_tasted,
                AVG(t.personal_rating) as avg_rating,
                MAX(t.personal_rating) as highest_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND r.country IS NOT NULL
            GROUP BY r.country
            HAVING COUNT(DISTINCT b.id) >= 1
            ORDER BY wines_tasted DESC, avg_rating DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(row) for row in cursor.fetchall()]

    countries = [
        CountryStats(
            country=row.get("country", "Unknown"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
        )
        for row in rows
    ]

    return CountriesResponse(countries=countries)


@router.get("/vintages", response_model=VintagesResponse)
async def get_vintages(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of vintages to return"),
) -> VintagesResponse:
    """Return top vintage preferences based on consumed wines.

    Returns vintages with at least 2 wines tasted, ordered by avg rating.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                w.vintage,
                COUNT(DISTINCT b.id) as wines_tasted,
                AVG(t.personal_rating) as avg_rating,
                MAX(t.personal_rating) as highest_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND w.vintage IS NOT NULL
            GROUP BY w.vintage
            HAVING COUNT(DISTINCT b.id) >= 2
            ORDER BY avg_rating DESC, wines_tasted DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(row) for row in cursor.fetchall()]

    vintages = [
        VintageStats(
            vintage=row.get("vintage", 0),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
        )
        for row in rows
    ]

    return VintagesResponse(vintages=vintages)


@router.get("/appellations", response_model=AppellationsResponse)
async def get_appellations(
    limit: int = Query(5, ge=1, le=50, description="Maximum number of appellations to return"),
) -> AppellationsResponse:
    """Return top appellation preferences based on consumed wines."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                w.appellation,
                r.country,
                COUNT(DISTINCT b.id) as wines_tasted,
                AVG(t.personal_rating) as avg_rating,
                MAX(t.personal_rating) as highest_rating
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed' AND w.appellation IS NOT NULL
            GROUP BY w.appellation
            HAVING COUNT(DISTINCT b.id) >= 1
            ORDER BY wines_tasted DESC, avg_rating DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(row) for row in cursor.fetchall()]

    appellations = [
        AppellationStats(
            appellation=row.get("appellation", "Unknown"),
            country=row.get("country"),
            wines_tasted=row.get("wines_tasted", 0),
            avg_rating=row.get("avg_rating"),
            highest_rating=row.get("highest_rating"),
        )
        for row in rows
    ]

    return AppellationsResponse(appellations=appellations)


@router.get("/rating-trends", response_model=RatingTrendsResponse)
async def get_rating_trends() -> RatingTrendsResponse:
    """Return rating trends over time (monthly).

    Wraps ``StatsRepository.get_rating_timeline()`` and computes
    the overall trend direction.
    """
    stats_repo = StatsRepository()
    timeline = stats_repo.get_rating_timeline()

    if not timeline or len(timeline) < 2:
        return RatingTrendsResponse()

    # Keep last 12 months
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
async def get_consumed(
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

    Consolidates the filtering logic from ``show_consumed_wines_inventory()``.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                w.id as wine_id,
                b.id as bottle_id,
                w.wine_name, w.wine_type, w.vintage, w.varietal,
                p.name as producer_name,
                r.country,
                COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                t.personal_rating, t.community_rating, t.tasting_notes, t.last_tasted_date,
                b.consumed_date
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN producers p ON w.producer_id = p.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed'
            ORDER BY b.consumed_date DESC
        """)
        all_consumed = [dict(row) for row in cursor.fetchall()]

    if not all_consumed:
        return ConsumedWinesResponse()

    # Build filter options from full unfiltered set
    filter_wine_types = sorted({w.get("wine_type") for w in all_consumed if w.get("wine_type")})
    filter_countries = sorted({w.get("country") for w in all_consumed if w.get("country")})
    filter_producers = sorted({w.get("producer_name") for w in all_consumed if w.get("producer_name")})
    vintages = [w.get("vintage") for w in all_consumed if w.get("vintage")]
    fmin = min(vintages) if vintages else 2000
    fmax = max(vintages) if vintages else 2025

    filter_options = ConsumedFilterOptions(
        wine_types=filter_wine_types,
        countries=filter_countries,
        producers=filter_producers,
        min_vintage=fmin,
        max_vintage=fmax,
    )

    # Apply filters
    filtered = all_consumed

    if wine_type:
        filtered = [w for w in filtered if w.get("wine_type") == wine_type]
    if country:
        filtered = [w for w in filtered if w.get("country") == country]
    if producer:
        filtered = [w for w in filtered if w.get("producer_name") == producer]

    if min_vintage is not None or max_vintage is not None:
        lo = min_vintage or 0
        hi = max_vintage or 9999
        filtered = [
            w for w in filtered
            if w.get("vintage") is None or (lo <= w["vintage"] <= hi)
        ]

    if rating_filter:
        rf = rating_filter.lower()
        if rf == "rated":
            filtered = [w for w in filtered if w.get("personal_rating") is not None]
        elif rf == "unrated":
            filtered = [w for w in filtered if w.get("personal_rating") is None]
        elif rf.endswith("+"):
            threshold = int(rf.rstrip("+"))
            filtered = [w for w in filtered if (w.get("personal_rating") or 0) >= threshold]

    if search:
        s = search.lower()
        filtered = [
            w for w in filtered
            if s in (w.get("wine_name") or "").lower()
            or s in (w.get("producer_name") or "").lower()
            or s in (w.get("varietal") or "").lower()
        ]

    # Sort
    filtered = _sort_consumed(filtered, sort_by)

    # Apply limit
    total = len(filtered)
    filtered = filtered[:limit]

    items = [
        ConsumedWineItem(
            wine_id=w.get("wine_id"),
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
        for w in filtered
    ]

    return ConsumedWinesResponse(
        items=items,
        total=total,
        filter_options=filter_options,
    )


