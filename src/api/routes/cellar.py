"""Cellar API endpoints.

Exposes cellar inventory, statistics, chart data, filter options,
and CellarTracker sync. Business logic lives in the repository
layer; this module handles HTTP concerns, filtering, and sorting.
"""
import os

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.cellar import (
    CellarOverview,
    CellarStatsResponse,
    CellarValueStats,
    ChartDataResponse,
    DrinkingWindowStats,
    FilterOptions,
    InventoryItem,
    InventoryResponse,
    SyncResponse,
)
from src.database.repository import BottleRepository, StatsRepository
from src.utils import logger

router = APIRouter(prefix="/api/cellar", tags=["cellar"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_index(w: dict) -> float | None:
    """Return the best drinking index for an inventory row on a 0-100 scale.

    Prefers the CellarTracker-sourced ``drink_index``. Falls back to a
    locally computed bell-curve value when the drinking window is known.

    Args:
        w: Inventory row dict from ``BottleRepository.get_inventory()``.

    Returns:
        Float in ``[0, 100]`` or ``None`` when no data is available.
    """
    idx = w.get("drink_index")
    src = w.get("drink_window_source")
    if idx is not None and src == "cellar_tracker":
        return max(0.0, min(100.0, float(idx) * 100))
    df, dt = w.get("drink_from_year"), w.get("drink_to_year")
    if df and dt:
        from src.agents.drinking_window_service import compute_drink_index

        return compute_drink_index(int(df), int(dt))
    return None


def _build_grouped_inventory(raw_inventory: list[dict]) -> list[dict]:
    """Group raw bottle rows by ``wine_id`` and aggregate quantities.

    Args:
        raw_inventory: Flat list from ``BottleRepository.get_inventory()``.

    Returns:
        Deduplicated list with one entry per wine, summed quantities,
        and the most recent ``created_at`` timestamp.
    """
    wine_groups: dict[int, dict] = {}
    for bottle in raw_inventory:
        wine_id = bottle.get("wine_id")
        if wine_id not in wine_groups:
            wine_groups[wine_id] = bottle.copy()
        else:
            wine_groups[wine_id]["quantity"] = (
                wine_groups[wine_id].get("quantity", 0) + bottle.get("quantity", 0)
            )
            existing_ca = wine_groups[wine_id].get("created_at")
            incoming_ca = bottle.get("created_at")
            if incoming_ca and (not existing_ca or str(incoming_ca) > str(existing_ca)):
                wine_groups[wine_id]["created_at"] = incoming_ca
    return list(wine_groups.values())


def _extract_filter_options(inventory: list[dict]) -> FilterOptions:
    """Derive available filter values from the current inventory.

    Args:
        inventory: Grouped inventory list (one row per wine).

    Returns:
        ``FilterOptions`` with sorted unique values for each dropdown.
    """
    wine_types = sorted({w.get("wine_type") for w in inventory if w.get("wine_type")})
    countries = sorted({w.get("country") for w in inventory if w.get("country")})
    locations = sorted({w.get("location") for w in inventory if w.get("location")})
    producers = sorted({w.get("producer_name") for w in inventory if w.get("producer_name")})

    vintages = [w.get("vintage") for w in inventory if w.get("vintage")]
    min_vintage = min(vintages) if vintages else 2000
    max_vintage = max(vintages) if vintages else 2025

    return FilterOptions(
        wine_types=wine_types,
        countries=countries,
        locations=locations,
        producers=producers,
        min_vintage=min_vintage,
        max_vintage=max_vintage,
    )


def _apply_filters(
    inventory: list[dict],
    *,
    wine_type: str | None,
    country: str | None,
    producer: str | None,
    location: str | None,
    min_vintage: int | None,
    max_vintage: int | None,
    rating_filter: str | None,
    search: str | None,
) -> list[dict]:
    """Apply all optional filters to the inventory list.

    Mirrors the filtering logic in ``cellar_stats.py show_cellar_inventory()``.

    Args:
        inventory: Grouped inventory list.
        wine_type: Exact wine type string, or None for all.
        country: Exact country string, or None for all.
        producer: Exact producer name, or None for all.
        location: Exact storage location, or None for all.
        min_vintage: Minimum vintage year (inclusive).
        max_vintage: Maximum vintage year (inclusive).
        rating_filter: One of ``rated``, ``unrated``, ``90+``, ``80+``, ``70+``, or None.
        search: Free-text search across wine name and producer name.

    Returns:
        Filtered inventory list.
    """
    result = inventory

    if wine_type:
        result = [w for w in result if w.get("wine_type") == wine_type]
    if country:
        result = [w for w in result if w.get("country") == country]
    if producer:
        result = [w for w in result if w.get("producer_name") == producer]
    if location:
        result = [w for w in result if w.get("location") == location]

    if min_vintage is not None or max_vintage is not None:
        lo = min_vintage or 0
        hi = max_vintage or 9999
        result = [
            w for w in result
            if w.get("vintage") is None or (lo <= w["vintage"] <= hi)
        ]

    if rating_filter:
        rf = rating_filter.lower()
        if rf == "rated":
            result = [w for w in result if w.get("personal_rating") is not None]
        elif rf == "unrated":
            result = [w for w in result if w.get("personal_rating") is None]
        elif rf.endswith("+"):
            threshold = int(rf.rstrip("+"))
            result = [w for w in result if (w.get("personal_rating") or 0) >= threshold]

    if search:
        s = search.lower()
        result = [
            w for w in result
            if s in (w.get("wine_name") or "").lower()
            or s in (w.get("producer_name") or "").lower()
            or s in (w.get("varietal") or "").lower()
        ]

    return result


# Sort key names that the frontend can pass
_SORT_KEYS = {
    "created_at_desc": lambda w: str(w.get("created_at") or ""),
    "producer": lambda w: (w.get("producer_name") or "", w.get("vintage") or 0),
    "wine_name": lambda w: w.get("wine_name") or "",
    "vintage_desc": lambda w: w.get("vintage") or 0,
    "vintage_asc": lambda w: w.get("vintage") or 9999,
    "rating_desc": lambda w: w.get("personal_rating") or 0,
    "rating_asc": lambda w: w.get("personal_rating") or 9999,
    "drink_desc": lambda w: _effective_index(w) or 0,
    "drink_asc": lambda w: _effective_index(w) if _effective_index(w) is not None else -9999,
}


def _apply_sort(inventory: list[dict], sort_by: str) -> list[dict]:
    """Sort inventory by the requested key.

    Args:
        inventory: Filtered inventory list.
        sort_by: Sort key name from ``_SORT_KEYS``.

    Returns:
        Sorted list (new list, does not mutate input).
    """
    reverse = sort_by.endswith("_desc")
    key_fn = _SORT_KEYS.get(sort_by, _SORT_KEYS["created_at_desc"])
    return sorted(inventory, key=key_fn, reverse=reverse)


def _row_to_item(w: dict) -> InventoryItem:
    """Convert a raw inventory dict to an ``InventoryItem`` schema.

    Args:
        w: Inventory row dict with all joined fields.

    Returns:
        Typed ``InventoryItem`` instance.
    """
    return InventoryItem(
        wine_id=w.get("wine_id", 0),
        wine_name=w.get("wine_name", ""),
        producer_name=w.get("producer_name"),
        vintage=w.get("vintage"),
        wine_type=w.get("wine_type"),
        varietal=w.get("varietal"),
        country=w.get("country"),
        region_name=w.get("region_name"),
        quantity=w.get("quantity", 0),
        personal_rating=w.get("personal_rating"),
        community_rating=w.get("community_rating"),
        drink_index=_effective_index(w),
        drink_from_year=w.get("drink_from_year"),
        drink_to_year=w.get("drink_to_year"),
        drink_window_source=w.get("drink_window_source"),
        location=w.get("location"),
        bin=w.get("bin"),
        purchase_price=w.get("purchase_price"),
        currency=w.get("currency"),
        description=w.get("description"),
        producer_description=w.get("producer_description"),
        producer_id=w.get("producer_id"),
        bottle_note=w.get("bottle_note"),
        last_tasted_date=str(w["last_tasted_date"]) if w.get("last_tasted_date") else None,
        like_votes=w.get("like_votes"),
        like_percentage=w.get("like_percentage"),
        q_purchased=w.get("q_purchased", 0),
        q_consumed=w.get("q_consumed", 0),
        q_quantity=w.get("q_quantity", 0),
        created_at=str(w["created_at"]) if w.get("created_at") else None,
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.get("/inventory", response_model=InventoryResponse)
async def get_inventory(
    wine_type: str | None = Query(None, description="Filter by wine type (e.g. Red, White)"),
    country: str | None = Query(None, description="Filter by country"),
    producer: str | None = Query(None, description="Filter by producer name"),
    location: str | None = Query(None, description="Filter by storage location"),
    min_vintage: int | None = Query(None, description="Minimum vintage year (inclusive)"),
    max_vintage: int | None = Query(None, description="Maximum vintage year (inclusive)"),
    rating_filter: str | None = Query(None, description="Rating filter: rated, unrated, 90+, 80+, 70+"),
    search: str | None = Query(None, description="Free-text search (wine name, producer, varietal)"),
    sort_by: str = Query("created_at_desc", description="Sort key (see _SORT_KEYS)"),
) -> InventoryResponse:
    """Return filtered, sorted cellar inventory with filter options.

    Consolidates the filtering, grouping, and sorting logic from
    ``cellar_stats.py show_cellar_inventory()`` (~160 lines).
    """
    bottle_repo = BottleRepository()
    raw = bottle_repo.get_inventory()
    grouped = _build_grouped_inventory(raw)

    # Build filter options from the full (unfiltered) inventory
    filter_options = _extract_filter_options(grouped)

    # Apply filters
    filtered = _apply_filters(
        grouped,
        wine_type=wine_type,
        country=country,
        producer=producer,
        location=location,
        min_vintage=min_vintage,
        max_vintage=max_vintage,
        rating_filter=rating_filter,
        search=search,
    )

    # Sort
    sorted_inv = _apply_sort(filtered, sort_by)

    # Build response
    items = [_row_to_item(w) for w in sorted_inv]
    total_bottles = sum(w.get("quantity", 0) for w in sorted_inv)

    return InventoryResponse(
        items=items,
        total_wines=len(items),
        total_bottles=total_bottles,
        filter_options=filter_options,
    )


@router.get("/filters", response_model=FilterOptions)
async def get_filters() -> FilterOptions:
    """Return available filter values for the inventory UI.

    Useful for populating dropdowns before the first inventory fetch.
    """
    bottle_repo = BottleRepository()
    raw = bottle_repo.get_inventory()
    grouped = _build_grouped_inventory(raw)
    return _extract_filter_options(grouped)


@router.get("/stats", response_model=CellarStatsResponse)
async def get_stats() -> CellarStatsResponse:
    """Return combined cellar overview metrics.

    Wraps ``StatsRepository.get_cellar_overview()``,
    ``get_drinking_window_stats()``, and ``get_cellar_value()``.
    """
    stats_repo = StatsRepository()

    overview_raw = stats_repo.get_cellar_overview()
    drinking_raw = stats_repo.get_drinking_window_stats()
    value_raw = stats_repo.get_cellar_value()

    return CellarStatsResponse(
        overview=CellarOverview(**overview_raw),
        drinking_stats=DrinkingWindowStats(**drinking_raw),
        value_stats=CellarValueStats(**value_raw),
    )


@router.get("/charts", response_model=ChartDataResponse)
async def get_charts() -> ChartDataResponse:
    """Return pre-computed data for all cellar statistics charts.

    Wraps multiple ``StatsRepository`` methods and returns JSON-ready
    data that the frontend passes directly to Plotly.
    """
    stats_repo = StatsRepository()

    overview = stats_repo.get_cellar_overview()
    varietal_dist = stats_repo.get_varietal_distribution(limit=10)
    region_dist = stats_repo.get_region_distribution(limit=10)
    drinking_wines = stats_repo.get_drinking_window_wines()
    cellar_timeline = stats_repo.get_cellar_size_over_time()
    top_rated = stats_repo.get_top_rated_wines(limit=10)

    return ChartDataResponse(
        wine_type_distribution=overview.get("by_type", []),
        country_distribution=overview.get("by_country", []),
        varietal_distribution=varietal_dist,
        region_distribution=region_dist,
        drinking_window_wines=drinking_wines,
        cellar_size_over_time=cellar_timeline,
        top_rated=top_rated,
    )


@router.post("/sync", response_model=SyncResponse)
async def sync_cellar_tracker() -> SyncResponse:
    """Trigger a CellarTracker data sync.

    Reads credentials from server-side environment variables
    (``CELLAR_TRACKER_USERNAME``, ``CELLAR_TRACKER_PASSWORD``).
    """
    from dotenv import load_dotenv

    load_dotenv()

    username = os.getenv("CELLAR_TRACKER_USERNAME")
    password = os.getenv("CELLAR_TRACKER_PASSWORD")

    if not username or not password:
        raise HTTPException(
            status_code=400,
            detail=(
                "CellarTracker credentials not configured. "
                "Set CELLAR_TRACKER_USERNAME and CELLAR_TRACKER_PASSWORD in .env."
            ),
        )

    try:
        from src.etl.cellartracker_importer import CellarTrackerImporter
        from src.utils import get_default_db_path

        db_path = get_default_db_path()
        importer = CellarTrackerImporter(username, password, str(db_path))
        stats = importer.import_all()

        return SyncResponse(
            success=True,
            wines_processed=stats.get("wines_processed", 0),
            wines_imported=stats.get("wines_imported", 0),
            bottles_processed=stats.get("bottles_processed", 0),
            bottles_imported=stats.get("bottles_imported", 0),
            producers_created=stats.get("producers_created", 0),
            regions_created=stats.get("regions_created", 0),
            errors=stats.get("errors", []),
        )

    except Exception as e:
        logger.error(f"CellarTracker sync failed: {e}", exc_info=True)
        return SyncResponse(
            success=False,
            error_message=str(e),
        )


