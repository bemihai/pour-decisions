"""Cellar API endpoints.

Exposes cellar inventory, statistics, chart data, filter options,
and CellarTracker sync. Business logic lives in the repository
layer; this module handles HTTP concerns, grouping, and sorting.

Note: all route handlers are synchronous (``def``). FastAPI runs them in a
thread-pool executor so the event loop remains unblocked. Migrating to async
I/O would require async database drivers and is tracked as a future improvement.
The CellarTracker sync endpoint may take 10-60+ seconds; consider migrating it
to a background-task / polling pattern for multi-user deployments.
"""
import os

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.cellar import (
    CellarOverview,
    CellarStatsResponse,
    CellarValueStats,
    ChartDataResponse,
    DrinkingWindowStats,
    DrinkNextItem,
    DrinkNextResponse,
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
            wine_groups[wine_id] = bottle.copy()  # ty:ignore[invalid-assignment]
        else:
            wine_groups[wine_id]["quantity"] = (
                wine_groups[wine_id].get("quantity", 0) + bottle.get("quantity", 0)
            )
            existing_ca = wine_groups[wine_id].get("created_at")
            incoming_ca = bottle.get("created_at")
            if incoming_ca and (not existing_ca or str(incoming_ca) > str(existing_ca)):
                wine_groups[wine_id]["created_at"] = incoming_ca
    return list(wine_groups.values())


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
    "drink_asc": lambda w: (lambda idx: idx if idx is not None else -9999)(_effective_index(w)),
}


def _apply_sort(inventory: list[dict], sort_by: str) -> list[dict]:
    """Sort inventory by the requested key.

    Args:
        inventory: Grouped inventory list.
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
        appellation=w.get("appellation"),
        vineyard=w.get("vineyard"),
        country=w.get("country"),
        region_name=w.get("region_name"),
        quantity=w.get("quantity", 0),
        personal_rating=w.get("personal_rating"),
        community_rating=w.get("community_rating"),
        do_like=bool(w["do_like"]) if w.get("do_like") is not None else None,
        drink_index=_effective_index(w),
        drink_from_year=w.get("drink_from_year"),
        drink_to_year=w.get("drink_to_year"),
        drink_window_source=w.get("drink_window_source"),
        location=w.get("location"),
        bin=w.get("bin"),
        purchase_date=str(w["purchase_date"]) if w.get("purchase_date") else None,
        purchase_price=w.get("purchase_price"),
        valuation_price=w.get("valuation_price"),
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
def get_inventory(
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

    Filtering is applied at the SQL level in the repository. Grouping by
    wine_id and sorting by computed fields (e.g. drink_index) are done in
    Python after the SQL fetch.
    """
    bottle_repo = BottleRepository()

    # Lightweight distinct queries for filter dropdowns (always over full inventory)
    raw_filter_opts = bottle_repo.get_inventory_filter_options()
    filter_options = FilterOptions(**raw_filter_opts)

    # SQL-filtered fetch
    raw = bottle_repo.get_inventory(
        location=location,
        wine_type=wine_type,
        country=country,
        producer=producer,
        min_vintage=min_vintage,
        max_vintage=max_vintage,
        rating_filter=rating_filter,
        search=search,
    )
    grouped = _build_grouped_inventory(raw)
    sorted_inv = _apply_sort(grouped, sort_by)

    items = [_row_to_item(w) for w in sorted_inv]
    total_bottles = sum(w.get("quantity", 0) for w in sorted_inv)

    return InventoryResponse(
        items=items,
        total_wines=len(items),
        total_bottles=total_bottles,
        filter_options=filter_options,
    )


@router.get("/filters", response_model=FilterOptions)
def get_filters() -> FilterOptions:
    """Return available filter values for the inventory UI.

    Uses lightweight ``SELECT DISTINCT`` queries instead of a full inventory
    scan so this endpoint is cheap to call for populating dropdowns.
    """
    bottle_repo = BottleRepository()
    return FilterOptions(**bottle_repo.get_inventory_filter_options())


@router.get("/stats", response_model=CellarStatsResponse)
def get_stats() -> CellarStatsResponse:
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
def get_charts() -> ChartDataResponse:
    """Return pre-computed data for all cellar statistics charts.

    Wraps multiple ``StatsRepository`` methods and returns JSON-ready
    data that the frontend passes directly to a charting library.
    """
    from datetime import datetime

    stats_repo = StatsRepository()

    overview         = stats_repo.get_cellar_overview()
    varietal_dist    = stats_repo.get_varietal_distribution(limit=10)
    region_dist      = stats_repo.get_region_distribution(limit=10)
    drinking_wines   = stats_repo.get_drinking_window_wines()
    cellar_timeline  = stats_repo.get_cellar_size_over_time()
    top_rated        = stats_repo.get_top_rated_wines(limit=10)
    vintage_dist     = stats_repo.get_cellar_vintage_distribution()
    rating_dist      = stats_repo.get_cellar_rating_distribution()

    # Compute wine-age buckets from the vintage distribution (no extra DB hit).
    current_year = datetime.now().year
    age_buckets: dict[str, int] = {
        "0-5 years": 0, "6-10 years": 0, "11-15 years": 0,
        "16-20 years": 0, "20+ years": 0,
    }
    for item in vintage_dist:
        vintage = item.get("vintage")
        bottles = item.get("bottles", 0)
        if vintage:
            age = current_year - int(vintage)
            if age <= 5:
                age_buckets["0-5 years"] += bottles
            elif age <= 10:
                age_buckets["6-10 years"] += bottles
            elif age <= 15:
                age_buckets["11-15 years"] += bottles
            elif age <= 20:
                age_buckets["16-20 years"] += bottles
            else:
                age_buckets["20+ years"] += bottles
    wine_age_dist = [{"range": k, "bottles": v} for k, v in age_buckets.items() if v > 0]

    return ChartDataResponse(
        wine_type_distribution=overview.get("by_type", []),
        country_distribution=overview.get("by_country", []),
        varietal_distribution=varietal_dist,
        region_distribution=region_dist,
        drinking_window_wines=drinking_wines,
        cellar_size_over_time=cellar_timeline,
        top_rated=top_rated,
        vintage_distribution=vintage_dist,
        rating_distribution=rating_dist,
        wine_age_distribution=wine_age_dist,
    )


@router.get("/drink-next", response_model=DrinkNextResponse)
def get_drink_next(
    limit: int = Query(50, ge=1, le=200, description="Maximum wines per type")
) -> DrinkNextResponse:
    """Get wines ready to drink now, grouped by wine type.

    Returns wines with the highest drink_index (closest to peak/past peak),
    prioritizing wines that should be consumed soon. Wines are grouped by
    type (Red, White, Rosé, Sparkling, etc.) and sorted by drink_index
    descending within each group.

    Args:
        limit: Maximum number of wines to return per wine type (default 50).

    Returns:
        DrinkNextResponse with wines grouped by type.
    """
    from datetime import datetime

    bottle_repo = BottleRepository()
    raw_inventory = bottle_repo.get_inventory()
    grouped = _build_grouped_inventory(raw_inventory)

    # Filter to wines with drinking window data and currently in drinking window
    current_year = datetime.now().year
    ready_wines = []
    for wine in grouped:
        idx = _effective_index(wine)
        df = wine.get("drink_from_year")
        dt = wine.get("drink_to_year")
        
        # Include wines with drink_index >= 50 (approaching or past peak)
        # OR wines currently in their drinking window
        if idx is not None and idx >= 50:
            ready_wines.append(wine)
        elif df and dt and df <= current_year <= dt:
            ready_wines.append(wine)

    # Sort by drink_index descending (highest = drink soonest)
    ready_wines.sort(key=lambda w: _effective_index(w) or 0, reverse=True)

    # Group by wine type
    by_type: dict[str, list[DrinkNextItem]] = {}
    for wine in ready_wines:
        wine_type = wine.get("wine_type") or "Other"
        if wine_type not in by_type:
            by_type[wine_type] = []
        
        # Limit per type
        if len(by_type[wine_type]) >= limit:
            continue

        item = DrinkNextItem(
            wine_id=wine.get("wine_id", 0),
            wine_name=wine.get("wine_name", ""),
            producer_name=wine.get("producer_name"),
            vintage=wine.get("vintage"),
            wine_type=wine_type,
            varietal=wine.get("varietal"),
            region_name=wine.get("region_name"),
            country=wine.get("country"),
            quantity=wine.get("quantity", 0),
            drink_index=_effective_index(wine),
            drink_from_year=wine.get("drink_from_year"),
            drink_to_year=wine.get("drink_to_year"),
            personal_rating=wine.get("personal_rating"),
            community_rating=wine.get("community_rating"),
            location=wine.get("location"),
        )
        by_type[wine_type].append(item)

    total_wines = sum(len(wines) for wines in by_type.values())

    return DrinkNextResponse(by_type=by_type, total_wines=total_wines)



@router.post("/sync", response_model=SyncResponse)
def sync_cellar_tracker() -> SyncResponse:
    """Trigger a CellarTracker data sync.

    Reads credentials from server-side environment variables
    (``CELLAR_TRACKER_USERNAME``, ``CELLAR_TRACKER_PASSWORD``).

    Note: this operation can take 10-60+ seconds. For multi-user deployments
    consider moving it to a background task with a status-polling endpoint.
    """
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
