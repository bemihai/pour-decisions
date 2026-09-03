"""
Wine agent tools for cellar management.
This module provides tools for querying and managing the user's wine cellar inventory.
"""

from typing import List, Dict
from datetime import datetime
from langchain_core.tools import tool

from src.agents.tools.registry import (
    CostClass,
    LatencyClass,
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolPrerequisite,
    ToolTier,
)
from src.database.repository import WineRepository, BottleRepository, StatsRepository
from src.agents.tools.utils import get_drink_status
from src.utils import get_default_db_path, logger


@tool
def get_cellar_wines(
    region: str | None = None,
    country: str | None = None,
    producer: str | None = None,
    wine_name: str | None = None,
    wine_type: str | None = None,
    varietal: str | None = None,
    appellation: str | None = None,
    vintage: int | None = None,
    vintage_min: int | None = None,
    vintage_max: int | None = None,
    ready_to_drink: bool = False,
    min_rating: int | None = None,
    limit: int = 50
) -> List[Dict]:
    """Get wines from user's cellar with optional and search filters.

    Query the local wine cellar database to find wines matching specified criteria.
    All filters are optional and can be combined for precise searching.

    Args:
        region: Filter by wine region (e.g., "Burgundy", "Piedmont", "Rioja").
                Supports partial matching (e.g., "Burg" matches "Burgundy").
        country: Filter by country (e.g., "France", "Italy", "Spain", "USA"). Exact match.
        producer: Filter by producer/winery name. Supports partial matching.
        wine_name: Filter by wine name. Supports partial matching.
        wine_type: Filter by wine type. Must be one of: "Red", "White", "Rosé",
                  "Sparkling", "Dessert", "Fortified". Exact match.
        varietal: Filter by grape variety (e.g., "Pinot Noir", "Cabernet Sauvignon").
                 Supports partial matching.
        appellation: Filter by appellation (e.g., "Pauillac", "Barolo DOCG"). Exact match.
        vintage: Filter by exact vintage year. Example: 2015
        vintage_min: Minimum vintage year (inclusive). Example: 2015
        vintage_max: Maximum vintage year (inclusive). Example: 2020
        ready_to_drink: If True, only return wines currently in their drinking window
                       (current year is between drink_from_year and drink_to_year).
        min_rating: Minimum personal rating (0-100 scale).
        limit: Maximum number of wines to return. Default 50, max 200.

    Returns:
        List of dictionaries, each containing:
        - wine_id: Unique identifier
        - name: Full wine name
        - producer: Producer/winery name
        - vintage: Vintage year (None for non-vintage wines)
        - wine_type: Type of wine (Red, White, etc.)
        - varietal: Grape variety/varieties
        - appellation: Specific appellation
        - region: Wine region
        - country: Country of origin
        - quantity: Number of bottles owned
        - location: Cellar location (e.g., "Rack A3")
        - drinking_window: String representation of optimal drinking period
        - drink_status: One of "ready", "aging", "past_peak", "unknown"
        - personal_rating: User's rating (0-100 scale) if tasted
        - purchase_price: Purchase price per bottle if available

    Example:
        >>> wines = get_cellar_wines(region="Burgundy", ready_to_drink=True)
        >>> wines = get_cellar_wines(wine_type="Red", vintage_min=2015, vintage_max=2020)
        >>> wines = get_cellar_wines(producer="Domaine Leflaive")
        >>> wines = get_cellar_wines(country="France", wine_type="White")
        >>> wines = get_cellar_wines(appellation="Barolo DOCG", ready_to_drink=True)

    Notes:
        - Returns empty list if no wines match criteria
        - Results ordered by producer, then vintage (descending)
        - Only returns wines with bottles in cellar (quantity > 0)
    """
    try:
        wine_repo = WineRepository(get_default_db_path())
        bottle_repo = BottleRepository(get_default_db_path())
        vintage_filter = vintage if vintage is not None else None

        wines = wine_repo.get_all(
            vintage=vintage_filter,
            wine_type=wine_type,
            appellation=appellation,
            country=country,
            min_rating=min_rating,
            ready_to_drink=ready_to_drink if ready_to_drink else None,
            producer_name=producer,
            region_name=region,
            wine_name=wine_name,
            varietal=varietal
        )

        results = []
        for wine in wines:
            if vintage is None:
                if vintage_min and wine.vintage and wine.vintage < vintage_min:
                    continue
                if vintage_max and wine.vintage and wine.vintage > vintage_max:
                    continue

            # Get bottle information
            bottles = bottle_repo.get_by_wine(wine.id, status="in_cellar")
            if not bottles:
                continue

            # Determine drink status
            drink_status = get_drink_status(wine.drink_from_year, wine.drink_to_year)

            location = bottles[0].location if bottles else None
            total_quantity = sum(b.quantity for b in bottles)

            results.append({
                "wine_id": wine.id,
                "name": wine.wine_name,
                "producer": wine.producer_name,
                "vintage": wine.vintage,
                "wine_type": wine.wine_type,
                "varietal": wine.varietal,
                "appellation": wine.appellation,
                "region": wine.region_name,
                "country": wine.country,
                "quantity": total_quantity,
                "location": location,
                "drinking_window": f"{wine.drink_from_year}-{wine.drink_to_year}" if wine.drink_from_year else None,
                "drink_status": drink_status,
                "personal_rating": wine.personal_rating,
                "purchase_price": bottles[0].purchase_price if bottles else None
            })

        logger.info(f"Found {len(results)} wines matching criteria")
        return results[:min(limit, 200)]

    except Exception:
        logger.exception("Unexpected failure while getting cellar wines")
        raise


@tool
def get_wine_details(
    wine_name: str | None = None,
    vintage: int | None = None
) -> Dict:
    """Get detailed information about a specific wine in the cellar.

    Retrieve comprehensive details for a single wine, including inventory,
    tasting notes, ratings, and purchase information.

    Args:
        wine_name: Wine name to search for. Supports partial matching.
                  Example: "Gevrey" will match "Gevrey-Chambertin"
        vintage: Vintage year to filter by (optional, use with wine_name).
                Example: 2019

    Returns:
        Dictionary containing detailed wine information:
        - name: Full wine name
        - producer: Producer/winery name with country
        - vintage: Vintage year
        - wine_type: Type (Red, White, Rosé, Sparkling, Dessert, Fortified)
        - varietal: Grape variety or blend
        - designation: Special designation (e.g., "Reserve", "Grand Cru")
        - appellation: Specific appellation (e.g., "Pouilly-Fumé")
        - vineyard: Specific vineyard name if applicable
        - region: Full region (primary + secondary)
        - country: Country of origin
        - bottle_size: Bottle size (e.g., "750ml", "1.5L")

        Inventory details:
        - quantity_owned: Current number of bottles owned
        - quantity_consumed: Total bottles consumed
        - location: Cellar location + bin  (e.g., "Cellar, Bin 2.1")
        - purchase_date: When purchased
        - purchase_price: Price paid per bottle

        Drinking window:
        - drink_from_year: Start of optimal drinking period
        - drink_to_year: End of optimal drinking period
        - drink_status: Current status ("ready", "aging", "past_peak", "unknown")
        - drink_index: Drinkability score

        Tasting information:
        - personal_rating: User's rating (0-100 scale)
        - community_rating: Average community rating (0-100 scale)
        - tasting_notes: User's personal tasting notes
        - last_tasted_date: Date of most recent tasting
        - is_defective: Whether bottle was defective
        - do_like: Simple like/dislike indicator

    Raises:
        ValueError: If neither wine_id nor wine_name is provided

    Example:
        >>> wine = get_wine_details(wine_id=42)
        >>> wine = get_wine_details(wine_name="Gevrey-Chambertin", vintage=2019)

    Notes:
        - Returns most recent match if multiple wines found with same name
        - Returns None values for missing data (e.g., not yet tasted)
        - Completely free operation (local SQLite query)
    """
    try:
        if not wine_name:
            return {"error": "Must provide either wine_id or wine_name"}

        wine_repo = WineRepository(get_default_db_path())
        bottle_repo = BottleRepository(get_default_db_path())


        wines = wine_repo.get_all(wine_name=wine_name, limit=10)
        if vintage:
            wines = [w for w in wines if w.vintage == vintage]
        wine = wines[0] if wines else None

        if not wine:
            return {"error": "Wine not found"}

        bottles = bottle_repo.get_by_wine(wine.id)
        in_cellar_bottles = [b for b in bottles if b.status == "in_cellar"]
        consumed_bottles = [b for b in bottles if b.status == "consumed"]
        drink_status = get_drink_status(wine.drink_from_year, wine.drink_to_year)
        locations = list(set(f"{b.location}-{b.bin}" for b in in_cellar_bottles if b.location))

        return {
            # Basic info
            "name": wine.wine_name,
            "producer": wine.producer_name,
            "vintage": wine.vintage,
            "wine_type": wine.wine_type,
            "varietal": wine.varietal,
            "designation": wine.designation,
            "appellation": wine.appellation,
            "vineyard": wine.vineyard,
            "region": wine.region_name,
            "country": wine.country,
            "bottle_size": wine.bottle_size,

            # Inventory
            "quantity_owned": sum(b.quantity for b in in_cellar_bottles),
            "quantity_consumed": sum(b.quantity for b in consumed_bottles),
            "location": ", ".join(locations) if locations else None,
            "purchase_date": in_cellar_bottles[0].purchase_date if in_cellar_bottles else None,
            "purchase_price": in_cellar_bottles[0].purchase_price if in_cellar_bottles else None,

            # Drinking window
            "drink_from_year": wine.drink_from_year,
            "drink_to_year": wine.drink_to_year,
            "drink_status": drink_status,
            "drink_index": wine.drink_index,

            # Tasting
            "personal_rating": wine.personal_rating,
            "community_rating": wine.community_rating,
            "tasting_notes": wine.tasting_notes,
            "last_tasted_date": str(wine.last_tasted_date) if wine.last_tasted_date else None,
        }

    except Exception:
        logger.exception("Unexpected failure while getting wine details")
        raise


@tool
def get_cellar_statistics() -> Dict:
    """Get comprehensive statistics about the wine cellar collection.

    Retrieve overview statistics, composition breakdown, and collection insights.

    Returns:
        Dictionary containing cellar statistics:

        Overview:
        - total_bottles: Total number of bottles in cellar
        - unique_wines: Number of unique wine entries
        - total_value: Estimated total collection value
        - average_bottle_price: Average price per bottle

        By wine type:
        - by_type: Dict with counts for Red, White, Rosé, Sparkling, etc.
        - type_percentages: Dict with percentage breakdown by type

        By region:
        - by_region: Dict with top 10 regions and bottle counts
        - by_country: Dict with country distribution

        By vintage:
        - oldest_vintage: Oldest vintage in collection
        - newest_vintage: Newest vintage in collection
        - average_vintage: Average vintage year
        - vintage_distribution: Dict of vintages and counts

        Drinking windows:
        - ready_to_drink: Number of bottles in drinking window
        - still_aging: Number of bottles not yet ready
        - past_peak: Number of bottles past optimal window
        - unknown_window: Bottles without drinking window data

        Value insights:
        - most_expensive: Details of most expensive bottle
        - total_invested: Total amount spent on collection
        - appreciation_estimate: Estimated value increase

        Top producers:
        - top_producers: List of top 5 producers by bottle count

    Example:
        >>> stats = get_cellar_statistics()
        >>> print(f"Total bottles: {stats['total_bottles']}")
        >>> print(f"Ready to drink: {stats['ready_to_drink']}")

    Notes:
        - Completely free operation (local SQLite aggregations)
        - Updates in real-time based on current database state
        - Useful for collection overview and analysis
    """
    try:
        stats_repo = StatsRepository(get_default_db_path())
        wine_repo = WineRepository(get_default_db_path())

        overview = stats_repo.get_cellar_overview()
        wines = wine_repo.get_all()
        current_year = datetime.now().year

        ready = sum(1 for w in wines if w.drink_from_year and w.drink_to_year
                   and w.drink_from_year <= current_year <= w.drink_to_year and w.q_quantity > 0)
        aging = sum(1 for w in wines if w.drink_from_year and current_year < w.drink_from_year
                   and w.q_quantity > 0)
        past_peak = sum(1 for w in wines if w.drink_to_year and current_year > w.drink_to_year
                       and w.q_quantity > 0)
        unknown = sum(1 for w in wines if (not w.drink_from_year or not w.drink_to_year)
                     and w.q_quantity > 0)

        # Calculate type percentages
        total = overview["total_bottles"]
        type_percentages = {}
        if total > 0:
            for type_info in overview["by_type"]:
                type_percentages[type_info["wine_type"]] = round(
                    (type_info["bottles"] / total) * 100, 1
                )

        return {
            "total_bottles": overview['total_bottles'],
            "unique_wines": overview['unique_wines'],
            "by_type": overview['by_type'],
            "type_percentages": type_percentages,
            "by_country": overview['by_country'],
            "ready_to_drink": ready,
            "still_aging": aging,
            "past_peak": past_peak,
            "unknown_window": unknown,
        }

    except Exception:
        logger.exception("Unexpected failure while getting cellar statistics")
        raise


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        tool=get_cellar_wines,
        metadata=ToolMetadata(
            name="get_cellar_wines",
            category=ToolCategory.CELLAR,
            tier=ToolTier.CORE,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Query the cellar inventory with optional wine and drinking-window filters.",
        ),
    ),
    ToolDefinition(
        tool=get_wine_details,
        metadata=ToolMetadata(
            name="get_wine_details",
            category=ToolCategory.CELLAR,
            tier=ToolTier.CORE,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Retrieve detailed cellar information for one wine.",
        ),
    ),
    ToolDefinition(
        tool=get_cellar_statistics,
        metadata=ToolMetadata(
            name="get_cellar_statistics",
            category=ToolCategory.CELLAR,
            tier=ToolTier.EXTENDED,
            prerequisites=(ToolPrerequisite.CELLAR_SCHEMA,),
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Summarize cellar inventory, composition, and drinking-window statistics.",
        ),
    ),
)
