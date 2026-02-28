"""Data models for wine cellar database."""
from datetime import date, datetime
from pydantic import BaseModel, ConfigDict, Field


class Producer(BaseModel):
    """Wine producer/winery agents."""
    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(None, description="Unique identifier")
    name: str = Field("", description="Producer/winery name")
    country: str | None = Field(None, description="Country of origin")
    region: str | None = Field(None, description="Full region name (primary + secondary)")
    description: str | None = Field(None, description="Additional notes about the producer")
    created_at: datetime | None = Field(None, description="Record creation timestamp")
    updated_at: datetime | None = Field(None, description="Record last update timestamp")


class Region(BaseModel):
    """Geographic wine region agents."""
    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(None, description="Unique identifier")
    primary_name: str = Field("", description="Primary region name (e.g., Loire Valley, Bordeaux)")
    secondary_name: str | None = Field(None, description="Secondary region name (e.g., Médoc, Sancerre)")
    country: str = Field("", description="Country where region is located")
    description: str | None = Field(None, description="Additional notes about the region")
    created_at: datetime | None = Field(None, description="Record creation timestamp")


class Tasting(BaseModel):
    """Tasting notes and ratings agents."""
    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(None, description="Unique identifier")
    wine_id: int = Field(0, description="Foreign key to wine table")
    is_defective: bool = Field(False, description="Indicates if the bottle was defective")
    personal_rating: int | None = Field(None, description="Personal rating on 0-100 scale")
    tasting_notes: str | None = Field(None, description="Personal tasting notes and review")
    do_like: bool | None = Field(None, description="Indicates if the taster liked the wine")
    community_rating: float | None = Field(None, description="Community average rating on 0-100 scale")
    like_votes: int | None = Field(0, description="Number of community 'like' votes")
    like_percentage: float | None = Field(None, description="Percentage of community 'like' votes")
    last_tasted_date: date | None = Field(None, description="Date of most recent tasting")
    created_at: datetime | None = Field(None, description="Record creation timestamp")
    updated_at: datetime | None = Field(None, description="Record last update timestamp")


class Wine(BaseModel):
    """Wine catalog agents."""
    model_config = ConfigDict(from_attributes=True)

    # Ids and foreign keys
    id: int | None = Field(None, description="Unique wine identifier")
    source: str = Field("manual", description="Data source: 'cellar_tracker', 'vivino', or 'manual'")
    external_id: str | None = Field(None, description="External ID from source system (e.g., CellarTracker iWine)")
    producer_id: int | None = Field(None, description="Foreign key to producer table")
    region_id: int | None = Field(None, description="Foreign key to region table")

    # Wine details
    wine_name: str = Field("", description="Full wine name (producer + designation)")
    vintage: int | None = Field(None, description="Vintage year (empty for non-vintage wines)")
    wine_type: str = Field("Red", description="Wine type: Red, White, Rosé, Sparkling, Dessert, Fortified")
    varietal: str | None = Field(None, description="Grape variety/varietal (e.g., Sauvignon Blanc, Merlot)")
    designation: str | None = Field(None, description="Special designation or cuvée name (e.g., Reserve, Grand Cru)")
    appellation: str | None = Field(None, description="Specific appellation (e.g., Pouilly-Fumé)")
    vineyard: str | None = Field(None, description="Vineyard name")
    bottle_size: str = Field("750ml", description="Bottle size (e.g., 750ml, 1.5L)")
    drink_from_year: int | None = Field(None, description="Start of optimal drinking window")
    drink_to_year: int | None = Field(None, description="End of optimal drinking window")
    drink_index: float | None = Field(None, description="Drinkability index score")
    description: str | None = Field(None, description="LLM-generated wine description")

    # Community inventory fields
    q_purchased: int = Field(0, description="Quantity purchased (community cellar-data)")
    q_quantity: int = Field(0, description="Quantity currently owned + pending (community cellar-data)")
    q_consumed: int = Field(0, description="Quantity consumed (community cellar-data)")

    # Producer and region fields (not in DB, populated via joins)
    producer_name: str | None = Field(None, description="Producer name")
    region_name: str | None = Field(None, description="Full region name (primary + secondary)")
    country: str | None = Field(None, description="Country")

    # Tasting fields (not in DB, populated via joins)
    personal_rating: int | None = Field(None, description="Personal rating on 0-100 scale")
    community_rating: float | None = Field(None, description="Community average rating on 0-100 scale")
    tasting_notes: str | None = Field(None, description="Personal tasting notes and review")
    last_tasted_date: date | None = Field(None, description="Date of most recent tasting")

    created_at: datetime | None = Field(None, description="Record creation timestamp")
    updated_at: datetime | None = Field(None, description="Record last update timestamp")


class Bottle(BaseModel):
    """Individual bottle inventory agents."""
    model_config = ConfigDict(from_attributes=True)

    # Ids and foreign keys
    id: int | None = Field(None, description="Unique identifier")
    wine_id: int = Field(0, description="Foreign key to wine table")
    source: str = Field("manual", description="Data source: 'cellar_tracker', 'vivino', or 'manual'")
    external_bottle_id: str | None = Field(None, description="External bottle ID (e.g., CellarTracker barcode)")

    quantity: int = Field(1, description="Number of bottles in this record")
    status: str = Field("in_cellar", description="Status: 'in_cellar', 'consumed', 'gifted', 'lost'")
    location: str | None = Field(None, description="Storage location (e.g., Cellar, Wine rack)")
    bin: str | None = Field(None, description="Specific bin/rack position")
    purchase_date: date | None = Field(None, description="Date of purchase")
    purchase_price: float | None = Field(None, description="Purchase price per bottle")
    valuation_price: float | None = Field(None, description="Current valuation price per bottle")
    currency: str = Field("RON", description="Currency code (e.g., RON, EUR, USD)")
    store_name: str | None = Field(None, description="Retailer/store name where purchased")
    consumed_date: date | None = Field(None, description="Date consumed (if status is 'consumed')")
    bottle_note: str | None = Field(None, description="Notes specific to this bottle")
    created_at: datetime | None = Field(None, description="Record creation timestamp")
    updated_at: datetime | None = Field(None, description="Record last update timestamp")

    # Related objects (not in DB, populated via joins)
    wine_name: str | None = Field(None, description="Wine name (populated via join)")
    producer_name: str | None = Field(None, description="Producer name (populated via join)")
    region_name: str | None = Field(None, description="Region name (populated via join)")
    vintage: int | None = Field(None, description="Vintage year (populated via join)")


class SyncLog(BaseModel):
    """Sync operation log agents."""
    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(None, description="Unique identifier")
    source: str = Field("", description="Data source: 'cellar_tracker' or 'vivino'")
    sync_type: str = Field("", description="Sync type: 'initial', 'incremental', or 'manual'")
    sync_started_at: datetime | None = Field(None, description="Timestamp when sync operation started")
    sync_completed_at: datetime | None = Field(None, description="Timestamp when sync operation completed")
    status: str = Field("", description="Sync status: 'success', 'failed', or 'partial'")
    records_processed: int = Field(0, description="Total number of records processed")
    records_imported: int = Field(0, description="Number of new records imported")
    records_updated: int = Field(0, description="Number of existing records updated")
    records_skipped: int = Field(0, description="Number of records skipped")
    records_failed: int = Field(0, description="Number of records that failed to import")
    error_message: str | None = Field(None, description="Error message if sync failed")
    error_details: str | None = Field(None, description="Detailed error information")
    created_at: datetime | None = Field(None, description="Record creation timestamp")


class FoodPairingRule(BaseModel):
    """Food pairing rule agents."""
    model_config = ConfigDict(from_attributes=True)

    id: int | None = Field(None, description="Unique identifier")
    food_name: str = Field("", description="Name of the food (e.g., 'steak', 'salmon')")
    category: str | None = Field(None, description="Food category (e.g., 'red_meats', 'seafood', 'cheese')")
    wine_types: str = Field("", description="Comma-separated wine types (e.g., 'Red,White')")
    varietals: str = Field("", description="Comma-separated varietals (e.g., 'Cabernet Sauvignon,Malbec')")
    characteristics: str | None = Field(None, description="Wine characteristics description")
    pairing_explanation: str | None = Field(None, description="Why this pairing works")
    created_at: datetime | None = Field(None, description="Record creation timestamp")
    updated_at: datetime | None = Field(None, description="Record last update timestamp")

    def get_wine_types_list(self) -> list[str]:
        """Parse wine_types string into list."""
        return [wt.strip() for wt in self.wine_types.split(",") if wt.strip()]

    def get_varietals_list(self) -> list[str]:
        """Parse varietals string into list."""
        return [v.strip() for v in self.varietals.split(",") if v.strip()]
