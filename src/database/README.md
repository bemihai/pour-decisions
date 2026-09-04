# Database Module

> **Project version**: 0.8.4 — last verified 2026-09-04.
> Schema and repository pattern are stable. Milestone 14 (knowledge graph and cellar
> intelligence) may introduce new tables or repositories. Update this README accordingly.

The `database` module manages the wine cellar SQLite database. It uses raw SQL (no ORM), Pydantic models for validation, and a repository-per-entity pattern.

## Components

| File / Directory | Purpose |
|------------------|---------|
| `db.py` | Connection management (`get_db_connection`), schema creation (`initialize_database`) |
| `models.py` | Pydantic data models for all entities |
| `utils.py` | Dynamic SQL query builder (`build_update_query`), string normalization |
| `repository/` | One repository class per database entity |
| `migrations/` | Standalone migration scripts (see `migrations/README.md`) |

## Schema

```
producers          regions            wines
 - id               - id               - id
 - name (UNIQUE)    - primary_name     - wine_name
 - country          - secondary_name   - vintage
 - region           - country          - wine_type
 - description      - description      - varietal / appellation
                                       - producer_id (FK)
                                       - region_id (FK)
                                       - description (LLM-generated)

bottles            tastings           food_pairing_rules
 - id               - id               - id
 - wine_id (FK)     - wine_id (FK)     - food_name
 - quantity          - personal_rating  - category
 - status            - community_rating - wine_types
 - location / bin    - tasting_notes    - varietals
 - purchase_price    - do_like          - pairing_explanation

sync_logs
 - id
 - source / sync_type
 - status
 - records_processed / imported / updated / skipped / failed
```

Foreign keys are enforced via `PRAGMA foreign_keys = ON` in every connection.

## Data Models (`models.py`)

All models use `pydantic.BaseModel` with `ConfigDict(from_attributes=True)`:

- `Wine` - Full wine catalog entry with joined producer/region/tasting fields
- `Bottle` - Individual bottle inventory record
- `Producer` - Wine producer / winery
- `Region` - Geographic wine region (primary + secondary name)
- `Tasting` - Tasting notes and ratings
- `SyncLog` - ETL sync operation log
- `FoodPairingRule` - Food-to-wine pairing rule

## Repositories (`repository/`)

Each repository wraps SQL queries for a single entity:

| Repository | Key Methods |
|------------|-------------|
| `WineRepository` | `get_by_id`, `get_by_name`, `get_by_external_id`, `get_all`, `find_duplicates`, `create`, `update` |
| `BottleRepository` | `get_by_id`, `get_by_wine`, `get_owned_quantity`, `create`, `update`, `consume` |
| `ProducerRepository` | `get_by_id`, `get_by_name`, `get_or_create`, `get_all`, `update` |
| `RegionRepository` | `get_by_id`, `get_by_name_and_country`, `get_or_create`, `get_all` |
| `TastingRepository` | `get_by_id`, `get_by_wine`, `get_latest_by_wine`, `get_all_with_wine_info`, `create`, `update` |
| `SyncLogRepository` | `start_sync_log`, `complete_sync_log` |
| `StatsRepository` | `get_cellar_overview`, `get_top_rated_wines`, `get_drinking_window_wines`, `get_rating_statistics`, `get_wine_type_stats`, ... |
| `FoodPairingRepository` | `get_by_id`, `get_by_food_name`, `search_by_food_name`, `get_by_wine_type`, `get_all` |

## Usage

### Connection

```python
from src.database import get_db_connection

with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM wines")
    count = cursor.fetchone()[0]
```

### Repository Pattern

```python
from src.database.repository import WineRepository

repo = WineRepository()
wine = repo.get_by_id(42)
wines = repo.get_all(wine_type="Red", country="France", limit=10)
```

### Initialization

```bash
make cellar-init   # Creates tables if not exist
```

Or programmatically:

```python
from src.database import initialize_database

initialize_database()  # Uses default path from config
```

## Migrations

Standalone scripts in `migrations/`. Each checks for existing changes before applying. See `migrations/README.md` for details.

```bash
python src/database/migrations/add_wine_description.py
python src/database/migrations/create_food_pairing_rules.py
```
