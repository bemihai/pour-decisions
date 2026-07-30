# ETL Module

> **Project version**: 0.7.3 — last verified 2026-05-27.
> ETL importers are stable. Subject to change if Milestone 14 (knowledge graph and cellar
> intelligence) extends the import pipeline.

The `etl` module handles importing wine data from external sources into the wine cellar SQLite database.

## Components

| File | Purpose |
|------|---------|
| `cellartracker_importer.py` | `CellarTrackerImporter` - Imports via CellarTracker API |
| `vivino_importer.py` | `VivinoImporter` - Imports from Vivino CSV exports |
| `import_cellartracker.py` | CLI entry point for CellarTracker import |
| `import_vivino.py` | CLI entry point for Vivino import |
| `utils.py` | Shared ETL utilities (normalization, parsing, deduplication) |

## CellarTracker Importer

Connects to the CellarTracker API and imports wines, bottles, and tasting notes.

```python
from src.etl import CellarTrackerImporter

importer = CellarTrackerImporter(username, password)
stats = importer.import_all()
```

**Features:**
- Creates producers and regions on-the-fly via `get_or_create`
- Deduplicates wines by external ID
- Imports bottles with location, purchase price, and status
- Imports tasting notes and ratings (personal + community)
- Logs every sync operation to `sync_logs` table

**Environment variables:** `CELLAR_TRACKER_USERNAME`, `CELLAR_TRACKER_PASSWORD`

## Vivino Importer

Parses Vivino CSV exports (`full_wine_list.csv` and/or `cellar.csv`).

```python
from src.etl import VivinoImporter

importer = VivinoImporter()
stats = importer.import_full_wine_list_csv("cellar-data/vivino/full_wine_list.csv")
```

**Features:**
- Parses wine type, vintage, drinking window, ratings
- Normalizes ratings from Vivino's 5-point scale to 0-100
- Creates producers and regions via repository pattern
- Generates stable external IDs for deduplication

## Shared Utilities (`utils.py`)

| Function | Purpose |
|----------|---------|
| `normalize_wine_type(type_str)` | Maps source-specific types to standard types (Red, White, etc.) |
| `clean_text(text)` | Strip HTML entities and whitespace |
| `parse_date(date_str)` | Parse various date formats |
| `parse_vintage(vintage_str)` | Extract vintage year from strings |
| `parse_drinking_window(window_str)` | Parse "2020-2030" style drinking windows |
| `parse_country(country_str)` | Normalize country codes/names to full names |
| `normalize_rating(rating, scale)` | Convert ratings to 0-100 scale |
| `generate_external_id(...)` | Create stable deduplication keys |

## CLI Usage

```bash
make import-ct          # Import from CellarTracker API
make import-vivino      # Import Vivino CSV data
make sync               # Sync all sources (with auto-backup)
```

Or directly:

```bash
PYTHONPATH=. python3 -m src.etl.import_cellartracker -u USERNAME -p PASSWORD
PYTHONPATH=. python3 -m src.etl.import_vivino
```

## Data Flow

```
CellarTracker API  ──>  CellarTrackerImporter  ──>  ProducerRepository.get_or_create()
                                                     RegionRepository.get_or_create()
Vivino CSV         ──>  VivinoImporter         ──>  WineRepository.create() / update()
                                                     BottleRepository.create()
                                                     TastingRepository.create()
                                                     SyncLogRepository.start/complete()
```
