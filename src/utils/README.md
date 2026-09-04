# Utils Module

> **Project version**: 0.8.4 — last verified 2026-09-04.
> Shared utilities are stable. New utilities (e.g., additional wine terminology files or
> caching helpers) may be added as milestones are implemented.

The `utils` module provides shared utilities re-exported for convenient access throughout the application.

## Canonical Import

```python
from src.utils import logger, get_config, get_embedder
```

All public symbols are re-exported from `src/utils/__init__.py`.

## Components

| File | Purpose |
|------|---------|
| `utils.py` | Config loading (`get_config`), project root detection, ChromaDB client init, hashing, cosine similarity, JSON loading |
| `logger.py` | Custom colour-coded logger with relative source paths. Log level via `LOG_LEVEL` env var (0=NOTSET, 1=INFO, 2=DEBUG) |
| `resources.py` | Cached HuggingFace embedder instances (`get_embedder()`) |
| `env.py` | Loads `.env` at import time via `python-dotenv`. Exports API key constants |
| `tracing.py` | Langfuse callback handler for LLM observability |
| `terms.py` | Wine terminology data loader (grape synonyms, misspellings, region variations, etc.) |
| `terminology/` | JSON dictionaries used by `terms.py` |

## Key Functions

### Configuration

```python
from src.utils import get_config

cfg = get_config()                          # OmegaConf DictConfig
host = cfg.chroma.client.host              # "localhost"
model = cfg.model.name                     # "gemini-2.5-flash" (production default)
local_model = cfg.model.ollama.name        # "gemma3:4b" (overridable via OLLAMA_MODEL env var)
```

Config is loaded from `app_config.yml` at the project root. Supports `${oc.env:VAR, default}` interpolation.

### Logging

```python
from src.utils import logger

logger.info("Processing complete")
logger.error("Connection failed")
```

Never use `print()` in application code. The logger provides colour-coded output with source file, function name, and line number.

### Embeddings

```python
from src.utils import get_embedder

embedder = get_embedder()                   # Uses model from config
embedder = get_embedder("custom-model")     # Specific model
```

Instances are cached at module level to avoid re-downloading.

### Project Paths

```python
from src.utils import get_project_root, get_default_db_path, find_project_root

root = get_project_root()                   # Path to project root
db = get_default_db_path()                  # Path to wine_cellar.db
```

### Hashing

```python
from src.utils import generate_hash, compute_file_hash

text_hash = generate_hash("some content")   # MD5 hex digest
file_hash = compute_file_hash(Path("f.pdf"))
```

## Wine Terminology (`terminology/`)

JSON dictionaries loaded by `terms.py` and re-exported from `src/utils`:

| File | Export | Type | Purpose |
|------|--------|------|---------|
| `grape_synonyms.json` | `GRAPE_SYNONYMS` | `dict[str, list[str]]` | Canonical grape name to synonym list |
| `misspellings.json` | `MISSPELLINGS` | `dict[str, str]` | Common misspelling to correct form |
| `region_variations.json` | `REGION_VARIATIONS` | `dict[str, list[str]]` | Canonical region to variations |
| `query_expansions.json` | `QUERY_EXPANSIONS` | `dict[str, str]` | Query term expansion mappings |
| `classifications.json` | `CLASSIFICATIONS` | `dict[str, str]` | Wine classification abbreviations |
| `wine_appellations.json` | `WINE_APPELLATIONS` | `list[str]` | Known wine appellations |

Derived lookup patterns (`GRAPE_PATTERNS`, `REGION_PATTERNS`, `CLASSIFICATION_PATTERNS`, `PRODUCER_PREFIXES`, `PRODUCER_SUFFIXES`) are built at import time in `terms.py`.

## Environment Variables (`env.py`)

Loaded at import time from `.env`:

| Variable | Required | Used By |
|----------|----------|---------|
| `GOOGLE_API_KEY` | No | `llm.py` (cloud fallback when provider is `google`) |
| `OBSERVABILITY_ENABLED` | No | `tracing.py`, `api/main.py` |
| `OBSERVABILITY_PROVIDER` | No | `tracing.py`, `api/main.py` |
| `PHOENIX_ENDPOINT` | No | `tracing.py` |
| `PHOENIX_ENDPOINT_DOCKER` | No | `tracing.py` |
| `PHOENIX_PROJECT_NAME` | No | `tracing.py` |
| `CELLAR_TRACKER_USERNAME` | No | `etl/cellartracker_importer.py` |
| `CELLAR_TRACKER_PASSWORD` | No | `etl/cellartracker_importer.py` |
