# AGENTS.md

## Project Overview

Pour Decisions is a RAG-powered wine chatbot with cellar management. **Cost minimization is the #1 architectural constraint** - prefer local models, free-tier services, caching, and batching over cloud API calls.

## Architecture

Five main subsystems connected through `app_config.yml` (OmegaConf):

1. **RAG Pipeline** (`src/chroma/` for indexing, `src/retrieval/` for querying) - ChromaDB vector store (Docker container, host port 8100 → container port 8000) with hybrid search (vector 70% + BM25 30%), cross-encoder reranking, metadata boosting, query compression, and semantic deduplication.
2. **Agentic LLM Layer** (`src/agents/`) - LangGraph ReAct agent (`src/agents/intelligent/agent.py`) that selects tools (cellar queries, RAG search, web search, taste profile, food pairing) via LLM planning. Targets 2-3 LLM calls per query max. Alternative: keyword agent (`src/agents/keyword/agent.py`) with pattern-matching routing and 1 LLM call per query.
3. **Wine Cellar DB** (`src/database/`) - SQLite with raw SQL (no ORM), Pydantic models for validation, repository pattern per entity (`src/database/repository/`). Tables: `producers`, `regions`, `wines`, `bottles`, `tastings`, `sync_logs`, `food_pairing_rules`.
4. **REST API Layer** (`src/api/`) - FastAPI backend (port 8000) exposing all business logic as stateless JSON endpoints. Pydantic request/response schemas in `src/api/schemas/`, route handlers in `src/api/routes/` (chat, cellar, taste_profile, wines). Resources preloaded in `lifespan()` startup and stored in `app.state`.
5. **Frontend** (`frontend/`) - Next.js 16 + TypeScript + Tailwind v4 + shadcn/ui. Typed API client (`lib/api.ts`), TanStack Query for data fetching, Zustand for state. Replaces Streamlit UI with React components.

## Key Patterns

- **Config**: All settings in `app_config.yml`, loaded via `get_config()` from `src/utils/utils.py`. Supports env var interpolation (`${oc.env:VAR, default}`).
- **Imports**: Use `from src.utils import logger, get_config, get_embedder` (re-exported from `src/utils/__init__.py`). Never use `print()`.
- **Embeddings**: Always use `get_embedder()` which caches model instances in `src/utils/resources.py`. Model name comes from config (`chroma.settings.embedder`).
- **DB access**: Use `with get_db_connection() as conn:` context manager. Foreign keys enforced via PRAGMA. Migrations are standalone scripts in `src/database/migrations/`.
- **Prompts**: Stored as markdown files in `src/agents/prompts/`, loaded at module level in `src/agents/llm.py`.
- **Tools**: LangChain `@tool` decorated functions in `src/agents/tools/`. Organized into `CORE_TOOLS` (5 essential), `EXTENDED_TOOLS` (12 additional), and `ALL_TOOLS` in `src/agents/tools/__init__.py`. Use `get_tools(extended=True)` to retrieve. Categories: cellar (`cellar_tools.py`), taste profile (`taste_profile_tools.py`), pairing (`pairing_tools.py`), RAG search (`rag_tools.py`), web search (`web_search_tools.py`).
- **Models**: Pydantic `BaseModel` with `ConfigDict(from_attributes=True)` for all data models (`src/database/models.py`): `Wine`, `Bottle`, `Producer`, `Region`, `Tasting`, `SyncLog`, `FoodPairingRule`.
- **Repositories**: One per entity in `src/database/repository/`: `WineRepository`, `BottleRepository`, `ProducerRepository`, `RegionRepository`, `TastingRepository`, `SyncLogRepository`, `StatsRepository`, `FoodPairingRepository`.
- **Description Service**: `src/agents/description_service.py` - lazy LLM generation of wine/producer descriptions, RAG-enhanced with wine book context, persisted in SQLite to avoid repeated calls.
- **Wine Terminology**: JSON dictionaries in `src/utils/terminology/` (grape synonyms, misspellings, region variations, query expansions, classifications, appellations). Loaded by `src/utils/terms.py` and re-exported via `src/utils/__init__.py`.
- **Web Search**: Tavily integration configured under `web_search` in `app_config.yml`. Results cached in a separate SQLite database (`cellar-data/web_cache.db`) with per-type TTL.
- **API Schemas**: TypeScript interfaces in `frontend/src/lib/types.ts` mirror Pydantic schemas in `src/api/schemas/`. Keep in sync manually when changing request/response shapes.
- **API Client**: `frontend/src/lib/api.ts` - typed wrappers around `fetch()` for every FastAPI endpoint. `ApiError` class with HTTP status, `toQueryString()` helper for filters.
- **Frontend State**: Zustand stores in `frontend/src/stores/` for client-side state (chat messages, agent mode, filters). TanStack Query (`@tanstack/react-query`) for server state with 60s `staleTime`.

## UI

Multi-page Streamlit app:

- **Entry point**: `src/ui/app.py` - configures navigation with 3 pages.
- **Pages** (`src/ui/pages/`): `chatbot.py` (default, wine Q&A with agent modes), `cellar.py` (inventory, stats, CellarTracker sync), `taste_profile.py` (analytics dashboard).
- **Sidebar** (`src/ui/sidebar.py`): Agent mode selector (Intelligent Agent, Keyword Agent, No Agent/RAG Only). RAG settings shown only in RAG-Only mode.
- **Helpers** (`src/ui/helper/`): `display.py` (display utilities, styles), `cellar_stats.py` (cellar widgets), `taste_profile_stats.py` (taste profile analytics widgets).
- **Cached resources** (`src/ui/resources.py`): `@st.cache_resource` for LLM, agents, ChromaDB client, retriever, reranker.

## Development Commands

```bash
make install          # uv sync
make run              # Start Streamlit app (auto-starts ChromaDB Docker if needed)
make api              # Start FastAPI on :8000 (auto-starts ChromaDB)
make frontend         # Start Next.js dev server on :3000
make dev-full         # Start ChromaDB + FastAPI + Next.js (all at once)
make dev-stop         # Kill any lingering processes on :8000 and :3000
make frontend-build   # Production build of Next.js app
make test-fast        # Quick test, no coverage, stop on first failure
make test             # Full test suite with coverage
make test-unit        # Tests with 80% coverage threshold
make test-watch       # Watch mode for continuous testing
make test-coverage    # Open HTML coverage report in browser
make chroma-up        # Start ChromaDB container only (polls until healthy)
make chroma-upload    # Incremental index wine books into ChromaDB
make chroma-reindex   # Force full reindex
make chroma-stats     # Collection statistics
make import-ct        # Import from CellarTracker API
make import-vivino    # Import Vivino CSV data
make sync             # Sync all sources (with auto-backup)
make web-cache-clear  # Clear web search result cache
```

All `make` targets set `PYTHONPATH=$(pwd)` automatically. Running scripts directly requires `PYTHONPATH=. python3 -m src.module.name`.

## Testing

- Fixtures in `tests/conftest.py`: `in_memory_chroma_client` (ephemeral, function-scoped), `temp_chroma_client` (persistent, temp dir), `test_collection`, `populated_collection`, `sample_chunks`, `sample_embeddings`, `test_data_dir`, `test_wine_pdf`, `temp_dir`.
- Test mirrors src: `tests/chroma/` (7 test files), `tests/agents/` (`test_web_search_tools.py`), etc.
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`.
- Coverage threshold: 80% on `make test-unit`.

## Style

- Type hints required on all functions. Google-style docstrings.
- Black formatting at 120 chars. isort for imports.
- No emojis in code comments, docstrings, or logs.
- Standard library, then third-party, then local imports - grouped and separated.

## Data Flow

1. **Indexing**: PDF/EPUB files -> `src/chroma/chunks.py` (split by strategy: basic/by_title/semantic) -> `src/chroma/metadata_extractor.py` (extract grapes, regions, vintages, classifications, producers, appellations) -> `src/chroma/loader.py` (batch upsert to ChromaDB with content-hash dedup) -> BM25 index pickle at `chroma-data/bm25_index.pkl`.
2. **Query**: User query -> `src/retrieval/query_utils.py` (normalize + expand wine terms) -> `src/retrieval/query_analyzer.py` (extract entities, build metadata filters) -> `HybridRetriever` (vector + BM25 via RRF) -> `DocumentReranker` (cross-encoder) -> metadata boosting -> `src/retrieval/query_compression.py` (optional TF-IDF compression) -> `src/retrieval/context_builder.py` (semantic dedup + format) -> LLM with prompt from `src/agents/prompts/`.
3. **Cellar Import**: Vivino CSV or CellarTracker API -> `src/etl/` importers (`VivinoImporter`, `CellarTrackerImporter`) -> SQLite via repository pattern, with sync logging.

## Environment

Requires `.env` file with `GOOGLE_API_KEY`, `EMBEDDING_MODEL`, and `WINE_BOOKS_PATH`. Optional: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `CELLAR_TRACKER_USERNAME`, `CELLAR_TRACKER_PASSWORD`, `CHROMA_HOST`, `CHROMA_PORT` (default 8100 for local dev). All loaded in `src/utils/env.py` at import time via `python-dotenv`.

Frontend environment: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000/api`) - can be set at build time or runtime.
