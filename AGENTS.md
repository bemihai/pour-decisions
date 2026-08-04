# AGENTS.md

> **Project version**: 0.7.3 — last updated 2026-08-04.
> Reflects the current architecture. Subject to change as Milestone 3–14 improvements are
> implemented. See `design/roadmap/agentic-ai/milestones/` for planned changes.

## Project Overview

Pour Decisions is a RAG-powered wine chatbot with cellar management. **Cost minimization is the #1 architectural constraint** - prefer local models, free-tier services, caching, and batching over cloud API calls. The current production API is an explicit temporary exception: it defaults to cloud Gemini for quality/reliability, while local API startup stays opt-in through config.

## Collaboration Policy

- The user retains **100% ownership** over architecture, structure, and design decisions.
- This repository is a **learning lab**. Favor explicit, understandable code and clear control flow over clever abstractions, hidden behavior, or convenience magic.
- **Do not** make architecture, structure, or design changes without explicit approval.
- For **medium or large changes**, present a short plan before implementation even when the direction appears obvious.
- For **small, localized tasks**, implement directly while remaining within established patterns and approved boundaries.
- Surface assumptions, identify tradeoffs, and do not silently broaden scope.

## Approval Gates

The following changes require explicit approval before implementation:

- architecture, structure, or design changes
- prompt changes
- API or schema contract changes
- config default changes
- dependency additions or removals
- database migrations or migration edits
- design doc updates
- major frontend refactors

If a task requires any of the above, stop, explain why, and request approval before proceeding.

## Decision Priorities

When evaluating alternatives, use this priority order:

1. **Cost-consciousness** - default to low-cost solutions, but paid options are acceptable when they are clearly the best choice.
2. **Maintainability** - prefer code that is easy to read, extend, and debug.
3. **Reliability** - favor predictable behavior and explicit failure handling.
4. **Learning value** - prefer implementations that help the user understand the system and the tradeoffs.
5. **Modernity** - prefer current, well-supported approaches over legacy or outdated patterns.

If these priorities conflict, explain the tradeoff explicitly. Do not resolve the conflict through an unstated assumption.

## Agent Checklist

Before starting work:

1. Read this file and any task-relevant design docs.
2. Identify whether the request is a small localized task or a medium/large change.
3. Check whether the task crosses any approval gate.
4. If architecture, structure, design, frontend direction, or reviewed docs may change, stop and ask first.

### Supplemental Instructions

- Load `planka.md` only when the user asks for Planka work, card creation, board organization, backlog management, or closely related project-tracking tasks. Do not load it by default for ordinary coding work.

While implementing:

1. Stay within the approved scope.
2. Prefer explicit code paths, clear naming, and understandable control flow.
3. Avoid implicit fallbacks, hidden side effects, and unnecessary abstraction.
4. Keep cost, maintainability, and reliability visible in design choices.

Before handoff:

1. Run the appropriate tests for the change size.
2. Report what was changed, what was verified, and what was not verified.
3. Call out assumptions, tradeoffs, and any follow-up decisions the user should review.

## LLM Development Workflow

We use a strict **Strategy → Design → Implementation** workflow for LLM-assisted feature development. See `design/llm-coding/workflow-guide.md` for the full process. 
**Key rules:** Implement step-by-step from phased design documents, and treat design specs as living documents that must be updated upon deviation.

### Design Authority

- Existing design documents are reviewed artifacts. **Do not update them without explicit permission.**
- If implementation reveals a needed design change, explain the divergence clearly and request approval before editing design docs.
- “Small doc cleanup” is **not** exempt from this rule.

## Architecture

Five main subsystems connected through `app_config.yml` (OmegaConf):

1. **RAG Pipeline** (`src/chroma/` for indexing, `src/retrieval/` for querying) - ChromaDB vector store (Docker container, host port 8100 → container port 8000) with hybrid search (vector 70% + BM25 30%), cross-encoder reranking, metadata boosting, query compression, and semantic deduplication.
2. **Agentic LLM Layer** (`src/agents/`) - LangGraph ReAct agent (`src/agents/intelligent/agent.py`) that selects tools (cellar queries, RAG search, web search, taste profile, food pairing) via LLM planning. Targets 2-3 LLM calls per query max.
3. **Wine Cellar DB** (`src/database/`) - SQLite with raw SQL (no ORM), Pydantic models for validation, repository pattern per entity (`src/database/repository/`). Tables: `producers`, `regions`, `wines`, `bottles`, `tastings`, `sync_logs`, `food_pairing_rules`.
4. **REST API Layer** (`src/api/`) - FastAPI backend (port 8000) exposing all business logic as stateless JSON endpoints. Pydantic request/response schemas in `src/api/schemas/`, route handlers in `src/api/routes/` (chat, cellar, taste_profile, wines). Resources preloaded in `lifespan()` startup and stored in `app.state`.
5. **Frontend** (`frontend/`) - Next.js 16 + TypeScript + Tailwind v4 + shadcn/ui. Typed API client (`lib/api.ts`), TanStack Query for data fetching, Zustand for state. 

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

React + Next.js 16 multi-page app (`frontend/`):

- **Framework**: Next.js 16 App Router, TypeScript (strict), Tailwind CSS v4, shadcn/ui.
- **Pages**: `/` (Chat — `app/page.tsx`), `/cellar` (Cellar inventory + charts), `/taste-profile` (analytics dashboard).
- **Routing**: File-based via `app/` directory. Layouts in `app/layout.tsx`. Navigation via `Navigation.tsx`.
- **Shared components** (`src/components/`): `ChatInterface`, `ChatMessage`, `ChatSidebar`, `SourceList`, `MetricCard`, `DrinkingIndex`, `WineCard`, `FilterPanel`, `PageHeader`, `Rating`, `Section`, `EmptyState`.
- **Cellar components** (`src/components/cellar/`): `CellarOverview`, `CellarTabs`, `CellarInventory`, `CellarStatistics`, `CellarSyncButton`.
- **Taste Profile components** (`src/components/taste-profile/`): `TasteOverview`, `TasteProfileContent`, `TasteAnalytics`, `TasteHistory`, `TasteFavorites`.
- **Charts** (`src/components/charts/`): Recharts-based chart wrappers for all analytics views.
- **API client** (`src/lib/api.ts`): typed `fetch()` wrappers; resources preloaded in FastAPI `lifespan()` at startup.

## Development Commands

```bash
make install          # uv sync (Python deps)
make run              # Start production stack: ChromaDB + FastAPI (:8000) + Next.js (:3000)
make api              # Start FastAPI on :8000 (auto-starts ChromaDB, --reload)
make frontend         # Start Next.js dev server on :3000
make dev-full         # Start ChromaDB + FastAPI + Next.js together (dev mode)
make dev-stop         # Kill any lingering processes on :8000 and :3000
make frontend-build   # Production build of Next.js app
make frontend-test    # Run frontend unit tests (Vitest, exits after one pass)
make test-fast        # Quick Python test, no coverage, stop on first failure
make test             # Python tests with coverage + frontend tests
make test-unit        # Python tests with 80% coverage threshold
make test-watch       # Watch mode for continuous Python testing
make test-coverage    # Open HTML coverage report in browser
make chroma-up        # Start ChromaDB container only (polls until healthy)
make chroma-upload    # Incremental index wine books into ChromaDB
make chroma-reindex   # Force full Chroma reindex + verified BM25 rebuild
make chroma-stats     # Sampled collection statistics
make chroma-stats-exact # Exact configured-corpus JSON artifact
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
- **Frontend tests**: Vitest + React Testing Library in `frontend/src/components/__tests__/`. Run with `make frontend-test` or `cd frontend && npm test`. Watch mode: `cd frontend && npm run test:watch`. Coverage: `cd frontend && npm run test:coverage`.

### Testing Policy

- Unit tests must stay fast.
- For **small changes**, run targeted tests for the touched area.
- For **large changes**, run the full relevant suite before handoff.
- If full validation is skipped because of scope, time, environment limits, or missing prerequisites, state that explicitly.

## Style

- Type hints required on all functions. Google-style docstrings.
- Black formatting at 120 chars. isort for imports.
- No emojis in code comments, docstrings, or logs.
- Standard library, then third-party, then local imports - grouped and separated.

### Design and Explicitness

- Avoid hidden magic, implicit assumptions, and surprising fallbacks.
- Prefer explicit data flow, explicit configuration, and clear interfaces.
- Code involving agents or orchestration must be easy to follow step-by-step.
- Keep logging balanced: log what helps debugging or analysis, avoid noisy or decorative logs.
- Challenge unclear, weak, or inconsistent technical assumptions instead of blindly implementing them.
- Do not prefer OOP over functional programming. Use classes when appropriate and mix with functional programming following Python's best practices. 
- Versioned documents use the exact project version from `pyproject.toml`; documents do not maintain
  independent semantic versions.
- Keep the document's `last updated` or `last verified` date separate from the project version and
  update that date when the document is edited or re-verified.

### Frontend Policy

- The user is primarily a backend engineer. For meaningful frontend changes, explain the intended UI direction and request approval before implementation.
- Keep the UI functional at all times.
- Do not perform major frontend refactors without a reviewed plan and explicit approval.
- Creative UI work is allowed only when the rationale is clear and usability remains intact.

### Working Style

- For short, focused tasks: be execution-focused and concise.
- For larger tasks, design work, research, or brainstorming: be collaborative, explicit about tradeoffs, and open about alternatives.
- Compliance is not passivity: follow instructions carefully, but raise concerns when a request is technically weak, risky, or inconsistent with repo goals.

## Data Flow

1. **Indexing**: PDF/EPUB files -> `src/chroma/chunks.py` (split by strategy: basic/by_title/semantic) -> `src/chroma/metadata_extractor.py` (extract grapes, regions, vintages, classifications, producers, appellations) -> `src/chroma/loader.py` (batch upsert to ChromaDB with content-hash dedup) -> verified BM25 index pickle and synchronization manifest under `chroma-data/`.
   > **Note**: `make chroma-reindex` resets and rebuilds Chroma first, then atomically replaces BM25 and its synchronization manifest. Retrieval falls back explicitly to vector-only when the manifest is missing or stale.
2. **Query**: User query -> `src/retrieval/query_utils.py` (normalize + expand wine terms) -> `src/retrieval/query_analyzer.py` (extract entities, build metadata filters) -> `HybridRetriever` (vector + verified BM25 via RRF) -> `DocumentReranker` (cross-encoder, threshold effectively 0.0) -> metadata boosting -> `src/retrieval/query_compression.py` (optional TF-IDF compression) -> `src/retrieval/context_builder.py` (semantic dedup + format) -> LLM with prompt from `src/agents/prompts/`.
   > **Note**: API, eval, and agent RAG tools share `execute_production_rag()`; agent tools call it with generation disabled.
3. **Cellar Import**: Vivino CSV or CellarTracker API -> `src/etl/` importers (`VivinoImporter`, `CellarTrackerImporter`) -> SQLite via repository pattern, with sync logging.

> For a step-by-step code trace of the full pipeline see [`docs/rag-pipeline-deep-dive.md`](docs/rag-pipeline-deep-dive.md).

## Environment

Requires `.env` file with `EMBEDDING_MODEL` and `WINE_BOOKS_PATH`. Optional: `GOOGLE_API_KEY` (cloud fallback), `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `OBSERVABILITY_ENABLED`, `OBSERVABILITY_PROVIDER`, `PHOENIX_ENDPOINT`, `PHOENIX_ENDPOINT_DOCKER`, `PHOENIX_PROJECT_NAME`, `CELLAR_TRACKER_USERNAME`, `CELLAR_TRACKER_PASSWORD`, `CHROMA_HOST`, `CHROMA_PORT` (default 8100 for local dev), `OLLAMA_MODEL` (default `gemma3:4b`), `OLLAMA_MEMORY_LIMIT` (default `3G`). All loaded in `src/utils/env.py` at import time via `python-dotenv`.

Frontend environment: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000/api`) - can be set at build time or runtime.
