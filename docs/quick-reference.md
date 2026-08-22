# Pour Decisions - Quick Reference

> **Project version**: 0.8.0 — last verified 2026-08-22.
> Commands and configs reflect the current stack. Subject to change as Milestones 4–14 land.

For an explanation of how indexing and retrieval work, see
[`pour-decisions-rag-pipeline.md`](pour-decisions-rag-pipeline.md).

## Common Commands

### Local Development
```bash
make install          # Install Python dependencies with uv
make dev-full         # ChromaDB + FastAPI :8000 + Next.js :3000 (hot-reload)
make api              # FastAPI only on :8000 (auto-starts ChromaDB)
make frontend         # Next.js dev server on :3000
make run              # Production stack: build Next.js then start all
make dev-stop         # Kill processes on :8000 and :3000
make chroma-health    # Check ChromaDB status
```

### Docker Deployment
```bash
make up              # Start all services (ChromaDB + api + frontend)
make down            # Stop all services
make logs            # View logs
make status          # Check service status
```

### ChromaDB Management
```bash
make chroma-up       # Start ChromaDB only
make chroma-down     # Stop ChromaDB
make chroma-health   # Check health status
make chroma-backup   # Backup ChromaDB data
make chroma-restore BACKUP_FILE=path/to/backup.tar.gz  # Restore from backup
make chroma-reset    # Complete reset (removes all data)
make chroma-upload   # Index wine books (incremental)
make chroma-reindex  # Force Chroma rebuild + atomic verified BM25 rebuild
make chroma-status   # Show index status
make chroma-stats    # Sampled collection statistics
make chroma-stats-exact # Exact configured-corpus JSON artifact
```

### Wine Cellar Database
```bash
make cellar-init     # Initialize database
make cellar-info     # Show database stats
make cellar-backup   # Backup database
make cellar-restore  # Restore from backup (BACKUP_FILE=path)
make import-vivino   # Import Vivino CSV
make import-ct       # Import CellarTracker
make sync            # Sync all sources
make web-cache-clear # Clear web search result cache
```

### Testing
```bash
make test            # Python tests with coverage + frontend tests
make test-unit       # Python tests with 80% coverage threshold
make test-fast       # Quick Python test run (no coverage, stop at first failure)
make test-watch      # Python watch mode for continuous testing
make test-coverage   # Open HTML coverage report in browser
make eval            # Eval harness (retrieval-only mode)
make eval-full       # Eval harness + Ragas scoring
make eval-report     # Compare latest eval result files
make eval-validate   # Check golden dataset for stale cellar-dependent items
make frontend-test   # Frontend unit tests (Vitest, exits after one pass)
cd frontend && npm run test:watch    # Frontend watch mode
cd frontend && npm run test:coverage # Frontend coverage report
```

## Port Configuration

| Service   | Local Port | Notes                              |
|-----------|------------|------------------------------------|
| Next.js   | 3000       | Frontend (dev and production)      |
| FastAPI   | 8000       | Backend REST API, `/docs` for Swagger |
| ChromaDB  | 8100       | Vector store (host port → container 8000) |

## Environment Variables

Required in `.env`:
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
WINE_BOOKS_PATH=data/wine-books
```

Optional cloud fallback:
```bash
GOOGLE_API_KEY=your_gemini_api_key
```

Optional for CellarTracker import:
```bash
CELLAR_TRACKER_USERNAME=your_username
CELLAR_TRACKER_PASSWORD=your_password
```

Optional for web search:
```bash
TAVILY_API_KEY=your_tavily_key
```

Description generation web search controls (`app_config.yml`):
```yaml
description_generation:
  enable_web_search: true

web_search:
  auto_fallback: false  # opt-in low-confidence RAG fallback
```

Note: DescriptionService and automatic RAG fallback share the cached web-search service and read
the API key env var name from `web_search.tavily.api_key_env` (default `TAVILY_API_KEY`). Automatic
fallback fails safe and leaves book results unchanged when the key or provider is unavailable.

## Tool Registry Diagnostics

The intelligent agent binds a readiness-filtered tool snapshot during construction. Inspect the
complete catalogue, current prerequisite readiness, and the default cloud agent's startup
selection with:

```bash
curl http://localhost:8000/api/tools
```

Readiness results are cached by shared prerequisite. Configure the cache TTL in `app_config.yml`:

```yaml
agents:
  tool_registry:
    health_check_ttl_seconds: 60
```

Readiness refreshes do not hot-swap tools on an existing agent. Restart or deliberately reconstruct
the agent to bind a new snapshot.

Optional for tracing:
```bash
OBSERVABILITY_ENABLED=false
OBSERVABILITY_PROVIDER=phoenix
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_ENDPOINT_DOCKER=http://phoenix:6006/v1/traces
PHOENIX_PROJECT_NAME=pour-decisions
```

Frontend environment (`.env.local` inside `frontend/`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api   # default
```

## Observability (Phoenix)

Phoenix runs locally on `http://localhost:6006`.

Start/stop commands:
```bash
make phoenix
make phoenix-down
```

Trace correlation tips:
- Filter by `agent_mode` (`intelligent`, `rag_only`)
- Filter by `request_id` to find one specific chat request
- Description jobs are tagged with `feature=description_generation`

If traces do not appear:
1. Confirm Phoenix is running (`make phoenix`, then open `http://localhost:6006`)
2. Confirm observability is enabled (`OBSERVABILITY_ENABLED=true`)
3. Confirm OTLP endpoint uses `/v1/traces`:
   - `PHOENIX_ENDPOINT=http://localhost:6006/v1/traces`
   - `PHOENIX_ENDPOINT_DOCKER=http://phoenix:6006/v1/traces`
4. Restart API after changing `.env`

## Directory Structure

```
chroma-data/          # ChromaDB persistent storage (mounted as volume)
  bm25_index.pkl      # BM25 keyword search index
  bm25_index.meta.json # Chroma/BM25 record-count and sorted-ID synchronization proof
  manifests/          # Incremental indexing manifests
cellar-data/          # Wine cellar SQLite database + web cache
  wine_cellar.db
  web_cache.db        # Web search result cache (Tavily)
frontend/             # Next.js app source
  src/components/     # React components
  src/lib/            # API client, types, utilities
  src/stores/         # Zustand state stores
archive/
  ui/                 # Archived Streamlit code (no longer active)
backups/
  chroma/             # ChromaDB backups
  wine_cellar/        # Database backups
```

## Troubleshooting

### ChromaDB won't start
1. Check logs: `docker logs pour_decisions_chromadb`
2. Reset and restore: `make chroma-reset`, then `make chroma-restore BACKUP_FILE=...`

### Collection not found
1. Restore from backup: `make chroma-restore BACKUP_FILE=backups/chroma/chroma-backup-YYYYMMDD-HHMMSS.tar.gz`
2. Or repopulate: `make chroma-upload`

### Connection refused
1. Ensure ChromaDB is running: `make chroma-health`
2. Start if needed: `make chroma-up`
3. Wait for health check (up to 60 seconds)

### API not responding
1. Check FastAPI is running: `curl http://localhost:8000/health`
2. Check logs from `make api` or `make dev-full`
3. Verify `.env` keys are set

For more details, see `TROUBLESHOOTING.md`
