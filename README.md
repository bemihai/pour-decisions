# Pour Decisions

> **Project version**: 0.13.0 — last verified 2026-10-26.

Pour Decisions is a wine knowledge and cellar-management application. It combines a curated
wine-book index, local retrieval and reranking, a direct Ollama Cloud model, and a personal SQLite
cellar so answers can use both published evidence and verified inventory data.

## What it provides

- Hybrid wine-book retrieval with dense search, BM25, local cross-encoder reranking, citations,
  and bounded corrective retrieval.
- Intelligent chat with readiness-filtered cellar, taste-profile, pairing, RAG, and optional
  Tavily web-search tools, plus a simpler RAG-only mode.
- Cellar inventory, drinking-window views, CellarTracker and Vivino imports, and persisted wine,
  producer, and grape descriptions.
- Taste analytics and explicit likes, dislikes, avoids, and price preferences for one local profile.
- A Next.js interface for chat, cellar browsing, wine detail, and taste-profile management.
- Local embeddings, reranking, caching, and SQLite persistence to keep routine operation
  understandable and cost-conscious.

## Architecture

```text
Next.js :3000
  -> FastAPI :8000
       -> intelligent agent -> readiness-filtered tools
       -> RAG-only service  -> shared retrieval runtime
            -> ChromaDB :8100 + synchronized BM25
            -> local embeddings and reranker
       -> SQLite cellar, preferences, memory, and web cache
       -> direct Ollama Cloud generation
       -> optional Tavily search and local Phoenix tracing :6006
```

The FastAPI lifespan owns the model, agent, retrieval resources, tool registry, and optional
conversation memory. Both chat modes reuse the same retrieval transformations. Indexing lives in
`src/chroma/`, query-time retrieval in `src/retrieval/`, agent orchestration in `src/agents/`,
SQLite access in `src/database/`, imports in `src/etl/`, API routes in `src/api/`, and the web app
in `frontend/`.

For the detailed indexing and retrieval path, read the
[RAG and retrieval pipeline guide](docs/pour-decisions-rag-pipeline.md). For commands, ports,
configuration, runtime controls, and troubleshooting, use the
[quick reference](docs/quick-reference.md).

## First run

The local development path requires Python 3.11, `uv`, Node.js/npm, and Docker with Compose.

1. Install dependencies:

   ```bash
   make install
   cd frontend && npm install && cd ..
   ```

2. Create a private `.env` from `.env.example` and replace its placeholders. The template targets
   Docker Compose (`CHROMA_HOST=chromadb`, `CHROMA_PORT=8000`); for host-based development use
   `CHROMA_HOST=localhost` and `CHROMA_PORT=8100`. At minimum, configure:

   ```bash
   OLLAMA_API_KEY=your_ollama_api_key
   EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   WINE_BOOKS_PATH=/absolute/path/to/wine-books
   CHROMA_HOST=localhost
   CHROMA_PORT=8100
   ```

   Keep `.env` private. `TAVILY_API_KEY` and CellarTracker credentials are optional.

3. Start ChromaDB and index the configured books:

   ```bash
   make chroma-up
   make chroma-upload
   ```

4. Optionally initialize the cellar, then start the development stack:

   ```bash
   make cellar-init
   make dev-full
   ```

Open <http://localhost:3000>. FastAPI health and interactive API documentation are available at
<http://localhost:8000/health> and <http://localhost:8000/docs>.

For a Compose deployment, keep the template's Chroma service values, ensure `WINE_BOOKS_PATH`
resolves inside the API container when indexing is needed, and follow the
[Docker deployment guide](DOCKER.md).

## Using the application

- `/` provides Intelligent and RAG-only chat. Intelligent mode uses SSE progress streaming when
  `streaming.enabled` is true; interrupted post-start streams are uncertain outcomes and are not
  replayed automatically.
- `/cellar` provides inventory filters, statistics, drinking-window views, and CellarTracker sync.
- `/cellar/<wine-id>` shows one wine and supports explicit description generation.
- `/taste-profile` provides analytics, history, favorites, and declared-preference management.

Run `make help` for the command index. Common development checks are:

```bash
make test-fast        # Fast Python tests, without coverage
make test-unit        # Python unit suite with the 80% coverage gate
make frontend-test    # Vitest, one pass
make frontend-build   # Next.js production build
make format-check     # Black/isort check for src and tests
```

The [quick reference](docs/quick-reference.md) is authoritative for the full command behavior,
environment variables, ports, and failure recovery.

## Documentation map

Start here, then use the maintained guide that owns the topic instead of duplicating its details.

### Operations and architecture

- [Docker deployment](DOCKER.md) — Compose services, setup, persistence, and operational commands.
- [Quick reference](docs/quick-reference.md) — canonical commands, configuration, ports,
  environment variables, entry points, and troubleshooting.
- [Ollama Cloud model configuration](docs/ollama-model-configuration.md) — supported generative
  model slots and cost boundaries.
- [RAG and retrieval pipeline](docs/pour-decisions-rag-pipeline.md) — indexing, retrieval,
  evaluation decisions, and failure behavior.
- [Frontend](frontend/README.md) — pages, components, data flow, type synchronization, and tests.

### Backend subsystems

- [Agents](src/agents/README.md) — intelligent-agent graph, tools, prompts, and descriptions.
- [Agent runtime guardrails](src/agents/guardrails/README.md) — budgets, loop protection,
  cancellation, deadlines, retries, and internal outcomes.
- [Chroma ingestion](src/chroma/README.md) — extraction, chunking, filtering, and index lifecycle.
- [Retrieval](src/retrieval/README.md) — query planning, candidate union, reranking, and context.
- [Database](src/database/README.md) — schema, Pydantic models, and raw-SQL repositories.
- [Database migrations](src/database/migrations/README.md) — explicit migration procedures and
  backup requirements.
- [ETL](src/etl/README.md) — Vivino and CellarTracker imports.
- [Evaluation harness](src/eval/README.md) — datasets, metrics, execution modes, and result review.
- [Shared utilities](src/utils/README.md) — configuration, logging, embeddings, paths, and tracing.

### Testing

- [Test suite](tests/README.md) — test commands, markers, fixtures, and authoring conventions.
- [API integration test plan](tests/api/TEST_PLAN.md) — maintained live-endpoint smoke plan.

Repository policy lives in `AGENTS.md`; it is not product documentation. Historical evidence,
archived material, prompts, generated artifacts, and local design documents are intentionally
outside this maintained documentation set.
