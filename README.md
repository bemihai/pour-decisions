# Pour Decisions

> **Project version**: 0.8.4 — last verified 2026-09-04.
> This document reflects the current state of the codebase. Components remain subject to change.

> A wine expert chatbot powered by RAG, an agentic LLM layer, and cellar management

Pour Decisions is an intelligent wine assistant that combines LLMs with a curated knowledge base of professional wine books and a personal wine cellar database. It uses RAG for accurate, source-cited answers; a LangGraph-based agentic layer for tool selection; and a full cellar management system with taste profile analytics.

## Features

### RAG Pipeline
- **Hybrid Search**: Balanced dense + BM25 candidate union followed by local cross-encoder reranking
- **Cross-Encoder Reranking**: `ms-marco-MiniLM-L-6-v2` for precision improvement
- **Wine Terminology**: Built-in normalization plus deterministic entity/intent query planning
- **Wine Metadata Extraction**: Grapes, regions, vintages, appellations, producers extracted from documents
- **Metadata Boosting**: Score boost for results matching query entities
- **Query Compression**: Local TF-IDF extractive compression to reduce token usage
- **Semantic Deduplication**: Removes near-duplicate chunks from context
- **Incremental Indexing**: Only processes new or modified files
- **Query Caching**: LRU cache for repeated queries
- **Source Citations**: Every answer references source material

### Agentic LLM Layer
- **Intelligent Agent**: LangGraph ReAct agent with LLM-driven tool selection (typically 1-3 calls; default hard budget 5)
- **Readiness-Aware Tools**: An explicit 18-tool catalogue filters unavailable dependencies at agent startup
- **Tool Introspection**: `GET /api/tools` reports current readiness and the agent's immutable startup selection
- **Async Agent Runtime**: One compiled graph supports `invoke()` and `ainvoke()`; FastAPI awaits intelligent-agent work and keeps synchronous RAG-only work off the event loop
- **Runtime Guardrails**: Pre-model call budgets, a graph-step backstop, exact duplicate-call blocking, conservative off-topic deflection, safe tool errors, bounded async admission/deadlines, narrow SQLite retry, and mandatory final-answer sanitization
- **RAG-Only Mode**: Traditional RAG without agents
- **Tool Categories**: Cellar queries, taste profile, food pairing, RAG search, web search
- **Web Search**: Tavily integration with SQLite-backed result caching

### Wine Cellar Management
- **SQLite Database**: Repository pattern, Pydantic models, raw SQL (no ORM)
- **ETL Importers**: CellarTracker API and Vivino CSV import pipelines
- **LLM Descriptions**: RAG-enhanced wine and producer descriptions with lazy generation and DB persistence
- **Food Pairing Rules**: Rule-based and LLM-assisted pairing recommendations

### UI
- **React + Next.js 16**: Multi-page App Router application (Chat, Cellar, Taste Profile)
- **Agent Mode Selector**: Switch between Intelligent and RAG-Only modes in sidebar
- **Cellar Dashboard**: Inventory browser, statistics, CellarTracker sync
- **Taste Profile Analytics**: Rating distributions, varietal analysis, regional preferences, trends
- **Dark Mode**: System-aware theme with manual toggle
- **Responsive**: Mobile-first layout with shadcn/ui components

## Table of Contents

- [Architecture](#architecture)
- [RAG Pipeline](#rag-pipeline)
- [Agentic LLM Layer](#agentic-llm-layer)
- [Wine Cellar & ETL](#wine-cellar--etl)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│              Frontend (Next.js 16, React, Tailwind v4)               │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐     │
│  │     Chat     │  │   Wine Cellar    │  │   Taste Profile    │     │
│  └──────┬───────┘  └──────────────────┘  └────────────────────┘     │
└─────────┼────────────────────────────────────────────────────────────┘
          │  HTTP/JSON  (src/lib/api.ts → NEXT_PUBLIC_API_URL)
          ▼
┌──────────────────────────────────────────────────────────────────────┐
│           REST API Layer  (FastAPI, src/api/, port :8000)            │
│  /api/chat  /api/cellar  /api/taste-profile  /api/wines  /api/tools  │
└─────────┬────────────────────────────────────────────────────────────┘
          │  Agent Mode: Intelligent / RAG-Only
          ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Agentic LLM Layer  (src/agents/)                        │
│                                                                      │
│  ┌────────────────────┐    ┌──────────────────────────────────────┐  │
│  │  Intelligent Agent │    │  Tools (src/agents/tools/)           │  │
│  │  (LangGraph ReAct) │───>│  - Cellar queries (SQLite)          │  │
│  │  1-3 typical/cap 5 │    │  - Taste profile analysis           │  │
│  └────────────────────┘    │  - Food & wine pairing              │  │
│                            │  - RAG search (wine knowledge)      │  │
│                            │  - Web search (Tavily + cache)      │  │
│                            └──────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌───────────────────────────┐  ┌──────────────────────────────────────┐
│   RAG Pipeline            │  │   Wine Cellar DB (src/database/)     │
│                           │  │                                      │
│  Query Preprocessing      │  │   SQLite + Repository Pattern        │
│  - Normalize wine terms   │  │   Tables: wines, bottles, producers, │
│  - Plan dense/sparse text │  │     regions, tastings, sync_logs,    │
│  - Analyze entities       │  │     food_pairing_rules               │
│                           │  │                                      │
│  Hybrid Retrieval         │  │   ETL (src/etl/)                     │
│  - Vector pool (ChromaDB) │  │   - CellarTracker API importer      │
│  - BM25 keyword pool      │  │   - Vivino CSV importer             │
│  - Balanced union         │  └──────────────────────────────────────┘
│                           │
│  Post-Retrieval           │
│  - Cross-encoder rerank   │
│  - Metadata boosting      │
│  - Compression (disabled) │
│  - Semantic deduplication │
│  - Context formatting     │
│                           │
│  Chroma + BM25 indexes    │
│  (Chroma host port 8100)  │
└───────────────────────────┘
```

## RAG Pipeline

Wine books are converted into searchable passages before users ask questions. At query time, the
system searches those passages by both meaning and exact terminology, reranks the combined results,
and gives the best evidence to the answering model. Extraction, indexing, search, and reranking run
locally; only final answer generation uses the configured application model.

```text
PDF / EPUB -> extract -> structured chunks -> Chroma + BM25
user question -> dense + keyword search -> rerank -> clean context -> LLM or agent
```

The production path has three important rules:

- Chroma and BM25 contain the same accepted chunks and use the same contextual search text.
- The API, evaluation harness, and agent tools all call `execute_production_rag()`.
- Missing or stale BM25 state causes an explicit vector-only fallback instead of mixing indexes.

Common indexing commands:

```bash
make chroma-upload       # Index new or changed books
make chroma-reindex      # Rebuild Chroma and the synchronized BM25 index
make chroma-stats        # Inspect a sample of the collection
make chroma-stats-exact  # Produce exact corpus statistics
```

Documentation:

- [`docs/pour-decisions-rag-pipeline.md`](docs/pour-decisions-rag-pipeline.md) — canonical plain-English and technical guide
- [`src/chroma/README.md`](src/chroma/README.md) — ingestion module responsibilities and contracts
- [`src/retrieval/README.md`](src/retrieval/README.md) — retrieval module responsibilities and usage
- [`src/eval/README.md`](src/eval/README.md) — evaluation commands, metrics, and result schema
- [`docs/rag-pipeline.md`](docs/rag-pipeline.md) — generic RAG tutorial, independent of this project

## Agentic LLM Layer

The agent layer (`src/agents/`) provides one active agent implementation plus RAG-only chat:

### Intelligent Agent (`src/agents/intelligent/agent.py`)
- LangGraph ReAct workflow with `StateGraph`
- LLM selects tools based on query analysis (planning call)
- Tools execute locally (DB queries, calculations)
- LLM generates final answer from tool outputs (generation call)
- Registry readiness is evaluated at construction; the bound tools and Jinja-rendered guidance use the same immutable snapshot
- Conservative relevance routing can deflect clear off-topic requests before any model or tool call
- Pre-model accounting enforces a default five-attempt budget plus a 30-step graph backstop
- Exact duplicate tool calls stop before the repeated pending batch executes
- Unexpected tool failures use stable safe messages, and every final answer passes mandatory sensitive-output sanitization
- Standard requests typically use 1-3 calls; hybrid planning and generation are counted separately

### RAG-Only Mode
- Traditional retrieval-augmented generation without LangGraph tool routing
- Uses the production retrieval path exposed by the chat API
- Useful for direct source-grounded answers and retrieval quality testing

### Tools (`src/agents/tools/`)

Tools are LangChain `@tool` decorated functions, organized by category:

| File | Tools | Description |
|------|-------|-------------|
| `cellar_tools.py` | `get_cellar_wines`, `get_wine_details`, `get_cellar_statistics` | Wine cellar inventory and management |
| `taste_profile_tools.py` | `get_user_taste_profile`, `get_top_rated_wines`, `get_wine_recommendations_from_profile`, `compare_wine_to_profile` | User preference analysis |
| `pairing_tools.py` | `get_food_pairing_wines`, `get_pairing_for_wine`, `get_wine_and_cheese_pairings` | Food and wine pairing |
| `rag_tools.py` | `search_wine_knowledge`, `search_wine_region_info`, `search_grape_variety_info`, `search_wine_term_definition`, `search_wine_producer_info` | RAG knowledge base search |
| `web_search_tools.py` | `search_web_for_wine`, `search_wine_price`, `search_wine_reviews` | Web search via Tavily with SQLite cache |

Module-local definitions are composed by `src/agents/tools/catalog.py` into an explicit registry.
The compatibility exports contain `CORE_TOOLS` (5 essential tools), `EXTENDED_TOOLS` (13 additional
tools), and `ALL_TOOLS` (18 active tools). Use `get_tools(extended=True)` to retrieve the complete
static catalogue when needed.

### Description Service (`src/agents/description_service.py`)
- Lazy-generates LLM descriptions for wines and producers
- RAG-enhanced: uses wine book context when available
- Persists descriptions in SQLite (no repeated LLM calls)
- Graceful fallback when RAG context is unavailable

## Wine Cellar & ETL

### Database (`src/database/`)
- **Schema**: `producers`, `regions`, `wines`, `bottles`, `tastings`, `sync_logs`, `food_pairing_rules`
- **Models**: Pydantic `BaseModel` with `ConfigDict(from_attributes=True)` in `src/database/models.py`
- **Repositories**: One per entity in `src/database/repository/` (`WineRepository`, `BottleRepository`, `ProducerRepository`, `RegionRepository`, `TastingRepository`, `SyncLogRepository`, `StatsRepository`, `FoodPairingRepository`)
- **Migrations**: Standalone scripts in `src/database/migrations/`
- **Connection**: `get_db_connection()` context manager with `PRAGMA foreign_keys = ON`

### ETL Importers (`src/etl/`)
- **CellarTracker**: `CellarTrackerImporter` - API-based import with sync logging
- **Vivino**: `VivinoImporter` - CSV-based import
- **Shared utilities**: `src/etl/utils.py` (normalization, parsing, deduplication helpers)

```bash
make import-ct        # Import from CellarTracker API
make import-vivino    # Import Vivino CSV data
make sync             # Sync all sources (with auto-backup)
```

## Setup & Installation

### Quick Start with Docker (Recommended)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd pour-decisions

# 2. Copy environment file and add your keys
cp .env.example .env
nano .env  # Add EMBEDDING_MODEL, WINE_BOOKS_PATH (GOOGLE_API_KEY optional)

# 3. Run the quick start script
./docker_quickstart.sh

# Or manually:
docker compose up --build
```

Access the app at: **http://localhost:3000** (Next.js frontend) and **http://localhost:8000/docs** (FastAPI docs).

Docker Compose starts the ChromaDB vector store, FastAPI backend, and Next.js frontend with persistent storage.

---

### Manual Installation (Development)

#### Prerequisites

- Python 3.11+
- Node.js 22+ and npm (for the Next.js frontend)
- Optional Ollama (local LLM runtime for eval/development or deliberate local API startup)
- `GOOGLE_API_KEY` for the cloud production default

#### 1. Clone and Install

```bash
git clone <your-repo-url>
cd pour-decisions

# Python dependencies (using uv)
pip install uv
make install   # runs uv sync

# Frontend dependencies
cd frontend && npm install && cd ..
```

#### 2. Configure Environment

Create a `.env` file:

```bash
# Required
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
WINE_BOOKS_PATH=/path/to/your/wine/books

# Optional: cloud fallback
GOOGLE_API_KEY=your_google_api_key_here

# ChromaDB (defaults shown)
CHROMA_HOST=localhost
CHROMA_PORT=8100


# Optional: Web search (Tavily)
TAVILY_API_KEY=your_tavily_key

# Optional: CellarTracker import
CELLAR_TRACKER_USERNAME=your_username
CELLAR_TRACKER_PASSWORD=your_password

# Optional: Phoenix tracing (local)
OBSERVABILITY_ENABLED=false
OBSERVABILITY_PROVIDER=phoenix
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_ENDPOINT_DOCKER=http://phoenix:6006/v1/traces
PHOENIX_PROJECT_NAME=pour-decisions
```

All environment variables are loaded via `src/utils/env.py` at import time using `python-dotenv`.

#### 3. Start ChromaDB

```bash
make chroma-up   # Docker container on host port 8100
```

#### 4. Load Wine Books

```bash
make chroma-upload   # Incremental indexing
```

#### 5. Initialize Wine Cellar (Optional)

```bash
make cellar-init     # Create SQLite database
make import-ct       # Import from CellarTracker
```

#### 6. Run the App

For development (hot-reload on both frontend and backend):
```bash
make dev-full   # ChromaDB + FastAPI :8000 + Next.js :3000
```

For production build:
```bash
make run   # Builds Next.js and starts the full stack
```

Open **http://localhost:3000**.

## Usage

### Chatbot Page (`/`)

The default page. Ask any wine question:
- "What is the difference between Merlot and Cabernet Sauvignon?"
- "What wines in my cellar pair well with lamb?"
- "Show me my taste profile for Italian wines"
- "Search for current prices of Barolo 2018"

### Sidebar — Agent Mode

Select the agent mode in the left sidebar:
- **Intelligent Agent**: LLM-driven tool selection. Best for complex, multi-step queries.
- **RAG Only**: Traditional RAG retrieval without agents.

### Cellar Page (`/cellar`)

Wine cellar dashboard with:
- Inventory browser with filters
- Cellar statistics and charts
- CellarTracker sync button

### Taste Profile Page (`/taste-profile`)

Analytics dashboard with:
- Rating distribution and trends
- Wine type performance
- Top varietals and varietal analysis
- Producer loyalty
- Favorite regions, countries, vintages, appellations
- Consumed wines inventory

## Configuration

### `app_config.yml`

```yaml
chroma:
  client:
    host: ${oc.env:CHROMA_HOST, localhost}
    port: ${oc.env:CHROMA_PORT, 8100}

  chunking:
    strategy: section_recursive         # section_recursive, section_semantic
    chunk_size: 1024
    chunk_overlap: 256
    extract_wine_metadata: true
    enable_small_to_big: false          # small-to-big retrieval pattern
    small_chunk_size: 256
    large_chunk_size: 1024

  indexing:
    quality_filter:
      mode: enforce
      min_score: 0.4
    bm25:
      rebuild_on_reindex: true
      sync_manifest_path: "chroma-data/bm25_index.meta.json"

  retrieval:
    n_results: 5
    similarity_threshold: 0.3
    # Deduplication
    use_deduplication: true
    deduplication_threshold: 0.9
    # Hybrid search
    enable_hybrid: true
    semantic_candidate_pool: 25
    bm25_candidate_pool: 25
    reranker_input_limit: 50
    bm25_index_path: "chroma-data/bm25_index.pkl"
    validate_bm25_sync: true
    # Reranking
    enable_reranking: true
    reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: 5
    rerank_threshold: 0.0               # accepted M3 cutoff for negative logits
    min_retrieval_confidence: 0.3       # provisional; fallback remains disabled
    # Context compression
    enable_compression: false
    compression_max_chars: 8000
    # Metadata boosting
    enable_metadata_boost: true
    metadata_boost_factor: 0.1

  settings:
    batch_size: 2500
    embedder: ${oc.env:EMBEDDING_MODEL}

  collections:
    - name: wine_books
      local_data_path: ${oc.env:WINE_BOOKS_PATH}
      metadata:
        description: "Professional wine books collection"
        hnsw:space: cosine
        hnsw:search_ef: 100
        hnsw:construction_ef: 200
        hnsw:num_threads: 8
        version: v1.1

agents:
  guardrails:
    call_budget:
      enabled: true
      max_llm_calls_per_query: 5
      max_graph_steps_per_query: 30
    loop_detection:
      enabled: true
    relevance:
      enabled: true
    tool_execution:
      enabled: true
      max_concurrent_calls: 4
      timeout_seconds:
        fast: 10
        slow: 30
      retry:
        enabled: true
        max_attempts: 2
        delay_seconds: 0.1
        min_remaining_seconds: 1.0
        allowed_cost_classes:
          - free
  tool_registry:
    health_check_ttl_seconds: 60   # dependency readiness cache TTL

model:
  provider: google                      # main app uses Google cloud models
  name: gemini-2.5-flash                # main app default model
  fallback_provider: google             # kept for compatibility in cloud-only app paths
  fallback_name: gemini-2.5-flash       # fallback cloud model
  hybrid_tool_calling: false            # unused while the app stays cloud-only
  ollama:
    base_url: ${oc.env:OLLAMA_BASE_URL, http://localhost:11434}

cellar:
  db_path: cellar-data/wine_cellar.db

web_search:
  provider: tavily
  max_results: 7
  auto_fallback: false
  cache:
    enabled: true
    max_entries: 1000
    db_path: cellar-data/web_cache.db
  tavily:
    api_key_env: TAVILY_API_KEY
```

Config is loaded via `get_config()` from `src/utils/utils.py` using OmegaConf. Supports environment variable interpolation with `${oc.env:VAR, default}`.

### Model Configuration

Pour Decisions currently uses Google Gemini for the main API path. Local Ollama execution remains
available for eval/development workflows and future local-routing work, but it is not the
production default.

The API startup policy is explicit in `app_config.yml`:

```yaml
api:
  enable_local_model_startup: false
```

With the default `false` value, the API loads the cloud model only and treats local Ollama as an
opt-in runtime path for deliberate experiments. Turning it on does not change the production
default request path automatically; it only makes local startup available.

**Local Ollama Models for Eval/Development:**

| Model | RAM | Speed | Use Case |
|-------|-----|-------|----------|
| `gemma3:4b` | 3.3GB | Fast | **Local dev/testing (RECOMMENDED)** |
| `gemma2:2b` | 1.6GB | Very fast | Lightweight (RAM-constrained machines) |
| `phi3:mini` | 2.3GB | Fast | Good balance |
| `llama3.2:3b` | 2.0GB | Fast | Excellent quality |
| `gemma4:e2b` | 5-6GB | Slow | Best quality (CPU-only) |

**Eval/Development Configuration:**

```bash
# Set in .env for local eval or manual experiments
OLLAMA_MODEL=gemma3:4b
OLLAMA_MEMORY_LIMIT=3G

# The main app remains cloud-first by default.
# Enable local API startup only when you want to test that path explicitly.
```

For full model configuration details, see [**Ollama Model Configuration Guide**](docs/ollama-model-configuration.md).

### Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `n_results` | Chunks to retrieve | 5 | 3-10 |
| `similarity_threshold` | Minimum similarity | 0.3 | 0.2-0.5 |
| `chunk_size` | Document chunk size | 1024 | 512-2048 |
| `chunk_overlap` | Overlap between chunks | 256 | 128-512 |
| `deduplication_threshold` | Similarity for dedup | 0.9 | 0.85-0.95 |
| `enable_hybrid` | Hybrid search | true | true/false |
| `enable_reranking` | Cross-encoder reranking | true | true/false |
| `rerank_top_k` | Results after reranking | 5 | 3-10 |
| `rerank_threshold` | Reranker filtering threshold | 0.0 | null or calibrated logit |
| `min_retrieval_confidence` | Normalized low-confidence cutoff | 0.3 | 0.0-1.0 |
| `enable_compression` | Context compression | false | true/false |
| `enable_metadata_boost` | Metadata score boost | true | true/false |
| `metadata_boost_factor` | Boost per entity match | 0.1 | 0.05-0.2 |

## Project Structure

```
pour-decisions/
├── src/
│   ├── agents/
│   │   ├── __init__.py                  # Exports WineAgent and create_wine_agent
│   │   ├── llm.py                       # LLM loading, invocation, prompt chain
│   │   ├── prompt_renderer.py           # Strict snapshot-aware Jinja prompt rendering
│   │   ├── description_service.py       # RAG-enhanced wine/producer descriptions
│   │   ├── intelligent/
│   │   │   └── agent.py                 # WineAgent (LangGraph ReAct)
│   │   ├── tools/
│   │   │   ├── __init__.py              # CORE_TOOLS, EXTENDED_TOOLS, get_tools()
│   │   │   ├── catalog.py               # Authoritative 18-tool catalogue composition
│   │   │   ├── registry.py              # Metadata, readiness cache, and selection snapshots
│   │   │   ├── cellar_tools.py          # Cellar inventory queries
│   │   │   ├── taste_profile_tools.py   # Taste preference analysis
│   │   │   ├── pairing_tools.py         # Food & wine pairing
│   │   │   ├── rag_tools.py             # RAG knowledge base search
│   │   │   ├── web_search_tools.py      # Tavily web search + SQLite cache
│   │   │   └── utils.py                 # Shared tool utilities
│   │   └── prompts/                     # Markdown and Jinja prompt assets
│   │
│   ├── chroma/
│   │   ├── extraction/                  # Layout-aware PDF and entry-aware EPUB
│   │   ├── chunking/                    # Block-aware section chunking
│   │   ├── loader.py                    # CollectionDataLoader (batch upsert)
│   │   ├── load_data.py                 # CLI entry point for indexing
│   │   ├── hierarchical_chunks.py       # Small-to-big retrieval
│   │   ├── index_tracker.py             # Manifest-based incremental indexing
│   │   ├── metadata_extractor.py        # Wine entity extraction
│   │   ├── deduplication.py             # Content deduplication
│   │   ├── stats.py                     # Collection diagnostics
│   │   └── utils.py                     # ChromaDB helpers
│   │
│   ├── retrieval/
│   │   ├── vector_retriever.py          # ChromaRetriever (vector search + cache)
│   │   ├── keyword_search.py            # BM25Index (keyword search)
│   │   ├── hybrid_retriever.py          # Balanced union + unweighted RRF fallback
│   │   ├── reranker.py                  # DocumentReranker (cross-encoder)
│   │   ├── confidence.py                # Normalized retrieval confidence primitive
│   │   ├── query_utils.py               # Wine term normalization & expansion
│   │   ├── query_analyzer.py            # Metadata entity extraction & filtering
│   │   ├── query_compression.py         # TF-IDF extractive context compression
│   │   └── context_builder.py           # Context formatting & semantic dedup
│   │
│   ├── database/
│   │   ├── db.py                        # SQLite connection, schema initialization
│   │   ├── models.py                    # Pydantic models (Wine, Bottle, Producer, etc.)
│   │   ├── utils.py                     # Dynamic SQL query builder
│   │   ├── repository/                  # Repository per entity
│   │   │   ├── wine.py                  # WineRepository
│   │   │   ├── bottle.py                # BottleRepository
│   │   │   ├── producer.py              # ProducerRepository
│   │   │   ├── region.py                # RegionRepository
│   │   │   ├── tasting.py              # TastingRepository
│   │   │   ├── stats.py                 # StatsRepository
│   │   │   ├── sync_logs.py             # SyncLogRepository
│   │   │   └── food_pairing.py          # FoodPairingRepository
│   │   └── migrations/                  # Standalone migration scripts
│   │
│   ├── etl/
│   │   ├── cellartracker_importer.py    # CellarTracker API importer
│   │   ├── vivino_importer.py           # Vivino CSV importer
│   │   ├── import_cellartracker.py      # CLI entry point for CT import
│   │   ├── import_vivino.py             # CLI entry point for Vivino import
│   │   └── utils.py                     # Shared ETL utilities
│   │
│   ├── api/
│   │   ├── main.py                      # FastAPI app, lifespan resource loading
│   │   ├── dependencies.py              # Shared dependency injection
│   │   ├── routes/                      # Route handlers (chat, cellar, taste_profile, wines, tools)
│   │   └── schemas/                     # Pydantic request/response schemas, including tool status
│   │
│   └── utils/
│       ├── __init__.py                  # Re-exports: logger, get_config, get_embedder, etc.
│       ├── utils.py                     # Config loading, project root, hashing
│       ├── resources.py                 # Cached HuggingFace embedder instances
│       ├── env.py                       # Environment variable loading (dotenv)
│       ├── logger.py                    # Logging setup
│       ├── tracing.py                   # Langfuse callback handler
│       ├── terms.py                     # Wine terminology data loader
│       └── terminology/                 # JSON dictionaries
│           ├── grape_synonyms.json
│           ├── misspellings.json
│           ├── region_variations.json
│           ├── query_expansions.json
│           ├── classifications.json
│           └── wine_appellations.json
│
├── frontend/                            # Next.js 16 + React + TypeScript frontend
│   ├── src/
│   │   ├── app/                         # Next.js App Router pages (layout, page.tsx files)
│   │   ├── components/                  # React components (shared, cellar, taste-profile)
│   │   ├── lib/                         # API client (api.ts), types, utilities
│   │   └── stores/                      # Zustand client-side state stores
│   ├── vitest.config.ts                 # Vitest test configuration
│   └── package.json                     # Node.js dependencies
│
├── archive/
│   └── ui/                              # Archived Streamlit code (no longer active)
│
├── tests/
│   ├── conftest.py                      # Shared fixtures
│   ├── chroma/                          # ChromaDB and indexing tests
│   └── agents/                          # Agent and tool tests
│
├── docs/                                # Documentation and diagrams
├── chroma-data/                         # ChromaDB storage + BM25 index + manifests
├── cellar-data/                         # Wine cellar SQLite DB + web cache
├── app_config.yml                       # Application configuration (OmegaConf)
├── docker-compose.yml                   # Docker setup (app + ChromaDB)
├── Makefile                             # Development commands
└── pyproject.toml                       # Dependencies (uv)
```

## Development

### Makefile Commands

```bash
# Application (local)
make run                # Production stack: ChromaDB + FastAPI + Next.js
make install            # Install Python dependencies (uv sync)

# Development mode (hot-reload)
make api                # FastAPI only on :8000 (auto-starts ChromaDB)
make frontend           # Next.js dev server on :3000
make dev-full           # ChromaDB + FastAPI + Next.js all together
make dev-stop           # Kill processes on :8000 and :3000

# Frontend
make frontend-build     # Production build of Next.js app
make frontend-test      # Run frontend unit tests (Vitest)

# Docker Compose
make up                 # Start all services (ChromaDB + api + frontend)
make down               # Stop all services
make restart            # Restart all services
make logs               # View all service logs
make status             # Check service status
make build              # Rebuild Docker images
make rebuild            # Stop, rebuild, and start

# ChromaDB Management
make chroma-up          # Start ChromaDB container
make chroma-down        # Stop ChromaDB container
make chroma-health      # Check container health
make chroma-reset       # Reset ChromaDB (clear all data)
make chroma-backup      # Backup ChromaDB data
make chroma-restore     # Restore from backup (BACKUP_FILE=path)

# Data Indexing
make chroma-upload      # Index new/modified files (incremental)
make chroma-reindex     # Force Chroma reindex + verified BM25 rebuild
make chroma-status      # Show index status
make chroma-stats       # Show sampled collection statistics
make chroma-stats-exact # Save exact configured-corpus JSON statistics

# Wine Cellar Database
make cellar-init        # Initialize database
make cellar-info        # Show database info
make cellar-backup      # Backup database
make cellar-restore     # Restore from backup (BACKUP_FILE=path)

# Data Import
make import-vivino      # Import Vivino CSV data
make import-ct          # Import from CellarTracker API
make sync               # Sync all sources (with auto-backup)

# Web Search
make web-cache-clear    # Clear web search result cache

# Testing
make test               # Python tests with coverage + frontend tests
make test-unit          # Python tests with 80% coverage threshold
make test-fast          # Quick run (no coverage, stop at first failure)
make test-watch         # Watch mode for continuous Python testing
make test-coverage      # Open HTML coverage report in browser
```

All `make` targets set `PYTHONPATH=$(pwd)` automatically. Running scripts directly requires `PYTHONPATH=. python3 -m src.module.name`.

### Testing

```bash
# Python tests
make test-fast   # Quick feedback loop
make test        # Full Python suite + frontend tests before committing

# Frontend tests (Vitest + React Testing Library)
make frontend-test             # Run once and exit
cd frontend && npm run test:watch    # Interactive watch mode
cd frontend && npm run test:coverage # Coverage report
```

Test structure mirrors `src/`: `tests/chroma/`, `tests/agents/`, etc. Frontend component tests live in `frontend/src/components/__tests__/`. See [`tests/README.md`](tests/README.md) for detailed testing guide.

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

Run from project root with correct PYTHONPATH:
```bash
PYTHONPATH=$(pwd) python3 -m src.chroma.load_data
# or use make targets which set PYTHONPATH automatically
make chroma-upload
```

### "Unable to connect to ChromaDB"

```bash
make chroma-health   # Check container status
make chroma-up       # Start container
```

### "No results found for query"

- Knowledge base empty: Run `make chroma-upload`
- Similarity threshold too high: Lower `similarity_threshold` in config
- Question unrelated to indexed content: Expected behavior

### App crashes or shows errors

- Check FastAPI logs in the terminal running `make api` or `make dev-full`
- Check Next.js logs in the terminal running `make frontend` or `make dev-full`
- Check ChromaDB logs: `docker logs pour_decisions_chromadb`
- Verify API keys in `.env`
- Test the API directly: `http://localhost:8000/health` and `http://localhost:8000/docs`

## License

[Your License Here]

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) for the agent framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Next.js](https://nextjs.org/) and [React](https://react.dev/) for the frontend
- [shadcn/ui](https://ui.shadcn.com/) for UI components
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Google Gemini](https://ai.google.dev/) for LLM capabilities
- [Tavily](https://tavily.com/) for web search API
