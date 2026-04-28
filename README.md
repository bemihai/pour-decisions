# Pour Decisions

> A wine expert chatbot powered by RAG, an agentic LLM layer, and cellar management

Pour Decisions is an intelligent wine assistant that combines LLMs with a curated knowledge base of professional wine books and a personal wine cellar database. It uses RAG for accurate, source-cited answers; a LangGraph-based agentic layer for tool selection; and a full cellar management system with taste profile analytics.

## Features

### RAG Pipeline
- **Hybrid Search**: Vector similarity (70%) + BM25 keyword (30%) with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: `ms-marco-MiniLM-L-6-v2` for precision improvement
- **Wine Terminology**: Built-in query normalization and expansion via JSON dictionaries
- **Wine Metadata Extraction**: Grapes, regions, vintages, appellations, producers extracted from documents
- **Metadata Boosting**: Score boost for results matching query entities
- **Query Compression**: Local TF-IDF extractive compression to reduce token usage
- **Semantic Deduplication**: Removes near-duplicate chunks from context
- **Incremental Indexing**: Only processes new or modified files
- **Query Caching**: LRU cache for repeated queries
- **Source Citations**: Every answer references source material

### Agentic LLM Layer
- **Intelligent Agent**: LangGraph ReAct agent with LLM-driven tool selection (2-3 LLM calls per query)
- **Keyword Agent**: Pattern-matching router with 1 LLM call per query (faster, ideal for testing)
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
- **Agent Mode Selector**: Switch between Intelligent, Keyword, and RAG-Only modes in sidebar
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
│  /api/chat   /api/cellar   /api/taste-profile   /api/wines           │
└─────────┬────────────────────────────────────────────────────────────┘
          │  Agent Mode: Intelligent / Keyword / RAG-Only
          ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Agentic LLM Layer  (src/agents/)                        │
│                                                                      │
│  ┌────────────────────┐    ┌──────────────────────────────────────┐  │
│  │  Intelligent Agent │    │  Tools (src/agents/tools/)           │  │
│  │  (LangGraph ReAct) │───>│  - Cellar queries (SQLite)          │  │
│  │  2-3 LLM calls/q   │    │  - Taste profile analysis           │  │
│  ├────────────────────┤    │  - Food & wine pairing              │  │
│  │  Keyword Agent     │    │  - RAG search (wine knowledge)      │  │
│  │  Pattern matching  │───>│  - Web search (Tavily + cache)      │  │
│  │  1 LLM call/query  │    └──────────────────────────────────────┘  │
│  └────────────────────┘                                              │
└──────────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌───────────────────────────┐  ┌──────────────────────────────────────┐
│   RAG Pipeline            │  │   Wine Cellar DB (src/database/)     │
│                           │  │                                      │
│  Query Preprocessing      │  │   SQLite + Repository Pattern        │
│  - Normalize wine terms   │  │   Tables: wines, bottles, producers, │
│  - Expand query           │  │     regions, tastings, sync_logs,    │
│  - Analyze metadata       │  │     food_pairing_rules               │
│                           │  │                                      │
│  Hybrid Retrieval         │  │   ETL (src/etl/)                     │
│  - Vector (ChromaDB 70%)  │  │   - CellarTracker API importer      │
│  - BM25 keyword (30%)     │  │   - Vivino CSV importer             │
│  - RRF fusion             │  └──────────────────────────────────────┘
│                           │
│  Post-Retrieval           │
│  - Cross-encoder rerank   │
│  - Metadata boosting      │
│  - Query compression      │
│  - Semantic deduplication │
│  - Context formatting     │
│                           │
│  ChromaDB Vector Store    │
│  (Docker, port 8100)      │
└───────────────────────────┘
```

## RAG Pipeline

### 1. Document Ingestion & Storage

Wine books are processed and stored in ChromaDB:

```
src/chroma/
├── load_data.py           # CLI for data ingestion (--force, --status)
├── loader.py              # CollectionDataLoader (batch upsert with content-hash dedup)
├── chunks.py              # Chunking strategies (basic, by_title, semantic)
├── hierarchical_chunks.py # Small-to-big retrieval pattern
├── index_tracker.py       # Incremental indexing with manifest tracking
├── metadata_extractor.py  # Wine entity extraction (grapes, regions, vintages, etc.)
├── deduplication.py       # Content deduplication utilities
├── stats.py               # Collection statistics and diagnostics
└── utils.py               # ChromaDB helper functions
```

**Features:**
- Multiple chunking strategies: Basic, By Title, Semantic
- Wine metadata extraction (grapes, regions, vintages, classifications, producers, appellations)
- Document context extraction (title, chapter, section)
- Incremental indexing via manifest files in `chroma-data/manifests/`
- Content hash-based duplicate detection
- BM25 index pickle generation at `chroma-data/bm25_index.pkl`

See [`src/chroma/README.md`](src/chroma/README.md) for detailed chunking strategy documentation.

**Run data loading:**
```bash
make chroma-upload    # Incremental (default)
make chroma-reindex   # Force reindex all
make chroma-status    # View index status
make chroma-stats     # Collection statistics
```

### 2. Retrieval Component

The retriever uses hybrid search combining vector and keyword matching:

```
src/retrieval/
├── vector_retriever.py    # ChromaRetriever (vector search with query expansion + caching)
├── keyword_search.py      # BM25Index (keyword search, persisted as pickle)
├── hybrid_retriever.py    # HybridRetriever (RRF fusion)
├── reranker.py            # DocumentReranker (cross-encoder)
├── query_utils.py         # Query normalization and expansion using wine terminology
├── query_analyzer.py      # Metadata-based filtering (extract entities -> ChromaDB where filters)
├── query_compression.py   # TF-IDF extractive compression to reduce context size
└── context_builder.py     # Context formatting, semantic deduplication, source display
```

**Key Features:**
- **Query Preprocessing**: Wine term normalization (misspelling correction, grape synonyms, region variations) and expansion via JSON dictionaries in `src/utils/terminology/`
- **Query Analysis**: Extracts grape, region, vintage, appellation entities from query and builds ChromaDB metadata filters
- **Hybrid Search**: Vector (70%) + BM25 (30%) with RRF fusion
- **Cross-Encoder Reranking**: `ms-marco-MiniLM-L-6-v2` for precision
- **Metadata Boosting**: Score boost for results matching detected query entities
- **Context Compression**: Local TF-IDF sentence scoring and deduplication (no LLM calls)
- **Query Caching**: LRU cache (100 queries default) in ChromaRetriever
- **Similarity Filtering**: Configurable threshold (default: 0.3)

### 3. Prompt Engineering

Custom prompts for different agent modes:

```
src/agents/prompts/
├── intelligent_agent_system_prompt.md  # ReAct agent system behavior
├── keyword_agent_generation_prompt.md  # Keyword agent answer generation
├── rag_only_system_prompt.md           # RAG-only system behavior
├── rag_only_user_prompt.md             # RAG-only context + question format
├── wine_description_prompt.md          # LLM wine description generation
└── producer_description_prompt.md      # LLM producer description generation
```

### 4. LLM Integration

Supports multiple LLM providers configured in `app_config.yml`:
- **Google Gemini** (default): `gemini-2.5-flash`
- **OpenAI**: GPT models (configurable)

### 5. Error Handling & Fallbacks

| Component | Error Scenario | Fallback Behavior |
|-----------|---------------|--------------------|
| ChromaDB Connection | Server unavailable | Disable RAG, use LLM only |
| Retriever | Query fails | Empty context, continue with LLM |
| Context Building | No results found | Empty context, LLM general knowledge |
| LLM | API error | Show error message, allow retry |
| Agent Tools | Tool execution fails | Agent retries or answers without tool |

## Agentic LLM Layer

The agent layer (`src/agents/`) provides two agent implementations:

### Intelligent Agent (`src/agents/intelligent/agent.py`)
- LangGraph ReAct workflow with `StateGraph`
- LLM selects tools based on query analysis (planning call)
- Tools execute locally (DB queries, calculations)
- LLM generates final answer from tool outputs (generation call)
- 2-3 LLM calls per query

### Keyword Agent (`src/agents/keyword/agent.py`)
- Pattern-matching router (no LLM for routing)
- Keyword patterns map queries to tool categories: cellar, taste, knowledge, pairing, web_search
- 1 LLM call per query (generation only)
- Better for testing and cost-sensitive usage

### Tools (`src/agents/tools/`)

Tools are LangChain `@tool` decorated functions, organized by category:

| File | Tools | Description |
|------|-------|-------------|
| `cellar_tools.py` | `get_cellar_wines`, `get_wine_details`, `get_cellar_statistics` | Wine cellar inventory and management |
| `taste_profile_tools.py` | `get_user_taste_profile`, `get_top_rated_wines`, `get_wine_recommendations_from_profile`, `compare_wine_to_profile` | User preference analysis |
| `pairing_tools.py` | `get_food_pairing_wines`, `get_pairing_for_wine`, `get_wine_and_cheese_pairings`, `suggest_dinner_menu_with_wines` | Food and wine pairing |
| `rag_tools.py` | `search_wine_knowledge`, `search_wine_region_info`, `search_grape_variety_info`, `search_wine_term_definition`, `search_wine_producer_info` | RAG knowledge base search |
| `web_search_tools.py` | `search_web_for_wine`, `search_wine_price`, `search_wine_reviews` | Web search via Tavily with SQLite cache |

Tools are registered in `src/agents/tools/__init__.py` as `CORE_TOOLS` (5 essential tools) and `EXTENDED_TOOLS` (12 additional tools). Use `get_tools(extended=True)` to get all tools.

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
nano .env  # Add GOOGLE_API_KEY, EMBEDDING_MODEL, WINE_BOOKS_PATH

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
- Google API Key (for Gemini) or OpenAI API Key

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
GOOGLE_API_KEY=your_google_api_key_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
WINE_BOOKS_PATH=/path/to/your/wine/books

# ChromaDB (defaults shown)
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Optional: OpenAI
OPENAI_API_KEY=your_key

# Optional: Web search (Tavily)
TAVILY_API_KEY=your_tavily_key

# Optional: CellarTracker import
CELLAR_TRACKER_USERNAME=your_username
CELLAR_TRACKER_PASSWORD=your_password

# Optional: Phoenix tracing (local)
OBSERVABILITY_ENABLED=false
OBSERVABILITY_PROVIDER=phoenix
PHOENIX_ENDPOINT=http://localhost:6006
PHOENIX_ENDPOINT_DOCKER=http://phoenix:6006
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
- **Keyword Agent**: Pattern-matching routing. Faster, fewer LLM calls, good for testing.
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
    port: ${oc.env:CHROMA_PORT, 8000}

  chunking:
    strategy: by_title                  # basic, by_title, semantic
    chunk_size: 1024
    chunk_overlap: 256
    extract_wine_metadata: true
    enable_small_to_big: false          # small-to-big retrieval pattern
    small_chunk_size: 256
    large_chunk_size: 1024

  retrieval:
    n_results: 5
    similarity_threshold: 0.3
    # Deduplication
    use_deduplication: true
    deduplication_threshold: 0.9
    # Hybrid search
    enable_hybrid: true
    hybrid_vector_weight: 0.7
    hybrid_keyword_weight: 0.3
    bm25_index_path: "chroma-data/bm25_index.pkl"
    # Reranking
    enable_reranking: true
    reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: 5
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

model:
  provider: google
  name: gemini-2.5-flash

initial_message:
  answer: "Hi there! Ask me anything about wine."

cellar:
  db_path: cellar-data/wine_cellar.db

web_search:
  provider: tavily
  max_results: 5
  cache:
    enabled: true
    max_entries: 1000
    db_path: cellar-data/web_cache.db
  tavily:
    api_key_env: TAVILY_API_KEY
```

Config is loaded via `get_config()` from `src/utils/utils.py` using OmegaConf. Supports environment variable interpolation with `${oc.env:VAR, default}`.

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
| `enable_compression` | Context compression | false | true/false |
| `enable_metadata_boost` | Metadata score boost | true | true/false |
| `metadata_boost_factor` | Boost per entity match | 0.1 | 0.05-0.2 |

## Project Structure

```
pour-decisions/
├── src/
│   ├── agents/
│   │   ├── __init__.py                  # Exports WineAgent, KeywordWineAgent, create_*
│   │   ├── llm.py                       # LLM loading, invocation, prompt chain
│   │   ├── description_service.py       # RAG-enhanced wine/producer descriptions
│   │   ├── intelligent/
│   │   │   └── agent.py                 # WineAgent (LangGraph ReAct)
│   │   ├── keyword/
│   │   │   └── agent.py                 # KeywordWineAgent (pattern matching)
│   │   ├── tools/
│   │   │   ├── __init__.py              # CORE_TOOLS, EXTENDED_TOOLS, get_tools()
│   │   │   ├── cellar_tools.py          # Cellar inventory queries
│   │   │   ├── taste_profile_tools.py   # Taste preference analysis
│   │   │   ├── pairing_tools.py         # Food & wine pairing
│   │   │   ├── rag_tools.py             # RAG knowledge base search
│   │   │   ├── web_search_tools.py      # Tavily web search + SQLite cache
│   │   │   └── utils.py                 # Shared tool utilities
│   │   └── prompts/                     # Markdown prompt files
│   │
│   ├── chroma/
│   │   ├── chunks.py                    # Chunking strategies (basic/by_title/semantic)
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
│   │   ├── hybrid_retriever.py          # HybridRetriever (RRF fusion)
│   │   ├── reranker.py                  # DocumentReranker (cross-encoder)
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
│   │   ├── routes/                      # Route handlers (chat, cellar, taste_profile, wines)
│   │   └── schemas/                     # Pydantic request/response schemas
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
├── design/                              # Design documents and plans
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
make chroma-reindex     # Force reindex all files
make chroma-status      # Show index status
make chroma-stats       # Show collection statistics

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
