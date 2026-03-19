# AGENTS.md

## Project Overview

Pour Decisions is a RAG-powered wine chatbot with cellar management. **Cost minimization is the #1 architectural constraint** - prefer local models, free-tier services, caching, and batching over cloud API calls.

## Architecture

Three main subsystems connected through `app_config.yml` (OmegaConf):

1. **RAG Pipeline** (`src/chroma/` for indexing, `src/retrieval/` for querying) - ChromaDB vector store (runs as Docker container on port 8000) with hybrid search (vector 70% + BM25 30%), cross-encoder reranking, and semantic deduplication.
2. **Agentic LLM Layer** (`src/agents/`) - LangGraph ReAct agent (`src/agents/intelligent/agent.py`) that selects tools (cellar queries, RAG search, web search, taste profile) via LLM planning. Targets 2-3 LLM calls per query max.
3. **Wine Cellar DB** (`src/database/`) - SQLite with raw SQL (no ORM), Pydantic models for validation, repository pattern per entity (`src/database/repository/`).

## Key Patterns

- **Config**: All settings in `app_config.yml`, loaded via `get_config()` from `src/utils/utils.py`. Supports env var interpolation (`${oc.env:VAR, default}`).
- **Imports**: Use `from src.utils import logger, get_config, get_embedder` (re-exported from `src/utils/__init__.py`). Never use `print()`.
- **Embeddings**: Always use `get_embedder()` which caches model instances in `src/utils/resources.py`. Model name comes from config.
- **DB access**: Use `with get_db_connection() as conn:` context manager. Foreign keys enforced via PRAGMA. Migrations are standalone scripts in `src/database/migrations/`.
- **Prompts**: Stored as markdown files in `src/agents/prompts/`, loaded at module level in `src/agents/llm.py`.
- **Tools**: LangChain `@tool` decorated functions in `src/agents/tools/`. Registered in `src/agents/tools/__init__.py` via `CORE_TOOLS` list.
- **Models**: Pydantic `BaseModel` with `ConfigDict(from_attributes=True)` for all data models (`src/database/models.py`).

## Development Commands

```bash
make install          # uv sync
make run              # Start Streamlit app (auto-starts ChromaDB Docker if needed)
make test-fast        # Quick test, no coverage, stop on first failure
make test             # Full test suite with coverage
make chroma-up        # Start ChromaDB container only
make chroma-upload    # Incremental index wine books into ChromaDB
make chroma-reindex   # Force full reindex
```

All `make` targets set `PYTHONPATH=$(pwd)` automatically. Running scripts directly requires `PYTHONPATH=. python3 -m src.module.name`.

## Testing

- Fixtures in `tests/conftest.py`: `in_memory_chroma_client` (ephemeral, function-scoped), `sample_chunks`, `test_data_dir`.
- Test mirrors src: `tests/chroma/`, `tests/agents/`, etc.
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`.
- Coverage threshold: 80% on `make test-unit`.

## Style

- Type hints required on all functions. Google-style docstrings.
- Black formatting at 120 chars. isort for imports.
- No emojis in code comments, docstrings, or logs.
- Standard library, then third-party, then local imports - grouped and separated.

## Data Flow

1. **Indexing**: PDF/EPUB files -> `src/chroma/chunks.py` (split by strategy: basic/by_title/semantic) -> wine metadata extraction -> `src/chroma/loader.py` (batch upsert to ChromaDB with content-hash dedup) -> BM25 index pickle at `chroma-data/bm25_index.pkl`.
2. **Query**: User query -> `src/retrieval/query_utils.py` (normalize + expand wine terms) -> `HybridRetriever` (vector + BM25 via RRF) -> `DocumentReranker` (cross-encoder) -> `context_builder.py` (dedup + format) -> LLM with prompt from `src/agents/prompts/`.
3. **Cellar Import**: Vivino CSV or CellarTracker API -> `src/etl/` importers -> SQLite via repository pattern.

## Environment

Requires `.env` file with `GOOGLE_API_KEY`. Optional: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGFUSE_*` keys, `CELLAR_TRACKER_USERNAME/PASSWORD`. All loaded in `src/utils/env.py` at import time.

