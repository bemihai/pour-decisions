# Docker Deployment Guide

> **Project version**: 0.9.0 — last verified 2026-09-25.

Docker Compose runs the FastAPI backend (`api`), Next.js frontend (`frontend`), ChromaDB (`chromadb`), and optional Phoenix observability (`phoenix`). Generative requests use the direct Ollama Cloud API. The Compose stack does not run a local Ollama daemon or pull a model.

## Setup

1. Copy `.env.example` to `.env` and set `OLLAMA_API_KEY`, `EMBEDDING_MODEL`, and `WINE_BOOKS_PATH`. Keep the API key private.
2. Set the application model in `app_config.yml` under `model`. The default is `gemma4:31b` at `https://ollama.com`; execution and judge models have separate settings under `eval`.
3. Run `make up` or `docker compose up -d --build`.
4. Check `http://localhost:3000`, `http://localhost:8000/health`, and `docker compose ps`.

The API waits for healthy ChromaDB; the frontend waits for the API. The API must have a valid Cloud key to load its generative model. Local embeddings and the ChromaDB vector store remain part of retrieval.

## Useful commands

```bash
make logs           # All services
make logs-app       # API service
make logs-chroma    # ChromaDB service
make status         # Container health
make down           # Stop services
make rebuild        # Stop, rebuild, and restart
```

`cellar-data/`, `chroma-data/`, and `app_config.yml` are bind-mounted into the API container. The Phoenix volume persists observability data. Removing the old Ollama service declaration does not delete an existing user model volume; `docker compose down -v` removes Compose-managed volumes, so use it only when that is intended.

For Cloud configuration and supported model selection, see [Ollama Cloud model configuration](docs/ollama-model-configuration.md).
