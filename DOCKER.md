# Docker Deployment Guide

This guide explains how to deploy Pour Decisions using Docker Compose with all services including the local LLM (Ollama).

## Services

The `docker-compose.yml` file defines the following services:

1. **API** (`api`) - FastAPI backend on port 8000
2. **Frontend** (`frontend`) - Next.js web app on port 3000
3. **ChromaDB** (`chromadb`) - Vector database on port 8100
4. **Ollama** (`ollama`) - Local LLM inference server on port 11434
5. **Ollama Init** (`ollama-init`) - One-time service to pull the Gemma 4 model

## Prerequisites

- Docker Engine 20.10+
- Docker Compose V2
- 16 GB RAM minimum (Ollama Gemma 4 e2b requires ~7 GB)
- `.env` file with required API keys (see `.env.example`)

## Quick Start

```bash
# Start all services (ChromaDB + Ollama + API + Frontend)
make up

# Or manually:
docker compose up -d

# View logs
make logs

# Check status
make status
```

## First-Time Setup

On first startup, the `ollama-init` service will automatically pull the configured model (default: `gemma2:2b`, ~1.6 GB download). This takes 1-5 minutes depending on your internet connection and model size.

You can monitor the download progress:

```bash
docker compose logs -f ollama-init
```

Once complete, the init container will exit and the model will be cached in the `ollama-data` Docker volume for future use.

**Note**: The model downloaded by `ollama-init` must match the `model.name` in `app_config.yml`. Set both via the `OLLAMA_MODEL` environment variable (see Resource Limits section).

## Service Dependencies

The services start in this order:

1. **ChromaDB** - starts first, waits for health check
2. **Ollama** - starts after ChromaDB, waits for health check  
3. **Ollama Init** - pulls the model after Ollama is healthy
4. **API** - starts after ChromaDB and Ollama are healthy
5. **Frontend** - starts after the API is healthy

## Environment Variables

The API service requires these environment variables (set in `.env`):

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here
EMBEDDING_MODEL=models/text-embedding-004
WINE_BOOKS_PATH=/path/to/wine/books

# Optional (for CellarTracker sync)
CELLAR_TRACKER_USERNAME=your_username
CELLAR_TRACKER_PASSWORD=your_password

# Optional (for web search)
TAVILY_API_KEY=your_tavily_api_key
```

The API service also receives these Docker-managed environment variables:

- `CHROMA_HOST=chromadb` - ChromaDB service hostname
- `CHROMA_PORT=8000` - ChromaDB container port
- `OLLAMA_BASE_URL=http://ollama:11434` - Ollama service URL

## Resource Limits

The Ollama service has a memory limit of 6 GB to prevent the container from consuming all available RAM. The Gemma 4 e2b model (Q4_K_M quantization) uses approximately 7.2 GB on disk and ~5-6 GB in RAM during inference.

If you have less than 16 GB of system RAM, consider:
- Closing other applications
- Using the cloud model instead (set `model.provider: google` in `app_config.yml`)
- Switching to the smaller `gemma4:2b` model (edit `ollama-init` entrypoint)

## Useful Commands

```bash
# Start all services
make up

# Stop all services
make down

# Restart services
make restart

# View all logs
make logs

# View specific service logs
make logs-app        # API logs
make logs-chroma     # ChromaDB logs
make logs-ollama     # Ollama logs

# Check service status
make status

# Rebuild images
make build

# Stop, rebuild, and restart
make rebuild

# Access container shells
make shell-app       # API container
make shell-chroma    # ChromaDB container
make shell-ollama    # Ollama container
```

## Volumes

The following Docker volumes persist data across container restarts:

- `ollama-data` - Ollama model files (~7.2 GB for Gemma 4 e2b)

The following host directories are bind-mounted:

- `./cellar-data` - SQLite databases (wine cellar, web cache)
- `./chroma-data` - ChromaDB vector store
- `./app_config.yml` - Application configuration

## Health Checks

All services define health checks:

- **API**: `curl http://localhost:8000/health` (30s interval)
- **Frontend**: `wget http://localhost:3000` (30s interval)
- **ChromaDB**: TCP connection test (30s interval)
- **Ollama**: `curl http://localhost:11434/api/tags` (10s interval)

Services won't be marked as healthy until their health check passes. Dependent services wait for upstream health before starting.

## Troubleshooting

### Ollama fails to start

If Ollama fails to start or the health check times out:

```bash
# Check Ollama logs
docker compose logs ollama

# Restart Ollama service only
docker compose restart ollama
```

### Model download fails

If `ollama-init` fails to pull the model:

```bash
# Check init logs
docker compose logs ollama-init

# Manually pull the model
docker compose exec ollama ollama pull gemma4:e2b

# Or restart the init service
docker compose up ollama-init
```

### Out of memory errors

If you see OOM errors:

1. Check available RAM: `docker stats`
2. Increase Docker's memory limit in Docker Desktop settings
3. Close other applications
4. Consider using the cloud model instead

### ChromaDB connection errors

If the API can't connect to ChromaDB:

```bash
# Verify ChromaDB is healthy
docker compose ps chromadb

# Check ChromaDB logs
docker compose logs chromadb

# Test ChromaDB directly
curl http://localhost:8100/api/v2/heartbeat
```

## Development vs Production

For local development, use the Makefile targets that start services directly on the host (faster iteration):

```bash
make dev-full  # ChromaDB + Ollama + FastAPI + Next.js (host processes)
```

For production deployment, use Docker Compose:

```bash
docker compose up -d  # All services containerized
```

## Updating

To update to the latest version:

```bash
# Pull latest code
git pull

# Rebuild and restart
make rebuild

# Or manually
docker compose down
docker compose build --no-cache
docker compose up -d
```

The Ollama model and ChromaDB data are preserved in volumes and will not be lost during updates.

## Removing Everything

To completely remove all containers, volumes, and data:

```bash
# Stop and remove containers
docker compose down

# Remove all volumes (including model data)
docker compose down -v

# Remove local data directories
rm -rf cellar-data chroma-data
```

**Warning**: This deletes all your wine cellar data and the Ollama model. You'll need to re-download the model (~7.2 GB) and re-import your cellar data.

