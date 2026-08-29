# Ollama Model Configuration Guide

> **Project version**: 0.8.2 — last verified 2026-08-29.
> Model selection and config paths are stable for now. Milestone 13 (local LLM routing and user
> memory) may alter how models are chosen at runtime.

## Overview

Pour Decisions supports multiple Ollama models for local LLM inference. The default is `gemma3:4b`
(3.3 GB RAM), which provides a good balance of speed and quality for local development and testing.
The model is configured via the `OLLAMA_MODEL` env var (or `model.ollama.name` in `app_config.yml`).

## Recommended Models

| Model | Disk Size | RAM Required | Speed (CPU) | Quality | Tool Calling | Use Case |
|-------|-----------|--------------|-------------|---------|--------------|----------|
| `gemma3:4b` | 3.3 GB | 3.3 GB | Fast | Good | No | **Local dev/testing (RECOMMENDED)** — use with `hybrid_tool_calling: true` |
| `gemma2:2b` | 1.6 GB | 1.6 GB | Very Fast | Good | No | RAM-constrained machines — use with `hybrid_tool_calling: true` |
| `phi3:mini` | 2.3 GB | 2.3 GB | Fast | Very Good | Yes | Good balance for production |
| `llama3.2:3b` | 2.0 GB | 2.0 GB | Fast | Excellent | Yes | Best quality for size |
| `gemma4:e2b` | 7.2 GB | 5-6 GB | Slow | Best | Yes | Extended thinking, production |

**Default**: `gemma3:4b` (requires 8+ GB total system RAM recommended)

> **Tool calling note**: `gemma3:4b` and `gemma2:2b` do not support tool calling in Ollama.
> Enable `hybrid_tool_calling: true` in `app_config.yml` if you want the cloud model (Gemini)
> to handle tool selection while the local model generates the final answer.
> For evaluation (Ragas), tool calling is not needed; `gemma3:4b` works correctly as a judge.

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Set in your `.env` file — this is the single source of truth and is interpolated into
`app_config.yml` automatically via `${oc.env:OLLAMA_MODEL, gemma3:4b}`:

```bash
# Model to use (overrides the default in app_config.yml)
OLLAMA_MODEL=gemma3:4b

# Memory limit for Docker container (adjust based on model)
OLLAMA_MEMORY_LIMIT=3G

# Ollama server URL
OLLAMA_BASE_URL=http://localhost:11434
```

No manual edit of `app_config.yml` is needed — `model.ollama.name` reads `OLLAMA_MODEL` at startup.

### Method 2: Direct Config Edit

Edit `app_config.yml` directly to hard-code a model (takes effect only when `OLLAMA_MODEL` is not
set in the environment):

```yaml
model:
  provider: google
  name: gemini-2.5-flash
  ollama:
    name: phi3:mini  # Change to desired local model
    base_url: http://localhost:11434
```

For Docker deployment, also update `.env`:

```bash
OLLAMA_MODEL=phi3:mini
OLLAMA_MEMORY_LIMIT=3G
```

## Local Development Setup

### 1. Pull the Model

```bash
# Start Ollama (if not running)
make ollama-up

# Pull your chosen model
ollama pull gemma3:4b
# OR
ollama pull phi3:mini
# OR
ollama pull llama3.2:3b
```

### 2. Update Configuration

Set `OLLAMA_MODEL` in `.env` (preferred):

```bash
OLLAMA_MODEL=gemma3:4b
```

Or edit `app_config.yml` directly:

```yaml
model:
  provider: ollama
  name: gemma3:4b
```

### 3. Restart API

```bash
# If API is running
make dev-stop
make api
```

## Docker Deployment

### 1. Configure Environment

Edit `.env` — this is sufficient; `app_config.yml` reads the value automatically:

```bash
OLLAMA_MODEL=gemma3:4b
OLLAMA_MEMORY_LIMIT=3G
```

### 2. Deploy Stack

```bash
# Stop services (if running)
make down

# Remove init container to force model re-pull
docker rm pour_decisions_ollama_init

# Start services (will pull the configured model)
make up

# Monitor model download
docker compose logs -f ollama-init
```

### 3. Verify

```bash
# Check Ollama has the model
docker exec pour_decisions_ollama ollama list

# Check API logs
docker compose logs -f api | grep "Loaded Ollama model"
```

## Memory Requirements

Adjust `OLLAMA_MEMORY_LIMIT` based on your model choice:

- `gemma3:4b`, `phi3:mini`, `llama3.2:3b`: **4G** recommended
- `gemma2:2b`: **3G** (default) is sufficient
- `gemma4:e2b`: **6G**
- Larger models (7B+): **8G+**

In `docker-compose.yml`, the limit is set via:

```yaml
ollama:
  deploy:
    resources:
      limits:
        memory: ${OLLAMA_MEMORY_LIMIT:-3G}
```

## Switching Models

### Quick Switch (Local)

```bash
# 1. Pull new model
ollama pull phi3:mini

# 2. Set env var (preferred) or edit app_config.yml
export OLLAMA_MODEL=phi3:mini   # or add to .env

# 3. Restart API
make dev-stop && make api
```

### Quick Switch (Docker)

```bash
# 1. Update .env
# Set: OLLAMA_MODEL=phi3:mini

# 2. Restart services
make down
docker rm pour_decisions_ollama_init
make up
```

## Performance Tips

1. **Small models for development**: Use `gemma2:2b` for very fast iteration on RAM-constrained machines
2. **CPU-only inference**: Even small models take 5-15s per query on CPU
3. **GPU acceleration**: Install Ollama with CUDA/ROCm support for 10x speedup
4. **Hybrid mode**: Use cloud model for tool calling (`hybrid_tool_calling: true`)
5. **Cloud fallback**: Always configure `fallback_provider: google` for reliability

## Troubleshooting

### "Model not found" Error

```bash
# Check available models
ollama list

# Pull the model manually
ollama pull gemma3:4b

# Verify in logs
make logs-ollama
```

### Slow Inference

- **Use smaller model**: Switch to `gemma2:2b` via `OLLAMA_MODEL=gemma2:2b` in `.env`
- **Enable hybrid mode**: Set `hybrid_tool_calling: true` in config
- **Use cloud model**: Set `provider: google` in config

### Out of Memory

```bash
# Reduce memory limit in .env
OLLAMA_MEMORY_LIMIT=2G

# Or use smaller model
OLLAMA_MODEL=gemma2:2b
```

### Docker Init Container Fails

```bash
# Check init logs
docker compose logs ollama-init

# Manually pull in running container
docker exec pour_decisions_ollama ollama pull gemma3:4b

# Remove init container and retry
docker rm pour_decisions_ollama_init
make up
```

## Model Comparison

### Inference Speed (CPU-only, M1 Mac)

| Model | Cold Start | Warm Query | Tool Calling |
|-------|-----------|-----------|--------------|
| `gemma3:4b` | ~12s | ~8s | ~15s |
| `gemma2:2b` | ~8s | ~5s | ~10s |
| `phi3:mini` | ~12s | ~8s | ~15s |
| `llama3.2:3b` | ~10s | ~7s | ~12s |
| `gemma4:e2b` | ~45s | ~30s | ~90s |

### Quality (Subjective, Wine Domain)

| Model | RAG Answers | Tool Calling | Structured Output |
|-------|------------|--------------|-------------------|
| `gemma3:4b` | Very Good | Good | Good |
| `gemma2:2b` | Good | Good | Moderate |
| `phi3:mini` | Very Good | Excellent | Good |
| `llama3.2:3b` | Excellent | Very Good | Very Good |
| `gemma4:e2b` | Best | Best | Best |

## Best Practices

1. **Use OLLAMA_MODEL env var**: Set the local model once in `.env`; `app_config.yml` reads it automatically
2. **Test locally first**: Pull and test the model with `ollama run <model>` before deploying
3. **Monitor logs**: Watch API startup logs to confirm the correct model loaded
4. **Use cloud fallback**: Always configure Gemini as fallback for reliability
5. **Document changes**: Update team documentation when changing production model

## Related Configuration

- `app_config.yml`: Local Ollama model configuration (`model.ollama.name = ${oc.env:OLLAMA_MODEL, gemma3:4b}`)
- `.env`: Environment-specific overrides — single source of truth when set
- `docker-compose.yml`: Container resource limits
- `src/agents/llm.py`: Model loading logic
- `DOCKER.md`: Deployment guide
- `AGENTS.md`: Architecture reference
