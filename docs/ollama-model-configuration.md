# Ollama Model Configuration Guide

## Overview

Pour Decisions supports multiple Ollama models for local LLM inference. The default is `gemma2:2b` (1.6 GB RAM), which provides a good balance of speed and quality for local development and testing.

## Recommended Models

| Model | Disk Size | RAM Required | Speed (CPU) | Quality | Use Case |
|-------|-----------|--------------|-------------|---------|----------|
| `gemma2:2b` | 1.6 GB | 1.6 GB | Very Fast | Good | **Local dev/testing (RECOMMENDED)** |
| `phi3:mini` | 2.3 GB | 2.3 GB | Fast | Very Good | Good balance for production |
| `llama3.2:3b` | 2.0 GB | 2.0 GB | Fast | Excellent | Best quality for size |
| `gemma4:e2b` | 7.2 GB | 5-6 GB | Slow | Best | Extended thinking, production |

**Default**: `gemma2:2b` (optimized for machines with 8+ GB RAM)

## Configuration Methods

### Method 1: Environment Variables (Recommended)

Set in your `.env` file:

```bash
# Model to use
OLLAMA_MODEL=gemma2:2b

# Memory limit for Docker container (adjust based on model)
OLLAMA_MEMORY_LIMIT=3G

# Ollama server URL
OLLAMA_BASE_URL=http://localhost:11434
```

Then update `app_config.yml` to match:

```yaml
model:
  name: gemma2:2b  # Must match OLLAMA_MODEL
```

### Method 2: Direct Config Edit

Edit `app_config.yml` directly:

```yaml
model:
  provider: ollama
  name: phi3:mini  # Change to desired model
  ollama:
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
ollama pull gemma2:2b
# OR
ollama pull phi3:mini
# OR
ollama pull llama3.2:3b
```

### 2. Update Configuration

Edit `app_config.yml`:

```yaml
model:
  provider: ollama
  name: gemma2:2b  # Match the model you pulled
```

### 3. Restart API

```bash
# If API is running
make dev-stop
make api
```

## Docker Deployment

### 1. Configure Environment

Edit `.env`:

```bash
OLLAMA_MODEL=gemma2:2b
OLLAMA_MEMORY_LIMIT=3G
```

Edit `app_config.yml`:

```yaml
model:
  name: gemma2:2b
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

- `gemma2:2b`, `phi3:mini`, `llama3.2:3b`: **3G** (default)
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

# 2. Update config
# Edit app_config.yml: model.name = phi3:mini

# 3. Restart API
make dev-stop && make api
```

### Quick Switch (Docker)

```bash
# 1. Update .env
# Set: OLLAMA_MODEL=phi3:mini

# 2. Update app_config.yml
# Set: model.name = phi3:mini

# 3. Restart services
make down
docker rm pour_decisions_ollama_init
make up
```

## Performance Tips

1. **Small models for development**: Use `gemma2:2b` for fast iteration
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
ollama pull gemma2:2b

# Verify in logs
make logs-ollama
```

### Slow Inference

- **Use smaller model**: Switch to `gemma2:2b`
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
docker exec pour_decisions_ollama ollama pull gemma2:2b

# Remove init container and retry
docker rm pour_decisions_ollama_init
make up
```

## Model Comparison

### Inference Speed (CPU-only, M1 Mac)

| Model | Cold Start | Warm Query | Tool Calling |
|-------|-----------|-----------|--------------|
| `gemma2:2b` | ~8s | ~5s | ~10s |
| `phi3:mini` | ~12s | ~8s | ~15s |
| `llama3.2:3b` | ~10s | ~7s | ~12s |
| `gemma4:e2b` | ~45s | ~30s | ~90s |

### Quality (Subjective, Wine Domain)

| Model | RAG Answers | Tool Calling | Structured Output |
|-------|------------|--------------|-------------------|
| `gemma2:2b` | Good | Good | Moderate |
| `phi3:mini` | Very Good | Excellent | Good |
| `llama3.2:3b` | Excellent | Very Good | Very Good |
| `gemma4:e2b` | Best | Best | Best |

## Best Practices

1. **Match config to environment**: Ensure `app_config.yml` and `.env` have the same model
2. **Test locally first**: Pull and test the model with `ollama run <model>` before deploying
3. **Monitor logs**: Watch API startup logs to confirm the correct model loaded
4. **Use cloud fallback**: Always configure Gemini as fallback for reliability
5. **Document changes**: Update team documentation when changing production model

## Related Configuration

- `app_config.yml`: Primary model configuration
- `.env`: Environment-specific overrides (Docker)
- `docker-compose.yml`: Container resource limits
- `src/agents/llm.py`: Model loading logic
- `DOCKER.md`: Deployment guide
- `AGENTS.md`: Architecture reference

