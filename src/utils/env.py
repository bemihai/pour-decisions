"""Environment variable loading and configuration.

Loads variables from ``.env`` file at import time using ``python-dotenv``.
Exposes API keys and service credentials as module-level constants.
"""

import os
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


load_env()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


# CellarTracker credentials
CELLAR_TRACKER_USERNAME = os.environ.get("CELLAR_TRACKER_USERNAME", "")
CELLAR_TRACKER_PASSWORD = os.environ.get("CELLAR_TRACKER_PASSWORD", "")

# Ollama configuration (defaults set for local development)
# The model name default is intentionally absent here — app_config.yml owns that default
# via model.ollama.name = ${oc.env:OLLAMA_MODEL, gemma3:4b}. This constant is kept for
# direct env-var consumers (e.g. docker-compose health checks) that bypass the config.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_MEMORY_LIMIT = os.environ.get("OLLAMA_MEMORY_LIMIT", "3G")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# agents observability
OBSERVABILITY_ENABLED = os.environ.get("OBSERVABILITY_ENABLED", "false").lower() == "true"
OBSERVABILITY_PROVIDER = os.environ.get("OBSERVABILITY_PROVIDER", "phoenix")
PHOENIX_ENDPOINT = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
PHOENIX_ENDPOINT_DOCKER = os.environ.get("PHOENIX_ENDPOINT_DOCKER", "http://phoenix:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "pour-decisions")
