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

LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

# CellarTracker credentials
CELLAR_TRACKER_USERNAME = os.environ.get("CELLAR_TRACKER_USERNAME", "")
CELLAR_TRACKER_PASSWORD = os.environ.get("CELLAR_TRACKER_PASSWORD", "")

# Ollama configuration (defaults set for local development)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_MEMORY_LIMIT = os.environ.get("OLLAMA_MEMORY_LIMIT", "3G")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

