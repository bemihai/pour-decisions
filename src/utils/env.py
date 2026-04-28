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

# agents observability
OBSERVABILITY_ENABLED = os.environ.get("OBSERVABILITY_ENABLED", "false").lower() == "true"
OBSERVABILITY_PROVIDER = os.environ.get("OBSERVABILITY_PROVIDER", "phoenix")
PHOENIX_ENDPOINT = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
PHOENIX_ENDPOINT_DOCKER = os.environ.get("PHOENIX_ENDPOINT_DOCKER", "http://phoenix:6006")
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "pour-decisions")
