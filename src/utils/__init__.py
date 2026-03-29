"""Shared utilities re-exported for convenient access.

Usage::

    from src.utils import logger, get_config, get_embedder
"""

from .logger import logger
from .utils import *
from .tracing import get_langfuse_callback
from .resources import get_embedder
from .terms import *