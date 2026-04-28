"""Shared utilities re-exported for convenient access.

Usage::

    from src.utils import logger, get_config, get_embedder
"""

from .logger import logger
from .utils import *
from .tracing import (
    compute_equivalent_cost,
    get_trace_context,
    get_tracing_callbacks,
    init_observability,
    set_span_attributes,
    start_request_span,
)
from .resources import get_embedder
from .terms import *