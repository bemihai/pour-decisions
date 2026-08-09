"""Explicit resolution of reviewed document chunking strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import DocumentChunker
from .section_recursive import SectionRecursiveChunker
from .section_semantic import SectionSemanticChunker


class ChunkerRegistry:
    """Construct only chunking strategies reviewed for Milestone 3."""

    @classmethod
    def resolve(cls, strategy: str, chunking_config: Mapping[str, Any] | Any) -> DocumentChunker:
        """Construct one chunker from an explicit strategy and configuration."""
        normalized_strategy = _normalize_strategy(strategy)
        if normalized_strategy == "section_recursive":
            return SectionRecursiveChunker(
                chunk_size=int(_config_value(chunking_config, "chunk_size", 1024)),
                chunk_overlap=int(_config_value(chunking_config, "chunk_overlap", 256)),
                min_chunk_chars=int(_config_value(chunking_config, "min_chunk_chars", 200)),
            )
        if normalized_strategy == "section_semantic":
            return SectionSemanticChunker.from_config(chunking_config)
        raise ValueError(f"Unsupported chunking strategy: {strategy!r}")

    @classmethod
    def resolve_from_config(cls, chunking_config: Mapping[str, Any] | Any) -> DocumentChunker:
        """Resolve the configured strategy from ``chroma.chunking``."""
        strategy = _config_value(chunking_config, "strategy", None)
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("chroma.chunking.strategy must be a non-empty string")
        return cls.resolve(strategy, chunking_config)


def _normalize_strategy(strategy: str) -> str:
    """Normalize and validate one strategy name."""
    if not isinstance(strategy, str):
        raise TypeError("strategy must be a string")
    normalized_strategy = strategy.strip().casefold()
    if not normalized_strategy:
        raise ValueError("strategy must not be empty")
    return normalized_strategy


def _config_value(config: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    """Read one value from a mapping or OmegaConf-like config object."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)
