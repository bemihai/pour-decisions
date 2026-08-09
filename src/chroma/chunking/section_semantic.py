"""Optional semantic chunking constrained to document section boundaries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from langchain_experimental.text_splitter import SemanticChunker

from src.chroma.extraction.base import DocumentElement
from src.utils import get_embedder, logger

from .base import ChunkCandidate, DocumentChunker
from .section_recursive import (
    SectionRecursiveChunker,
    _build_candidate,
    _build_section_text,
    _group_elements,
)


_BREAKPOINT_THRESHOLD_TYPES = {"gradient", "interquartile", "percentile", "standard_deviation"}


class SectionSemanticChunker(DocumentChunker):
    """Split prose semantically inside sections using the cached local embedder."""

    strategy = "section_semantic"

    def __init__(
        self,
        *,
        enabled: bool = False,
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
        min_chunk_chars: int = 200,
        embedding_model: str | None = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
    ) -> None:
        """Configure opt-in semantic splitting and deterministic fallback."""
        normalized_threshold_type = breakpoint_threshold_type.strip().casefold()
        if normalized_threshold_type not in _BREAKPOINT_THRESHOLD_TYPES:
            raise ValueError(f"Unsupported semantic breakpoint threshold type: {breakpoint_threshold_type!r}")
        if not isinstance(breakpoint_threshold_amount, (int, float)):
            raise TypeError("breakpoint_threshold_amount must be numeric")

        self._enabled = enabled
        self._embedding_model = embedding_model
        self._breakpoint_threshold_type = normalized_threshold_type
        self._breakpoint_threshold_amount = float(breakpoint_threshold_amount)
        self._min_chunk_chars = min_chunk_chars
        self._recursive_fallback = SectionRecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )

    @classmethod
    def from_config(cls, chunking_config: Mapping[str, Any] | Any) -> SectionSemanticChunker:
        """Construct the optional strategy from reviewed chunking settings."""
        semantic_config = _config_value(chunking_config, "semantic", {})
        return cls(
            enabled=bool(_config_value(semantic_config, "enabled", False)),
            chunk_size=int(_config_value(chunking_config, "chunk_size", 1024)),
            chunk_overlap=int(_config_value(chunking_config, "chunk_overlap", 256)),
            min_chunk_chars=int(_config_value(chunking_config, "min_chunk_chars", 200)),
            embedding_model=_config_value(semantic_config, "embedding_model", None),
            breakpoint_threshold_type=str(
                _config_value(semantic_config, "breakpoint_threshold_type", "percentile")
            ),
            breakpoint_threshold_amount=float(
                _config_value(semantic_config, "breakpoint_threshold_amount", 95.0)
            ),
        )

    def chunk(self, elements: list[DocumentElement]) -> list[ChunkCandidate]:
        """Build semantic candidates, or deterministic candidates when disabled."""
        if not self._enabled:
            return self._recursive_fallback.chunk(elements)

        try:
            embedder = get_embedder(self._embedding_model)
            semantic_splitter = SemanticChunker(
                embeddings=embedder,
                add_start_index=False,
                breakpoint_threshold_type=self._breakpoint_threshold_type,
                breakpoint_threshold_amount=self._breakpoint_threshold_amount,
            )
        except Exception as error:
            logger.warning("Semantic chunking unavailable; using section_recursive: %s", error)
            return self._recursive_fallback.chunk(elements)

        candidates: list[ChunkCandidate] = []
        for group in _group_elements(elements):
            section_text, spans = _build_section_text(group.elements)
            try:
                semantic_documents = semantic_splitter.create_documents([section_text])
            except Exception as error:
                logger.warning(
                    "Semantic chunking failed for section '%s'; using section_recursive: %s",
                    group.heading_path,
                    error,
                )
                _append_reindexed(candidates, self._recursive_fallback.chunk(group.elements))
                continue

            search_start = 0
            for document in semantic_documents:
                chunk_text = document.page_content.strip()
                if not chunk_text:
                    continue
                start, end = _locate_chunk_range(section_text, chunk_text, search_start)
                candidate = _build_candidate(
                    group=group,
                    spans=spans,
                    text=chunk_text,
                    start=start,
                    end=end,
                    chunk_index=len(candidates),
                    strategy=self.strategy,
                    min_chunk_chars=self._min_chunk_chars,
                )
                if candidate is not None:
                    candidates.append(candidate)
                search_start = end
        return candidates


def _locate_chunk_range(section_text: str, chunk_text: str, search_start: int) -> tuple[int, int]:
    """Map whitespace-normalized semantic output back to source character offsets."""
    pattern = r"\s+".join(re.escape(part) for part in chunk_text.split())
    match = re.search(pattern, section_text[search_start:])
    if match is None:
        match = re.search(pattern, section_text)
        if match is None:
            return 0, len(section_text)
        return match.start(), match.end()
    return search_start + match.start(), search_start + match.end()


def _append_reindexed(target: list[ChunkCandidate], candidates: list[ChunkCandidate]) -> None:
    """Append fallback candidates while keeping file-level indexes contiguous."""
    for candidate in candidates:
        target.append(replace(candidate, chunk_index=len(target)))


def _config_value(config: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    """Read one value from a mapping or OmegaConf-like config object."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)
