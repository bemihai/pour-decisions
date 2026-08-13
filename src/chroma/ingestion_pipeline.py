"""Provider-neutral document ingestion orchestration for ChromaDB."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from src.utils import generate_hash, logger

from .chunking import ChunkCandidate, ChunkerRegistry, DocumentChunker
from .contextual_text import build_contextual_search_text
from .extraction import DocumentElement, ExtractorRegistry
from .metadata_extractor import extract_wine_metadata


@dataclass
class ChunkMetadata:
    """Flat Chroma metadata assembled from one normalized chunk candidate."""

    filename: str
    file_path: str
    file_type: str
    chunk_index: int
    chunk_id: str
    content_hash: str
    page_number: int = -1
    language: str = "unknown"
    category: str = "unknown"
    topic: str = "unknown"
    summary: str = "none"
    word_count: int = 0
    char_count: int = 0
    document_title: str = ""
    chapter: str = ""
    section: str = ""
    extraction_provider: str = ""
    chunking_strategy: str = ""
    heading_path: str = ""
    structural_role: str = "unknown"
    entry_title: str = ""
    column_id: int | str = -1
    start_block_id: int = -1
    end_block_id: int = -1
    start_page: int = -1
    end_page: int = -1
    layout_audit_required: bool = False
    reading_order_confidence: float = 1.0
    grapes: str = ""
    regions: str = ""
    vintages: str = ""
    classifications: str = ""
    producers: str = ""
    appellations: str = ""


class DocumentExtractionPipeline:
    """Resolve and run one configured source-format extractor."""

    def __init__(self, extraction_config: Mapping[str, Any] | Any) -> None:
        """Store the explicit ``chroma.extraction`` configuration."""
        self._extraction_config = extraction_config

    def extract(self, path: Path) -> list[DocumentElement]:
        """Extract normalized elements, skipping unsupported formats when configured."""
        source_path = Path(path)
        extractor = ExtractorRegistry.resolve_from_config(source_path.suffix, self._extraction_config)
        if extractor is None:
            logger.warning("Skipping unsupported document type: %s", source_path)
            return []
        return [
            element
            if element.document_title
            else replace(element, document_title=source_path.stem)
            for element in extractor.extract(source_path)
        ]


class DocumentChunkingPipeline:
    """Resolve and run one configured provider-neutral chunker."""

    def __init__(self, chunking_config: Mapping[str, Any] | Any) -> None:
        """Resolve the strategy once for repeated document processing."""
        self._chunker: DocumentChunker = ChunkerRegistry.resolve_from_config(chunking_config)

    def chunk(self, elements: list[DocumentElement]) -> list[ChunkCandidate]:
        """Build normalized candidates from extracted elements."""
        if not elements:
            return []
        return self._chunker.chunk(elements)


def assemble_chroma_chunks(
    candidates: list[ChunkCandidate],
    *,
    extract_metadata: bool = True,
) -> list[dict[str, Any]]:
    """Convert candidates to the stable dictionary contract consumed by the loader."""
    chunks: list[dict[str, Any]] = []
    for candidate in candidates:
        content_hash = generate_hash(candidate.text)
        source_path = Path(candidate.source_path)
        chunk_id = f"{source_path.stem}_{candidate.chunk_index}_{content_hash[:8]}"
        contextual_text = build_contextual_search_text(
            candidate.text,
            {
                "document_title": candidate.document_title,
                "chapter": candidate.chapter,
                "entry_title": candidate.metadata.get("entry_title", ""),
                "section": candidate.section,
                "structural_role": candidate.structural_role,
            },
        )
        wine_metadata = extract_wine_metadata(contextual_text) if extract_metadata else None
        metadata = ChunkMetadata(
            filename=source_path.name,
            file_path=candidate.source_path,
            file_type=source_path.suffix.lower() or f".{candidate.file_type.lstrip('.')}",
            chunk_index=candidate.chunk_index,
            chunk_id=chunk_id,
            content_hash=content_hash,
            page_number=_page_or_default(candidate.page_number),
            word_count=len(candidate.text.split()),
            char_count=len(candidate.text),
            document_title=candidate.document_title,
            chapter=candidate.chapter,
            section=candidate.section,
            extraction_provider=candidate.extraction_provider,
            chunking_strategy=candidate.chunking_strategy,
            heading_path=candidate.heading_path,
            structural_role=candidate.structural_role,
            entry_title=str(candidate.metadata.get("entry_title", "")),
            column_id=candidate.metadata.get("column_id", -1),
            start_block_id=int(candidate.metadata.get("start_block_id", -1)),
            end_block_id=int(candidate.metadata.get("end_block_id", -1)),
            start_page=_page_or_default(candidate.start_page),
            end_page=_page_or_default(candidate.end_page),
            layout_audit_required=bool(candidate.metadata.get("layout_audit_required", False)),
            reading_order_confidence=float(candidate.metadata.get("reading_order_confidence", 1.0)),
            grapes=_join_metadata_values(wine_metadata.grapes) if wine_metadata else "",
            regions=_join_metadata_values(wine_metadata.regions) if wine_metadata else "",
            vintages=_join_metadata_values(wine_metadata.vintages) if wine_metadata else "",
            classifications=_join_metadata_values(wine_metadata.classifications) if wine_metadata else "",
            producers=_join_metadata_values(wine_metadata.producers) if wine_metadata else "",
            appellations=_join_metadata_values(wine_metadata.appellations) if wine_metadata else "",
        )
        chunks.append(
            {
                "id": chunk_id,
                "text": candidate.text,
                "metadata": asdict(metadata),
                "importance_score": 1.0,
            }
        )
    return chunks


def _page_or_default(page_number: int | None) -> int:
    """Convert optional page lineage to the legacy Chroma sentinel."""
    return page_number if page_number is not None else -1


def _join_metadata_values(values: set[str]) -> str:
    """Serialize extracted wine values deterministically for Chroma metadata."""
    return ",".join(sorted(values))
