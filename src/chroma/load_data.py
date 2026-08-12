"""Runnable script for processing external data and loading to ChromaDB.

Usage:
    python -m src.chroma.load_data                    # Incremental mode (default)
    python -m src.chroma.load_data --force            # Force reindex all files
    python -m src.chroma.load_data --status           # Show index status only
"""
import argparse
import os
from pathlib import Path
from typing import Any

from src.chroma.bm25_builder import rebuild_bm25_from_collection
from src.chroma.index_tracker import IndexTracker
from src.chroma.loader import CollectionDataLoader
from src.utils import get_config, logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def show_index_status(collection_name: str) -> None:
    """Display current index status for a collection."""
    tracker = IndexTracker(collection_name=collection_name)
    stats = tracker.get_stats()

    print(f"\nIndex Status for '{collection_name}':")
    print(f"   Files indexed: {stats['total_files']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Last updated: {stats['last_updated']}")

    if stats['total_files'] > 0:
        print(f"\n   Indexed files:")
        for file_path in sorted(tracker.get_indexed_files()):
            info = tracker.manifest.files[file_path]
            print(f"   - {info.file_path.split('/')[-1]} ({info.chunk_count} chunks)")


def main() -> None:
    """CLI entry point for indexing wine books into ChromaDB."""
    parser = argparse.ArgumentParser(description="Load wine data into ChromaDB")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reindex all files, ignoring existing index"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show index status without processing"
    )
    args = parser.parse_args()

    cfg = get_config()
    chroma_cfg = cfg.chroma

    for collection in chroma_cfg.collections:
        if args.status:
            show_index_status(collection.name)
            continue

        logger.info(f"Loading collection '{collection.name}' to ChromaDB")

        if args.force:
            logger.warning("Force reindex mode: all files will be reprocessed")
            _validate_force_reindex_source(
                data_path=collection.local_data_path,
                file_extensions=[".epub", ".pdf"],
            )

        loader = CollectionDataLoader(
            collection_name=collection.name,
            collection_metadata=collection.metadata,
            chroma_host=chroma_cfg.client.host,
            chroma_port=chroma_cfg.client.port,
            embedding_model=chroma_cfg.settings.embedder,
            batch_size=chroma_cfg.settings.batch_size,
            extraction_config=chroma_cfg.extraction,
            chunking_config=chroma_cfg.chunking,
            indexing_config=chroma_cfg.indexing,
        )
        if args.force:
            loader.reset_collection()

        extract_wine_metadata = getattr(chroma_cfg.chunking, "extract_wine_metadata", True)

        stats = loader.load_directory(
            file_extensions=[".epub", ".pdf"],
            data_path=collection.local_data_path,
            strategy=chroma_cfg.chunking.strategy,
            chunk_size=chroma_cfg.chunking.chunk_size,
            overlap_size=chroma_cfg.chunking.chunk_overlap,
            extract_metadata=extract_wine_metadata,
            incremental=True,
            force_reindex=args.force,
        )

        if args.force:
            _raise_for_indexing_errors(collection.name, stats)
            _rebuild_bm25_if_enabled(
                chroma_cfg=chroma_cfg,
                collection_name=collection.name,
                collection=loader.collection,
            )

        print(f"\nCollection '{collection.name}' processing complete:")
        print(f"   Total files: {stats.get('total_files', 0)}")
        print(f"   Files processed: {stats.get('files_processed', 0)}")
        print(f"   Files skipped (already indexed): {stats.get('files_skipped', 0)}")
        print(f"   Chunks added: {stats.get('total_chunks_added', 0)}")


def _raise_for_indexing_errors(collection_name: str, stats: dict[str, Any]) -> None:
    """Fail a forced reindex before BM25 replacement when Chroma indexing failed."""
    errors = list(stats.get("errors") or [])
    if errors:
        raise RuntimeError(
            f"Forced Chroma reindex failed for collection {collection_name!r} with {len(errors)} error(s)"
        )


def _validate_force_reindex_source(*, data_path: str | Path, file_extensions: list[str]) -> None:
    """Refuse to reset Chroma unless the reindex source is available and non-empty."""
    data_directory = Path(data_path)
    if not data_directory.is_dir():
        raise ValueError(f"Data directory {data_path} does not exist or is not a directory")

    has_supported_file = any(
        path.is_file()
        for extension in file_extensions
        for path in data_directory.glob(f"**/*{extension}")
    )
    if not has_supported_file:
        raise ValueError(f"No files found with extensions {file_extensions} in {data_path}")


def _rebuild_bm25_if_enabled(*, chroma_cfg: Any, collection_name: str, collection: Any) -> None:
    """Rebuild BM25 after a successful forced Chroma reindex when configured."""
    indexing_cfg = getattr(chroma_cfg, "indexing", None)
    bm25_cfg = getattr(indexing_cfg, "bm25", None)
    if not bool(getattr(bm25_cfg, "rebuild_on_reindex", False)):
        logger.info("BM25 rebuild after forced reindex is disabled")
        return

    rebuild_bm25_from_collection(
        collection=collection,
        collection_name=collection_name,
        index_path=str(chroma_cfg.retrieval.bm25_index_path),
        manifest_path=str(bm25_cfg.sync_manifest_path),
        batch_size=int(chroma_cfg.settings.batch_size),
    )


if __name__ == "__main__":
    main()
