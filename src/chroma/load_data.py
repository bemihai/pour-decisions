"""Runnable script for processing external data and loading to ChromaDB.

Usage:
    python -m src.chroma.load_data                    # Incremental mode (default)
    python -m src.chroma.load_data --force            # Force reindex all files
    python -m src.chroma.load_data --status           # Show index status only
"""
import os
import argparse

from src.chroma.index_tracker import IndexTracker
from src.chroma.loader import CollectionDataLoader
from src.utils import get_config, logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def show_index_status(collection_name: str) -> None:
    """Display current index status for a collection."""
    tracker = IndexTracker(collection_name=collection_name)
    stats = tracker.get_stats()

    print(f"\n📊 Index Status for '{collection_name}':")
    print(f"   Files indexed: {stats['total_files']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Last updated: {stats['last_updated']}")

    if stats['total_files'] > 0:
        print(f"\n   Indexed files:")
        for file_path in sorted(tracker.get_indexed_files()):
            info = tracker.manifest.files[file_path]
            print(f"   - {info.file_path.split('/')[-1]} ({info.chunk_count} chunks)")


def main():
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

        loader = CollectionDataLoader(
            collection_name=collection.name,
            collection_metadata=collection.metadata,
            chroma_host=chroma_cfg.client.host,
            chroma_port=chroma_cfg.client.port,
            embedding_model=chroma_cfg.settings.embedder,
            batch_size=chroma_cfg.settings.batch_size,
        )

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

        print(f"\n✅ Collection '{collection.name}' processing complete:")
        print(f"   Total files: {stats.get('total_files', 0)}")
        print(f"   Files processed: {stats.get('files_processed', 0)}")
        print(f"   Files skipped (already indexed): {stats.get('files_skipped', 0)}")
        print(f"   Chunks added: {stats.get('total_chunks_added', 0)}")


if __name__ == "__main__":
    main()
