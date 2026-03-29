"""ChromaDB statistics and diagnostics script.

Displays detailed statistics about ChromaDB collections including:
- Number of records per collection
- Embedding dimensions
- Metadata field distribution
- Storage information
"""
import argparse
import sys
from typing import Any

from .utils import get_collection_stats, get_all_stats
from src.utils import get_config, initialize_chroma_client


def print_stats(stats: dict[str, Any]) -> None:
    """Print formatted statistics for a collection."""
    print(f"\n{'='*60}")
    print(f"Collection: {stats['name']}")
    print(f"{'='*60}")

    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    print(f"  Records: {stats['record_count']:,}")
    print(f"  Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")

    if "avg_document_length" in stats:
        print(f"\n  Document Length (chars):")
        print(f"    Average: {stats['avg_document_length']:,}")
        print(f"    Min: {stats['min_document_length']:,}")
        print(f"    Max: {stats['max_document_length']:,}")

    if stats.get("metadata"):
        print(f"\n  Collection Metadata:")
        for key, value in stats["metadata"].items():
            print(f"    {key}: {value}")

    if stats.get("metadata_fields"):
        print(f"\n  Metadata Fields:")
        for field in sorted(stats["metadata_fields"]):
            print(f"    {field}")

    wine_fields = ["grapes", "regions", "vintages", "appellations", "producers"]
    sample_values = stats.get("metadata_sample_values", {})

    wine_metadata_present = any(f in sample_values for f in wine_fields)
    if wine_metadata_present:
        print(f"\n  Wine Metadata Samples:")
        for field in wine_fields:
            if field in sample_values and sample_values[field]:
                values = list(sample_values[field].keys())[:3]
                if values:
                    print(f"    {field}: {', '.join(values)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Display ChromaDB statistics")
    parser.add_argument(
        "--collection", "-c",
        type=str,
        help="Specific collection name (default: all collections)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON"
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point for displaying ChromaDB collection statistics."""

    args = parse_args()
    cfg = get_config()
    chroma_cfg = cfg.chroma

    try:
        client = initialize_chroma_client(chroma_cfg.client.host, chroma_cfg.client.port)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print(f"Make sure ChromaDB is running on {chroma_cfg.client.host}:{chroma_cfg.client.port}")
        sys.exit()

    if args.collection:
        all_stats = [get_collection_stats(client, args.collection)]
    else:
        all_stats = get_all_stats(client)

    if not all_stats:
        print("No collections found in ChromaDB")
        sys.exit()

    if args.json:
        import json
        print(json.dumps(all_stats, indent=2, default=str))
    else:
        print("\n" + "="*60)
        print("ChromaDB Statistics")
        print(f"Server: {chroma_cfg.client.host}:{chroma_cfg.client.port}")
        print(f"Total Collections: {len(all_stats)}")

        total_records = sum(s.get("record_count", 0) for s in all_stats if "error" not in s)
        print(f"Total Records: {total_records:,}")

        for stats in all_stats:
            print_stats(stats)

        print("\n" + "="*60)


if __name__ == "__main__":
    main()

