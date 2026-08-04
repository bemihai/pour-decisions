"""ChromaDB statistics and diagnostics script.

Displays detailed statistics about ChromaDB collections including:
- Number of records per collection
- Embedding dimensions
- Metadata field distribution
- Storage information
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.chroma.bm25_builder import compute_chunk_ids_sha256, read_collection_documents
from src.utils import get_config, initialize_chroma_client

from .utils import get_all_stats, get_collection_stats

NEAR_EMPTY_THRESHOLD_CHARS = 200
SOURCE_METADATA_FIELDS = ("file_path", "filename", "document_title", "source")


def get_exact_collection_stats(
    client: Any,
    collection_name: str,
    *,
    batch_size: int = 2500,
) -> dict[str, Any]:
    """Calculate exact corpus diagnostics from every record in bounded batches.

    Args:
        client: ChromaDB client exposing ``get_collection()``.
        collection_name: Collection to inspect.
        batch_size: Maximum records requested from Chroma per call.

    Returns:
        Exact collection statistics, or an error payload when the collection
        cannot be read consistently.
    """
    try:
        collection = client.get_collection(collection_name)
        records = read_collection_documents(collection, batch_size=batch_size)
    except Exception as exc:
        return {
            "name": collection_name,
            "statistics_mode": "exact",
            "error": str(exc),
        }

    record_count = len(records)
    document_lengths = [len(record["document"]) for record in records]
    empty_document_count = sum(length == 0 for length in document_lengths)
    near_empty_document_count = sum(length < NEAR_EMPTY_THRESHOLD_CHARS for length in document_lengths)
    source_counts: Counter[str] = Counter()
    records_missing_source = 0
    for record in records:
        source_identifier = _source_identifier(record["metadata"])
        if source_identifier is None:
            records_missing_source += 1
        else:
            source_counts[source_identifier] += 1

    chunks_per_source = list(source_counts.values())
    stats: dict[str, Any] = {
        "name": collection_name,
        "statistics_mode": "exact",
        "record_count": record_count,
        "metadata": dict(collection.metadata) if collection.metadata else {},
        "avg_document_length": _average(document_lengths),
        "min_document_length": min(document_lengths, default=0),
        "max_document_length": max(document_lengths, default=0),
        "empty_document_count": empty_document_count,
        "empty_document_rate": _rate(empty_document_count, record_count),
        "near_empty_threshold_chars": NEAR_EMPTY_THRESHOLD_CHARS,
        "near_empty_includes_empty": True,
        "near_empty_document_count": near_empty_document_count,
        "near_empty_document_rate": _rate(near_empty_document_count, record_count),
        "source_document_count": len(source_counts),
        "records_missing_source": records_missing_source,
        "min_chunks_per_source": min(chunks_per_source, default=0),
        "avg_chunks_per_source": _average(chunks_per_source),
        "max_chunks_per_source": max(chunks_per_source, default=0),
        "chunk_ids_sha256": compute_chunk_ids_sha256(record["id"] for record in records),
    }
    return stats


def _source_identifier(metadata: dict[str, Any]) -> str | None:
    """Resolve one stable source-document identifier from chunk metadata."""
    for field_name in SOURCE_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _average(values: list[int]) -> float:
    """Return an exact arithmetic mean, or zero for an empty input."""
    return sum(values) / len(values) if values else 0.0


def _rate(count: int, total: int) -> float:
    """Return a zero-safe fraction."""
    return count / total if total else 0.0


def _write_json_artifact(stats: list[dict[str, Any]], output_path: Path) -> None:
    """Write machine-readable statistics to the requested artifact path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, default=str) + "\n", encoding="utf-8")


def print_stats(stats: dict[str, Any]) -> None:
    """Print formatted statistics for a collection."""
    print(f"\n{'='*60}")
    print(f"Collection: {stats['name']}")
    print(f"{'='*60}")

    if "error" in stats:
        print(f"  Error: {stats['error']}")
        return

    print(f"  Statistics Mode: {stats.get('statistics_mode', 'sampled')}")
    print(f"  Records: {stats['record_count']:,}")
    print(f"  Embedding Dimension: {stats.get('embedding_dimension', 'N/A')}")

    if "avg_document_length" in stats:
        print(f"\n  Document Length (chars):")
        average_length = stats["avg_document_length"]
        average_text = f"{average_length:,.2f}" if isinstance(average_length, float) else f"{average_length:,}"
        print(f"    Average: {average_text}")
        print(f"    Min: {stats['min_document_length']:,}")
        print(f"    Max: {stats['max_document_length']:,}")

    if stats.get("statistics_mode") == "exact":
        print("\n  Corpus Quality:")
        print(f"    Empty: {stats['empty_document_count']:,} ({stats['empty_document_rate']:.6f})")
        print(
            f"    Near-empty (<{stats['near_empty_threshold_chars']} chars, includes empty): "
            f"{stats['near_empty_document_count']:,} ({stats['near_empty_document_rate']:.6f})"
        )
        print("\n  Source Distribution:")
        print(f"    Source documents: {stats['source_document_count']:,}")
        print(f"    Records missing source: {stats['records_missing_source']:,}")
        print(
            "    Chunks per source (min/avg/max): "
            f"{stats['min_chunks_per_source']:,}/"
            f"{stats['avg_chunks_per_source']:.2f}/"
            f"{stats['max_chunks_per_source']:,}"
        )
        print(f"    Sorted chunk-ID SHA-256: {stats['chunk_ids_sha256']}")

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
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Read every record in bounded batches instead of sampling up to 100 records",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum records per Chroma request in exact mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON payload to this path",
    )

    return parser.parse_args()


def main() -> int:
    """CLI entry point for displaying ChromaDB collection statistics."""

    args = parse_args()
    cfg = get_config()
    chroma_cfg = cfg.chroma

    try:
        client = initialize_chroma_client(chroma_cfg.client.host, chroma_cfg.client.port)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print(f"Make sure ChromaDB is running on {chroma_cfg.client.host}:{chroma_cfg.client.port}")
        return 1

    if args.exact:
        collection_names = (
            [args.collection]
            if args.collection
            else [str(collection.name) for collection in chroma_cfg.collections]
        )
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else int(chroma_cfg.settings.batch_size)
        )
        all_stats = [
            get_exact_collection_stats(client, collection_name, batch_size=batch_size)
            for collection_name in collection_names
        ]
    elif args.collection:
        all_stats = [get_collection_stats(client, args.collection)]
    else:
        all_stats = get_all_stats(client)

    if not all_stats:
        print("No collections found in ChromaDB")
        return 1

    if args.output:
        _write_json_artifact(all_stats, args.output)

    if args.json:
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

        if args.output:
            print(f"\nSaved JSON statistics to {args.output}")
        print("\n" + "="*60)

    return 1 if any("error" in stats for stats in all_stats) else 0


if __name__ == "__main__":
    sys.exit(main())
