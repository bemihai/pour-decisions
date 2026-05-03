"""Chunk ID lookup utility for golden dataset authoring.

This script helps populate ``ground_truth_chunk_ids`` for ``rag_only`` samples by
querying the existing ChromaDB index locally and reporting candidate chunk IDs.

It is intended as a one-off developer tool and makes no LLM calls.

Usage::

    python -m src.eval.chunk_id_lookup --question "What is the minimum aging for Barolo?"
    python -m src.eval.chunk_id_lookup --question "What is terroir?" --top-k 8
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.retrieval.vector_retriever import ChromaRetriever
from src.utils import get_config, initialize_chroma_client, logger


def _build_retriever(collection_name: str | None = None) -> ChromaRetriever:
    """Build a Chroma retriever using project configuration.

    Args:
        collection_name: Optional override for the Chroma collection name. If not
            provided, the first configured collection is used.

    Returns:
        Configured :class:`~src.retrieval.vector_retriever.ChromaRetriever` instance.

    Raises:
        ValueError: If no collection is configured.
        Exception: If Chroma client initialization or collection retrieval fails.
    """
    cfg = get_config()
    host = cfg.chroma.client.host
    port = int(cfg.chroma.client.port)
    embedder = cfg.chroma.settings.embedder

    if collection_name is None:
        if not cfg.chroma.collections:
            raise ValueError("No Chroma collections configured in app_config.yml")
        collection_name = cfg.chroma.collections[0].name

    # Type narrowing for static analysis: collection_name is guaranteed here.
    resolved_collection_name: str = collection_name

    client = initialize_chroma_client(host=host, port=port)
    return ChromaRetriever(
        client=client,
        collection_name=resolved_collection_name,
        embedding_model=embedder,
        n_results=int(cfg.chroma.retrieval.n_results),
        similarity_threshold=float(cfg.chroma.retrieval.similarity_threshold),
        enable_query_expansion=True,
        enable_cache=False,
    )


def _format_candidate(doc: dict[str, Any], rank: int) -> dict[str, Any]:
    """Format a retrieved document into a compact candidate payload.

    Args:
        doc: Retrieval result dictionary produced by ``ChromaRetriever.retrieve``.
        rank: 1-based ranking position.

    Returns:
        Dict containing rank, chunk ID, similarity, selected metadata fields, and
        a short text preview.
    """
    metadata = doc.get("metadata") or {}
    text = doc.get("document") or ""

    return {
        "rank": rank,
        "chunk_id": doc.get("id"),
        "similarity": doc.get("similarity"),
        "source": metadata.get("source"),
        "title": metadata.get("title"),
        "chunk_index": metadata.get("chunk_index"),
        "preview": text[:180].replace("\n", " "),
    }


def lookup_chunk_ids(question: str, top_k: int, collection_name: str | None = None) -> list[dict[str, Any]]:
    """Retrieve candidate chunk IDs for a natural-language question.

    Args:
        question: Question text used as the retrieval query.
        top_k: Number of candidate chunks to return.
        collection_name: Optional Chroma collection override.

    Returns:
        List of candidate dictionaries sorted by retrieval rank.

    Raises:
        ValueError: If ``question`` is blank or ``top_k`` is non-positive.
    """
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    retriever = _build_retriever(collection_name=collection_name)
    docs = retriever.retrieve(question, n_results=top_k)
    return [_format_candidate(doc=doc, rank=index) for index, doc in enumerate(docs, start=1)]


def main() -> int:
    """Run the chunk ID lookup CLI.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Lookup candidate Chroma chunk IDs for a golden dataset question.",
    )
    parser.add_argument("--question", required=True, help="Question text to search for")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates to return (default: 10)")
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional collection override (default: first configured collection)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit compact JSON output for copy/paste into ground_truth_chunk_ids workflows",
    )
    args = parser.parse_args()

    try:
        candidates = lookup_chunk_ids(
            question=args.question,
            top_k=args.top_k,
            collection_name=args.collection,
        )
    except Exception as exc:
        logger.error("Chunk ID lookup failed: %s", exc)
        return 1

    logger.info("Found %d candidate chunks for question: %s", len(candidates), args.question)

    if args.json:
        logger.info(json.dumps(candidates, ensure_ascii=True, indent=2))
        return 0

    for candidate in candidates:
        logger.info(
            "rank=%d id=%s similarity=%.4f source=%s chunk_index=%s",
            candidate["rank"],
            candidate["chunk_id"],
            candidate["similarity"] if candidate["similarity"] is not None else -1.0,
            candidate["source"],
            candidate["chunk_index"],
        )
        logger.info("preview=%s", candidate["preview"])

    return 0


if __name__ == "__main__":
    sys.exit(main())

