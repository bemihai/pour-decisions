"""Chunk ID lookup utility for golden dataset authoring.

This script helps populate ``ground_truth_chunk_ids`` for ``rag_only`` samples by
querying the existing ChromaDB index locally and reporting candidate chunk IDs.

It is intended as a one-off developer tool and makes no LLM calls.

Usage::

    python -m src.eval.scripts.chunk_id_lookup --question "What is the minimum aging for Barolo?"
    python -m src.eval.scripts.chunk_id_lookup --question "What is terroir?" --top-k 8
"""

import argparse
import json
import sys
from typing import Any, Literal

from src.retrieval import (
    HybridRetriever,
    build_reranker_from_config,
    build_retrieval_query_plan,
    build_retriever_from_config,
    execute_production_rag,
)
from src.utils import get_config, logger


RetrievalMode = Literal["hybrid", "vector", "bm25"]


def _format_candidate(
    doc: dict[str, Any],
    rank: int,
    preview_chars: int | None = 180,
) -> dict[str, Any]:
    """Format a retrieved document into a candidate payload.

    Args:
        doc: Retrieval result dictionary produced by ``ChromaRetriever.retrieve``.
        rank: 1-based ranking position.
        preview_chars: Maximum preview length. ``None`` preserves the complete
            chunk text for interactive review.

    Returns:
        Dict containing rank, chunk ID, similarity, selected metadata fields,
        and either a bounded preview or the complete chunk text.
    """
    metadata = doc.get("metadata") or {}
    text = doc.get("document") or ""

    preview = text if preview_chars is None else text[:preview_chars].replace("\n", " ")

    return {
        "rank": rank,
        "chunk_id": doc.get("id"),
        "similarity": doc.get("similarity"),
        "source": metadata.get("source") or metadata.get("file_path") or metadata.get("filename"),
        "title": metadata.get("title") or metadata.get("document_title"),
        "chunk_index": metadata.get("chunk_index"),
        "retrieval_channels": list(doc.get("retrieval_channels", []) or []),
        "dense_rank": doc.get("dense_rank"),
        "sparse_rank": doc.get("sparse_rank"),
        "dense_similarity": doc.get("dense_similarity"),
        "bm25_score": doc.get("bm25_score"),
        "rrf_score": doc.get("rrf_score"),
        "rerank_score": doc.get("rerank_score"),
        "metadata_matches": doc.get("metadata_matches"),
        "structural_role": metadata.get("structural_role"),
        "heading_path": metadata.get("heading_path"),
        "retrieval_diagnostics": dict(doc.get("retrieval_diagnostics", {}) or {}),
        "preview": preview,
    }


def lookup_chunk_ids(
    question: str,
    top_k: int,
    collection_name: str | None = None,
    preview_chars: int | None = 180,
    retrieval_mode: RetrievalMode = "hybrid",
) -> list[dict[str, Any]]:
    """Retrieve candidate chunk IDs for a natural-language question.

    Args:
        question: Question text used as the retrieval query.
        top_k: Number of candidate chunks to return.
        collection_name: Optional Chroma collection override.
        preview_chars: Maximum characters included in each candidate preview.
            ``None`` returns the complete chunk text.
        retrieval_mode: Shared production hybrid retrieval or an explicit
            single-channel diagnostic mode.

    Returns:
        List of candidate dictionaries sorted by retrieval rank.

    Raises:
        ValueError: If ``question`` is blank or ``top_k`` is non-positive.
    """
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if retrieval_mode not in {"hybrid", "vector", "bm25"}:
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode!r}")

    config = get_config()
    retriever = build_retriever_from_config(
        config,
        collection_name=collection_name,
        enable_cache=False,
        enable_query_expansion=False,
    )
    plan = build_retrieval_query_plan(question)

    if retrieval_mode == "hybrid":
        if not isinstance(retriever, HybridRetriever):
            raise RuntimeError(
                "Hybrid lookup requested, but synchronized hybrid retrieval is unavailable. "
                "Use --mode vector for an explicit dense-only diagnostic."
            )
        result = execute_production_rag(
            prompt=question,
            config=config,
            model=None,
            retriever=retriever,
            reranker=build_reranker_from_config(config),
            message_history=[],
            n_results_override=top_k,
            generation_enabled=False,
        )
        if result.retrieval_error:
            raise RuntimeError(result.retrieval_error)
        docs = [_artifact_document(artifact) for artifact in result.context_chunks[:top_k]]
    elif retrieval_mode == "vector":
        vector_retriever = retriever.vector_retriever if isinstance(retriever, HybridRetriever) else retriever
        docs = vector_retriever.retrieve(plan.semantic_query, n_results=top_k)
        docs = [
            {
                **document,
                "dense_rank": rank,
                "dense_similarity": document.get("similarity"),
                "retrieval_channels": ["dense"],
            }
            for rank, document in enumerate(docs, start=1)
        ]
    else:
        if not isinstance(retriever, HybridRetriever):
            raise RuntimeError("BM25 lookup requested, but a synchronized BM25 index is unavailable")
        docs = retriever.bm25_index.search(plan.sparse_query, top_k=top_k)
        docs = [
            {
                **document,
                "sparse_rank": rank,
                "retrieval_channels": ["sparse"],
            }
            for rank, document in enumerate(docs, start=1)
        ]

    return [
        _format_candidate(doc=doc, rank=index, preview_chars=preview_chars)
        for index, doc in enumerate(docs, start=1)
    ]


def _artifact_document(artifact: Any) -> dict[str, Any]:
    """Convert a production artifact back to the lookup formatter shape."""
    return {
        "id": artifact.id,
        "document": artifact.text,
        "metadata": artifact.metadata,
        "similarity": artifact.similarity,
        "rerank_score": artifact.rerank_score,
        "rrf_score": artifact.rrf_score,
        "dense_rank": artifact.dense_rank,
        "sparse_rank": artifact.sparse_rank,
        "dense_similarity": artifact.dense_similarity,
        "bm25_score": artifact.bm25_score,
        "metadata_matches": artifact.metadata_matches,
        "retrieval_channels": artifact.retrieval_channels,
        "retrieval_diagnostics": artifact.retrieval_diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for chunk ID lookup."""
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
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Include complete chunk text instead of the compact 180-character preview",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "vector", "bm25"),
        default="hybrid",
        help="Retrieval mode (default: production hybrid)",
    )
    return parser


def main() -> int:
    """Run the chunk ID lookup CLI.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    args = build_parser().parse_args()

    try:
        candidates = lookup_chunk_ids(
            question=args.question,
            top_k=args.top_k,
            collection_name=args.collection,
            preview_chars=None if args.full_text else 180,
            retrieval_mode=args.mode,
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
