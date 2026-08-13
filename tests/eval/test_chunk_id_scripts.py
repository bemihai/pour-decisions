"""Tests for chunk-ID lookup modes and resumable interactive curation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.scripts.chunk_id_curator import (
    _default_dataset_path,
    _load_jsonl,
    _print_candidates,
    run_curation,
)
from src.eval.scripts.chunk_id_lookup import _format_candidate, build_parser, lookup_chunk_ids
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rag_service import RAGChunkArtifact


def test_format_candidate_can_preserve_complete_chunk_and_resolve_current_metadata() -> None:
    """Interactive lookup retains full text and uses current ingestion metadata."""
    text = "first line\nsecond line with enough text to exceed a short preview"
    candidate = _format_candidate(
        {
            "id": "book_7_deadbeef",
            "document": text,
            "similarity": 0.75,
            "metadata": {
                "file_path": "/books/example.pdf",
                "document_title": "Example Book",
                "chunk_index": 7,
                "structural_role": "prose",
                "heading_path": "Example > Nebbiolo > taste",
            },
            "dense_rank": 2,
            "sparse_rank": 1,
            "dense_similarity": 0.75,
            "bm25_score": 4.5,
            "retrieval_channels": ["dense", "sparse"],
        },
        rank=1,
        preview_chars=None,
    )

    assert candidate["preview"] == text
    assert candidate["source"] == "/books/example.pdf"
    assert candidate["title"] == "Example Book"
    assert candidate["retrieval_channels"] == ["dense", "sparse"]
    assert candidate["structural_role"] == "prose"


def test_format_candidate_keeps_compact_default() -> None:
    """Standalone lookup output remains bounded by its existing default."""
    candidate = _format_candidate(
        {"id": "chunk-1", "document": "x" * 250, "metadata": {}},
        rank=1,
    )

    assert candidate["preview"] == "x" * 180


def test_lookup_cli_exposes_explicit_diagnostic_modes() -> None:
    """The CLI should default to hybrid and accept both single-channel modes."""
    default_args = build_parser().parse_args(["--question", "Nebbiolo"])
    bm25_args = build_parser().parse_args(
        ["--question", "Nebbiolo", "--mode", "bm25", "--full-text"]
    )

    assert default_args.mode == "hybrid"
    assert bm25_args.mode == "bm25"
    assert bm25_args.full_text is True


def test_curator_default_dataset_resolves_from_repository_root() -> None:
    """The default path must not duplicate the repository's ``src`` segment."""
    config = SimpleNamespace(eval=SimpleNamespace(dataset_path="src/eval/wine_qa_golden.jsonl"))

    dataset_path = _default_dataset_path(config)

    assert dataset_path.name == "wine_qa_golden.jsonl"
    assert dataset_path.parent.name == "eval"
    assert dataset_path.parent.parent.name == "src"
    assert dataset_path.exists()


def test_print_candidates_displays_chunk_id_and_complete_multiline_text(capsys) -> None:
    """Curator output exposes all evidence needed for a manual choice."""
    _print_candidates(
        [
            {
                "rank": 1,
                "chunk_id": "book_7_deadbeef",
                "similarity": 0.75,
                "source": "/books/example.pdf",
                "retrieval_channels": ["dense", "sparse"],
                "dense_rank": 2,
                "sparse_rank": 1,
                "dense_similarity": 0.75,
                "bm25_score": 4.5,
                "rerank_score": 2.25,
                "metadata_matches": 1,
                "structural_role": "prose",
                "heading_path": "Example > Nebbiolo > taste",
                "preview": "first line\nsecond line",
            }
        ]
    )

    output = capsys.readouterr().out
    assert "book_7_deadbeef" in output
    assert "first line" in output
    assert "second line" in output
    assert "/books/example.pdf" in output
    assert "channels=dense,sparse" in output
    assert "dense_rank=2" in output
    assert "sparse_rank=1" in output
    assert "bm25=4.5000" in output
    assert "rerank=2.2500" in output
    assert "Example > Nebbiolo > taste" in output


def _hybrid_retriever(mocker) -> tuple[HybridRetriever, object, object]:
    """Build a real hybrid wrapper around isolated channel mocks."""
    vector = mocker.Mock()
    sparse = mocker.Mock()
    return HybridRetriever(vector, sparse), vector, sparse


def test_hybrid_lookup_uses_shared_production_path_without_unwrapping(mocker) -> None:
    """Default lookup must preserve production hybrid retrieval and reranking."""
    retriever, vector, _sparse = _hybrid_retriever(mocker)
    mocker.patch("src.eval.scripts.chunk_id_lookup.get_config", return_value=object())
    mocker.patch("src.eval.scripts.chunk_id_lookup.build_retriever_from_config", return_value=retriever)
    reranker = object()
    mocker.patch("src.eval.scripts.chunk_id_lookup.build_reranker_from_config", return_value=reranker)
    execute = mocker.patch(
        "src.eval.scripts.chunk_id_lookup.execute_production_rag",
        return_value=SimpleNamespace(
            retrieval_error=None,
            context_chunks=[
                RAGChunkArtifact(
                    id="nebbiolo",
                    text="Tar and roses.",
                    metadata={"chapter": "NEBBIOLO"},
                    rerank_score=2.0,
                    sparse_rank=1,
                    retrieval_channels=["sparse"],
                )
            ],
        ),
    )

    candidates = lookup_chunk_ids("Nebbiolo aromas?", top_k=10, preview_chars=None)

    assert [candidate["chunk_id"] for candidate in candidates] == ["nebbiolo"]
    assert candidates[0]["rerank_score"] == 2.0
    assert candidates[0]["sparse_rank"] == 1
    execute.assert_called_once()
    assert execute.call_args.kwargs["retriever"] is retriever
    assert execute.call_args.kwargs["reranker"] is reranker
    assert execute.call_args.kwargs["generation_enabled"] is False
    vector.retrieve.assert_not_called()


def test_hybrid_lookup_fails_explicitly_when_synchronized_bm25_is_unavailable(mocker) -> None:
    """Hybrid mode must not silently become vector-only."""
    mocker.patch("src.eval.scripts.chunk_id_lookup.get_config", return_value=object())
    mocker.patch(
        "src.eval.scripts.chunk_id_lookup.build_retriever_from_config",
        return_value=mocker.Mock(spec=["retrieve"]),
    )

    with pytest.raises(RuntimeError, match="Hybrid lookup requested"):
        lookup_chunk_ids("Nebbiolo", top_k=5)


def test_explicit_vector_and_bm25_modes_use_channel_specific_queries(mocker) -> None:
    """Diagnostic modes should be opt-in and expose their native ranks."""
    retriever, vector, sparse = _hybrid_retriever(mocker)
    vector.retrieve.return_value = [
        {"id": "dense", "document": "Dense evidence", "metadata": {}, "similarity": 0.8}
    ]
    sparse.search.return_value = [
        {"id": "sparse", "document": "Sparse evidence", "metadata": {}, "bm25_score": 5.0}
    ]
    mocker.patch("src.eval.scripts.chunk_id_lookup.get_config", return_value=object())
    mocker.patch("src.eval.scripts.chunk_id_lookup.build_retriever_from_config", return_value=retriever)
    question = "What are the primary flavour characteristics of Nebbiolo?"

    vector_candidates = lookup_chunk_ids(question, top_k=3, retrieval_mode="vector")
    sparse_candidates = lookup_chunk_ids(question, top_k=4, retrieval_mode="bm25")

    vector.retrieve.assert_called_once_with(
        "nebbiolo aroma flavor taste sensory profile tannin acidity body",
        n_results=3,
    )
    sparse.search.assert_called_once_with("nebbiolo aroma taste tannin acidity body", top_k=4)
    assert vector_candidates[0]["dense_rank"] == 1
    assert vector_candidates[0]["retrieval_channels"] == ["dense"]
    assert sparse_candidates[0]["sparse_rank"] == 1
    assert sparse_candidates[0]["retrieval_channels"] == ["sparse"]


def test_interrupted_session_keeps_completed_selection_and_leaves_remaining_resumable(
    tmp_path: Path,
    mocker,
) -> None:
    """Each accepted choice should be atomic even if the next prompt is interrupted."""
    dataset = tmp_path / "golden.jsonl"
    rows = [
        {
            "id": "first",
            "category": "rag_only",
            "question": "First question",
            "ground_truth": "First answer",
            "ground_truth_chunk_ids": [],
        },
        {
            "id": "second",
            "category": "rag_only",
            "question": "Second question",
            "ground_truth": "Second answer",
            "ground_truth_chunk_ids": [],
        },
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mocker.patch(
        "src.eval.scripts.chunk_id_curator.lookup_chunk_ids",
        return_value=[
            {
                "rank": 1,
                "chunk_id": "chosen",
                "similarity": 0.8,
                "source": "book.pdf",
                "preview": "Complete evidence.",
            }
        ],
    )
    mocker.patch("builtins.input", side_effect=["1", KeyboardInterrupt])

    run_curation(dataset, top_k=1)

    saved = _load_jsonl(dataset)
    assert saved[0]["ground_truth_chunk_ids"] == ["chosen"]
    assert saved[1]["ground_truth_chunk_ids"] == []
    assert not dataset.with_suffix(".jsonl.tmp").exists()
