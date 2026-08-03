"""Tests for verified BM25 rebuilding from ChromaDB."""

from pathlib import Path
from uuid import uuid4

import pytest

from src.chroma.bm25_builder import (
    BM25SyncError,
    compute_chunk_ids_sha256,
    load_bm25_sync_manifest,
    read_collection_documents,
    rebuild_bm25_from_collection,
    validate_bm25_sync,
)
from src.retrieval.keyword_search import BM25Index


@pytest.fixture
def bm25_collection(in_memory_chroma_client, sample_chunks, sample_embeddings):
    """Create a uniquely named populated collection for BM25 tests."""
    collection_name = f"bm25_{uuid4().hex}"
    collection = in_memory_chroma_client.create_collection(name=collection_name)
    collection.add(
        ids=[f"test_chunk_{index}" for index in range(len(sample_chunks))],
        documents=[chunk["text"] for chunk in sample_chunks],
        metadatas=[chunk["metadata"] for chunk in sample_chunks],
        embeddings=sample_embeddings,
    )
    yield collection
    in_memory_chroma_client.delete_collection(collection_name)


def test_rebuild_bm25_produces_matching_index_and_manifest(bm25_collection, tmp_path: Path) -> None:
    """A completed rebuild should persist the exact Chroma record set."""
    index_path = tmp_path / "bm25_index.pkl"
    manifest_path = tmp_path / "bm25_index.meta.json"

    manifest = rebuild_bm25_from_collection(
        collection=bm25_collection,
        collection_name=bm25_collection.name,
        index_path=index_path,
        manifest_path=manifest_path,
        batch_size=2,
    )

    bm25 = BM25Index(index_path=index_path)
    persisted_manifest = load_bm25_sync_manifest(manifest_path)
    expected_ids = ["test_chunk_0", "test_chunk_1", "test_chunk_2"]
    is_synchronized, validation_error = validate_bm25_sync(
        collection=bm25_collection,
        collection_name=bm25_collection.name,
        bm25=bm25,
        index_path=index_path,
        manifest_path=manifest_path,
        batch_size=2,
    )

    assert [document["id"] for document in bm25.documents] == expected_ids
    assert manifest == persisted_manifest
    assert manifest.record_count == bm25_collection.count() == len(bm25)
    assert manifest.chunk_ids_sha256 == compute_chunk_ids_sha256(expected_ids)
    assert is_synchronized is True
    assert validation_error is None


def test_validate_bm25_sync_rejects_stale_collection(bm25_collection, tmp_path: Path) -> None:
    """A Chroma ID change after the build should disable synchronized hybrid use."""
    index_path = tmp_path / "bm25_index.pkl"
    manifest_path = tmp_path / "bm25_index.meta.json"
    rebuild_bm25_from_collection(
        collection=bm25_collection,
        collection_name=bm25_collection.name,
        index_path=index_path,
        manifest_path=manifest_path,
        batch_size=2,
    )
    bm25_collection.add(
        ids=["test_chunk_new"],
        documents=["A new wine document."],
        metadatas=[{"source": "new"}],
        embeddings=[[0.4, 0.5, 0.6, 0.7, 0.8]],
    )

    is_synchronized, validation_error = validate_bm25_sync(
        collection=bm25_collection,
        collection_name=bm25_collection.name,
        bm25=BM25Index(index_path=index_path),
        index_path=index_path,
        manifest_path=manifest_path,
        batch_size=2,
    )

    assert is_synchronized is False
    assert validation_error is not None
    assert "count" in validation_error or "hash" in validation_error


def test_failed_temporary_build_preserves_live_index(bm25_collection, tmp_path: Path, mocker) -> None:
    """A temporary write failure must not replace the last live pickle or manifest."""
    index_path = tmp_path / "bm25_index.pkl"
    manifest_path = tmp_path / "bm25_index.meta.json"
    index_path.write_bytes(b"existing-index")
    manifest_path.write_text("existing-manifest", encoding="utf-8")

    def fail_save(_bm25: BM25Index, temporary_path: str | Path | None = None) -> None:
        assert temporary_path is not None
        Path(temporary_path).write_bytes(b"partial-index")
        raise OSError("simulated disk failure")

    mocker.patch.object(BM25Index, "save", autospec=True, side_effect=fail_save)

    with pytest.raises(OSError, match="simulated disk failure"):
        rebuild_bm25_from_collection(
            collection=bm25_collection,
            collection_name=bm25_collection.name,
            index_path=index_path,
            manifest_path=manifest_path,
            batch_size=2,
        )

    assert index_path.read_bytes() == b"existing-index"
    assert manifest_path.read_text(encoding="utf-8") == "existing-manifest"


def test_read_collection_documents_rejects_incomplete_batch(mocker) -> None:
    """The builder should fail instead of silently indexing partial Chroma data."""
    collection = mocker.Mock()
    collection.count.return_value = 2
    collection.get.return_value = {
        "ids": ["chunk-1", "chunk-2"],
        "documents": ["Only one document"],
        "metadatas": [{}, {}],
    }

    with pytest.raises(BM25SyncError, match="incomplete documents"):
        read_collection_documents(collection, batch_size=2)
