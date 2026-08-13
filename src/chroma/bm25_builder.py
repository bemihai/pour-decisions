"""Verified BM25 rebuilding from a completed Chroma collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Iterable

from src.utils import logger

from .contextual_text import build_contextual_search_text

if TYPE_CHECKING:
    from src.retrieval.keyword_search import BM25Index


class BM25SyncError(RuntimeError):
    """Raised when Chroma and BM25 synchronization cannot be verified."""


@dataclass(frozen=True)
class BM25SyncManifest:
    """Sidecar metadata proving which Chroma snapshot produced a BM25 index."""

    collection_name: str
    record_count: int
    chunk_ids_sha256: str
    built_at: str
    bm25_path: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable manifest dictionary."""
        return asdict(self)


def compute_chunk_ids_sha256(chunk_ids: Iterable[str]) -> str:
    """Hash sorted chunk IDs using an unambiguous JSON representation."""
    payload = json.dumps(sorted(str(chunk_id) for chunk_id in chunk_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_collection_documents(collection: Any, *, batch_size: int = 2500) -> list[dict[str, Any]]:
    """Read every collection record in bounded batches for BM25 construction.

    Args:
        collection: Chroma collection exposing ``count()`` and ``get()``.
        batch_size: Maximum records requested from Chroma per call.

    Returns:
        Documents in the shape required by :class:`BM25Index`.

    Raises:
        ValueError: If ``batch_size`` is invalid.
        BM25SyncError: If Chroma returns incomplete or inconsistent records.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    expected_count = int(collection.count())
    records: list[dict[str, Any]] = []
    for offset in range(0, expected_count, batch_size):
        batch = collection.get(
            limit=min(batch_size, expected_count - offset),
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = list(batch.get("ids") or [])
        documents = batch.get("documents")
        metadatas = batch.get("metadatas")
        if documents is None or len(documents) != len(ids):
            raise BM25SyncError(f"Chroma returned incomplete documents at offset {offset}")
        if metadatas is not None and len(metadatas) != len(ids):
            raise BM25SyncError(f"Chroma returned incomplete metadata at offset {offset}")

        for index, chunk_id in enumerate(ids):
            document = documents[index]
            if document is None:
                raise BM25SyncError(f"Chroma record {chunk_id!r} has no document text")
            metadata = metadatas[index] if metadatas is not None else None
            normalized_metadata = dict(metadata or {})
            records.append(
                {
                    "id": str(chunk_id),
                    "document": str(document),
                    "search_text": build_contextual_search_text(str(document), normalized_metadata),
                    "metadata": normalized_metadata,
                }
            )

    final_count = int(collection.count())
    if final_count != expected_count or len(records) != expected_count:
        raise BM25SyncError(
            "Chroma count changed or batched read was incomplete: "
            f"initial={expected_count}, final={final_count}, read={len(records)}"
        )
    if len({record["id"] for record in records}) != len(records):
        raise BM25SyncError("Chroma batched read returned duplicate chunk IDs")
    return records


def read_collection_ids(collection: Any, *, batch_size: int = 2500) -> list[str]:
    """Read every collection ID in bounded batches without document payloads."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    expected_count = int(collection.count())
    chunk_ids: list[str] = []
    for offset in range(0, expected_count, batch_size):
        batch = collection.get(
            limit=min(batch_size, expected_count - offset),
            offset=offset,
            include=[],
        )
        chunk_ids.extend(str(chunk_id) for chunk_id in (batch.get("ids") or []))

    final_count = int(collection.count())
    if final_count != expected_count or len(chunk_ids) != expected_count:
        raise BM25SyncError(
            "Chroma count changed or ID read was incomplete: "
            f"initial={expected_count}, final={final_count}, read={len(chunk_ids)}"
        )
    if len(set(chunk_ids)) != len(chunk_ids):
        raise BM25SyncError("Chroma ID read returned duplicate chunk IDs")
    return chunk_ids


def rebuild_bm25_from_collection(
    *,
    collection: Any,
    collection_name: str,
    index_path: str | Path,
    manifest_path: str | Path,
    batch_size: int = 2500,
) -> BM25SyncManifest:
    """Atomically rebuild BM25 from one verified Chroma collection snapshot.

    The live pickle and manifest are replaced only after the temporary index,
    its document IDs, and a second Chroma ID snapshot all match.
    """
    target_index_path = Path(index_path)
    target_manifest_path = Path(manifest_path)
    target_index_path.parent.mkdir(parents=True, exist_ok=True)
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index_path = _temporary_path(target_index_path)
    temporary_manifest_path = _temporary_path(target_manifest_path)

    try:
        from src.retrieval.keyword_search import BM25Index

        documents = read_collection_documents(collection, batch_size=batch_size)
        expected_ids = [str(document["id"]) for document in documents]
        expected_hash = compute_chunk_ids_sha256(expected_ids)

        bm25 = BM25Index()
        bm25.build_index(documents)
        bm25.save(temporary_index_path)

        temporary_bm25 = BM25Index(index_path=temporary_index_path)
        _validate_bm25_snapshot(
            bm25=temporary_bm25,
            expected_count=len(documents),
            expected_hash=expected_hash,
        )

        current_ids = read_collection_ids(collection, batch_size=batch_size)
        current_hash = compute_chunk_ids_sha256(current_ids)
        if len(current_ids) != len(documents) or current_hash != expected_hash:
            raise BM25SyncError("Chroma collection changed while the BM25 index was being built")

        manifest = BM25SyncManifest(
            collection_name=collection_name,
            record_count=len(documents),
            chunk_ids_sha256=expected_hash,
            built_at=datetime.now(timezone.utc).isoformat(),
            bm25_path=str(target_index_path),
        )
        _write_manifest(temporary_manifest_path, manifest)

        temporary_index_path.replace(target_index_path)
        temporary_manifest_path.replace(target_manifest_path)

        persisted_bm25 = BM25Index(index_path=target_index_path)
        _validate_bm25_snapshot(
            bm25=persisted_bm25,
            expected_count=manifest.record_count,
            expected_hash=manifest.chunk_ids_sha256,
        )
        persisted_manifest = load_bm25_sync_manifest(target_manifest_path)
        if persisted_manifest != manifest:
            raise BM25SyncError("Persisted BM25 synchronization manifest does not match the build")

        logger.info(
            "Rebuilt synchronized BM25 index for collection=%s records=%d hash=%s",
            collection_name,
            manifest.record_count,
            manifest.chunk_ids_sha256,
        )
        return manifest
    finally:
        temporary_index_path.unlink(missing_ok=True)
        temporary_manifest_path.unlink(missing_ok=True)


def load_bm25_sync_manifest(manifest_path: str | Path) -> BM25SyncManifest:
    """Load and validate a BM25 synchronization manifest."""
    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return BM25SyncManifest(
            collection_name=str(payload["collection_name"]),
            record_count=int(payload["record_count"]),
            chunk_ids_sha256=str(payload["chunk_ids_sha256"]),
            built_at=str(payload["built_at"]),
            bm25_path=str(payload["bm25_path"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BM25SyncError(f"Invalid BM25 synchronization manifest at {path}: {exc}") from exc


def validate_bm25_sync(
    *,
    collection: Any,
    collection_name: str,
    bm25: "BM25Index",
    index_path: str | Path,
    manifest_path: str | Path,
    batch_size: int = 2500,
) -> tuple[bool, str | None]:
    """Validate the live Chroma IDs, BM25 documents, and sidecar manifest."""
    try:
        manifest = load_bm25_sync_manifest(manifest_path)
        if manifest.collection_name != collection_name:
            raise BM25SyncError(
                f"manifest collection={manifest.collection_name!r} does not match {collection_name!r}"
            )
        if Path(manifest.bm25_path) != Path(index_path):
            raise BM25SyncError(
                f"manifest BM25 path={manifest.bm25_path!r} does not match {str(index_path)!r}"
            )

        chunk_ids = read_collection_ids(collection, batch_size=batch_size)
        chunk_ids_hash = compute_chunk_ids_sha256(chunk_ids)
        if len(chunk_ids) != manifest.record_count:
            raise BM25SyncError(
                f"Chroma count={len(chunk_ids)} does not match manifest count={manifest.record_count}"
            )
        if chunk_ids_hash != manifest.chunk_ids_sha256:
            raise BM25SyncError("Chroma chunk-ID hash does not match the synchronization manifest")

        _validate_bm25_snapshot(
            bm25=bm25,
            expected_count=manifest.record_count,
            expected_hash=manifest.chunk_ids_sha256,
        )
        return True, None
    except BM25SyncError as exc:
        return False, str(exc)


def _validate_bm25_snapshot(*, bm25: "BM25Index", expected_count: int, expected_hash: str) -> None:
    """Raise when a BM25 object does not match an expected ID snapshot."""
    bm25_ids = [str(document.get("id", "")) for document in bm25.documents]
    if len(bm25_ids) != expected_count:
        raise BM25SyncError(f"BM25 count={len(bm25_ids)} does not match expected count={expected_count}")
    if any(not chunk_id for chunk_id in bm25_ids):
        raise BM25SyncError("BM25 contains a document without a chunk ID")
    if compute_chunk_ids_sha256(bm25_ids) != expected_hash:
        raise BM25SyncError("BM25 chunk-ID hash does not match the Chroma snapshot")


def _write_manifest(path: Path, manifest: BM25SyncManifest) -> None:
    """Write one manifest to an already-created temporary path."""
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _temporary_path(target_path: Path) -> Path:
    """Create an empty temporary file beside its eventual atomic-replace target."""
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    return Path(temporary_name)
