"""Tests for exact Chroma corpus diagnostics."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.chroma.bm25_builder import compute_chunk_ids_sha256
from src.chroma import stats as stats_module
from src.chroma.stats import _write_json_artifact, get_exact_collection_stats


class _ExactStatsCollection:
    """Small deterministic collection implementing the Chroma stats contract."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.metadata = {"version": "test"}
        self.get_calls: list[dict[str, Any]] = []

    def count(self) -> int:
        """Return the current record count."""
        return len(self.records)

    def get(self, **kwargs: Any) -> dict[str, Any]:
        """Return one requested record window."""
        self.get_calls.append(kwargs)
        offset = int(kwargs.get("offset", 0))
        limit = int(kwargs.get("limit", len(self.records)))
        selected = self.records[offset : offset + limit]
        return {
            "ids": [record["id"] for record in selected],
            "documents": [record["document"] for record in selected],
            "metadatas": [record["metadata"] for record in selected],
        }


def _client_for(collection: _ExactStatsCollection) -> SimpleNamespace:
    """Build a client returning the controlled collection."""
    return SimpleNamespace(get_collection=lambda _name: collection)


def test_exact_stats_cover_every_record_in_bounded_batches() -> None:
    """Counts, rates, source aggregates, and ID hash should use the full corpus."""
    records = [
        {"id": "chunk-c", "document": "", "metadata": {"file_path": "/books/a.pdf"}},
        {"id": "chunk-a", "document": "a" * 50, "metadata": {"file_path": "/books/a.pdf"}},
        {"id": "chunk-e", "document": "e" * 199, "metadata": {"filename": "b.epub"}},
        {"id": "chunk-b", "document": "b" * 200, "metadata": {"filename": "b.epub"}},
        {"id": "chunk-d", "document": "d" * 300, "metadata": {}},
    ]
    collection = _ExactStatsCollection(records)

    stats = get_exact_collection_stats(
        _client_for(collection),
        "wine_books",
        batch_size=2,
    )

    assert stats["statistics_mode"] == "exact"
    assert stats["record_count"] == 5
    assert stats["avg_document_length"] == 149.8
    assert stats["min_document_length"] == 0
    assert stats["max_document_length"] == 300
    assert stats["empty_document_count"] == 1
    assert stats["empty_document_rate"] == 0.2
    assert stats["near_empty_threshold_chars"] == 200
    assert stats["near_empty_includes_empty"] is True
    assert stats["near_empty_document_count"] == 3
    assert stats["near_empty_document_rate"] == 0.6
    assert stats["source_document_count"] == 2
    assert stats["records_missing_source"] == 1
    assert stats["min_chunks_per_source"] == 2
    assert stats["avg_chunks_per_source"] == 2.0
    assert stats["max_chunks_per_source"] == 2
    assert stats["chunk_ids_sha256"] == compute_chunk_ids_sha256(record["id"] for record in records)
    assert [call["offset"] for call in collection.get_calls] == [0, 2, 4]
    assert [call["limit"] for call in collection.get_calls] == [2, 2, 1]


def test_exact_stats_are_zero_safe_for_empty_collection() -> None:
    """An empty collection should produce explicit zero values and a stable hash."""
    collection = _ExactStatsCollection([])

    stats = get_exact_collection_stats(_client_for(collection), "empty", batch_size=2)

    assert stats["record_count"] == 0
    assert stats["avg_document_length"] == 0.0
    assert stats["min_document_length"] == 0
    assert stats["max_document_length"] == 0
    assert stats["empty_document_rate"] == 0.0
    assert stats["near_empty_document_rate"] == 0.0
    assert stats["source_document_count"] == 0
    assert stats["avg_chunks_per_source"] == 0.0
    assert stats["chunk_ids_sha256"] == compute_chunk_ids_sha256([])
    assert collection.get_calls == []


def test_exact_stats_hash_and_source_aggregates_are_order_independent() -> None:
    """Record order must not affect the sorted-ID hash or source distribution."""
    records = [
        {"id": "chunk-z", "document": "Zinfandel", "metadata": {"filename": "one.pdf"}},
        {"id": "chunk-a", "document": "Albariño", "metadata": {"filename": "two.pdf"}},
        {"id": "chunk-m", "document": "Merlot", "metadata": {"filename": "one.pdf"}},
    ]

    forward = get_exact_collection_stats(_client_for(_ExactStatsCollection(records)), "wine_books", batch_size=2)
    reverse = get_exact_collection_stats(
        _client_for(_ExactStatsCollection(list(reversed(records)))),
        "wine_books",
        batch_size=2,
    )

    assert forward["chunk_ids_sha256"] == reverse["chunk_ids_sha256"]
    assert forward["source_document_count"] == reverse["source_document_count"] == 2
    assert forward["min_chunks_per_source"] == reverse["min_chunks_per_source"] == 1
    assert forward["avg_chunks_per_source"] == reverse["avg_chunks_per_source"] == 1.5
    assert forward["max_chunks_per_source"] == reverse["max_chunks_per_source"] == 2


def test_exact_stats_report_collection_read_errors() -> None:
    """Exact mode should label failures instead of returning sampled-looking output."""
    client = SimpleNamespace(get_collection=lambda _name: (_ for _ in ()).throw(RuntimeError("unavailable")))

    stats = get_exact_collection_stats(client, "missing", batch_size=2)

    assert stats == {
        "name": "missing",
        "statistics_mode": "exact",
        "error": "unavailable",
    }


def test_write_json_artifact_preserves_exact_mode(tmp_path: Path) -> None:
    """The saved corpus artifact should be valid JSON and identify exact mode."""
    output_path = tmp_path / "m3_gate0_corpus_test.json"
    stats = [{"name": "wine_books", "statistics_mode": "exact", "record_count": 3}]

    _write_json_artifact(stats, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == stats
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_main_exact_writes_configured_collection_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exact CLI mode should inspect configured collections and save JSON."""
    collection = _ExactStatsCollection(
        [{"id": "chunk-1", "document": "Wine context", "metadata": {"filename": "wine.pdf"}}]
    )
    client = _client_for(collection)
    output_path = tmp_path / "corpus.json"
    config = SimpleNamespace(
        chroma=SimpleNamespace(
            client=SimpleNamespace(host="localhost", port=8100),
            collections=[SimpleNamespace(name="wine_books")],
            settings=SimpleNamespace(batch_size=2500),
        )
    )
    monkeypatch.setattr(stats_module, "get_config", lambda: config)
    monkeypatch.setattr(stats_module, "initialize_chroma_client", lambda _host, _port: client)
    monkeypatch.setattr(
        stats_module,
        "parse_args",
        lambda: SimpleNamespace(
            collection=None,
            json=False,
            exact=True,
            batch_size=1,
            output=output_path,
        ),
    )

    exit_code = stats_module.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload[0]["name"] == "wine_books"
    assert payload[0]["statistics_mode"] == "exact"
    assert payload[0]["record_count"] == 1
    assert f"Saved JSON statistics to {output_path}" in capsys.readouterr().out
