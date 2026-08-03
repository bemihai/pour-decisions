"""Tests for the synchronized forced-reindex CLI helpers."""

from types import SimpleNamespace

import pytest

from src.chroma import load_data


def _chroma_config(*, rebuild_on_reindex: bool = True) -> SimpleNamespace:
    """Build the config fields required by the BM25 post-reindex hook."""
    return SimpleNamespace(
        indexing=SimpleNamespace(
            bm25=SimpleNamespace(
                rebuild_on_reindex=rebuild_on_reindex,
                sync_manifest_path="chroma-data/test.meta.json",
            )
        ),
        retrieval=SimpleNamespace(bm25_index_path="chroma-data/test.pkl"),
        settings=SimpleNamespace(batch_size=100),
    )


def _app_config() -> SimpleNamespace:
    """Build the complete CLI config for one test collection."""
    chroma_cfg = _chroma_config()
    chroma_cfg.client = SimpleNamespace(host="localhost", port=8100)
    chroma_cfg.collections = [
        SimpleNamespace(
            name="wine_books",
            metadata={"version": "test"},
            local_data_path="/books",
        )
    ]
    chroma_cfg.settings.embedder = "test-embedder"
    chroma_cfg.chunking = SimpleNamespace(
        extract_wine_metadata=True,
        strategy="by_title",
        chunk_size=1024,
        chunk_overlap=256,
    )
    return SimpleNamespace(chroma=chroma_cfg)


def test_raise_for_indexing_errors_rejects_partial_chroma_reindex() -> None:
    """Any file error should stop before the live BM25 index is replaced."""
    with pytest.raises(RuntimeError, match="2 error"):
        load_data._raise_for_indexing_errors(
            "wine_books",
            {"errors": ["first", "second"]},
        )


def test_validate_force_reindex_source_rejects_missing_directory(tmp_path) -> None:
    """A missing source must be rejected before the live collection is reset."""
    with pytest.raises(ValueError, match="does not exist or is not a directory"):
        load_data._validate_force_reindex_source(
            data_path=tmp_path / "missing",
            file_extensions=[".epub", ".pdf"],
        )


def test_validate_force_reindex_source_rejects_empty_directory(tmp_path) -> None:
    """A source without supported books must be rejected before reset."""
    with pytest.raises(ValueError, match="No files found"):
        load_data._validate_force_reindex_source(
            data_path=tmp_path,
            file_extensions=[".epub", ".pdf"],
        )


def test_validate_force_reindex_source_accepts_nested_book(tmp_path) -> None:
    """Supported books nested below the source directory should pass preflight."""
    book_directory = tmp_path / "books"
    book_directory.mkdir()
    (book_directory / "wine.pdf").touch()

    load_data._validate_force_reindex_source(
        data_path=tmp_path,
        file_extensions=[".epub", ".pdf"],
    )


def test_rebuild_bm25_if_enabled_uses_completed_collection(mocker) -> None:
    """The post-reindex hook should pass the live collection and configured paths."""
    collection = object()
    rebuild = mocker.patch.object(load_data, "rebuild_bm25_from_collection")

    load_data._rebuild_bm25_if_enabled(
        chroma_cfg=_chroma_config(),
        collection_name="wine_books",
        collection=collection,
    )

    rebuild.assert_called_once_with(
        collection=collection,
        collection_name="wine_books",
        index_path="chroma-data/test.pkl",
        manifest_path="chroma-data/test.meta.json",
        batch_size=100,
    )


def test_rebuild_bm25_if_enabled_respects_disabled_config(mocker) -> None:
    """Disabled post-reindex rebuilding should not write an index."""
    rebuild = mocker.patch.object(load_data, "rebuild_bm25_from_collection")

    load_data._rebuild_bm25_if_enabled(
        chroma_cfg=_chroma_config(rebuild_on_reindex=False),
        collection_name="wine_books",
        collection=object(),
    )

    rebuild.assert_not_called()


def test_main_force_resets_chroma_before_rebuilding_bm25(mocker, monkeypatch) -> None:
    """The forced CLI path should reset, load successfully, then rebuild BM25."""
    config = _app_config()
    collection = object()
    loader = mocker.Mock(collection=collection)
    loader.load_directory.return_value = {"errors": [], "total_chunks_added": 3}
    mocker.patch.object(load_data, "get_config", return_value=config)
    mocker.patch.object(load_data, "_validate_force_reindex_source")
    loader_class = mocker.patch.object(load_data, "CollectionDataLoader", return_value=loader)
    rebuild = mocker.patch.object(load_data, "_rebuild_bm25_if_enabled")
    monkeypatch.setattr("sys.argv", ["load_data", "--force"])

    load_data.main()

    loader.reset_collection.assert_called_once_with()
    loader.load_directory.assert_called_once_with(
        file_extensions=[".epub", ".pdf"],
        data_path="/books",
        strategy="by_title",
        chunk_size=1024,
        overlap_size=256,
        extract_metadata=True,
        incremental=True,
        force_reindex=True,
    )
    rebuild.assert_called_once_with(
        chroma_cfg=config.chroma,
        collection_name="wine_books",
        collection=collection,
    )
    loader_class.assert_called_once()


def test_main_force_does_not_replace_bm25_after_chroma_error(mocker, monkeypatch) -> None:
    """A partial Chroma failure should stop before the BM25 post-hook."""
    config = _app_config()
    loader = mocker.Mock(collection=object())
    loader.load_directory.return_value = {"errors": ["failed document"]}
    mocker.patch.object(load_data, "get_config", return_value=config)
    mocker.patch.object(load_data, "_validate_force_reindex_source")
    mocker.patch.object(load_data, "CollectionDataLoader", return_value=loader)
    rebuild = mocker.patch.object(load_data, "_rebuild_bm25_if_enabled")
    monkeypatch.setattr("sys.argv", ["load_data", "--force"])

    with pytest.raises(RuntimeError, match="Forced Chroma reindex failed"):
        load_data.main()

    rebuild.assert_not_called()
