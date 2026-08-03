"""Unit tests for src/chroma/loader.py"""

from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.chroma.loader import CollectionDataLoader


class TestCollectionDataLoaderInit:
    """Test CollectionDataLoader initialization."""

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_init_success(self, mock_get_collection, mock_init_client, mock_get_embedder):
        """Test successful initialization of CollectionDataLoader."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_embedder = Mock()

        mock_init_client.return_value = mock_client
        mock_get_collection.return_value = mock_collection
        mock_get_embedder.return_value = mock_embedder

        loader = CollectionDataLoader(
            collection_name="test_collection",
            collection_metadata={"test": "meta"},
            chroma_host="localhost",
            chroma_port=8000,
            embedding_model="test-model",
            batch_size=1000,
        )

        assert loader.collection_name == "test_collection"
        assert loader.batch_size == 1000
        assert loader.embedding_model == "test-model"
        assert loader.embedder == mock_embedder
        assert loader.client == mock_client
        assert loader.collection == mock_collection

        mock_init_client.assert_called_once_with("localhost", 8000)
        mock_get_collection.assert_called_once_with(mock_client, "test_collection", {"test": "meta"})
        mock_get_embedder.assert_called_once_with(model_name="test-model")

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_init_default_batch_size(self, mock_get_collection, mock_init_client, mock_get_embedder):
        """Test initialization with default batch size."""
        mock_init_client.return_value = Mock()
        mock_get_collection.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        loader = CollectionDataLoader(
            collection_name="test",
            collection_metadata={},
            chroma_host="localhost",
            chroma_port=8000,
            embedding_model="model",
        )

        assert loader.batch_size == 2500

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_reset_collection_recreates_empty_target(
        self,
        mock_get_collection,
        mock_init_client,
        mock_get_embedder,
    ):
        """Forced reindex reset should replace the loader's collection handle."""
        original_collection = Mock()
        replacement_collection = Mock()
        mock_get_collection.side_effect = [original_collection, replacement_collection]
        mock_client = Mock()
        mock_init_client.return_value = mock_client
        mock_get_embedder.return_value = Mock()
        loader = CollectionDataLoader("test", {"version": "test"}, "localhost", 8000, "model")

        loader.reset_collection()

        mock_client.delete_collection.assert_called_once_with("test")
        assert loader.collection is replacement_collection
        assert mock_get_collection.call_args_list[-1].args == (
            mock_client,
            "test",
            {"version": "test"},
        )


class TestCheckDuplicate:
    """Test _check_duplicate method."""

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_check_duplicate_found(self, mock_get_collection, mock_init_client, mock_get_embedder):
        """Test duplicate detection when duplicate exists."""
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": ["existing_id"]}
        mock_get_collection.return_value = mock_collection
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        result = loader._check_duplicate("hash123")

        assert result is True
        mock_collection.get.assert_called_once_with(where={"content_hash": "hash123"}, limit=1)

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_check_duplicate_not_found(self, mock_get_collection, mock_init_client, mock_get_embedder):
        """Test duplicate detection when no duplicate exists."""
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        result = loader._check_duplicate("hash456")

        assert result is False

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_check_duplicate_error_handling(self, mock_get_collection, mock_init_client, mock_get_embedder):
        """Test duplicate check handles errors gracefully."""
        mock_collection = Mock()
        mock_collection.get.side_effect = Exception("Database error")
        mock_get_collection.return_value = mock_collection
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        result = loader._check_duplicate("hash789")

        assert result is False


class TestProcessFile:
    """Test process_file method."""

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.split_file")
    @patch("src.chroma.loader.validate_chunks")
    @patch("src.chroma.loader.create_batches")
    def test_process_file_success(
        self,
        mock_create_batches,
        mock_validate,
        mock_split,
        mock_get_collection,
        mock_init_client,
        mock_get_embedder,
    ):
        """Test successful file processing."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_init_client.return_value = Mock()

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_get_embedder.return_value = mock_embedder

        chunks = [
            {"id": "chunk1", "text": "Text 1", "metadata": {"content_hash": "hash1"}},
            {"id": "chunk2", "text": "Text 2", "metadata": {"content_hash": "hash2"}},
        ]
        mock_split.return_value = chunks
        mock_validate.return_value = chunks

        mock_create_batches.return_value = [
            (["chunk1", "chunk2"], [[0.1, 0.2], [0.3, 0.4]], [{"content_hash": "hash1"}, {"content_hash": "hash2"}], ["Text 1", "Text 2"])
        ]

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader._check_duplicate = Mock(return_value=False)

        result = loader.process_file(Path("test.pdf"))

        assert result["filename"] == "test.pdf"
        assert result["chunks_generated"] == 2
        assert result["chunks_added"] == 2
        assert result["chunks_skipped"] == 0
        assert len(result["errors"]) == 0
        assert result["processing_time"] > 0

        mock_split.assert_called_once()
        mock_collection.add.assert_called_once()

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.split_file")
    def test_process_file_no_chunks(
        self, mock_split, mock_get_collection, mock_init_client, mock_get_embedder
    ):
        """Test processing when no chunks are generated."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()
        mock_split.return_value = []

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        result = loader.process_file(Path("empty.pdf"))

        assert result["chunks_generated"] == 0
        assert result["chunks_added"] == 0
        assert "No chunks generated" in result["errors"]

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.split_file")
    @patch("src.chroma.loader.validate_chunks")
    def test_process_file_all_duplicates(
        self, mock_validate, mock_split, mock_get_collection, mock_init_client, mock_get_embedder
    ):
        """Test processing when all chunks are duplicates."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        chunks = [
            {"id": "chunk1", "text": "Text 1", "metadata": {"content_hash": "hash1"}},
            {"id": "chunk2", "text": "Text 2", "metadata": {"content_hash": "hash2"}},
        ]
        mock_split.return_value = chunks
        mock_validate.return_value = chunks

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader._check_duplicate = Mock(return_value=True)

        result = loader.process_file(Path("test.pdf"), skip_duplicates=True)

        assert result["chunks_generated"] == 2
        assert result["chunks_added"] == 0
        assert result["chunks_skipped"] == 2

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.split_file")
    def test_process_file_error_handling(
        self, mock_split, mock_get_collection, mock_init_client, mock_get_embedder
    ):
        """Test error handling during file processing."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()
        mock_split.side_effect = Exception("Split error")

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        result = loader.process_file(Path("bad.pdf"))

        assert result["chunks_added"] == 0
        assert len(result["errors"]) > 0
        assert "Split error" in str(result["errors"])

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.split_file")
    @patch("src.chroma.loader.validate_chunks")
    @patch("src.chroma.loader.create_batches")
    def test_process_file_batch_error(
        self,
        mock_create_batches,
        mock_validate,
        mock_split,
        mock_get_collection,
        mock_init_client,
        mock_get_embedder,
    ):
        """Test handling of batch addition errors."""
        mock_collection = Mock()
        mock_collection.add.side_effect = Exception("Batch error")
        mock_get_collection.return_value = mock_collection
        mock_init_client.return_value = Mock()

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2]]
        mock_get_embedder.return_value = mock_embedder

        chunks = [{"id": "chunk1", "text": "Text 1", "metadata": {"content_hash": "hash1"}}]
        mock_split.return_value = chunks
        mock_validate.return_value = chunks
        mock_create_batches.return_value = [
            (["chunk1"], [[0.1, 0.2]], [{"content_hash": "hash1"}], ["Text 1"])
        ]

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader._check_duplicate = Mock(return_value=False)

        result = loader.process_file(Path("test.pdf"))

        assert len(result["errors"]) > 0
        assert "Batch error" in str(result["errors"])

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.split_file")
    @patch("src.chroma.loader.validate_chunks")
    def test_process_file_skip_duplicates_false(
        self, mock_validate, mock_split, mock_get_collection, mock_init_client, mock_get_embedder
    ):
        """Test processing with skip_duplicates disabled."""
        mock_collection = Mock()
        mock_get_collection.return_value = mock_collection
        mock_init_client.return_value = Mock()

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2]]
        mock_get_embedder.return_value = mock_embedder

        chunks = [{"id": "chunk1", "text": "Text 1", "metadata": {"content_hash": "hash1"}}]
        mock_split.return_value = chunks
        mock_validate.return_value = chunks

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader._check_duplicate = Mock(return_value=True)

        with patch("src.chroma.loader.create_batches") as mock_batches:
            mock_batches.return_value = [(["chunk1"], [[0.1, 0.2]], [{"content_hash": "hash1"}], ["Text 1"])]
            result = loader.process_file(Path("test.pdf"), skip_duplicates=False)

        assert result["chunks_skipped"] == 0
        assert result["chunks_added"] == 1

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.split_file")
    def test_process_file_with_strategy(
        self, mock_split, mock_get_collection, mock_init_client, mock_get_embedder
    ):
        """Test processing with custom chunking strategy."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()
        mock_split.return_value = []

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        loader.process_file(
            Path("test.pdf"),
            strategy="semantic",
            chunk_size=256,
            overlap_size=64,
            extract_metadata=False,
        )

        mock_split.assert_called_once()
        call_kwargs = mock_split.call_args[1]
        assert call_kwargs["strategy"] == "semantic"
        assert call_kwargs["chunk_size"] == 256
        assert call_kwargs["overlap_size"] == 64
        assert call_kwargs["extract_metadata"] is False


class TestLoadDirectory:
    """Test load_directory method."""

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_load_directory_not_exists(self, mock_get_collection, mock_init_client, mock_get_embedder):
        """Test load_directory with non-existent directory."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        with pytest.raises(ValueError, match="does not exist"):
            loader.load_directory("/nonexistent/path")

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_load_directory_no_files(self, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path):
        """Test load_directory with no matching files."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        result = loader.load_directory(tmp_path)

        assert result["total_files"] == 0
        assert result["files_processed"] == 0

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.IndexTracker")
    def test_load_directory_success(
        self, mock_tracker_class, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test successful directory loading."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        # Create test files
        (tmp_path / "file1.pdf").write_text("test")
        (tmp_path / "file2.pdf").write_text("test")

        mock_tracker = Mock()
        mock_tracker.get_files_to_index.return_value = [
            tmp_path / "file1.pdf",
            tmp_path / "file2.pdf",
        ]
        mock_tracker.get_stats.return_value = {"total_chunks": 10}
        mock_tracker_class.return_value = mock_tracker

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader.process_file = Mock(return_value={
            "filename": "test.pdf",
            "chunks_generated": 5,
            "chunks_added": 5,
            "chunks_skipped": 0,
            "processing_time": 1.0,
            "errors": [],
        })

        result = loader.load_directory(tmp_path)

        assert result["total_files"] == 2
        assert result["files_processed"] == 2
        assert result["successful_files"] == 2
        assert result["failed_files"] == 0
        assert result["total_chunks_added"] == 10

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.IndexTracker")
    def test_load_directory_incremental_all_indexed(
        self, mock_tracker_class, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test incremental loading when all files are already indexed."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        (tmp_path / "file1.pdf").write_text("test")

        mock_tracker = Mock()
        mock_tracker.get_files_to_index.return_value = []
        mock_tracker.get_stats.return_value = {"total_chunks": 5}
        mock_tracker_class.return_value = mock_tracker

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        result = loader.load_directory(tmp_path, incremental=True)

        assert result["files_processed"] == 0
        assert result["files_skipped"] == 1
        assert result["message"] == "All files already indexed"

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_load_directory_force_reindex(
        self, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test force reindex ignores index tracker."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        (tmp_path / "file1.pdf").write_text("test")

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader.process_file = Mock(return_value={
            "filename": "test.pdf",
            "chunks_generated": 3,
            "chunks_added": 3,
            "chunks_skipped": 0,
            "processing_time": 1.0,
            "errors": [],
        })

        result = loader.load_directory(tmp_path, force_reindex=True)

        assert result["files_processed"] == 1
        assert result["files_skipped"] == 0

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.IndexTracker")
    def test_load_directory_with_errors(
        self, mock_tracker_class, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test directory loading with file processing errors."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        (tmp_path / "file1.pdf").write_text("test")
        (tmp_path / "file2.pdf").write_text("test")

        mock_tracker = Mock()
        mock_tracker.get_files_to_index.return_value = [
            tmp_path / "file1.pdf",
            tmp_path / "file2.pdf",
        ]
        mock_tracker_class.return_value = mock_tracker

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")

        # First file succeeds, second fails
        loader.process_file = Mock(side_effect=[
            {
                "filename": "file1.pdf",
                "chunks_generated": 5,
                "chunks_added": 5,
                "chunks_skipped": 0,
                "processing_time": 1.0,
                "errors": [],
            },
            {
                "filename": "file2.pdf",
                "chunks_generated": 0,
                "chunks_added": 0,
                "chunks_skipped": 0,
                "processing_time": 0.5,
                "errors": ["Processing error"],
            },
        ])

        result = loader.load_directory(tmp_path)

        assert result["files_processed"] == 2
        assert result["successful_files"] == 1
        assert result["failed_files"] == 1
        assert len(result["errors"]) > 0

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_load_directory_custom_extensions(
        self, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test loading with custom file extensions."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.md").write_text("test")
        (tmp_path / "file3.pdf").write_text("test")

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader.process_file = Mock(return_value={
            "filename": "test",
            "chunks_generated": 1,
            "chunks_added": 1,
            "chunks_skipped": 0,
            "processing_time": 1.0,
            "errors": [],
        })

        # This test validates extension filtering only; disable incremental mode
        # to avoid shared manifest state affecting file selection.
        result = loader.load_directory(tmp_path, file_extensions=[".txt", ".md"], incremental=False)

        assert result["total_files"] == 2
        assert result["files_processed"] == 2

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.IndexTracker")
    def test_load_directory_tracker_updates(
        self, mock_tracker_class, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test that index tracker is updated after successful processing."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        test_file = tmp_path / "file1.pdf"
        test_file.write_text("test")

        mock_tracker = Mock()
        mock_tracker.get_files_to_index.return_value = [test_file]
        mock_tracker.get_stats.return_value = {"total_chunks": 5}
        mock_tracker_class.return_value = mock_tracker

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader.process_file = Mock(return_value={
            "filename": "file1.pdf",
            "chunks_generated": 5,
            "chunks_added": 5,
            "chunks_skipped": 0,
            "processing_time": 1.0,
            "errors": [],
        })

        loader.load_directory(tmp_path)

        mock_tracker.mark_indexed.assert_called_once_with(test_file, 5)
        assert mock_tracker.save.call_count >= 1

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    @patch("src.chroma.loader.IndexTracker")
    def test_load_directory_tracker_not_updated_on_error(
        self, mock_tracker_class, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test that index tracker is not updated when file processing fails."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        test_file = tmp_path / "file1.pdf"
        test_file.write_text("test")

        mock_tracker = Mock()
        mock_tracker.get_files_to_index.return_value = [test_file]
        mock_tracker_class.return_value = mock_tracker

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader.process_file = Mock(return_value={
            "filename": "file1.pdf",
            "chunks_generated": 0,
            "chunks_added": 0,
            "chunks_skipped": 0,
            "processing_time": 1.0,
            "errors": ["Error processing"],
        })

        loader.load_directory(tmp_path)

        mock_tracker.mark_indexed.assert_not_called()

    @patch("src.chroma.loader.get_embedder")
    @patch("src.chroma.loader.initialize_chroma_client")
    @patch("src.chroma.loader.get_or_create_collection")
    def test_load_directory_no_incremental(
        self, mock_get_collection, mock_init_client, mock_get_embedder, tmp_path
    ):
        """Test loading without incremental processing."""
        mock_get_collection.return_value = Mock()
        mock_init_client.return_value = Mock()
        mock_get_embedder.return_value = Mock()

        (tmp_path / "file1.pdf").write_text("test")

        loader = CollectionDataLoader("test", {}, "localhost", 8000, "model")
        loader.process_file = Mock(return_value={
            "filename": "file1.pdf",
            "chunks_generated": 3,
            "chunks_added": 3,
            "chunks_skipped": 0,
            "processing_time": 1.0,
            "errors": [],
        })

        result = loader.load_directory(tmp_path, incremental=False)

        assert result["files_processed"] == 1
        assert result["files_skipped"] == 0
