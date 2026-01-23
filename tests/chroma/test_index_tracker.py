"""Unit tests for src/chroma/index_tracker.py"""

import time
from pathlib import Path


from src.chroma.index_tracker import IndexManifest, IndexTracker, IndexedFileInfo


class TestIndexedFileInfo:
    """Test IndexedFileInfo dataclass."""

    def test_create_indexed_file_info(self):
        """Test creating IndexedFileInfo with all required fields."""
        info = IndexedFileInfo(
            file_path="/path/to/file.pdf",
            file_hash="abc123",
            file_size=1024,
            modified_time=123456.789,
            indexed_at="2024-01-01T12:00:00",
            chunk_count=10,
            collection_name="test_collection"
        )

        assert info.file_path == "/path/to/file.pdf"
        assert info.file_hash == "abc123"
        assert info.file_size == 1024
        assert info.modified_time == 123456.789
        assert info.indexed_at == "2024-01-01T12:00:00"
        assert info.chunk_count == 10
        assert info.collection_name == "test_collection"


class TestIndexManifest:
    """Test IndexManifest dataclass and methods."""

    def test_create_manifest_defaults(self):
        """Test creating manifest with default values."""
        manifest = IndexManifest(collection_name="test")

        assert manifest.collection_name == "test"
        assert isinstance(manifest.created_at, str)
        assert isinstance(manifest.updated_at, str)
        assert manifest.files == {}

    def test_to_dict(self):
        """Test converting manifest to dictionary."""
        manifest = IndexManifest(
            collection_name="test",
            created_at="2024-01-01T12:00:00",
            updated_at="2024-01-01T12:00:00"
        )
        manifest.files["/path/to/file.pdf"] = IndexedFileInfo(
            file_path="/path/to/file.pdf",
            file_hash="abc123",
            file_size=1024,
            modified_time=123456.789,
            indexed_at="2024-01-01T12:00:00",
            chunk_count=10,
            collection_name="test"
        )

        result = manifest.to_dict()

        assert result["collection_name"] == "test"
        assert result["created_at"] == "2024-01-01T12:00:00"
        assert result["updated_at"] == "2024-01-01T12:00:00"
        assert "/path/to/file.pdf" in result["files"]
        assert result["files"]["/path/to/file.pdf"]["file_hash"] == "abc123"

    def test_from_dict(self):
        """Test creating manifest from dictionary."""
        data = {
            "collection_name": "test",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T13:00:00",
            "files": {
                "/path/to/file.pdf": {
                    "file_path": "/path/to/file.pdf",
                    "file_hash": "abc123",
                    "file_size": 1024,
                    "modified_time": 123456.789,
                    "indexed_at": "2024-01-01T12:00:00",
                    "chunk_count": 10,
                    "collection_name": "test"
                }
            }
        }

        manifest = IndexManifest.from_dict(data)

        assert manifest.collection_name == "test"
        assert manifest.created_at == "2024-01-01T12:00:00"
        assert manifest.updated_at == "2024-01-01T13:00:00"
        assert len(manifest.files) == 1
        assert "/path/to/file.pdf" in manifest.files
        assert manifest.files["/path/to/file.pdf"].file_hash == "abc123"

    def test_from_dict_no_files(self):
        """Test creating manifest from dictionary without files."""
        data = {
            "collection_name": "test",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T13:00:00"
        }

        manifest = IndexManifest.from_dict(data)

        assert manifest.collection_name == "test"
        assert manifest.files == {}


class TestIndexTracker:
    """Test IndexTracker class."""

    def test_init_no_manifest_path(self, temp_dir: Path):
        """Test initialization without specifying manifest path."""
        tracker = IndexTracker(
            manifest_path=temp_dir / "manifests" / "test_manifest.json",
            collection_name="test"
        )

        assert tracker.collection_name == "test"
        assert tracker.manifest is not None
        assert tracker.manifest.collection_name == "test"

    def test_init_with_manifest_path(self, temp_dir: Path):
        """Test initialization with custom manifest path."""
        manifest_path = temp_dir / "custom_manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        assert tracker.manifest_path == manifest_path
        assert tracker.collection_name == "test"

    def test_save_and_load_manifest(self, temp_dir: Path):
        """Test saving and loading manifest from disk."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        tracker.manifest.files["/test/file.pdf"] = IndexedFileInfo(
            file_path="/test/file.pdf",
            file_hash="abc123",
            file_size=1024,
            modified_time=123456.789,
            indexed_at="2024-01-01T12:00:00",
            chunk_count=10,
            collection_name="test"
        )

        tracker.save()
        assert manifest_path.exists()

        tracker2 = IndexTracker(manifest_path=manifest_path, collection_name="test")
        assert len(tracker2.manifest.files) == 1
        assert "/test/file.pdf" in tracker2.manifest.files
        assert tracker2.manifest.files["/test/file.pdf"].file_hash == "abc123"

    def test_save_updates_timestamp(self, temp_dir: Path):
        """Test that save updates the updated_at timestamp."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        original_time = tracker.manifest.updated_at
        time.sleep(0.01)
        tracker.save()

        assert tracker.manifest.updated_at > original_time

    def test_is_file_indexed_not_indexed(self, temp_dir: Path, test_data_dir: Path):
        """Test checking if file is indexed when it's not in manifest."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = test_data_dir / "knowledge" / "wine.pdf"
        assert not tracker.is_file_indexed(test_file)

    def test_is_file_indexed_file_exists_unchanged(self, temp_dir: Path, test_data_dir: Path):
        """Test checking if file is indexed when file exists and unchanged."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = test_data_dir / "knowledge" / "wine.pdf"
        tracker.mark_indexed(test_file, chunk_count=5)

        assert tracker.is_file_indexed(test_file)

    def test_is_file_indexed_file_changed(self, temp_dir: Path):
        """Test checking if file is indexed when file has changed."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = temp_dir / "test.txt"
        test_file.write_text("Original content")

        tracker.mark_indexed(test_file, chunk_count=1)
        assert tracker.is_file_indexed(test_file)

        time.sleep(0.01)
        test_file.write_text("Modified content")

        assert not tracker.is_file_indexed(test_file)

    def test_is_file_indexed_file_not_exists(self, temp_dir: Path):
        """Test checking if file is indexed when file no longer exists."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = temp_dir / "test.txt"
        test_file.write_text("Content")

        tracker.mark_indexed(test_file, chunk_count=1)
        test_file.unlink()

        assert not tracker.is_file_indexed(test_file)

    def test_get_files_to_index_all_new(self, temp_dir: Path):
        """Test filtering files when all are new."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        files_to_index = tracker.get_files_to_index([file1, file2])

        assert len(files_to_index) == 2
        assert file1 in files_to_index
        assert file2 in files_to_index

    def test_get_files_to_index_some_indexed(self, temp_dir: Path):
        """Test filtering files when some are already indexed."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file3 = temp_dir / "file3.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        file3.write_text("Content 3")

        tracker.mark_indexed(file1, chunk_count=1)
        tracker.mark_indexed(file2, chunk_count=2)

        files_to_index = tracker.get_files_to_index([file1, file2, file3])

        assert len(files_to_index) == 1
        assert file3 in files_to_index
        assert file1 not in files_to_index
        assert file2 not in files_to_index

    def test_get_files_to_index_all_indexed(self, temp_dir: Path):
        """Test filtering files when all are already indexed."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        tracker.mark_indexed(file1, chunk_count=1)
        tracker.mark_indexed(file2, chunk_count=2)

        files_to_index = tracker.get_files_to_index([file1, file2])

        assert len(files_to_index) == 0

    def test_get_files_to_index_modified_file(self, temp_dir: Path):
        """Test filtering files when a file has been modified."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        tracker.mark_indexed(file1, chunk_count=1)
        tracker.mark_indexed(file2, chunk_count=2)

        time.sleep(0.01)
        file1.write_text("Modified content")

        files_to_index = tracker.get_files_to_index([file1, file2])

        assert len(files_to_index) == 1
        assert file1 in files_to_index

    def test_mark_indexed(self, temp_dir: Path):
        """Test marking a file as indexed."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        tracker.mark_indexed(test_file, chunk_count=5)

        abs_path = str(test_file.absolute())
        assert abs_path in tracker.manifest.files
        info = tracker.manifest.files[abs_path]
        assert info.file_path == abs_path
        assert info.chunk_count == 5
        assert info.collection_name == "test"
        assert info.file_size == len("Test content")
        assert isinstance(info.file_hash, str)
        assert isinstance(info.modified_time, float)
        assert isinstance(info.indexed_at, str)

    def test_mark_indexed_updates_existing(self, temp_dir: Path):
        """Test marking a file as indexed updates existing entry."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = temp_dir / "test.txt"
        test_file.write_text("Original content")

        tracker.mark_indexed(test_file, chunk_count=3)
        original_hash = tracker.manifest.files[str(test_file.absolute())].file_hash

        time.sleep(0.01)
        test_file.write_text("Modified content")
        tracker.mark_indexed(test_file, chunk_count=5)

        abs_path = str(test_file.absolute())
        info = tracker.manifest.files[abs_path]
        assert info.chunk_count == 5
        assert info.file_hash != original_hash

    def test_remove_file(self, temp_dir: Path):
        """Test removing a file from manifest."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        tracker.mark_indexed(test_file, chunk_count=5)
        assert str(test_file.absolute()) in tracker.manifest.files

        result = tracker.remove_file(test_file)
        assert result is True
        assert str(test_file.absolute()) not in tracker.manifest.files

    def test_remove_file_not_in_manifest(self, temp_dir: Path):
        """Test removing a file that's not in manifest."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        result = tracker.remove_file(test_file)
        assert result is False

    def test_get_indexed_files(self, temp_dir: Path):
        """Test getting set of all indexed file paths."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        tracker.mark_indexed(file1, chunk_count=1)
        tracker.mark_indexed(file2, chunk_count=2)

        indexed_files = tracker.get_indexed_files()

        assert len(indexed_files) == 2
        assert str(file1.absolute()) in indexed_files
        assert str(file2.absolute()) in indexed_files

    def test_get_indexed_files_empty(self, temp_dir: Path):
        """Test getting indexed files when manifest is empty."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        indexed_files = tracker.get_indexed_files()

        assert len(indexed_files) == 0
        assert isinstance(indexed_files, set)

    def test_get_stats(self, temp_dir: Path):
        """Test getting statistics about indexed files."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        tracker.mark_indexed(file1, chunk_count=5)
        tracker.mark_indexed(file2, chunk_count=3)

        stats = tracker.get_stats()

        assert stats["total_files"] == 2
        assert stats["total_chunks"] == 8
        assert stats["collection_name"] == "test"
        assert isinstance(stats["last_updated"], str)

    def test_get_stats_empty(self, temp_dir: Path):
        """Test getting statistics when no files indexed."""
        manifest_path = temp_dir / "manifest.json"
        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        stats = tracker.get_stats()

        assert stats["total_files"] == 0
        assert stats["total_chunks"] == 0
        assert stats["collection_name"] == "test"

    def test_load_corrupt_manifest(self, temp_dir: Path):
        """Test loading a corrupt manifest file creates new manifest."""
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text("invalid json content {")

        tracker = IndexTracker(manifest_path=manifest_path, collection_name="test")

        assert tracker.manifest.collection_name == "test"
        assert len(tracker.manifest.files) == 0

    def test_manifest_persistence_across_instances(self, temp_dir: Path):
        """Test manifest persists across multiple tracker instances."""
        manifest_path = temp_dir / "manifest.json"

        tracker1 = IndexTracker(manifest_path=manifest_path, collection_name="test")
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")
        tracker1.mark_indexed(test_file, chunk_count=5)
        tracker1.save()

        tracker2 = IndexTracker(manifest_path=manifest_path, collection_name="test")
        assert len(tracker2.manifest.files) == 1
        assert str(test_file.absolute()) in tracker2.manifest.files

    def test_multiple_collections_separate_manifests(self, temp_dir: Path):
        """Test that different collections have separate manifests."""
        tracker1 = IndexTracker(
            manifest_path=temp_dir / "manifest1.json",
            collection_name="collection1"
        )
        tracker2 = IndexTracker(
            manifest_path=temp_dir / "manifest2.json",
            collection_name="collection2"
        )

        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        tracker1.mark_indexed(file1, chunk_count=1)
        tracker2.mark_indexed(file2, chunk_count=2)

        assert len(tracker1.manifest.files) == 1
        assert len(tracker2.manifest.files) == 1
        assert str(file1.absolute()) in tracker1.manifest.files
        assert str(file2.absolute()) in tracker2.manifest.files
        assert str(file1.absolute()) not in tracker2.manifest.files
        assert str(file2.absolute()) not in tracker1.manifest.files

    def test_default_manifest_dir_creation(self, temp_dir: Path, monkeypatch):
        """Test that default manifest directory is created if it doesn't exist."""
        manifest_dir = temp_dir / "manifests"
        assert not manifest_dir.exists()

        monkeypatch.setattr(
            "src.chroma.index_tracker.IndexTracker.DEFAULT_MANIFEST_DIR",
            manifest_dir
        )

        tracker = IndexTracker(collection_name="test")

        assert manifest_dir.exists()
        assert tracker.manifest_path.parent == manifest_dir
