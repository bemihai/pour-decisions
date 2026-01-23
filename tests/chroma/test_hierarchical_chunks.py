"""Unit tests for src/chroma/hierarchical_chunks.py"""

import pytest

from src.chroma.hierarchical_chunks import (
    HierarchicalChunk,
    create_hierarchical_chunks,
    expand_to_parent_context,
    prepare_chunks_for_indexing,
)


class TestHierarchicalChunk:
    """Test HierarchicalChunk dataclass."""

    def test_hierarchical_chunk_creation(self):
        """Test creating a HierarchicalChunk with all fields."""
        chunk = HierarchicalChunk(
            small_text="Small chunk text",
            large_text="Large context containing the small chunk text",
            chunk_id="chunk_0",
            metadata={"page": 1, "source": "test.pdf"},
        )

        assert chunk.small_text == "Small chunk text"
        assert chunk.large_text == "Large context containing the small chunk text"
        assert chunk.chunk_id == "chunk_0"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["source"] == "test.pdf"

    def test_hierarchical_chunk_empty_metadata(self):
        """Test creating chunk with empty metadata."""
        chunk = HierarchicalChunk(
            small_text="Text",
            large_text="Larger text",
            chunk_id="chunk_1",
            metadata={},
        )

        assert chunk.metadata == {}
        assert isinstance(chunk.metadata, dict)


class TestCreateHierarchicalChunks:
    """Test create_hierarchical_chunks function."""

    def test_create_chunks_basic(self):
        """Test creating hierarchical chunks with default parameters."""
        text = "a" * 1000  # 1000 characters
        chunks = create_hierarchical_chunks(text)

        assert len(chunks) > 0
        assert all(isinstance(c, HierarchicalChunk) for c in chunks)
        assert all(len(c.small_text) <= 256 for c in chunks)
        assert all(len(c.large_text) <= 1024 for c in chunks)

    def test_create_chunks_with_custom_sizes(self):
        """Test creating chunks with custom sizes."""
        text = "This is a test. " * 100  # ~1600 characters
        chunks = create_hierarchical_chunks(
            text,
            small_chunk_size=100,
            large_chunk_size=300,
            overlap=20,
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.small_text) <= 100
            assert len(chunk.large_text) <= 300
            assert chunk.small_text in chunk.large_text

    def test_create_chunks_empty_text(self):
        """Test creating chunks from empty text."""
        chunks = create_hierarchical_chunks("")

        assert len(chunks) == 0

    def test_create_chunks_whitespace_only(self):
        """Test creating chunks from whitespace-only text."""
        chunks = create_hierarchical_chunks("   \n\t   ")

        assert len(chunks) == 0

    def test_create_chunks_small_text(self):
        """Test creating chunks from text smaller than chunk size."""
        text = "Short text"
        chunks = create_hierarchical_chunks(text, small_chunk_size=256, large_chunk_size=1024)

        assert len(chunks) == 1
        assert chunks[0].small_text == "Short text"
        assert chunks[0].large_text == "Short text"

    def test_create_chunks_overlap(self):
        """Test that overlap creates overlapping chunks."""
        text = "a" * 500
        chunks = create_hierarchical_chunks(text, small_chunk_size=200, overlap=50)

        assert len(chunks) >= 2
        # Check that chunks have proper IDs
        assert chunks[0].chunk_id == "chunk_0"
        assert chunks[1].chunk_id == "chunk_1"

    def test_create_chunks_metadata_positions(self):
        """Test that metadata contains correct position information."""
        text = "a" * 600
        chunks = create_hierarchical_chunks(text, small_chunk_size=200, large_chunk_size=400)

        for chunk in chunks:
            assert "small_start" in chunk.metadata
            assert "small_end" in chunk.metadata
            assert "large_start" in chunk.metadata
            assert "large_end" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["small_end"] > chunk.metadata["small_start"]
            assert chunk.metadata["large_end"] > chunk.metadata["large_start"]

    def test_create_chunks_large_contains_small(self):
        """Test that large chunk always contains the small chunk."""
        text = "The quick brown fox jumps over the lazy dog. " * 50
        chunks = create_hierarchical_chunks(text, small_chunk_size=100, large_chunk_size=300)

        for chunk in chunks:
            assert chunk.small_text.strip() in chunk.large_text

    def test_create_chunks_sequential_ids(self):
        """Test that chunk IDs are sequential."""
        text = "Text " * 300
        chunks = create_hierarchical_chunks(text, small_chunk_size=150, large_chunk_size=400)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"chunk_{i}"
            assert chunk.metadata["chunk_index"] == i

    def test_create_chunks_no_overlap(self):
        """Test creating chunks with no overlap."""
        text = "a" * 500
        chunks = create_hierarchical_chunks(text, small_chunk_size=100, overlap=0)

        # With no overlap, we should get 5 chunks (500 / 100)
        assert len(chunks) == 5

    def test_create_chunks_context_padding(self):
        """Test that context padding is applied correctly."""
        text = "x" * 1000
        chunks = create_hierarchical_chunks(text, small_chunk_size=200, large_chunk_size=600)

        # Context padding = (600 - 200) / 2 = 200
        # First chunk should have large_start = 0 (bounded by text start)
        assert chunks[0].metadata["large_start"] == 0
        # Large text should be bigger than small text
        assert len(chunks[0].large_text) > len(chunks[0].small_text)

    def test_create_chunks_strips_whitespace(self):
        """Test that chunks are stripped of leading/trailing whitespace."""
        text = "  word1  " + (" " * 300) + "  word2  "
        chunks = create_hierarchical_chunks(text, small_chunk_size=50, large_chunk_size=150, overlap=10)

        for chunk in chunks:
            if chunk.small_text:  # Only check non-empty chunks
                assert chunk.small_text == chunk.small_text.strip()
                assert chunk.large_text == chunk.large_text.strip()

    def test_create_chunks_exact_boundary(self):
        """Test chunking when text length is exact multiple of chunk size."""
        text = "a" * 512
        chunks = create_hierarchical_chunks(text, small_chunk_size=256, overlap=0)

        assert len(chunks) == 2
        assert len(chunks[0].small_text) == 256
        assert len(chunks[1].small_text) == 256

    def test_create_chunks_invalid_overlap(self):
        """Test that ValueError is raised when overlap >= chunk_size."""
        text = "Test text for validation"

        # Test overlap equal to chunk_size
        with pytest.raises(ValueError, match="overlap.*must be less than small_chunk_size"):
            create_hierarchical_chunks(text, small_chunk_size=50, overlap=50)

        # Test overlap greater than chunk_size
        with pytest.raises(ValueError, match="overlap.*must be less than small_chunk_size"):
            create_hierarchical_chunks(text, small_chunk_size=50, overlap=100)


class TestExpandToParentContext:
    """Test expand_to_parent_context function."""

    def test_expand_with_parent_context(self):
        """Test expanding documents with parent context."""
        docs = [
            {
                "document": "Small text",
                "metadata": {"parent_context": "Large context with small text"},
            },
            {
                "document": "Another small",
                "metadata": {"parent_context": "Another large context with another small"},
            },
        ]

        result = expand_to_parent_context(docs, use_large_context=True)

        assert len(result) == 2
        assert result[0]["document"] == "Large context with small text"
        assert result[1]["document"] == "Another large context with another small"
        assert result[0]["used_parent_context"] is True
        assert result[1]["used_parent_context"] is True

    def test_expand_without_parent_context(self):
        """Test expanding documents without parent context."""
        docs = [
            {
                "document": "Text without parent",
                "metadata": {},
            },
        ]

        result = expand_to_parent_context(docs, use_large_context=True)

        assert len(result) == 1
        assert result[0]["document"] == "Text without parent"
        assert result[0]["used_parent_context"] is False

    def test_expand_disabled(self):
        """Test that expansion can be disabled."""
        docs = [
            {
                "document": "Small text",
                "metadata": {"parent_context": "Large context"},
            },
        ]

        result = expand_to_parent_context(docs, use_large_context=False)

        assert result[0]["document"] == "Small text"
        assert result[0]["used_parent_context"] is False

    def test_expand_empty_list(self):
        """Test expanding empty document list."""
        result = expand_to_parent_context([], use_large_context=True)

        assert len(result) == 0

    def test_expand_preserves_original_docs(self):
        """Test that original documents are not modified."""
        original_docs = [
            {
                "document": "Original",
                "metadata": {"parent_context": "Expanded"},
            },
        ]

        result = expand_to_parent_context(original_docs, use_large_context=True)

        # Original should not be modified
        assert original_docs[0]["document"] == "Original"
        # Result should be modified
        assert result[0]["document"] == "Expanded"

    def test_expand_mixed_documents(self):
        """Test expanding mix of documents with and without parent context."""
        docs = [
            {
                "document": "Text 1",
                "metadata": {"parent_context": "Large 1"},
            },
            {
                "document": "Text 2",
                "metadata": {},
            },
            {
                "document": "Text 3",
                "metadata": {"parent_context": "Large 3"},
            },
        ]

        result = expand_to_parent_context(docs, use_large_context=True)

        assert result[0]["document"] == "Large 1"
        assert result[0]["used_parent_context"] is True
        assert result[1]["document"] == "Text 2"
        assert result[1]["used_parent_context"] is False
        assert result[2]["document"] == "Large 3"
        assert result[2]["used_parent_context"] is True

    def test_expand_empty_parent_context(self):
        """Test that empty parent context is not used."""
        docs = [
            {
                "document": "Text",
                "metadata": {"parent_context": ""},
            },
        ]

        result = expand_to_parent_context(docs, use_large_context=True)

        assert result[0]["document"] == "Text"
        assert result[0]["used_parent_context"] is False

    def test_expand_preserves_other_metadata(self):
        """Test that other metadata fields are preserved."""
        docs = [
            {
                "document": "Text",
                "metadata": {
                    "parent_context": "Large",
                    "page": 5,
                    "source": "book.pdf",
                    "score": 0.95,
                },
            },
        ]

        result = expand_to_parent_context(docs, use_large_context=True)

        assert result[0]["metadata"]["page"] == 5
        assert result[0]["metadata"]["source"] == "book.pdf"
        assert result[0]["metadata"]["score"] == 0.95
        assert result[0]["metadata"]["parent_context"] == "Large"


class TestPrepareChunksForIndexing:
    """Test prepare_chunks_for_indexing function."""

    def test_prepare_basic(self):
        """Test preparing hierarchical chunks for indexing."""
        chunks = [
            HierarchicalChunk(
                small_text="Small 1",
                large_text="Large context 1",
                chunk_id="chunk_0",
                metadata={"chunk_index": 0},
            ),
            HierarchicalChunk(
                small_text="Small 2",
                large_text="Large context 2",
                chunk_id="chunk_1",
                metadata={"chunk_index": 1},
            ),
        ]

        file_metadata = {"filename": "test.pdf", "source": "wine_books"}

        result = prepare_chunks_for_indexing(chunks, file_metadata)

        assert len(result) == 2
        assert result[0]["id"] == "chunk_0"
        assert result[0]["text"] == "Small 1"
        assert result[0]["metadata"]["parent_context"] == "Large context 1"
        assert result[0]["metadata"]["filename"] == "test.pdf"
        assert result[1]["id"] == "chunk_1"
        assert result[1]["text"] == "Small 2"

    def test_prepare_merges_metadata(self):
        """Test that file metadata and chunk metadata are merged."""
        chunks = [
            HierarchicalChunk(
                small_text="Text",
                large_text="Context",
                chunk_id="chunk_0",
                metadata={"chunk_index": 0, "page": 1},
            ),
        ]

        file_metadata = {"filename": "book.pdf", "author": "Wine Expert"}

        result = prepare_chunks_for_indexing(chunks, file_metadata)

        assert result[0]["metadata"]["filename"] == "book.pdf"
        assert result[0]["metadata"]["author"] == "Wine Expert"
        assert result[0]["metadata"]["chunk_index"] == 0
        assert result[0]["metadata"]["page"] == 1

    def test_prepare_adds_size_metadata(self):
        """Test that chunk sizes are added to metadata."""
        chunks = [
            HierarchicalChunk(
                small_text="Small",
                large_text="Large context text",
                chunk_id="chunk_0",
                metadata={},
            ),
        ]

        result = prepare_chunks_for_indexing(chunks, {})

        assert result[0]["metadata"]["small_chunk_size"] == len("Small")
        assert result[0]["metadata"]["large_chunk_size"] == len("Large context text")

    def test_prepare_empty_chunks(self):
        """Test preparing empty list of chunks."""
        result = prepare_chunks_for_indexing([], {"filename": "test.pdf"})

        assert len(result) == 0

    def test_prepare_empty_file_metadata(self):
        """Test preparing chunks with empty file metadata."""
        chunks = [
            HierarchicalChunk(
                small_text="Text",
                large_text="Context",
                chunk_id="chunk_0",
                metadata={"page": 1},
            ),
        ]

        result = prepare_chunks_for_indexing(chunks, {})

        assert len(result) == 1
        assert result[0]["metadata"]["page"] == 1
        assert result[0]["metadata"]["parent_context"] == "Context"

    def test_prepare_indexes_small_text(self):
        """Test that small text is used as the indexed text."""
        chunks = [
            HierarchicalChunk(
                small_text="This is the small chunk for embedding",
                large_text="This is a much larger context that contains the small chunk for embedding and more",
                chunk_id="chunk_0",
                metadata={},
            ),
        ]

        result = prepare_chunks_for_indexing(chunks, {})

        assert result[0]["text"] == "This is the small chunk for embedding"
        assert result[0]["metadata"]["parent_context"] == "This is a much larger context that contains the small chunk for embedding and more"

    def test_prepare_preserves_chunk_ids(self):
        """Test that chunk IDs are preserved correctly."""
        chunks = [
            HierarchicalChunk(
                small_text="Text 1",
                large_text="Context 1",
                chunk_id="custom_id_0",
                metadata={},
            ),
            HierarchicalChunk(
                small_text="Text 2",
                large_text="Context 2",
                chunk_id="custom_id_1",
                metadata={},
            ),
        ]

        result = prepare_chunks_for_indexing(chunks, {})

        assert result[0]["id"] == "custom_id_0"
        assert result[1]["id"] == "custom_id_1"

    def test_prepare_metadata_override(self):
        """Test that chunk metadata can override file metadata."""
        chunks = [
            HierarchicalChunk(
                small_text="Text",
                large_text="Context",
                chunk_id="chunk_0",
                metadata={"source": "chunk_specific"},
            ),
        ]

        file_metadata = {"source": "file_level"}

        result = prepare_chunks_for_indexing(chunks, file_metadata)

        # Chunk metadata should override file metadata
        assert result[0]["metadata"]["source"] == "chunk_specific"

    def test_prepare_multiple_chunks_different_sizes(self):
        """Test preparing chunks with different text sizes."""
        chunks = [
            HierarchicalChunk(
                small_text="a" * 50,
                large_text="b" * 200,
                chunk_id="chunk_0",
                metadata={},
            ),
            HierarchicalChunk(
                small_text="c" * 100,
                large_text="d" * 500,
                chunk_id="chunk_1",
                metadata={},
            ),
        ]

        result = prepare_chunks_for_indexing(chunks, {})

        assert result[0]["metadata"]["small_chunk_size"] == 50
        assert result[0]["metadata"]["large_chunk_size"] == 200
        assert result[1]["metadata"]["small_chunk_size"] == 100
        assert result[1]["metadata"]["large_chunk_size"] == 500


class TestIntegrationHierarchicalChunking:
    """Integration tests for the full hierarchical chunking workflow."""

    def test_full_workflow(self):
        """Test complete workflow from text to indexed chunks."""
        text = "Wine is a fascinating beverage. " * 50

        # Step 1: Create hierarchical chunks
        hierarchical_chunks = create_hierarchical_chunks(
            text,
            small_chunk_size=100,
            large_chunk_size=300,
            overlap=20,
        )

        assert len(hierarchical_chunks) > 0

        # Step 2: Prepare for indexing
        file_metadata = {"filename": "wine.txt", "category": "wine_education"}
        indexed_chunks = prepare_chunks_for_indexing(hierarchical_chunks, file_metadata)

        assert len(indexed_chunks) == len(hierarchical_chunks)
        assert all("parent_context" in c["metadata"] for c in indexed_chunks)
        assert all(c["metadata"]["filename"] == "wine.txt" for c in indexed_chunks)

        # Step 3: Simulate retrieval and expansion
        retrieved = indexed_chunks[:2]  # Simulate retrieving first 2 chunks
        expanded = expand_to_parent_context(
            [{"document": c["text"], "metadata": c["metadata"]} for c in retrieved],
            use_large_context=True,
        )

        assert len(expanded) == 2
        assert all(d["used_parent_context"] is True for d in expanded)
        assert all(len(d["document"]) > len(indexed_chunks[i]["text"]) for i, d in enumerate(expanded))

    def test_workflow_with_short_text(self):
        """Test workflow with text shorter than chunk sizes."""
        text = "Short wine description."

        hierarchical_chunks = create_hierarchical_chunks(text)
        assert len(hierarchical_chunks) == 1

        indexed = prepare_chunks_for_indexing(hierarchical_chunks, {"source": "test"})
        assert len(indexed) == 1
        assert indexed[0]["text"] == "Short wine description."

        expanded = expand_to_parent_context(
            [{"document": indexed[0]["text"], "metadata": indexed[0]["metadata"]}],
            use_large_context=True,
        )
        assert expanded[0]["document"] == "Short wine description."
        assert expanded[0]["used_parent_context"] is True

    def test_workflow_disabled_expansion(self):
        """Test workflow with expansion disabled."""
        text = "Test text. " * 100

        hierarchical_chunks = create_hierarchical_chunks(text, small_chunk_size=50, overlap=10)
        indexed = prepare_chunks_for_indexing(hierarchical_chunks, {})

        retrieved = [{"document": c["text"], "metadata": c["metadata"]} for c in indexed[:1]]
        expanded = expand_to_parent_context(retrieved, use_large_context=False)

        assert expanded[0]["used_parent_context"] is False
        assert expanded[0]["document"] == indexed[0]["text"]
