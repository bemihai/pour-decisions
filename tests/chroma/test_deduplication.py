"""Unit tests for src/chroma/deduplication.py"""

from unittest.mock import Mock, patch

import numpy as np

from src.chroma.deduplication import (
    deduplicate_by_content_hash,
    deduplicate_chunks,
    deduplicate_context,
)


class TestDeduplicateByContentHash:
    """Test deduplicate_by_content_hash function."""

    def test_no_duplicates(self):
        """Test deduplication when there are no duplicates."""
        chunks = [
            {"document": "First chunk", "metadata": {"content_hash": "hash1"}},
            {"document": "Second chunk", "metadata": {"content_hash": "hash2"}},
            {"document": "Third chunk", "metadata": {"content_hash": "hash3"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 3
        assert result == chunks

    def test_with_duplicates(self):
        """Test deduplication when there are exact duplicates."""
        chunks = [
            {"document": "First chunk", "metadata": {"content_hash": "hash1"}},
            {"document": "Second chunk", "metadata": {"content_hash": "hash2"}},
            {"document": "First chunk duplicate", "metadata": {"content_hash": "hash1"}},
            {"document": "Third chunk", "metadata": {"content_hash": "hash3"}},
            {"document": "Second chunk duplicate", "metadata": {"content_hash": "hash2"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 3
        assert result[0]["metadata"]["content_hash"] == "hash1"
        assert result[1]["metadata"]["content_hash"] == "hash2"
        assert result[2]["metadata"]["content_hash"] == "hash3"
        assert result[0]["document"] == "First chunk"

    def test_preserves_order(self):
        """Test that deduplication preserves order of first occurrence."""
        chunks = [
            {"document": "A", "metadata": {"content_hash": "hash1"}},
            {"document": "B", "metadata": {"content_hash": "hash2"}},
            {"document": "C", "metadata": {"content_hash": "hash3"}},
            {"document": "A_dup", "metadata": {"content_hash": "hash1"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 3
        assert result[0]["document"] == "A"
        assert result[1]["document"] == "B"
        assert result[2]["document"] == "C"

    def test_missing_content_hash(self):
        """Test handling chunks without content_hash."""
        chunks = [
            {"document": "First chunk", "metadata": {"content_hash": "hash1"}},
            {"document": "No hash chunk", "metadata": {}},
            {"document": "Another no hash", "metadata": {"other_field": "value"}},
            {"document": "Second chunk", "metadata": {"content_hash": "hash2"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 4
        assert result[1]["document"] == "No hash chunk"
        assert result[2]["document"] == "Another no hash"

    def test_missing_metadata(self):
        """Test handling chunks without metadata."""
        chunks = [
            {"document": "First chunk", "metadata": {"content_hash": "hash1"}},
            {"document": "No metadata chunk"},
            {"document": "Second chunk", "metadata": {"content_hash": "hash2"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 3
        assert result[1]["document"] == "No metadata chunk"

    def test_empty_list(self):
        """Test deduplication with empty list."""
        chunks = []

        result = deduplicate_by_content_hash(chunks)

        assert result == []

    def test_single_chunk(self):
        """Test deduplication with single chunk."""
        chunks = [{"document": "Only chunk", "metadata": {"content_hash": "hash1"}}]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 1
        assert result[0] == chunks[0]

    def test_all_duplicates(self):
        """Test when all chunks are duplicates of first."""
        chunks = [
            {"document": "First", "metadata": {"content_hash": "hash1"}},
            {"document": "Second", "metadata": {"content_hash": "hash1"}},
            {"document": "Third", "metadata": {"content_hash": "hash1"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 1
        assert result[0]["document"] == "First"

    def test_empty_hash_values(self):
        """Test handling of empty hash values."""
        chunks = [
            {"document": "First chunk", "metadata": {"content_hash": ""}},
            {"document": "Second chunk", "metadata": {"content_hash": ""}},
            {"document": "Third chunk", "metadata": {"content_hash": "hash1"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 3


class TestDeduplicateChunks:
    """Test deduplicate_chunks function for semantic deduplication."""

    def test_empty_list(self):
        """Test deduplication with empty list."""
        chunks = []

        result = deduplicate_chunks(chunks)

        assert result == []

    def test_single_chunk(self):
        """Test deduplication with single chunk."""
        chunks = [{"document": "Only chunk"}]

        result = deduplicate_chunks(chunks)

        assert len(result) == 1
        assert result[0] == chunks[0]

    @patch("src.chroma.deduplication.get_embedder")
    @patch("src.chroma.deduplication.cosine_similarity")
    def test_no_semantic_duplicates(self, mock_cosine, mock_get_embedder):
        """Test when chunks are semantically different."""
        chunks = [
            {"document": "Red wine from France"},
            {"document": "White wine from Italy"},
            {"document": "Champagne from Spain"},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        mock_get_embedder.return_value = mock_embedder
        mock_cosine.return_value = 0.1

        result = deduplicate_chunks(chunks, similarity_threshold=0.90)

        assert len(result) == 3
        assert result == chunks

    @patch("src.chroma.deduplication.get_embedder")
    @patch("src.chroma.deduplication.cosine_similarity")
    def test_with_semantic_duplicates(self, mock_cosine, mock_get_embedder):
        """Test when chunks are semantically similar."""
        chunks = [
            {"document": "Red wine from Bordeaux"},
            {"document": "Bordeaux red wine"},
            {"document": "White wine from Italy"},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0]),
            np.array([0.95, 0.05]),
            np.array([0.0, 1.0]),
        ]
        mock_get_embedder.return_value = mock_embedder

        def cosine_side_effect(vec1, vec2):
            similarity = np.dot(vec1, vec2)
            return float(similarity)

        mock_cosine.side_effect = cosine_side_effect

        result = deduplicate_chunks(chunks, similarity_threshold=0.90)

        assert len(result) == 2
        assert result[0]["document"] == "Red wine from Bordeaux"
        assert result[1]["document"] == "White wine from Italy"

    @patch("src.chroma.deduplication.get_embedder")
    @patch("src.chroma.deduplication.cosine_similarity")
    def test_preserves_first_occurrence(self, mock_cosine, mock_get_embedder):
        """Test that first occurrence is kept for duplicates."""
        chunks = [
            {"document": "First version", "score": 10},
            {"document": "Duplicate version", "score": 5},
            {"document": "Another duplicate", "score": 3},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
        ]
        mock_get_embedder.return_value = mock_embedder
        mock_cosine.return_value = 0.99

        result = deduplicate_chunks(chunks, similarity_threshold=0.90)

        assert len(result) == 1
        assert result[0]["document"] == "First version"
        assert result[0]["score"] == 10

    @patch("src.chroma.deduplication.get_embedder")
    @patch("src.chroma.deduplication.cosine_similarity")
    def test_custom_similarity_threshold(self, mock_cosine, mock_get_embedder):
        """Test custom similarity threshold."""
        chunks = [
            {"document": "Chunk 1"},
            {"document": "Chunk 2"},
            {"document": "Chunk 3"},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0]),
            np.array([0.85, 0.15]),
            np.array([0.0, 1.0]),
        ]
        mock_get_embedder.return_value = mock_embedder

        mock_cosine.return_value = 0.85
        result = deduplicate_chunks(chunks, similarity_threshold=0.80)
        assert len(result) < 3

        mock_cosine.return_value = 0.10
        result = deduplicate_chunks(chunks, similarity_threshold=0.90)
        assert len(result) == 3

    @patch("src.chroma.deduplication.get_embedder")
    def test_missing_document_key(self, mock_get_embedder):
        """Test handling chunks without 'document' key."""
        chunks = [
            {"document": "Chunk 1"},
            {"text": "Chunk 2"},
            {"document": "Chunk 3"},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
        ]
        mock_get_embedder.return_value = mock_embedder

        result = deduplicate_chunks(chunks, similarity_threshold=0.90)

        assert len(result) == 3

    @patch("src.chroma.deduplication.get_embedder")
    def test_custom_embedding_model(self, mock_get_embedder):
        """Test passing custom embedding model."""
        chunks = [
            {"document": "Chunk 1"},
            {"document": "Chunk 2"},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ]
        mock_get_embedder.return_value = mock_embedder

        deduplicate_chunks(chunks, embedding_model="custom-model")

        mock_get_embedder.assert_called_once_with("custom-model")

    @patch("src.chroma.deduplication.get_embedder")
    @patch("src.chroma.deduplication.cosine_similarity")
    def test_preserves_order(self, mock_cosine, mock_get_embedder):
        """Test that original order is preserved."""
        chunks = [
            {"document": "A", "id": 1},
            {"document": "B", "id": 2},
            {"document": "C", "id": 3},
            {"document": "D", "id": 4},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            np.array([0.2, 0.8]),
        ]
        mock_get_embedder.return_value = mock_embedder
        mock_cosine.return_value = 0.1

        result = deduplicate_chunks(chunks, similarity_threshold=0.90)

        assert len(result) == 4
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
        assert result[2]["id"] == 3
        assert result[3]["id"] == 4


class TestDeduplicateContext:
    """Test deduplicate_context full pipeline."""

    def test_empty_list(self):
        """Test with empty chunk list."""
        chunks = []

        result = deduplicate_context(chunks)

        assert result == []

    @patch("src.chroma.deduplication.deduplicate_by_content_hash")
    @patch("src.chroma.deduplication.deduplicate_chunks")
    def test_uses_hash_first_by_default(self, mock_semantic, mock_hash):
        """Test that hash deduplication runs first by default."""
        chunks = [
            {"document": "Chunk 1", "metadata": {"content_hash": "hash1"}},
            {"document": "Chunk 2", "metadata": {"content_hash": "hash2"}},
        ]

        mock_hash.return_value = chunks
        mock_semantic.return_value = chunks

        result = deduplicate_context(chunks)

        mock_hash.assert_called_once_with(chunks)
        mock_semantic.assert_called_once()

    @patch("src.chroma.deduplication.deduplicate_by_content_hash")
    @patch("src.chroma.deduplication.deduplicate_chunks")
    def test_skip_hash_when_disabled(self, mock_semantic, mock_hash):
        """Test skipping hash deduplication when disabled."""
        chunks = [
            {"document": "Chunk 1"},
            {"document": "Chunk 2"},
        ]

        mock_semantic.return_value = chunks

        result = deduplicate_context(chunks, use_hash_first=False)

        mock_hash.assert_not_called()
        mock_semantic.assert_called_once_with(
            chunks,
            similarity_threshold=0.90,
            embedding_model=None,
        )

    @patch("src.chroma.deduplication.deduplicate_by_content_hash")
    @patch("src.chroma.deduplication.deduplicate_chunks")
    def test_passes_parameters_correctly(self, mock_semantic, mock_hash):
        """Test that parameters are passed correctly to sub-functions."""
        chunks = [
            {"document": "Chunk 1"},
            {"document": "Chunk 2"},
        ]

        mock_hash.return_value = chunks
        mock_semantic.return_value = chunks

        deduplicate_context(
            chunks,
            similarity_threshold=0.85,
            embedding_model="test-model",
            use_hash_first=True,
        )

        mock_hash.assert_called_once_with(chunks)
        mock_semantic.assert_called_once_with(
            chunks,
            similarity_threshold=0.85,
            embedding_model="test-model",
        )

    @patch("src.chroma.deduplication.deduplicate_by_content_hash")
    @patch("src.chroma.deduplication.deduplicate_chunks")
    def test_skips_semantic_for_single_chunk(self, mock_semantic, mock_hash):
        """Test that semantic dedup is skipped for single chunk after hash dedup."""
        chunks = [
            {"document": "Only chunk", "metadata": {"content_hash": "hash1"}},
        ]

        mock_hash.return_value = chunks

        result = deduplicate_context(chunks, use_hash_first=True)

        mock_hash.assert_called_once()
        mock_semantic.assert_not_called()

    @patch("src.chroma.deduplication.deduplicate_by_content_hash")
    @patch("src.chroma.deduplication.deduplicate_chunks")
    def test_full_pipeline_integration(self, mock_semantic, mock_hash):
        """Test full pipeline with both deduplication methods."""
        initial_chunks = [
            {"document": "A", "metadata": {"content_hash": "hash1"}},
            {"document": "A_dup", "metadata": {"content_hash": "hash1"}},
            {"document": "B", "metadata": {"content_hash": "hash2"}},
            {"document": "C", "metadata": {"content_hash": "hash3"}},
        ]

        after_hash = [
            {"document": "A", "metadata": {"content_hash": "hash1"}},
            {"document": "B", "metadata": {"content_hash": "hash2"}},
            {"document": "C", "metadata": {"content_hash": "hash3"}},
        ]

        final_chunks = [
            {"document": "A", "metadata": {"content_hash": "hash1"}},
            {"document": "B", "metadata": {"content_hash": "hash2"}},
        ]

        mock_hash.return_value = after_hash
        mock_semantic.return_value = final_chunks

        result = deduplicate_context(initial_chunks)

        assert len(result) == 2
        mock_hash.assert_called_once_with(initial_chunks)
        mock_semantic.assert_called_once_with(
            after_hash,
            similarity_threshold=0.90,
            embedding_model=None,
        )

    def test_single_chunk_no_dedup_needed(self):
        """Test that single chunk is returned without processing."""
        chunks = [{"document": "Only chunk"}]

        result = deduplicate_context(chunks)

        assert len(result) == 1
        assert result[0] == chunks[0]

    @patch("src.chroma.deduplication.deduplicate_by_content_hash")
    @patch("src.chroma.deduplication.deduplicate_chunks")
    def test_empty_after_hash_dedup(self, mock_semantic, mock_hash):
        """Test handling when hash dedup returns empty list."""
        chunks = [{"document": "Chunk"}]

        mock_hash.return_value = []

        result = deduplicate_context(chunks, use_hash_first=True)

        assert result == []
        mock_semantic.assert_not_called()


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    @patch("src.chroma.deduplication.get_embedder")
    @patch("src.chroma.deduplication.cosine_similarity")
    def test_mixed_duplicates_scenario(self, mock_cosine, mock_get_embedder):
        """Test scenario with both exact and semantic duplicates."""
        chunks = [
            {"document": "Bordeaux wines are excellent", "metadata": {"content_hash": "hash1"}},
            {"document": "Bordeaux wines are excellent", "metadata": {"content_hash": "hash1"}},
            {"document": "Bordeaux wine quality is superb", "metadata": {"content_hash": "hash2"}},
            {"document": "Italian Chianti is great", "metadata": {"content_hash": "hash3"}},
        ]

        mock_embedder = Mock()
        mock_embedder.embed_documents.return_value = [
            np.array([1.0, 0.0]),
            np.array([0.95, 0.05]),
            np.array([0.0, 1.0]),
        ]
        mock_get_embedder.return_value = mock_embedder
        mock_cosine.side_effect = [0.95, 0.1, 0.1]

        result = deduplicate_context(chunks, similarity_threshold=0.90)

        assert len(result) <= 2

    def test_no_duplicates_scenario(self):
        """Test scenario with completely different chunks."""
        chunks = [
            {"document": "Red wine", "metadata": {"content_hash": "hash1"}},
            {"document": "White wine", "metadata": {"content_hash": "hash2"}},
            {"document": "Rosé wine", "metadata": {"content_hash": "hash3"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 3

    @patch("src.chroma.deduplication.get_embedder")
    @patch("src.chroma.deduplication.cosine_similarity")
    def test_all_duplicates_scenario(self, mock_cosine, mock_get_embedder):
        """Test scenario where all chunks are duplicates."""
        chunks = [
            {"document": "Same content", "metadata": {"content_hash": "hash1"}},
            {"document": "Same content", "metadata": {"content_hash": "hash1"}},
            {"document": "Same content", "metadata": {"content_hash": "hash1"}},
        ]

        result = deduplicate_by_content_hash(chunks)

        assert len(result) == 1
        assert result[0]["document"] == "Same content"

    @patch("src.chroma.deduplication.get_embedder")
    def test_large_chunk_list(self, mock_get_embedder):
        """Test deduplication with large number of chunks."""
        chunks = [
            {"document": f"Chunk {i}", "metadata": {"content_hash": f"hash{i}"}}
            for i in range(100)
        ]

        mock_embedder = Mock()
        embeddings = [np.random.rand(128) for _ in range(100)]
        mock_embedder.embed_documents.return_value = embeddings
        mock_get_embedder.return_value = mock_embedder

        result = deduplicate_context(chunks, similarity_threshold=0.95)

        assert len(result) <= 100
        assert len(result) > 0
