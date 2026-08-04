"""Unit tests for src/chroma/utils.py"""

import chromadb as cdb

from src.chroma.utils import (
    create_batches,
    get_all_stats,
    get_collection_stats,
    get_or_create_collection,
    split_text_into_sentences,
    validate_chunks,
)


class TestCreateBatches:
    """Test create_batches function."""

    def test_create_batches_under_limit(self):
        """Test creating batches when data is under batch size limit."""
        ids = ["id1", "id2", "id3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        metadata = [{"a": 1}, {"b": 2}, {"c": 3}]
        documents = ["doc1", "doc2", "doc3"]

        batches = create_batches(
            ids=ids,
            embeddings=embeddings,
            metadata=metadata,
            documents=documents,
            batch_size=1000
        )

        assert len(batches) == 1
        assert batches[0] == (ids, embeddings, metadata, documents)

    def test_create_batches_over_limit(self):
        """Test creating batches when data exceeds batch size limit."""
        ids = [f"id_{i}" for i in range(2500)]
        embeddings = [[0.1, 0.2] for _ in range(2500)]
        metadata = [{"index": i} for i in range(2500)]
        documents = [f"doc_{i}" for i in range(2500)]

        batches = create_batches(
            ids=ids,
            embeddings=embeddings,
            metadata=metadata,
            documents=documents,
            batch_size=1000
        )

        assert len(batches) == 3
        assert len(batches[0][0]) == 1000
        assert len(batches[1][0]) == 1000
        assert len(batches[2][0]) == 500

    def test_create_batches_with_none_values(self):
        """Test creating batches with None values for optional parameters."""
        ids = ["id1", "id2", "id3"]

        batches = create_batches(
            ids=ids,
            embeddings=None,
            metadata=None,
            documents=None,
            batch_size=1000
        )

        assert len(batches) == 1
        assert batches[0][0] == ids
        assert batches[0][1] is None
        assert batches[0][2] is None
        assert batches[0][3] is None

    def test_create_batches_exact_batch_size(self):
        """Test creating batches when data size equals batch size."""
        ids = [f"id_{i}" for i in range(1000)]
        embeddings = [[0.1] for _ in range(1000)]

        batches = create_batches(
            ids=ids,
            embeddings=embeddings,
            batch_size=1000
        )

        assert len(batches) == 1
        assert len(batches[0][0]) == 1000

    def test_create_batches_multiple_full_batches(self):
        """Test creating multiple full batches."""
        ids = [f"id_{i}" for i in range(3000)]

        batches = create_batches(ids=ids, batch_size=1000)

        assert len(batches) == 3
        for batch in batches:
            assert len(batch[0]) == 1000


class TestGetOrCreateCollection:
    """Test get_or_create_collection function."""

    def test_get_existing_collection(self, in_memory_chroma_client: cdb.ClientAPI):
        """Test getting an existing collection."""
        collection_name = "existing_collection"
        in_memory_chroma_client.create_collection(name=collection_name)

        collection = get_or_create_collection(
            client=in_memory_chroma_client,
            name=collection_name
        )

        assert collection.name == collection_name
        assert len(in_memory_chroma_client.list_collections()) == 1

    def test_create_new_collection(self, in_memory_chroma_client: cdb.ClientAPI):
        """Test creating a new collection when it doesn't exist."""
        collection_name = "new_collection"

        collection = get_or_create_collection(
            client=in_memory_chroma_client,
            name=collection_name
        )

        assert collection.name == collection_name
        collections = in_memory_chroma_client.list_collections()
        collection_names = [c.name for c in collections]
        assert collection_name in collection_names

    def test_create_collection_with_metadata(self, in_memory_chroma_client: cdb.ClientAPI):
        """Test creating a collection with custom metadata."""
        collection_name = "metadata_collection"
        custom_metadata = {"source": "test", "version": "1.0"}

        collection = get_or_create_collection(
            client=in_memory_chroma_client,
            name=collection_name,
            metadata=custom_metadata
        )

        assert collection.name == collection_name
        assert collection.metadata["source"] == "test"
        assert collection.metadata["version"] == "1.0"
        assert "created" in collection.metadata

    def test_get_existing_collection_ignores_new_metadata(
        self,
        in_memory_chroma_client: cdb.ClientAPI
    ):
        """Test that getting an existing collection doesn't update its metadata."""
        collection_name = "existing_metadata_collection"
        original_metadata = {"original": "true"}

        in_memory_chroma_client.create_collection(
            name=collection_name,
            metadata=original_metadata
        )

        new_metadata = {"new": "metadata"}
        collection = get_or_create_collection(
            client=in_memory_chroma_client,
            name=collection_name,
            metadata=new_metadata
        )

        assert collection.metadata["original"] == "true"
        assert "new" not in collection.metadata


class TestValidateChunks:
    """Test validate_chunks function."""

    def test_validate_normal_chunks(self):
        """Test validating normal, valid chunks."""
        chunks = [
            {"text": "This is a valid chunk with enough content."},
            {"text": "Another valid chunk with sufficient text content here."},
            {"text": "Third chunk also has plenty of text to be valid."},
        ]

        valid_chunks = validate_chunks(chunks)

        assert len(valid_chunks) == 3

    def test_filter_empty_chunks(self):
        """Test that empty chunks are filtered out."""
        chunks = [
            {"text": "Valid chunk with content."},
            {"text": ""},
            {"text": "   "},
            {"text": "Another valid chunk."},
        ]

        valid_chunks = validate_chunks(chunks)

        assert len(valid_chunks) == 2
        assert all(len(chunk["text"].strip()) >= 10 for chunk in valid_chunks)

    def test_filter_short_chunks(self):
        """Test that very short chunks are filtered out."""
        chunks = [
            {"text": "Valid chunk with enough content."},
            {"text": "Short"},
            {"text": "A"},
            {"text": "This is also a valid chunk with content."},
        ]

        valid_chunks = validate_chunks(chunks)

        assert len(valid_chunks) == 2

    def test_filter_low_word_count(self):
        """Test that chunks with too few words are filtered out."""
        chunks = [
            {"text": "Valid chunk with multiple words here."},
            {"text": "One two"},
            {"text": "Only two"},
            {"text": "Another valid chunk with sufficient words."},
        ]

        valid_chunks = validate_chunks(chunks)

        assert len(valid_chunks) == 2
        assert all(len(chunk["text"].split()) >= 3 for chunk in valid_chunks)

    def test_validate_chunks_preserves_metadata(self):
        """Test that validation preserves chunk metadata."""
        chunks = [
            {
                "text": "Valid chunk with content.",
                "metadata": {"source": "test", "page": 1}
            },
            {
                "text": "Short",
                "metadata": {"source": "test", "page": 2}
            },
        ]

        valid_chunks = validate_chunks(chunks)

        assert len(valid_chunks) == 1
        assert valid_chunks[0]["metadata"]["source"] == "test"
        assert valid_chunks[0]["metadata"]["page"] == 1

    def test_validate_empty_list(self):
        """Test validating an empty list of chunks."""
        valid_chunks = validate_chunks([])
        assert len(valid_chunks) == 0

    def test_validate_chunks_with_missing_text_key(self):
        """Test handling chunks without 'text' key."""
        chunks = [
            {"text": "Valid chunk with content."},
            {"content": "Missing text key"},
            {"text": "Another valid chunk."},
        ]

        valid_chunks = validate_chunks(chunks)

        assert len(valid_chunks) == 2


class TestGetCollectionStats:
    """Test get_collection_stats function."""

    def test_stats_for_empty_collection(
        self,
        in_memory_chroma_client: cdb.ClientAPI,
        test_collection: cdb.Collection
    ):
        """Test getting stats for an empty collection."""
        stats = get_collection_stats(
            client=in_memory_chroma_client,
            collection_name=test_collection.name
        )

        assert stats["name"] == test_collection.name
        assert stats["statistics_mode"] == "sampled"
        assert stats["record_count"] == 0
        assert stats["embedding_dimension"] == "N/A (empty collection)"
        assert "metadata" in stats

    def test_stats_for_populated_collection(
        self,
        in_memory_chroma_client: cdb.ClientAPI,
        populated_collection: cdb.Collection
    ):
        """Test getting stats for a populated collection."""
        stats = get_collection_stats(
            client=in_memory_chroma_client,
            collection_name=populated_collection.name
        )

        assert stats["name"] == populated_collection.name
        assert stats["statistics_mode"] == "sampled"
        assert stats["record_count"] == 3
        assert stats["embedding_dimension"] == 5
        assert "avg_document_length" in stats
        assert "metadata_fields" in stats

    def test_stats_for_nonexistent_collection(
        self,
        in_memory_chroma_client: cdb.ClientAPI
    ):
        """Test getting stats for a collection that doesn't exist."""
        stats = get_collection_stats(
            client=in_memory_chroma_client,
            collection_name="nonexistent_collection"
        )

        assert "error" in stats
        assert stats["name"] == "nonexistent_collection"

    def test_stats_includes_document_length_stats(
        self,
        in_memory_chroma_client: cdb.ClientAPI
    ):
        """Test that stats include document length statistics."""
        collection = in_memory_chroma_client.create_collection(name="length_test")
        collection.add(
            ids=["id1", "id2", "id3"],
            documents=["Short", "Medium length document", "This is a longer document with more content"],
            embeddings=[[0.1] * 5, [0.2] * 5, [0.3] * 5]
        )

        stats = get_collection_stats(
            client=in_memory_chroma_client,
            collection_name="length_test"
        )

        assert "avg_document_length" in stats
        assert "min_document_length" in stats
        assert "max_document_length" in stats
        assert stats["min_document_length"] <= stats["avg_document_length"]
        assert stats["avg_document_length"] <= stats["max_document_length"]

    def test_stats_includes_metadata_fields(
        self,
        in_memory_chroma_client: cdb.ClientAPI
    ):
        """Test that stats include metadata field names."""
        collection = in_memory_chroma_client.create_collection(name="metadata_test")
        collection.add(
            ids=["id1", "id2"],
            documents=["doc1", "doc2"],
            metadatas=[
                {"source": "test", "page": 1},
                {"source": "test", "chapter": "intro"}
            ],
            embeddings=[[0.1] * 5, [0.2] * 5]
        )

        stats = get_collection_stats(
            client=in_memory_chroma_client,
            collection_name="metadata_test"
        )

        assert "metadata_fields" in stats
        metadata_fields = stats["metadata_fields"]
        assert "source" in metadata_fields
        assert "page" in metadata_fields
        assert "chapter" in metadata_fields


class TestGetAllStats:
    """Test get_all_stats function."""

    def test_get_stats_for_multiple_collections(self, in_memory_chroma_client: cdb.ClientAPI):
        """Test getting stats for all collections."""
        in_memory_chroma_client.create_collection(name="col_stats_1")
        in_memory_chroma_client.create_collection(name="col_stats_2")
        in_memory_chroma_client.create_collection(name="col_stats_3")

        all_stats = get_all_stats(client=in_memory_chroma_client)

        assert len(all_stats) >= 3
        collection_names = [stat["name"] for stat in all_stats]
        assert "col_stats_1" in collection_names
        assert "col_stats_2" in collection_names
        assert "col_stats_3" in collection_names

    def test_get_stats_for_empty_database(self, in_memory_chroma_client: cdb.ClientAPI):
        """Test getting stats when no collections exist."""
        all_stats = get_all_stats(client=in_memory_chroma_client)

        assert isinstance(all_stats, list)

    def test_get_stats_includes_all_collection_info(
        self,
        in_memory_chroma_client: cdb.ClientAPI,
        sample_chunks: list[dict],
        sample_embeddings: list[list[float]]
    ):
        """Test that stats include complete information for each collection."""
        collection = in_memory_chroma_client.create_collection(name="test_col_info")
        collection.add(
            ids=["id1"],
            documents=[sample_chunks[0]["text"]],
            embeddings=[sample_embeddings[0]]
        )

        all_stats = get_all_stats(client=in_memory_chroma_client)

        col_stats = [s for s in all_stats if s["name"] == "test_col_info"]
        assert len(col_stats) == 1
        assert col_stats[0]["record_count"] == 1
        assert "embedding_dimension" in col_stats[0]


class TestSplitTextIntoSentences:
    """Test split_text_into_sentences function."""

    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        text = "This is a short text. It has two sentences."

        chunks = split_text_into_sentences(text)

        assert len(chunks) == 1
        assert text in chunks[0]

    def test_split_long_text(self):
        """Test splitting text longer than chunk size."""
        sentences = [f"Sentence number {i}." for i in range(100)]
        text = " ".join(sentences)

        chunks = split_text_into_sentences(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 1100 for chunk in chunks)

    def test_split_handles_newlines(self):
        """Test that newlines are replaced with spaces."""
        text = "First line.\nSecond line.\nThird line."

        chunks = split_text_into_sentences(text)

        assert all("\n" not in chunk for chunk in chunks)

    def test_split_empty_text(self):
        """Test splitting empty text returns the original empty string."""
        text = ""

        chunks = split_text_into_sentences(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_maintains_sentence_endings(self):
        """Test that sentence endings are preserved."""
        text = "First sentence. Second sentence. Third sentence."

        chunks = split_text_into_sentences(text)

        assert len(chunks) == 1
        for sentence in ["First sentence", "Second sentence", "Third sentence"]:
            assert sentence in chunks[0]

    def test_split_creates_chunks_under_1000_chars(self):
        """Test that chunks are created when content exceeds 1000 chars."""
        long_sentence = "A" * 600
        text = f"{long_sentence}. {long_sentence}. {long_sentence}."

        chunks = split_text_into_sentences(text)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 1100

    def test_split_text_no_periods(self):
        """Test splitting text without sentence endings."""
        text = "This is text without proper sentence endings and just keeps going"

        chunks = split_text_into_sentences(text)

        assert len(chunks) == 1
        assert text in chunks[0]

    def test_split_preserves_content(self):
        """Test that all content is preserved after splitting."""
        text = "Sentence one. Sentence two. Sentence three."

        chunks = split_text_into_sentences(text)
        combined = " ".join(chunks)

        assert "Sentence one" in combined
        assert "Sentence two" in combined
        assert "Sentence three" in combined
