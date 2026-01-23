"""Unit tests for src/chroma/chunks.py"""

from pathlib import Path
from unittest.mock import Mock, patch

from src.chroma.chunks import (
    ChunkMetadata,
    semantic_chunking,
    split_file,
)


class TestChunkMetadata:
    """Test ChunkMetadata dataclass."""

    def test_chunk_metadata_creation(self):
        """Test creating ChunkMetadata with required fields."""
        metadata = ChunkMetadata(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_type=".pdf",
            chunk_index=0,
            chunk_id="test_0_abc123",
            content_hash="hash123",
        )

        assert metadata.filename == "test.pdf"
        assert metadata.file_path == "/path/to/test.pdf"
        assert metadata.file_type == ".pdf"
        assert metadata.chunk_index == 0
        assert metadata.chunk_id == "test_0_abc123"
        assert metadata.content_hash == "hash123"
        assert metadata.page_number == -1
        assert metadata.language == "unknown"
        assert metadata.word_count == 0

    def test_chunk_metadata_with_all_fields(self):
        """Test creating ChunkMetadata with all optional fields."""
        metadata = ChunkMetadata(
            filename="wine.pdf",
            file_path="/data/wine.pdf",
            file_type=".pdf",
            chunk_index=5,
            chunk_id="wine_5_xyz789",
            content_hash="hash456",
            page_number=10,
            language="en",
            category="wine",
            topic="bordeaux",
            summary="About Bordeaux wines",
            word_count=250,
            char_count=1500,
            document_title="The Wine Bible",
            chapter="Chapter 5",
            section="French Wines",
            grapes="Cabernet Sauvignon,Merlot",
            regions="Bordeaux,Medoc",
            vintages="2015,2016",
            classifications="AOC",
            producers="Chateau Margaux",
            appellations="Margaux",
        )

        assert metadata.page_number == 10
        assert metadata.language == "en"
        assert metadata.category == "wine"
        assert metadata.word_count == 250
        assert metadata.document_title == "The Wine Bible"
        assert metadata.grapes == "Cabernet Sauvignon,Merlot"
        assert metadata.regions == "Bordeaux,Medoc"
        assert metadata.vintages == "2015,2016"

    def test_chunk_metadata_to_dict(self):
        """Test converting ChunkMetadata to dictionary."""
        metadata = ChunkMetadata(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_type=".pdf",
            chunk_index=0,
            chunk_id="test_0",
            content_hash="hash123",
            grapes="Chardonnay",
        )

        metadata_dict = metadata.__dict__

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["filename"] == "test.pdf"
        assert metadata_dict["grapes"] == "Chardonnay"
        assert "chunk_index" in metadata_dict


class TestSemanticChunking:
    """Test semantic_chunking function."""

    @patch("src.chroma.chunks.get_embedder")
    @patch("src.chroma.chunks.SemanticChunker")
    def test_semantic_chunking_success(self, mock_chunker_class, mock_get_embedder):
        """Test successful semantic chunking."""
        mock_embedder = Mock()
        mock_get_embedder.return_value = mock_embedder

        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker

        mock_doc1 = Mock()
        mock_doc1.page_content = "First semantic chunk."
        mock_doc2 = Mock()
        mock_doc2.page_content = "Second semantic chunk."
        mock_chunker.create_documents.return_value = [mock_doc1, mock_doc2]

        content = "This is a long text that will be chunked semantically."
        result = semantic_chunking(content)

        assert len(result) == 2
        assert result[0] == "First semantic chunk."
        assert result[1] == "Second semantic chunk."
        mock_get_embedder.assert_called_once_with(None)
        mock_chunker.create_documents.assert_called_once_with([content])

    @patch("src.chroma.chunks.get_embedder")
    @patch("src.chroma.chunks.SemanticChunker")
    def test_semantic_chunking_with_custom_model(self, mock_chunker_class, mock_get_embedder):
        """Test semantic chunking with custom embedding model."""
        mock_embedder = Mock()
        mock_get_embedder.return_value = mock_embedder

        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker

        mock_doc = Mock()
        mock_doc.page_content = "Chunk text."
        mock_chunker.create_documents.return_value = [mock_doc]

        content = "Test content."
        custom_model = "custom-embedding-model"
        result = semantic_chunking(content, embedding_model=custom_model)

        mock_get_embedder.assert_called_once_with(custom_model)
        assert len(result) == 1

    @patch("src.chroma.chunks.get_embedder")
    @patch("src.chroma.chunks.SemanticChunker")
    def test_semantic_chunking_with_custom_thresholds(self, mock_chunker_class, mock_get_embedder):
        """Test semantic chunking with custom breakpoint thresholds."""
        mock_embedder = Mock()
        mock_get_embedder.return_value = mock_embedder

        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker

        mock_doc = Mock()
        mock_doc.page_content = "Chunk."
        mock_chunker.create_documents.return_value = [mock_doc]

        result = semantic_chunking(
            "Content",
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=2.5,
        )

        mock_chunker_class.assert_called_once_with(
            embeddings=mock_embedder,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=2.5,
        )
        assert len(result) == 1

    @patch("src.chroma.chunks.get_embedder")
    @patch("src.chroma.chunks.SemanticChunker")
    @patch("src.chroma.chunks.split_text_into_sentences")
    def test_semantic_chunking_fallback_on_error(
        self, mock_split, mock_chunker_class, mock_get_embedder
    ):
        """Test fallback to simple split when semantic chunking fails."""
        mock_get_embedder.side_effect = Exception("Embedding error")
        mock_split.return_value = ["Fallback chunk 1.", "Fallback chunk 2."]

        content = "Content to chunk."
        result = semantic_chunking(content)

        assert len(result) == 2
        assert result[0] == "Fallback chunk 1."
        assert result[1] == "Fallback chunk 2."
        mock_split.assert_called_once_with(content)

    @patch("src.chroma.chunks.get_embedder")
    @patch("src.chroma.chunks.SemanticChunker")
    def test_semantic_chunking_empty_content(self, mock_chunker_class, mock_get_embedder):
        """Test semantic chunking with empty content."""
        mock_embedder = Mock()
        mock_get_embedder.return_value = mock_embedder

        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        mock_chunker.create_documents.return_value = []

        result = semantic_chunking("")

        assert len(result) == 0


class TestSplitFile:
    """Test split_file function."""

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.semantic_chunking")
    @patch("src.chroma.chunks.extract_wine_metadata")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_semantic_strategy(
        self,
        mock_hash,
        mock_wine_meta,
        mock_semantic,
        mock_doc_context,
        mock_partition,
    ):
        """Test split_file with semantic strategy."""
        mock_partition.return_value = [
            Mock(__str__=lambda self: "Element 1"),
            Mock(__str__=lambda self: "Element 2"),
        ]
        mock_doc_context.return_value = {
            "document_title": "Wine Book",
            "chapter": "Chapter 1",
            "section": "Introduction",
        }
        mock_semantic.return_value = ["Semantic chunk 1", "Semantic chunk 2"]
        mock_hash.side_effect = ["hash1234", "hash1234", "hash5678", "hash5678"]

        mock_wine = Mock()
        mock_wine.grapes = ["Chardonnay"]
        mock_wine.regions = ["Burgundy"]
        mock_wine.vintages = ["2020"]
        mock_wine.classifications = ["AOC"]
        mock_wine.producers = ["Domaine"]
        mock_wine.appellations = ["Chablis"]
        mock_wine_meta.return_value = mock_wine

        test_file = Path("test.pdf")
        result = split_file(test_file, strategy="semantic")

        assert len(result) == 2
        assert result[0]["text"] == "Semantic chunk 1"
        assert result[1]["text"] == "Semantic chunk 2"
        assert result[0]["metadata"]["filename"] == "test.pdf"
        assert result[0]["metadata"]["document_title"] == "Wine Book"
        assert result[0]["metadata"]["grapes"] == "Chardonnay"
        assert result[0]["metadata"]["regions"] == "Burgundy"
        assert result[0]["importance_score"] == 1.0

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.chunk_elements")
    @patch("src.chroma.chunks.extract_wine_metadata")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_basic_strategy(
        self,
        mock_hash,
        mock_wine_meta,
        mock_chunk_elements,
        mock_doc_context,
        mock_partition,
    ):
        """Test split_file with basic strategy."""
        mock_partition.return_value = [Mock(), Mock()]
        mock_doc_context.return_value = {"document_title": "Test Doc"}

        mock_chunk1 = Mock()
        mock_chunk1.__str__ = lambda self: "Basic chunk 1"
        mock_chunk1.metadata = Mock()
        mock_chunk1.metadata.to_dict = lambda: {"page_number": 1, "languages": ["en"]}

        mock_chunk2 = Mock()
        mock_chunk2.__str__ = lambda self: "Basic chunk 2"
        mock_chunk2.metadata = Mock()
        mock_chunk2.metadata.to_dict = lambda: {"page_number": 2, "languages": ["en"]}

        mock_chunk_elements.return_value = [mock_chunk1, mock_chunk2]
        mock_hash.side_effect = ["hash1111", "hash1111", "hash2222", "hash2222"]

        mock_wine = Mock()
        mock_wine.grapes = []
        mock_wine.regions = []
        mock_wine.vintages = []
        mock_wine.classifications = []
        mock_wine.producers = []
        mock_wine.appellations = []
        mock_wine_meta.return_value = mock_wine

        test_file = Path("test.txt")
        result = split_file(test_file, strategy="basic", chunk_size=512, overlap_size=128)

        assert len(result) == 2
        assert result[0]["text"] == "Basic chunk 1"
        assert result[1]["text"] == "Basic chunk 2"
        assert result[0]["metadata"]["page_number"] == 1
        assert result[0]["metadata"]["language"] == "en"
        mock_chunk_elements.assert_called_once()

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.chunk_by_title")
    @patch("src.chroma.chunks.extract_wine_metadata")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_by_title_strategy(
        self,
        mock_hash,
        mock_wine_meta,
        mock_chunk_by_title,
        mock_doc_context,
        mock_partition,
    ):
        """Test split_file with by_title strategy."""
        mock_partition.return_value = [Mock()]
        mock_doc_context.return_value = {}

        mock_chunk = Mock()
        mock_chunk.__str__ = lambda self: "Title chunk"
        mock_chunk.metadata = Mock()
        mock_chunk.metadata.__dict__ = {"page_number": 5}
        mock_chunk_by_title.return_value = [mock_chunk]

        mock_hash.side_effect = ["hashaaaa", "hashaaaa"]
        mock_wine_meta.return_value = Mock(
            grapes=[], regions=[], vintages=[], classifications=[], producers=[], appellations=[]
        )

        test_file = Path("document.pdf")
        result = split_file(test_file, strategy="by_title")

        assert len(result) == 1
        assert result[0]["text"] == "Title chunk"
        mock_chunk_by_title.assert_called_once()

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    def test_split_file_invalid_strategy(self, mock_doc_context, mock_partition):
        """Test split_file with invalid strategy raises error."""
        mock_partition.return_value = [Mock()]
        mock_doc_context.return_value = {}

        test_file = Path("test.pdf")
        result = split_file(test_file, strategy="invalid_strategy")

        assert len(result) == 0

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.semantic_chunking")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_without_metadata_extraction(
        self, mock_hash, mock_semantic, mock_doc_context, mock_partition
    ):
        """Test split_file with metadata extraction disabled."""
        mock_partition.return_value = [Mock(__str__=lambda self: "Text")]
        mock_doc_context.return_value = {}
        mock_semantic.return_value = ["Chunk without metadata"]
        mock_hash.side_effect = ["hash9999", "hash9999"]

        test_file = Path("test.pdf")
        result = split_file(test_file, strategy="semantic", extract_metadata=False)

        assert len(result) == 1
        assert result[0]["metadata"]["grapes"] == ""
        assert result[0]["metadata"]["regions"] == ""
        assert result[0]["metadata"]["vintages"] == ""

    @patch("src.chroma.chunks.partition")
    def test_split_file_partition_error(self, mock_partition):
        """Test split_file handles partition errors gracefully."""
        mock_partition.side_effect = Exception("Partition failed")

        test_file = Path("corrupted.pdf")
        result = split_file(test_file)

        assert len(result) == 0

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.semantic_chunking")
    @patch("src.chroma.chunks.extract_wine_metadata")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_chunk_id_generation(
        self,
        mock_hash,
        mock_wine_meta,
        mock_semantic,
        mock_doc_context,
        mock_partition,
    ):
        """Test that chunk IDs are generated correctly."""
        mock_partition.return_value = [Mock(__str__=lambda self: "Text")]
        mock_doc_context.return_value = {}
        mock_semantic.return_value = ["Chunk 1", "Chunk 2"]
        mock_hash.side_effect = ["abcd1234567890", "fullhash1", "efgh5678901234", "fullhash2"]
        mock_wine_meta.return_value = Mock(
            grapes=[], regions=[], vintages=[], classifications=[], producers=[], appellations=[]
        )

        test_file = Path("winebook.pdf")
        result = split_file(test_file, strategy="semantic")

        assert result[0]["id"] == "winebook_0_abcd1234"
        assert result[1]["id"] == "winebook_1_efgh5678"
        assert result[0]["metadata"]["chunk_id"] == "winebook_0_abcd1234"
        assert result[1]["metadata"]["chunk_id"] == "winebook_1_efgh5678"

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.semantic_chunking")
    @patch("src.chroma.chunks.extract_wine_metadata")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_word_and_char_count(
        self,
        mock_hash,
        mock_wine_meta,
        mock_semantic,
        mock_doc_context,
        mock_partition,
    ):
        """Test that word count and character count are calculated correctly."""
        mock_partition.return_value = [Mock(__str__=lambda self: "Text")]
        mock_doc_context.return_value = {}
        chunk_text = "This is a test chunk with ten words in total."
        mock_semantic.return_value = [chunk_text]
        mock_hash.side_effect = ["hash", "hash"]
        mock_wine_meta.return_value = Mock(
            grapes=[], regions=[], vintages=[], classifications=[], producers=[], appellations=[]
        )

        test_file = Path("test.pdf")
        result = split_file(test_file, strategy="semantic")

        assert result[0]["metadata"]["word_count"] == 10
        assert result[0]["metadata"]["char_count"] == len(chunk_text)

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.semantic_chunking")
    @patch("src.chroma.chunks.extract_wine_metadata")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_with_multiple_wine_metadata(
        self,
        mock_hash,
        mock_wine_meta,
        mock_semantic,
        mock_doc_context,
        mock_partition,
    ):
        """Test split_file correctly handles multiple wine metadata values."""
        mock_partition.return_value = [Mock(__str__=lambda self: "Wine text")]
        mock_doc_context.return_value = {}
        mock_semantic.return_value = ["Wine chunk"]
        mock_hash.side_effect = ["hash", "hash"]

        mock_wine = Mock()
        mock_wine.grapes = ["Cabernet Sauvignon", "Merlot", "Cabernet Franc"]
        mock_wine.regions = ["Bordeaux", "Pauillac"]
        mock_wine.vintages = ["2015", "2016", "2018"]
        mock_wine.classifications = ["AOC", "Grand Cru"]
        mock_wine.producers = ["Chateau Latour", "Chateau Margaux"]
        mock_wine.appellations = ["Pauillac", "Margaux"]
        mock_wine_meta.return_value = mock_wine

        test_file = Path("wine.pdf")
        result = split_file(test_file, strategy="semantic")

        assert result[0]["metadata"]["grapes"] == "Cabernet Sauvignon,Merlot,Cabernet Franc"
        assert result[0]["metadata"]["regions"] == "Bordeaux,Pauillac"
        assert result[0]["metadata"]["vintages"] == "2015,2016,2018"
        assert result[0]["metadata"]["classifications"] == "AOC,Grand Cru"
        assert result[0]["metadata"]["producers"] == "Chateau Latour,Chateau Margaux"
        assert result[0]["metadata"]["appellations"] == "Pauillac,Margaux"

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.semantic_chunking")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_semantic_with_custom_kwargs(
        self, mock_hash, mock_semantic, mock_doc_context, mock_partition
    ):
        """Test split_file passes custom kwargs to semantic chunking."""
        mock_partition.return_value = [Mock(__str__=lambda self: "Text")]
        mock_doc_context.return_value = {}
        mock_semantic.return_value = ["Chunk"]
        mock_hash.side_effect = ["hash", "hash"]

        test_file = Path("test.pdf")
        result = split_file(
            test_file,
            strategy="semantic",
            extract_metadata=False,
            breakpoint_threshold_type="interquartile",
            breakpoint_threshold_amount=85.0,
        )

        mock_semantic.assert_called_once()
        call_kwargs = mock_semantic.call_args[1]
        assert call_kwargs["breakpoint_threshold_type"] == "interquartile"
        assert call_kwargs["breakpoint_threshold_amount"] == 85.0

    @patch("src.chroma.chunks.partition")
    @patch("src.chroma.chunks.extract_document_context")
    @patch("src.chroma.chunks.chunk_elements")
    @patch("src.chroma.chunks.extract_wine_metadata")
    @patch("src.chroma.chunks.generate_hash")
    def test_split_file_handles_missing_chunk_metadata(
        self,
        mock_hash,
        mock_wine_meta,
        mock_chunk_elements,
        mock_doc_context,
        mock_partition,
    ):
        """Test split_file handles chunks without metadata attribute."""
        mock_partition.return_value = [Mock()]
        mock_doc_context.return_value = {}

        mock_chunk = Mock()
        mock_chunk.__str__ = lambda self: "Chunk without metadata"
        mock_chunk.metadata = None
        mock_chunk_elements.return_value = [mock_chunk]

        mock_hash.side_effect = ["hash", "hash"]
        mock_wine_meta.return_value = Mock(
            grapes=[], regions=[], vintages=[], classifications=[], producers=[], appellations=[]
        )

        test_file = Path("test.pdf")
        result = split_file(test_file, strategy="basic")

        assert len(result) == 1
        assert result[0]["metadata"]["page_number"] == -1
        assert result[0]["metadata"]["language"] == "unknown"
