"""Shared pytest fixtures for all tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import chromadb as cdb
import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_wine_pdf(test_data_dir: Path) -> Path:
    """Return the path to the test wine PDF file."""
    pdf_path = test_data_dir / "knowledge" / "wine.pdf"
    assert pdf_path.exists(), f"Test PDF not found at {pdf_path}"
    return pdf_path


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def in_memory_chroma_client() -> cdb.ClientAPI:
    """
    Create an in-memory ChromaDB client for testing.

    Uses EphemeralClient which stores data in memory only. Each test function
    gets its own client instance (function scope), and collections persist
    only within that client instance for the duration of a single test.

    This provides isolation between tests: collections created in one test
    are not visible to subsequent tests because each test receives a new
    EphemeralClient instance.
    """
    client = cdb.EphemeralClient()
    return client


@pytest.fixture(scope="function")
def temp_chroma_client(temp_dir: Path) -> Generator[cdb.ClientAPI, None, None]:
    """Create a persistent ChromaDB client in a temporary directory."""
    client = cdb.PersistentClient(path=str(temp_dir / "chroma"))
    yield client
    del client


@pytest.fixture(scope="function")
def test_collection(in_memory_chroma_client: cdb.ClientAPI) -> cdb.Collection:
    """Create a test collection in the in-memory ChromaDB client."""
    return in_memory_chroma_client.create_collection(
        name="test_collection",
        metadata={"test": "true"}
    )


@pytest.fixture(scope="function")
def sample_chunks() -> list[dict]:
    """Return sample chunk data for testing."""
    return [
        {
            "text": "Cabernet Sauvignon is a red wine grape variety grown worldwide.",
            "metadata": {"source": "test", "chunk_index": 0}
        },
        {
            "text": "Chardonnay is the most widely planted white wine grape variety.",
            "metadata": {"source": "test", "chunk_index": 1}
        },
        {
            "text": "Bordeaux wines are typically blends of Cabernet Sauvignon and Merlot.",
            "metadata": {"source": "test", "chunk_index": 2}
        },
    ]


@pytest.fixture(scope="function")
def sample_embeddings() -> list[list[float]]:
    """Return sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
    ]


@pytest.fixture(scope="function")
def populated_collection(
    in_memory_chroma_client: cdb.ClientAPI,
    sample_chunks: list[dict],
    sample_embeddings: list[list[float]]
) -> cdb.Collection:
    """Create a collection populated with sample data."""
    collection = in_memory_chroma_client.create_collection(
        name="populated_test_collection",
        metadata={"test": "true", "populated": "true"}
    )

    ids = [f"test_chunk_{i}" for i in range(len(sample_chunks))]
    documents = [chunk["text"] for chunk in sample_chunks]
    metadatas = [chunk["metadata"] for chunk in sample_chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=sample_embeddings
    )

    return collection

