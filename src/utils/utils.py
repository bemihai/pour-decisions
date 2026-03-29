"""Core utility functions for the Pour Decisions application.

Provides configuration loading (OmegaConf), project root detection,
ChromaDB client initialization, hashing, cosine similarity, and JSON loading.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
import chromadb as cdb

from src.utils import logger


def initialize_chroma_client(host: str, port: int) -> cdb.ClientAPI:
    """Initialize and verify a connection to the ChromaDB server.

    Args:
        host: The host address of the ChromaDB server.
        port: The port number of the ChromaDB server.

    Returns:
        An instance of the ChromaDB HTTP client.

    Raises:
        Exception: If the connection or heartbeat check fails.
    """
    client = cdb.HttpClient(host=host, port=port)
    client.heartbeat()
    logger.info(f"Connected to ChromaDB at {host}:{port}")

    return client


def find_project_root(marker: str = "pyproject.toml") -> str:
    """Walk up from the current working directory to find the project root.

    Args:
        marker: Filename or directory name that identifies the project root.

    Returns:
        Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If no directory containing the marker is found.
    """
    current_path = os.path.abspath(os.getcwd())
    while current_path != os.path.dirname(current_path):
        if marker in os.listdir(current_path):
            return current_path
        current_path = os.path.dirname(current_path)
    raise FileNotFoundError(f"Project root with {marker} not found.")


def get_project_root() -> Path:
    """Return the project root as a Path object.

    Returns:
        Path to the project root directory (contains ``pyproject.toml``).
    """
    return Path(find_project_root())


def get_default_db_path() -> Path:
    """Return the default wine cellar database path from config.

    Returns:
        Absolute path to the SQLite database file.
    """
    cfg = get_config()
    return get_project_root() / cfg.cellar.db_path


def get_config() -> DictConfig:
    """Load and return the application configuration from ``app_config.yml``.

    Returns:
        OmegaConf DictConfig with all application settings.
    """
    return OmegaConf.load(Path(find_project_root()) / "app_config.yml")


def get_initial_message() -> list[dict]:
    """Return the initial chatbot greeting message from config.

    Returns:
        List containing a single message dict with 'role' and 'answer' keys.
    """
    cfg = get_config()
    msg = cfg.initial_message
    return [
        {
            "role": msg["role"] if "role" in msg else "ai",
            "answer": msg["answer"] if "answer" in msg else "Welcome! Ask me anything about wine."
        }
    ]


def generate_hash(content: str) -> str:
    """Generate an MD5 hash for a content string.

    Args:
        content: Text content to hash.

    Returns:
        Hexadecimal MD5 digest string.
    """
    return hashlib.md5(content.encode()).hexdigest()


def compute_file_hash(file_path: Path) -> str:
    """Compute the MD5 hash of a file's contents.

    Args:
        file_path: Path to the file.

    Returns:
        Hexadecimal MD5 digest string.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1. Returns 0.0 if either
        vector has zero magnitude.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def load_json(filepath: str | Path) -> dict | list:
    """Load and parse a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON content as a dict or list.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
