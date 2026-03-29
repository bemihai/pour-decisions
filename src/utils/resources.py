"""Cached embedding model instances.

Provides ``get_embedder()`` which returns a cached HuggingFace embedder,
avoiding re-instantiation across the application. Model name is read from
``app_config.yml`` under ``chroma.settings.embedder``.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import get_config


# Module-level cache for embedder
embedder_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embedder(model_name: str | None = None) -> HuggingFaceEmbeddings:
    """Get or create a cached HuggingFace embedder instance.

    Args:
        model_name: HuggingFace model name. If None, reads from config
            (``chroma.settings.embedder``).

    Returns:
        Cached HuggingFaceEmbeddings instance for the given model name.
    """
    if model_name is None:
        cfg = get_config()
        model_name = cfg.chroma.settings.embedder

    if model_name not in embedder_cache:
        embedder_cache[model_name] = HuggingFaceEmbeddings(model_name=model_name)

    return embedder_cache[model_name]
