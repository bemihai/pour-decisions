"""Eval CLI preflight checks.

These checks fail fast on obviously misconfigured environments before sample
execution starts. They are intentionally small and deterministic so they can
be tested in isolation from the eval runner.
"""

import argparse
import importlib
import urllib.request

from omegaconf import DictConfig

from src.eval.utils import resolve_eval_model_config, resolve_execution_model_config
from src.utils import initialize_chroma_client
from src.utils.env import GOOGLE_API_KEY


def _ollama_available(base_url: str) -> bool:
    """Return whether the Ollama endpoint is reachable."""
    try:
        urllib.request.urlopen(base_url, timeout=2)
        return True
    except Exception:
        return False


def _preflight_model_backend(
    parser: argparse.ArgumentParser,
    provider: str,
    model_name: str,
    kwargs: dict[str, object],
    *,
    label: str,
) -> None:
    """Fail fast on unsupported or unreachable model backends."""
    provider = provider.lower()
    if not provider or not model_name:
        parser.error(f"{label} model provider/name are not configured.")

    if provider == "ollama":
        base_url = str(kwargs.get("base_url", ""))
        if not _ollama_available(base_url):
            parser.error(
                f"Ollama is unreachable at {base_url}. Start Ollama or change {label.lower()} model provider settings."
            )
        return

    if provider == "google":
        if not GOOGLE_API_KEY:
            parser.error(f"GOOGLE_API_KEY is not set. Configure it or switch {label.lower()} model provider.")
        return

    parser.error(f"Unsupported {label.lower()} model provider: {provider}")


def preflight_model_backend(parser: argparse.ArgumentParser, config: DictConfig) -> None:
    """Fail fast on unsupported or unreachable execution model backends."""
    provider, model_name, kwargs = resolve_execution_model_config(config)
    _preflight_model_backend(
        parser,
        provider,
        model_name,
        kwargs,
        label="Execution",
    )


def preflight_rag_backend(parser: argparse.ArgumentParser, config: DictConfig) -> None:
    """Fail fast when the configured Chroma collection is unavailable."""
    collections = getattr(config.chroma, "collections", None)
    if not collections:
        parser.error("No Chroma collections are configured in app_config.yml.")

    collection_name = str(collections[0].name)
    try:
        client = initialize_chroma_client(
            host=config.chroma.client.host,
            port=int(config.chroma.client.port),
        )
        client.get_collection(collection_name)
    except Exception as exc:
        parser.error(
            f"Chroma preflight failed for collection '{collection_name}': {exc}. "
            "Start Chroma and verify the collection has been indexed."
        )


def preflight_full_mode(parser: argparse.ArgumentParser, config: DictConfig) -> None:
    """Fail fast when full-mode scoring dependencies are unavailable."""
    try:
        importlib.import_module("ragas")
    except ImportError:
        parser.error("Full eval requires `ragas`. Install the eval extra before using `--mode full`.")

    provider, model_name, _ = resolve_eval_model_config(config)
    if not str(provider).strip() or not str(model_name).strip():
        parser.error("Full eval requires a configured evaluator provider and model.")
    provider, model_name, kwargs = resolve_eval_model_config(config)
    _preflight_model_backend(
        parser,
        provider,
        model_name,
        kwargs,
        label="Evaluator",
    )


def run_preflight(
    parser: argparse.ArgumentParser,
    config: DictConfig,
    mode: str,
    backend: str,
) -> None:
    """Run fail-fast environment checks before sample execution."""
    if not (mode == "retrieval" and backend == "rag"):
        preflight_model_backend(parser, config)

    if backend == "rag":
        preflight_rag_backend(parser, config)

    if mode == "full":
        preflight_full_mode(parser, config)
