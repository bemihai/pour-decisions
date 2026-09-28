"""Eval CLI preflight checks.

These checks fail fast on obviously misconfigured environments before sample
execution starts. They are intentionally small and deterministic so they can
be tested in isolation from the eval runner.
"""

import argparse
import importlib

from omegaconf import DictConfig

from src.agents.llm import validate_cloud_model_config
from src.eval.utils import resolve_eval_model_config, resolve_execution_model_config
from src.utils import initialize_chroma_client
from src.utils.env import load_env


def preflight_eval_cloud_only_guardrail(parser: argparse.ArgumentParser, config: DictConfig) -> None:
    """Reject unsupported execution and judge providers or endpoints without network calls."""
    execution_provider, _execution_model, execution_kwargs = resolve_execution_model_config(config)
    evaluator_provider, _evaluator_model, evaluator_kwargs = resolve_eval_model_config(config)

    if execution_provider.lower() != "ollama":
        parser.error("Eval execution must use direct Ollama Cloud only.")

    if evaluator_provider.lower() != "ollama":
        parser.error("Eval judge scoring must use direct Ollama Cloud only.")
    if execution_kwargs.get("base_url") != "https://ollama.com":
        parser.error("Eval execution endpoint must be https://ollama.com.")
    if evaluator_kwargs.get("base_url") != "https://ollama.com":
        parser.error("Eval judge endpoint must be https://ollama.com.")


def _preflight_model_backend(
    parser: argparse.ArgumentParser,
    provider: str,
    model_name: str,
    kwargs: dict[str, object],
    *,
    label: str,
) -> None:
    """Validate Cloud configuration and credentials without contacting the provider."""
    load_env()
    try:
        validate_cloud_model_config(
            provider,
            model_name,
            str(kwargs.get("base_url", "")),
            float(kwargs.get("timeout", 0)),
        )
    except ValueError as error:
        parser.error(f"{label} Cloud model configuration: {error}")


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
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "ragas":
            parser.error("Full eval requires `ragas`. Install the eval extra before using `--mode full`.")
        parser.error(f"Full eval cannot import `ragas`: {exc}. Check eval dependency compatibility.")

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
    preflight_eval_cloud_only_guardrail(parser, config)

    if backend == "retriever" and mode != "retrieval":
        parser.error("The `retriever` backend only supports `--mode retrieval`.")

    if not (mode == "retrieval" and backend in {"rag", "retriever"}):
        preflight_model_backend(parser, config)

    if backend in {"rag", "retriever"}:
        preflight_rag_backend(parser, config)

    if mode == "full":
        preflight_full_mode(parser, config)
