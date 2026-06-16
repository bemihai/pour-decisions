"""Shared eval configuration and execution helpers."""

import subprocess
from typing import Any

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig

from src.agents.llm import invoke_llm, load_base_model
from src.eval.models import GoldenSample
from src.retrieval import ChromaRetriever, HybridRetriever


def resolve_execution_model_config(cfg: DictConfig) -> tuple[str, str, dict[str, Any]]:
    """Resolve the model config used to execute eval samples.

    This is the model under test for both the direct RAG backend and the agent
    backend. It always comes from the main application model config.

    Args:
        cfg: Application configuration.

    Returns:
        Tuple of ``(provider, model_name, kwargs)`` for ``load_base_model``.
    """
    provider = str(cfg.model.provider)
    model_name = str(cfg.model.name)

    kwargs: dict[str, Any] = {}
    if provider.lower() == "ollama":
        base_url = str(getattr(getattr(cfg.model, "ollama", None), "base_url", "http://localhost:11434"))
        kwargs["base_url"] = base_url

    return provider, model_name, kwargs


def resolve_eval_model_config(cfg: DictConfig) -> tuple[str, str, dict[str, Any]]:
    """Resolve evaluator model config for full-mode judge scoring.

    Priority order:
    1. ``eval.ragas.evaluator_provider`` / ``eval.ragas.evaluator_model``
    2. ``model.provider`` / ``model.name``

    Args:
        cfg: Application configuration.

    Returns:
        Tuple of ``(provider, model_name, kwargs)`` for ``load_base_model``.
    """
    eval_ragas = getattr(cfg.eval, "ragas", None)
    provider = str(getattr(eval_ragas, "evaluator_provider", "")).strip() or str(cfg.model.provider)
    model_name = str(getattr(eval_ragas, "evaluator_model", "")).strip() or str(cfg.model.name)

    kwargs: dict[str, Any] = {}
    if provider.lower() == "ollama":
        base_url = str(getattr(getattr(cfg.model, "ollama", None), "base_url", "http://localhost:11434"))
        kwargs["base_url"] = base_url

    return provider, model_name, kwargs


def load_execution_model(cfg: DictConfig) -> BaseChatModel:
    """Load the configured execution model used by the runner."""
    provider, model_name, kwargs = resolve_execution_model_config(cfg)
    return load_base_model(provider, model_name, **kwargs)


def load_eval_model(cfg: DictConfig) -> BaseChatModel:
    """Load the configured evaluator model used by Ragas scoring."""
    provider, model_name, kwargs = resolve_eval_model_config(cfg)
    return load_base_model(provider, model_name, **kwargs)


def get_retrieved_chunk_ids(retrieved_docs: list[dict[str, Any]]) -> list[str]:
    """Extract retrieved chunk IDs from retriever documents."""
    chunk_ids: list[str] = []
    for doc in retrieved_docs:
        chunk_id = doc.get("id")
        if isinstance(chunk_id, str) and chunk_id:
            chunk_ids.append(chunk_id)
    return chunk_ids


def run_rag_sample_sync(
    sample: GoldenSample,
    retriever: HybridRetriever | ChromaRetriever,
    model: BaseChatModel,
    retrieval_count: int,
) -> tuple[str, list[str], list[str], list[str]]:
    """Execute one sample against the RAG backend."""
    retrieved_docs = retriever.retrieve(sample.question, n_results=retrieval_count)
    contexts = [doc.get("document", "") for doc in retrieved_docs if doc.get("document")]
    retrieved_chunk_ids = get_retrieved_chunk_ids(retrieved_docs)

    context_text = "\n\n".join(contexts)
    answer = invoke_llm(
        question=sample.question,
        context=context_text,
        model=model,
        message_history=[],
    )

    return answer, contexts, retrieved_chunk_ids, []


def run_rag_retrieval_only_sync(
    sample: GoldenSample,
    retriever: HybridRetriever | ChromaRetriever,
    retrieval_count: int,
) -> tuple[str, list[str], list[str], list[str]]:
    """Execute one sample against the RAG retriever without LLM generation."""
    retrieved_docs = retriever.retrieve(sample.question, n_results=retrieval_count)
    contexts = [doc.get("document", "") for doc in retrieved_docs if doc.get("document")]
    retrieved_chunk_ids = get_retrieved_chunk_ids(retrieved_docs)
    return "", contexts, retrieved_chunk_ids, []


def extract_contexts_from_agent_messages(messages: list[Any]) -> list[str]:
    """Extract text contexts from agent tool messages."""
    contexts: list[str] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type != "tool":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            contexts.append(content)
        elif isinstance(content, list):
            parts = [str(item) for item in content if str(item).strip()]
            if parts:
                contexts.append(" ".join(parts))
        elif content:
            contexts.append(str(content))
    return contexts


def run_agent_sample_sync(agent: Any, sample: GoldenSample) -> tuple[str, list[str], list[str], list[str]]:
    """Execute one sample against the agent backend."""
    result = agent.invoke(sample.question)
    answer = str(result.get("final_answer", ""))
    tool_calls_made = [str(name) for name in result.get("tools_used", [])]
    contexts = extract_contexts_from_agent_messages(result.get("messages", []))
    return answer, contexts, [], tool_calls_made


def extract_eval_config_snapshot(cfg: DictConfig) -> dict[str, Any]:
    """Extract a stable config snapshot for reproducible eval runs."""
    provider, model_name, _ = resolve_execution_model_config(cfg)
    eval_provider, eval_model, _ = resolve_eval_model_config(cfg)
    return {
        "model": model_name,
        "provider": provider,
        "eval_model": eval_model,
        "eval_provider": eval_provider,
        "embedder": str(cfg.chroma.settings.embedder),
        "retrieval": {
            "n_results": int(cfg.chroma.retrieval.n_results),
            "similarity_threshold": float(cfg.chroma.retrieval.similarity_threshold),
            "use_deduplication": bool(cfg.chroma.retrieval.use_deduplication),
            "deduplication_threshold": float(cfg.chroma.retrieval.deduplication_threshold),
            "enable_hybrid": bool(cfg.chroma.retrieval.enable_hybrid),
            "hybrid_vector_weight": float(cfg.chroma.retrieval.hybrid_vector_weight),
            "hybrid_keyword_weight": float(cfg.chroma.retrieval.hybrid_keyword_weight),
            "enable_reranking": bool(cfg.chroma.retrieval.enable_reranking),
            "reranker_model": str(cfg.chroma.retrieval.reranker_model),
            "rerank_top_k": int(cfg.chroma.retrieval.rerank_top_k),
            "enable_compression": bool(cfg.chroma.retrieval.enable_compression),
            "compression_max_chars": int(cfg.chroma.retrieval.compression_max_chars),
            "enable_metadata_boost": bool(cfg.chroma.retrieval.enable_metadata_boost),
            "metadata_boost_factor": float(cfg.chroma.retrieval.metadata_boost_factor),
        },
        "eval": {
            "ragas_metrics": [str(metric) for metric in getattr(cfg.eval.ragas, "metrics", [])],
            "sample_timeout_seconds": float(getattr(cfg.eval, "sample_timeout_seconds", 0) or 0),
            "skip_cellar_samples_if_empty": bool(getattr(cfg.eval, "skip_cellar_samples_if_empty", True)),
            "validate_tag_filters": bool(getattr(cfg.eval, "validate_tag_filters", True)),
        },
    }


def _run_git_command(args: list[str]) -> str | None:
    """Run a git command and return stripped stdout, or ``None`` on failure."""
    try:
        return subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def get_git_metadata() -> dict[str, Any]:
    """Return git metadata for reproducibility, safely handling missing git state."""
    sha = _run_git_command(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    status_output = _run_git_command(["git", "status", "--porcelain"])
    is_dirty = None if status_output is None else bool(status_output)
    return {
        "sha": sha,
        "branch": branch,
        "is_dirty": is_dirty,
    }


def get_git_sha() -> str:
    """Get short git SHA for the current working tree."""
    return str(get_git_metadata()["sha"])
