"""Shared eval configuration and execution helpers."""

import subprocess
from typing import Any

from langchain_core.language_models import BaseChatModel
from omegaconf import DictConfig

from src.agents.llm import invoke_llm, load_base_model
from src.eval.models import GoldenSample
from src.retrieval import ChromaRetriever, HybridRetriever


def resolve_eval_model_config(cfg: DictConfig) -> tuple[str, str, dict[str, Any]]:
    """Resolve eval execution model config.

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


def load_eval_model(cfg: DictConfig) -> BaseChatModel:
    """Load the configured eval model."""
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
    eval_provider, eval_model, _ = resolve_eval_model_config(cfg)
    return {
        "model": str(cfg.model.name),
        "provider": str(cfg.model.provider),
        "eval_model": eval_model,
        "eval_provider": eval_provider,
        "embedder": str(cfg.chroma.settings.embedder),
        "n_results": int(cfg.chroma.retrieval.n_results),
        "enable_reranking": bool(cfg.chroma.retrieval.enable_reranking),
        "enable_hybrid": bool(cfg.chroma.retrieval.enable_hybrid),
    }


def get_git_sha() -> str:
    """Get short git SHA for the current working tree."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"
