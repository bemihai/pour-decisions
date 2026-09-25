"""Direct Ollama Cloud loading, prompt construction, and RAG invocation."""

import math
import os
import re

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from src.agents.prompt_registry import PromptRegistry, get_prompt_registry
from src.agents.provenance import build_rag_execution_provenance
from src.utils import get_tracing_callbacks, logger
from src.utils.env import load_env


_DIRECT_CLOUD_URL = "https://ollama.com"
_DIRECT_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9][A-Za-z0-9._-]*)?")
_MODEL_OPTIONS = frozenset({"temperature", "top_p", "top_k", "reasoning", "num_predict"})


def validate_cloud_model_config(
    provider: str,
    model_name: str,
    base_url: str,
    timeout_seconds: float,
    *,
    api_key: str | None = None,
) -> None:
    """Reject unsupported generation settings before constructing a client.

    Args:
        provider: Internal model provider identifier.
        model_name: Direct Ollama API model identifier.
        base_url: Native Cloud API endpoint.
        timeout_seconds: Per-request transport timeout.
        api_key: Optional test credential; production reads ``OLLAMA_API_KEY``.

    Raises:
        ValueError: If the Cloud connection contract is invalid.
    """
    if provider.strip().lower() != "ollama":
        raise ValueError("Only the Ollama Cloud model provider is supported")
    if not isinstance(model_name, str) or _DIRECT_MODEL_NAME.fullmatch(model_name) is None:
        raise ValueError("A direct Ollama Cloud model identifier is required")
    if base_url != _DIRECT_CLOUD_URL:
        raise ValueError("The Ollama Cloud endpoint must be https://ollama.com")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("The Ollama Cloud timeout must be positive and finite")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("The Ollama Cloud timeout must be positive and finite")
    credential = os.environ.get("OLLAMA_API_KEY") if api_key is None else api_key
    if not credential or not credential.strip():
        raise ValueError("OLLAMA_API_KEY must be configured for Ollama Cloud")


class ModelInternalError(Exception):
    """Gen AI Model error."""
    def __init__(self, message: str | None = None) -> None:
        self.message = message or "Model internal error"
        super().__init__(self.message)

    @property
    def default_message(self) -> str:
        """Default answer when agents raises this error."""
        return "I can't answer your question due to an internal error, please try again later."


def load_base_model(model_provider: str, model_name: str, **kwargs: object) -> BaseChatModel:
    """Construct the one supported generative client for sync and async use.

    Args:
        model_provider: Must be ``ollama``.
        model_name: Direct Ollama Cloud model identifier.
        **kwargs: Optional ``base_url``, ``timeout``, and reviewed sampling controls.

    Returns:
        A ``ChatOllama`` client with both native transport modes configured.

    Raises:
        ValueError: If the endpoint, credential, or model options are unsupported.
    """
    base_url = kwargs.pop("base_url", _DIRECT_CLOUD_URL)
    timeout_seconds = kwargs.pop("timeout", 60)
    unsupported_options = set(kwargs) - _MODEL_OPTIONS
    if unsupported_options:
        raise ValueError("Unsupported Ollama Cloud model option")
    load_env()
    validate_cloud_model_config(model_provider, model_name, base_url, timeout_seconds)

    options = dict(kwargs)
    if model_name.lower().startswith("gemma4:"):
        options.setdefault("temperature", 1.0)
        options.setdefault("top_p", 0.95)
        options.setdefault("top_k", 64)
    else:
        options.setdefault("temperature", 0.7)
    model = ChatOllama(
        model=model_name,
        base_url=base_url,
        client_kwargs={"timeout": timeout_seconds},
        **options,
    )
    logger.info("Loaded direct Ollama Cloud model: %s", model_name)
    return model


def _build_rag_messages(
    question: str,
    context: str,
    message_history: list[dict[str, object]],
    prompt_registry: PromptRegistry,
) -> list[SystemMessage | HumanMessage | AIMessage]:
    """Build the exact registered RAG prompt and conversation messages."""
    system_prompt = prompt_registry.get("rag_only_system").source.strip()
    user_prompt = prompt_registry.get("rag_only_user").source.strip()
    messages: list[SystemMessage | HumanMessage | AIMessage] = [SystemMessage(content=system_prompt)]
    for message in message_history:
        role = message.get("role")
        content = message.get("content")
        if content is None:
            if role == "human":
                content = message.get("question")
            elif role == "ai":
                content = message.get("answer")
        if not content:
            continue
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))
    user_content = user_prompt.replace("{context}", context).replace("{question}", question)
    messages.append(HumanMessage(content=user_content))
    return messages


def _coerce_model_output(model_output: object) -> str:
    """Preserve the existing conversion of provider output into answer text."""
    if hasattr(model_output, "content"):
        content = model_output.content
        return content if isinstance(content, str) else str(content)
    if isinstance(model_output, dict) and "content" in model_output:
        return model_output["content"]
    return str(model_output)


def _build_rag_invoke_config(
    model: BaseChatModel,
    prompt_registry: PromptRegistry,
    trace_context: dict[str, str] | None,
    callbacks: list[BaseCallbackHandler],
) -> RunnableConfig | None:
    """Build identical tracing and provenance config for sync and async calls."""
    if not trace_context and not callbacks:
        return None
    provenance_metadata = build_rag_execution_provenance(
        model,
        prompt_registry=prompt_registry,
    ).to_trace_attributes()
    return RunnableConfig(
        metadata={**(trace_context or {}), **provenance_metadata},
        callbacks=callbacks,
    )


def invoke_llm(
    question: str,
    context: str,
    model: BaseChatModel,
    message_history: list,
    trace_context: dict[str, str] | None = None,
) -> str:
    """
    Invoke the LLM agents with the provided question, context, and full message history.

    Args:
        question (str): The user's question to be answered by the agents.
        context (str): The context retrieved by the RAG pipeline.
        model (BaseChatModel): The loaded LLM agents instance.
        message_history (list): List of dicts with previous messages, each having
            ``"role"`` and ``"content"`` keys.
        trace_context: Optional request trace metadata forwarded to the chain runtime.

    Returns: The agents's answer as a string.
    """
    prompt_registry = get_prompt_registry()
    lc_messages = _build_rag_messages(question, context, message_history, prompt_registry)

    callbacks = get_tracing_callbacks()

    try:
        invoke_config = _build_rag_invoke_config(model, prompt_registry, trace_context, callbacks)
        if invoke_config:
            model_output = model.invoke(lc_messages, config=invoke_config)
        else:
            model_output = model.invoke(lc_messages)
        return _coerce_model_output(model_output)
    except Exception as error:
        raise ModelInternalError() from error


async def ainvoke_llm(
    question: str,
    context: str,
    model: BaseChatModel,
    message_history: list,
    trace_context: dict[str, str] | None = None,
) -> str:
    """Invoke the RAG model through its native async entry point.

    Args:
        question: User question.
        context: Retrieved context supplied to the model.
        model: Loaded LangChain chat model.
        message_history: Prior conversation turns.
        trace_context: Optional request trace metadata.

    Returns:
        Model answer coerced with the synchronous output contract.

    Raises:
        ModelInternalError: If config construction or model invocation fails.
    """
    prompt_registry = get_prompt_registry()
    lc_messages = _build_rag_messages(question, context, message_history, prompt_registry)
    callbacks = get_tracing_callbacks()

    try:
        invoke_config = _build_rag_invoke_config(model, prompt_registry, trace_context, callbacks)
        if invoke_config:
            model_output = await model.ainvoke(lc_messages, config=invoke_config)
        else:
            model_output = await model.ainvoke(lc_messages)
        return _coerce_model_output(model_output)
    except Exception as error:
        raise ModelInternalError() from error


def process_user_prompt(
    model: BaseChatModel,
    prompt: str,
    context: str,
    message_history: list,
    trace_context: dict[str, str] | None = None,
) -> str:
    """Process a user prompt with optional trace metadata.

    Args:
        model: The loaded LLM model instance.
        prompt: User question.
        context: Retrieved context used to answer the question.
        message_history: Prior conversation turns.
        trace_context: Optional request trace metadata forwarded to invoke_llm.

    Returns:
        Model answer text, or fallback error message.
    """
    try:
        answer = invoke_llm(prompt, context, model, message_history, trace_context=trace_context)
    except ModelInternalError as err:
        answer = err.default_message
        logger.error(f"ModelInternalError: {err}")
    return answer


async def process_user_prompt_async(
    model: BaseChatModel,
    prompt: str,
    context: str,
    message_history: list,
    trace_context: dict[str, str] | None = None,
) -> str:
    """Process a user prompt through the native async RAG model helper.

    Args:
        model: Loaded LangChain chat model.
        prompt: User question.
        context: Retrieved context used to answer the question.
        message_history: Prior conversation turns.
        trace_context: Optional request trace metadata forwarded to ``ainvoke_llm``.

    Returns:
        Model answer text, or the synchronous fail-soft error message.
    """
    try:
        answer = await ainvoke_llm(prompt, context, model, message_history, trace_context=trace_context)
    except ModelInternalError as err:
        answer = err.default_message
        logger.error(f"ModelInternalError: {err}")
    return answer
