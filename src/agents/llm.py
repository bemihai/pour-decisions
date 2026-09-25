"""LLM loading, prompt construction, and invocation for RAG and agent pipelines.

Supports Google Gemini (cloud) and Ollama (local) providers. RAG prompts are
resolved from the process-cached prompt registry for each invocation.

Provider notes:
- ``"ollama"``: Local inference via Ollama server (``localhost:11434``).
  Sampling parameters are model-family aware:
  - Gemma 4 (``gemma4:*``): temperature=1.0, top_p=0.95, top_k=64 as recommended
    by Google. Do NOT set ``num_predict`` — the model's internal reasoning pass
    consumes tokens before visible output; a strict limit produces empty responses.
  - All other Ollama models: temperature=0.7 (standard default).
  Tool calling: only models that declare tool support in Ollama work with
  ``bind_tools()``. As of 2026-05, ``gemma3:4b`` does NOT support tool calling.
  Use ``hybrid_tool_calling: true`` in ``app_config.yml`` to route tool-selection
  calls through the cloud model while keeping generation local.
- ``"google"``: Google Gemini API. Requires ``GOOGLE_API_KEY`` in the environment.
"""

import math
import os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from src.agents.prompt_registry import PromptRegistry, get_prompt_registry
from src.agents.provenance import build_rag_execution_provenance
from src.utils import get_tracing_callbacks, logger
from src.utils.env import GOOGLE_API_KEY


_DIRECT_CLOUD_URL = "https://ollama.com"


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
    if not isinstance(model_name, str) or not model_name or model_name != model_name.strip() or any(
        char.isspace() for char in model_name
    ):
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


def load_base_model(model_provider: str, model_name: str, **kwargs) -> BaseChatModel:
    """Load the base LLM based on the provider.

    Supports ``"ollama"`` (local) and ``"google"`` (Gemini API).

    Sampling parameters for Ollama are model-family aware:

    - ``gemma4:*`` models: temperature=1.0, top_p=0.95, top_k=64 (Google-recommended).
      ``num_predict`` must NOT be set — the internal reasoning pass consumes tokens
      before visible output is produced; a strict limit produces empty responses.
    - All other Ollama models: temperature=0.7 (standard default).

    Tool calling: not all Ollama models support ``bind_tools()``. As of 2026-05,
    ``gemma3:4b`` does NOT. Enable ``hybrid_tool_calling: true`` in ``app_config.yml``
    to use the cloud model for tool selection when running a non-tool-capable local model.

    Args:
        model_provider: One of ``"ollama"`` or ``"google"``.
        model_name: Model identifier passed to the underlying client,
            e.g. ``"gemma3:4b"`` for Ollama or ``"gemini-2.5-flash"`` for Google.
        **kwargs: Additional keyword arguments forwarded to the model constructor.
            For ``"ollama"``, ``base_url`` is popped and defaults to
            ``"http://localhost:11434"`` if not provided.

    Returns:
        An initialised ``BaseChatModel`` instance.

    Raises:
        ValueError: If ``model_provider`` is not ``"ollama"`` or ``"google"``.
    """
    match model_provider.lower():
        case "ollama":
            base_url = kwargs.pop("base_url", "http://localhost:11434")
            is_gemma4 = model_name.lower().startswith("gemma4")
            if is_gemma4:
                # Google-recommended sampling params for Gemma 4 extended-thinking models.
                # Do NOT add num_predict — see module docstring.
                temperature = float(kwargs.pop("temperature", 1.0))
                model = ChatOllama(
                    model=model_name,
                    base_url=base_url,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=64,
                    **kwargs,
                )
            else:
                temperature = float(kwargs.pop("temperature", 0.7))
                model = ChatOllama(
                    model=model_name,
                    base_url=base_url,
                    temperature=temperature,
                    **kwargs,
                )
            logger.info(f"Loaded Ollama model: {model_name} at {base_url}")
            return model
        case "google":
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                max_retries=2,
                google_api_key=GOOGLE_API_KEY,
                **kwargs,
            )
            logger.info(f"Loaded Google model successfully: {model_name}")
            return model
        case _:
            raise ValueError(f"Unsupported model provider: {model_provider}")


def load_model_with_fallback(
    primary_provider: str,
    primary_name: str,
    fallback_provider: str | None = None,
    fallback_name: str | None = None,
) -> BaseChatModel:
    """Load the primary model, falling back to the secondary on failure.

    Intended for API startup: ensures a model is always available even when the
    local Ollama server is offline or the cloud API key is missing.

    Args:
        primary_provider: Primary model provider (e.g. ``"ollama"``).
        primary_name: Primary model name (e.g. ``"gemma3:4b"``).
        fallback_provider: Fallback provider (e.g. ``"google"``). Pass ``None``
            to raise immediately on primary failure instead of falling back.
        fallback_name: Fallback model name (e.g. ``"gemini-2.5-flash"``).

    Returns:
        An initialised ``BaseChatModel`` instance (primary or fallback).

    Raises:
        RuntimeError: If both primary and fallback fail to load.
        Exception: The original primary exception if no fallback is configured.

    Example:
        >>> model = load_model_with_fallback(
        ...     "ollama", "gemma3:4b",
        ...     fallback_provider="google",
        ...     fallback_name="gemini-2.5-flash",
        ... )
    """
    try:
        model = load_base_model(primary_provider, primary_name)
        logger.info(f"Primary model loaded: {primary_provider}/{primary_name}")
        return model
    except Exception as primary_err:
        logger.warning(
            f"Primary model ({primary_provider}/{primary_name}) failed to load: {primary_err}"
        )
        if fallback_provider and fallback_name:
            logger.info(f"Falling back to {fallback_provider}/{fallback_name}")
            try:
                model = load_base_model(fallback_provider, fallback_name)
                logger.info(f"Fallback model loaded: {fallback_provider}/{fallback_name}")
                return model
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Both primary ({primary_provider}/{primary_name}) and fallback "
                    f"({fallback_provider}/{fallback_name}) models failed to load. "
                    f"Primary error: {primary_err}. Fallback error: {fallback_err}"
                ) from fallback_err
        raise


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
    except Exception as e:
        raise ModelInternalError(str(e)) from e


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
    except Exception as e:
        raise ModelInternalError(str(e)) from e


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
