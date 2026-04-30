"""LLM loading, prompt construction, and invocation for RAG and agent pipelines.

Supports Google Gemini (cloud) and Ollama (local) providers. Prompts are loaded
from markdown files in ``src/agents/prompts/`` at module import time.

Provider notes:
- ``"ollama"``: Local inference via Ollama server (``localhost:11434``). Use
  ``gemma4:e2b`` with Google-recommended sampling params (temperature=1.0,
  top_p=0.95, top_k=64). Do NOT set ``num_predict`` -- Gemma 4 uses an internal
  reasoning pass whose tokens count against the budget before visible output is
  produced; a strict limit produces empty responses.
- ``"google"``: Google Gemini API. Requires ``GOOGLE_API_KEY`` in the environment.
"""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from src.utils import get_tracing_callbacks, logger
from src.utils.env import GOOGLE_API_KEY
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from src.utils import logger
from src.utils.env import GOOGLE_API_KEY

_prompt_dir = Path(__file__).parent / "prompts"

try:
    with open(_prompt_dir / "rag_only_system_prompt.md", 'r') as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    logger.warning("RAG system prompt file not found. Using default.")
    SYSTEM_PROMPT = "You are a helpful wine expert assistant."

try:
    with open(_prompt_dir / "rag_only_user_prompt.md", 'r') as f:
        USER_PROMPT = f.read().strip()
except FileNotFoundError:
    logger.warning("RAG user prompt file not found. Using default.")
    USER_PROMPT = "Context: {context}\n\nQuestion: {question}"


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

    Supports ``"ollama"`` (local Gemma 4) and ``"google"`` (Gemini API).

    Args:
        model_provider: One of ``"ollama"`` or ``"google"``.
        model_name: Model identifier passed to the underlying client,
            e.g. ``"gemma4:e2b"`` for Ollama or ``"gemini-2.5-flash"`` for Google.
        **kwargs: Additional keyword arguments forwarded to the model constructor.
            For ``"ollama"``, ``base_url`` is popped and defaults to
            ``"http://localhost:11434"`` if not provided.

    Returns:
        An initialised ``BaseChatModel`` instance.

    Raises:
        ValueError: If ``model_provider`` is not ``"ollama"`` or ``"google"``.

    Note:
        Gemma 4 uses an internal reasoning pass before producing visible output.
        Do not set ``num_predict`` -- it truncates the thinking tokens and results
        in an empty response. Google recommends temperature=1.0, top_p=0.95,
        top_k=64 for Gemma 4, and these are applied automatically for the
        ``"ollama"`` provider.
    """
    match model_provider.lower():
        case "ollama":
            base_url = kwargs.pop("base_url", "http://localhost:11434")
            model = ChatOllama(
                model=model_name,
                base_url=base_url,
                # Google-recommended sampling parameters for Gemma 4
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                # Do NOT add num_predict here -- see module docstring
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
        primary_name: Primary model name (e.g. ``"gemma4:e2b"``).
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
        ...     "ollama", "gemma4:e2b",
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
        message_history (list): List of dicts with previous messages, each having 'role' and 'question'/'answer'.
        trace_context: Optional request trace metadata forwarded to the chain runtime.

    Returns: The agents's answer as a string.
    """
    messages = [("system", SYSTEM_PROMPT)]
    for msg in message_history:
        if msg["role"] == "human" and "question" in msg:
            messages.append(("human", msg["question"]))
        elif msg["role"] == "ai" and "answer" in msg:
            messages.append(("ai", msg["answer"]))
    # Inject the user prompt with context and question
    # Escape literal braces in context so str.format() does not misinterpret them
    # as format placeholders (retrieved documents may contain JSON, recipes, etc.)
    safe_context = context.replace("{", "{{").replace("}", "}}")
    messages.append(("human", USER_PROMPT.format(question=question, context=safe_context)))
    prompt = ChatPromptTemplate.from_messages(messages)
    tagging_chain = prompt | model
    callbacks = get_tracing_callbacks()

    try:
        invoke_config: RunnableConfig | None = None
        if trace_context or callbacks:
            invoke_config = RunnableConfig(
                metadata=trace_context or {},
                callbacks=callbacks,
            )
        if invoke_config:
            model_output = tagging_chain.invoke({"question": question, "context": context}, config=invoke_config)
        else:
            model_output = tagging_chain.invoke({"question": question, "context": context})
        if hasattr(model_output, "content"):
            content = model_output.content
            return content if isinstance(content, str) else str(content)
        elif isinstance(model_output, dict) and "content" in model_output:
            return model_output["content"]
        else:
            return str(model_output)
    except Exception as e:
        raise ModelInternalError() from e


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
