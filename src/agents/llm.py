"""LLM loading, prompt construction, and invocation for RAG and agent pipelines.

Supports Google Gemini and OpenAI providers. Prompts are loaded from markdown
files in ``src/agents/prompts/`` at module import time.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils import get_tracing_callbacks, logger
from src.utils.env import GOOGLE_API_KEY

# Load prompts from markdown files
from pathlib import Path

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

    Currently only Google Gemini is supported. The API key is read from the
    ``GOOGLE_API_KEY`` environment variable (loaded via ``src/utils/env.py``).

    Args:
        model_provider: The model provider. Only ``"google"`` is supported.
        model_name: The model name to load (e.g. ``"gemini-2.0-flash"``).
        **kwargs: Additional keyword arguments forwarded to the model constructor.

    Returns:
        An initialised ``BaseChatModel`` instance.

    Raises:
        ValueError: If ``model_provider`` is not ``"google"``.
    """
    match model_provider.lower():
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
