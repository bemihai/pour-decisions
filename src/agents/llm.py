"""LLM loading, prompt construction, and invocation for RAG and agent pipelines.

Supports Google Gemini and OpenAI providers. Prompts are loaded from markdown
files in ``src/agents/prompts/`` at module import time.
"""
import streamlit as st

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import CallbackManager

from src.utils import logger, get_langfuse_callback
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
    """
    Loads the base LLM agents based on the provider.

    Args:
        model_provider (str): The agents provider, e.g., "google", "openai".
        model_name (str): The name of the agents to load.
        **kwargs: Additional keyword arguments to pass to the agents constructor.

    Returns: An instance of the loaded chat agents.
    """
    # TODO: fix langfuse with langchain v1
    # callback_manager = CallbackManager([get_langfuse_callback()])
    match model_provider.lower():
        case "google":
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0,
                max_retries=2,
                google_api_key=GOOGLE_API_KEY,
                # callback_manager=callback_manager,
                **kwargs,
            )
            logger.info(f"Loaded Google agents successfully: {model_name}")
            return model
        case "openai":
            api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in Streamlit secrets.")
            model = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                max_retries=2,
                api_key=api_key,
                # callback_manager=callback_manager,
                **kwargs,
            )
            logger.info(f"Loaded OpenAI agents successfully: {model_name}")
            return model
        case _:
            raise ValueError(f"Unsupported agents provider: {model_provider}")


def invoke_llm(question: str, context: str, model: BaseChatModel, message_history: list) -> str:
    """
    Invoke the LLM agents with the provided question, context, and full message history.

    Args:
        question (str): The user's question to be answered by the agents.
        context (str): The context retrieved by the RAG pipeline.
        model (BaseChatModel): The loaded LLM agents instance.
        message_history (list): List of dicts with previous messages, each having 'role' and 'question'/'answer'.

    Returns: The agents's answer as a string.
    """
    messages = [("system", SYSTEM_PROMPT)]
    for msg in message_history:
        if msg["role"] == "human" and "question" in msg:
            messages.append(("human", msg["question"]))
        elif msg["role"] == "ai" and "answer" in msg:
            messages.append(("ai", msg["answer"]))
    # Inject the user prompt with context and question
    messages.append(("human", USER_PROMPT.format(question=question, context=context)))
    prompt = ChatPromptTemplate.from_messages(messages)
    tagging_chain = prompt | model

    try:
        model_output = tagging_chain.invoke({"question": question, "context": context})
        if hasattr(model_output, "content"):
            return model_output.content
        elif isinstance(model_output, dict) and "content" in model_output:
            return model_output["content"]
        else:
            return str(model_output)
    except Exception as e:
        raise ModelInternalError() from e


def process_user_prompt(model: BaseChatModel, prompt: str, context: str, message_history: list) -> str:
    """Process a user prompt with context"""
    try:
        answer = invoke_llm(prompt, context, model, message_history)
    except ModelInternalError as err:
        answer = err.default_message
        logger.error(f"ModelInternalError: {err}")
    return answer
