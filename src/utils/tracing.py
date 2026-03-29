"""Langfuse tracing integration for LLM observability.

Provides a callback handler for LangChain that sends traces to Langfuse
for monitoring LLM calls, latency, and retrieval quality.
"""
from src.utils import logger
from src.utils.env import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


def get_langfuse_callback():
    """Create and return a Langfuse callback handler for LangChain.

    Returns:
        CallbackHandler instance if Langfuse credentials are configured,
        or None if initialization fails.
    """
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host="https://cloud.langfuse.com"
        )

        langfuse_handler = CallbackHandler()
    except Exception as err:
        langfuse_handler = None
        logger.warning(f"Cannot instantiate the Langfuse handler. Langfuse logging is disabled: {err}")

    return langfuse_handler

