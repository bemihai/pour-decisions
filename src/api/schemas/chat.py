"""Pydantic request/response schemas for the chat API."""
from typing import Literal

from pydantic import BaseModel, Field

AgentMode = Literal["intelligent", "keyword", "rag_only"]
ModelProvider = Literal["local", "cloud"]


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: Literal["human", "ai"] = Field(..., description="Message role: 'human' or 'ai'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    """Chat message request."""

    message: str = Field(..., min_length=1, description="User message text")
    agent_mode: AgentMode = Field(
        "intelligent",
        description="Agent mode: 'intelligent', 'keyword', or 'rag_only'",
    )
    model_provider: ModelProvider = Field(
        "local",
        description="LLM backend: 'local' (Ollama/Gemma 4) or 'cloud' (Gemini)",
    )
    message_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages for context",
    )
    enable_rag: bool = Field(True, description="Enable RAG retrieval (rag_only mode)")
    n_results: int | None = Field(None, description="Override number of retrieved chunks (rag_only mode)")


class Source(BaseModel):
    """RAG source citation returned with a chat response."""

    name: str = Field(..., description="Source document name (stem, no extension)")
    page: int | None = Field(None, description="Page number within the source document")
    relevance: float | None = Field(None, description="Similarity / reranking score")


class WebSource(BaseModel):
    """Web search source returned with a chat response."""

    title: str = Field(..., description="Page title or URL fallback")
    url: str = Field(..., description="Full URL of the web source")


class ChatResponse(BaseModel):
    """Chat response with answer and source citations."""

    answer: str = Field(..., description="Generated answer text")
    sources: list[Source] = Field(default_factory=list, description="RAG source citations")
    web_sources: list[WebSource] = Field(default_factory=list, description="Web search sources")
    agent_mode: AgentMode = Field(..., description="Agent mode that produced this response")
    model_provider: ModelProvider | None = Field(
        None,
        description="LLM backend that produced this response ('local' or 'cloud')",
    )
    error: str | None = Field(None, description="Error message if the request failed gracefully")


class InitialMessageResponse(BaseModel):
    """Initial welcome message returned to new chat sessions."""

    role: str = Field("ai", description="Message role (always 'ai')")
    content: str = Field(..., description="Welcome message text")
