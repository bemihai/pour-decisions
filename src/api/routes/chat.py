"""Chat API endpoints.

Consolidates agent invocation and RAG-only query logic from
``src/ui/pages/chatbot.py`` into stateless REST endpoints.
Each request carries its own message history; no server-side session.

The POST dispatcher is asynchronous. Intelligent mode awaits the compiled
LangGraph runtime directly, while the synchronous RAG-only pipeline is bridged
through ``asyncio.to_thread()`` until native async retrieval arrives in M6B.
"""
import asyncio
import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.language_models import BaseChatModel

from src.agents.provenance import ExecutionProvenance, build_rag_execution_provenance
from src.api.dependencies import (
    get_reranker,
    get_retriever,
)
from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    InitialMessageResponse,
    ModelProvider,
    Source,
    WebSource,
)
from src.retrieval import execute_production_rag
from src.utils import (
    get_trace_context,
    is_observability_active,
    logger,
    set_execution_provenance_attributes,
    set_span_attributes,
    start_request_span,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


_WEB_SEARCH_TOOLS = {"search_web_for_wine", "search_wine_price", "search_wine_reviews"}
_SOURCE_RE = re.compile(r"Source:\s*(https?://\S+)")
_DEFAULT_INITIAL_MESSAGE = InitialMessageResponse(
    role="assistant",
    content="Hello. How can I help you with wine today?",
)


def _extract_web_sources_from_messages(messages: list) -> list[WebSource]:
    """Extract web source URLs and titles from agent ToolMessage objects.

    Mirrors ``_extract_web_sources()`` in ``pages/chatbot.py`` but returns
    typed ``WebSource`` models instead of raw dicts.

    Args:
        messages: LangGraph message list from an agent invocation result.

    Returns:
        Deduplicated list of ``WebSource`` instances.
    """
    seen: set[str] = set()
    sources: list[WebSource] = []

    for msg in messages:
        tool_name = getattr(msg, "name", None)
        if tool_name not in _WEB_SEARCH_TOOLS:
            continue
        content = getattr(msg, "content", "") or ""
        urls = _SOURCE_RE.findall(content)
        lines = content.splitlines()

        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            title = url  # fallback
            for i, line in enumerate(lines):
                if url in line and i >= 2:
                    title_line = lines[i - 2].strip()
                    if title_line:
                        title = re.sub(r"^\[\d+] ", "", title_line)
                    break
            sources.append(WebSource(title=title, url=url))

    return sources


def _format_sources(retrieved_docs: list[dict]) -> list[Source]:
    """Convert raw retrieved docs to typed ``Source`` models.

    Mirrors ``format_sources_for_display()`` in ``context_builder.py``
    but returns ``Source`` Pydantic models.

    Args:
        retrieved_docs: Documents returned by the retriever.

    Returns:
        List of ``Source`` models with name, page, and relevance.
    """
    from pathlib import Path as _Path

    sources: list[Source] = []
    for doc in retrieved_docs:
        metadata = doc.get("metadata", {})
        similarity = doc.get("similarity")

        raw_source: str = str(
            metadata.get("source", metadata.get("filename", "Unknown")) or "Unknown"
        )
        if "/" in raw_source:
            raw_source = raw_source.split("/")[-1]
        name = _Path(raw_source).stem

        page = metadata.get("page", metadata.get("page_number"))

        sources.append(Source(name=name, page=page, relevance=similarity))
    return sources


def _filter_cited_sources(answer: str, sources: list[Source]) -> list[Source]:
    """Keep only sources actually cited in the LLM answer text.

    Detects citation patterns like ``[1]``, ``[2, 3]``, ``[1, 4, 5]``.

    Args:
        answer: The generated answer text.
        sources: Full list of retrieved sources.

    Returns:
        Filtered list containing only cited sources, or the original list
        if no valid citations are found.
    """
    citation_pattern = r"\[(\d+(?:\s*,\s*\d+)*)\]"
    matches = re.findall(citation_pattern, answer)
    if not matches:
        return sources

    cited_numbers: set[int] = set()
    for match in matches:
        cited_numbers.update(int(n.strip()) for n in match.split(","))

    cited: list[Source] = []
    for num in sorted(cited_numbers):
        if 1 <= num <= len(sources):
            cited.append(sources[num - 1])

    return cited if cited else sources


async def _ainvoke_intelligent_agent(
    agent: Any,
    prompt: str,
    message_history: list[dict],
    trace_context: dict[str, str] | None = None,
) -> tuple[str, list[Source], list[WebSource]]:
    """Await the intelligent agent and preserve the route helper result shape."""
    result = await agent.ainvoke(
        prompt,
        message_history=message_history,
        trace_context=trace_context,
    )
    answer = result.get("final_answer", "")
    web_sources = _extract_web_sources_from_messages(result.get("messages", []))
    return answer, [], web_sources


def _invoke_rag_only(
    prompt: str,
    cfg,
    model: BaseChatModel,
    retriever,
    reranker,
    message_history: list[dict],
    enable_rag: bool,
    n_results_override: int | None,
    trace_context: dict[str, str] | None = None,
) -> tuple[str, list[Source], list[WebSource]]:
    """Run the shared production RAG pipeline (no agent).

    Args:
        prompt: User question.
        model: Pre-loaded LLM.
        retriever: Pre-loaded retriever (hybrid or vector-only), or None.
        reranker: Pre-loaded reranker, or None.
        message_history: Conversation history as list of dicts.
        enable_rag: Whether to perform retrieval.
        n_results_override: Optional override for number of retrieved chunks.
        trace_context: Optional request trace metadata.

    Returns:
        Tuple of (answer, rag_sources, empty web_sources).
    """
    result = execute_production_rag(
        prompt=prompt,
        config=cfg,
        model=model,
        retriever=retriever,
        reranker=reranker,
        message_history=message_history,
        enable_retrieval=enable_rag,
        n_results_override=n_results_override,
        generation_enabled=True,
        trace_context=trace_context,
    )
    sources = [
        Source(name=source.name, page=source.page, relevance=source.relevance)
        for source in result.sources
    ]
    return result.answer, sources, []


async def _ainvoke_rag_only(
    prompt: str,
    cfg: Any,
    model: BaseChatModel,
    retriever: Any,
    reranker: Any,
    message_history: list[dict],
    enable_rag: bool,
    n_results_override: int | None,
    trace_context: dict[str, str] | None = None,
) -> tuple[str, list[Source], list[WebSource]]:
    """Run the synchronous RAG-only pipeline on a worker thread.

    Cancelling the await does not stop work already executing in the thread.
    Native async retrieval remains deferred to M6B.
    """
    return await asyncio.to_thread(
        _invoke_rag_only,
        prompt=prompt,
        cfg=cfg,
        model=model,
        retriever=retriever,
        reranker=reranker,
        message_history=message_history,
        enable_rag=enable_rag,
        n_results_override=n_results_override,
        trace_context=trace_context,
    )


def _is_observability_enabled() -> bool:
    """Return True when observability is active.

    Thin wrapper around ``is_observability_active()`` so tests can monkeypatch
    this at the module level without affecting the imported symbol.
    """
    return is_observability_active()


def _resolve_request_execution_provenance(
    *,
    mode: str,
    model: BaseChatModel | None,
    intelligent_agent: Any,
) -> ExecutionProvenance | None:
    """Return provenance for the actual resource selected after fallback."""
    if mode == "intelligent":
        provenance = getattr(intelligent_agent, "execution_provenance", None)
        return provenance if isinstance(provenance, ExecutionProvenance) else None
    if model is not None:
        return build_rag_execution_provenance(model)
    return None


_QUOTA_KEYWORDS = ("429", "RESOURCE_EXHAUSTED", "quota")


def _friendly_error_message(error: Exception, agent_label: str) -> str:
    """Produce a user-friendly error string from an agent exception.

    Args:
        error: The caught exception.
        agent_label: Human-readable agent name for the error message.

    Returns:
        A user-facing error string.
    """
    error_type = type(error).__name__
    error_msg = str(error)

    if any(kw in error_msg for kw in _QUOTA_KEYWORDS) or "quota" in error_msg.lower():
        return (
            "The AI service quota has been exceeded. Please try again later "
            "or switch to 'rag_only' mode."
        )
    if "ChatGoogleGenerativeAI" in error_type or "APIError" in error_type:
        return (
            f"There was an issue with the AI service. Please try again later "
            f"or switch to 'rag_only' mode. (Error: {error_type})"
        )
    return (
        f"Error processing your request with the {agent_label}. "
        f"Please try again or switch to a different agent mode. (Error: {error_type})"
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/", response_model=ChatResponse)
async def send_message(
    http_request: Request,
    request: ChatRequest,
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
) -> ChatResponse:
    """Send a chat message and get a response from the selected agent.

    The ``agent_mode`` field selects the execution path:

    * ``intelligent`` -- LangGraph ReAct agent with tool selection (2-3 LLM calls).
    * ``rag_only`` -- Traditional RAG pipeline, no agent.

    The ``model_provider`` field selects the LLM backend:

    * ``cloud`` -- Google Gemini API (production default).
    * ``local`` -- Ollama, only when local startup is enabled explicitly; otherwise falls back to cloud.
    """
    mode = request.agent_mode
    provider = request.model_provider
    prompt = request.message
    state = http_request.app.state
    request_id = http_request.headers.get("X-Request-Id") or str(uuid.uuid4())
    session_id = http_request.headers.get("X-Session-Id")
    trace_context = get_trace_context(request_id=request_id, session_id=session_id, agent_mode=mode)

    # Select model and agents based on the requested provider.
    # "local" falls back to cloud automatically when local startup is disabled
    # or Ollama is unavailable.
    if provider == "cloud":
        local_model = getattr(state, "local_model", None)
        local_intelligent_agent = getattr(state, "local_intelligent_agent", None)
        model = getattr(state, "cloud_model", None) or getattr(state, "model", None)
        intelligent_agent = getattr(state, "cloud_intelligent_agent", None) or getattr(
            state, "intelligent_agent", None
        )

        if mode == "intelligent":
            actual_provider = (
                "local"
                if (
                    local_intelligent_agent is not None
                    and intelligent_agent is local_intelligent_agent
                )
                else "cloud"
            )
        else:
            actual_provider = "local" if local_model is not None and model is local_model else "cloud"
    else:
        local_model = getattr(state, "local_model", None)
        cloud_model = getattr(state, "cloud_model", None) or getattr(state, "model", None)
        local_intelligent_agent = getattr(state, "local_intelligent_agent", None)
        cloud_intelligent_agent = getattr(state, "cloud_intelligent_agent", None) or getattr(
            state, "intelligent_agent", None
        )

        model = local_model or cloud_model
        intelligent_agent = local_intelligent_agent or cloud_intelligent_agent

        # Report provider for the active execution path (agent/model actually selected).
        if mode == "intelligent":
            actual_provider = "local" if local_intelligent_agent is not None else "cloud"
        else:
            actual_provider = "local" if local_model is not None else "cloud"

    message_history = [{"role": m.role, "content": m.content} for m in request.message_history]

    answer = ""
    sources: list[Source] = []
    web_sources: list[WebSource] = []
    error: str | None = None

    with start_request_span(trace_context) as span:
        set_span_attributes(span, {"route": "/api/chat/", "agent_mode": mode})
        if span is not None:
            execution_provenance = _resolve_request_execution_provenance(
                mode=mode,
                model=model,
                intelligent_agent=intelligent_agent,
            )
            if execution_provenance is not None:
                set_execution_provenance_attributes(
                    span,
                    execution_provenance.to_trace_attributes(),
                )

        try:
            if mode == "intelligent":
                if intelligent_agent is None:
                    raise HTTPException(status_code=503, detail="Intelligent agent not available")
                answer, sources, web_sources = await _ainvoke_intelligent_agent(
                    intelligent_agent, prompt, message_history, trace_context=trace_context
                )
            else:  # rag_only (default fallback)
                if model is None:
                    raise HTTPException(
                        status_code=503,
                        detail="LLM model not available. Check startup logs for loading errors.",
                    )
                answer, sources, web_sources = await _ainvoke_rag_only(
                    prompt=prompt,
                    cfg=getattr(state, "config"),
                    model=model,
                    retriever=retriever,
                    reranker=reranker,
                    message_history=message_history,
                    enable_rag=request.enable_rag,
                    n_results_override=request.n_results,
                    trace_context=trace_context,
                )

            set_span_attributes(
                span,
                {
                    "http_status_code": 200,
                    "retrieval_enabled": request.enable_rag,
                },
            )

        except HTTPException as http_error:
            set_span_attributes(
                span,
                {
                    "http_status_code": http_error.status_code,
                    "error_class": type(http_error).__name__,
                },
            )
            raise
        except Exception as e:
            agent_label = {"intelligent": "intelligent agent"}.get(mode, "RAG pipeline")
            logger.error(f"Error in chat ({mode}): {e}", exc_info=True)
            error = _friendly_error_message(e, agent_label)
            answer = error
            set_span_attributes(
                span,
                {
                    "http_status_code": 500,
                    "error_class": type(e).__name__,
                },
            )

    return ChatResponse(
        answer=answer or "",
        sources=sources,
        web_sources=web_sources,
        agent_mode=mode,
        model_provider=actual_provider,
        error=error,
        trace_id=request_id if _is_observability_enabled() else None,
    )


@router.get("/initial-message", response_model=InitialMessageResponse)
def get_initial() -> InitialMessageResponse:
    """Return the initial welcome message for new chat sessions."""
    return _DEFAULT_INITIAL_MESSAGE
