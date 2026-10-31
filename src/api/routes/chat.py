"""Chat API endpoints.

Consolidates agent invocation and RAG-only query logic from
``src/ui/pages/chatbot.py`` into REST endpoints. Intelligent requests may opt
into durable server-side threads; all other requests remain stateless and use
the client-supplied message history.

The POST dispatcher awaits both the compiled LangGraph runtime and the shared
asynchronous production RAG service directly.
"""

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from src.agents.guardrails import (
    RELEVANCE_REDIRECT,
    ToolProgressReporter,
    evaluate_relevance,
    load_relevance_config,
)
from src.agents.provenance import ExecutionProvenance, build_rag_execution_provenance
from src.api.dependencies import (
    get_async_rag_runtime,
    get_conversation_memory_manager,
    get_optional_conversation_service,
)
from src.api.schemas.chat import (
    AgentDoneStreamEvent,
    ChatRequest,
    ChatResponse,
    InitialMessageResponse,
    ModelProvider,
    Source,
    StreamErrorEvent,
    ThreadAction,
    ToolProgressStreamEvent,
    WebSource,
)
from src.database.repository.conversation import ConversationConflict, ConversationOperation
from src.retrieval import (
    AsyncRAGRuntimeResources,
    correction_request_scope,
    execute_production_rag_async,
)
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
_STREAM_HEARTBEAT_SECONDS = 15.0
_STREAM_CACHE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


async def _reject_saved_thread(thread_id: str | None, conversation_service: Any) -> None:
    """Prevent legacy endpoints from bypassing saved-session lifecycle."""
    if thread_id is not None and conversation_service is not None and await conversation_service.is_owned(thread_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "saved_conversation_requires_lifecycle_api"},
        )


def _select_execution_resources(
    state: Any,
) -> tuple[BaseChatModel | None, Any, ModelProvider]:
    """Select the single Cloud model and agent for either chat transport."""
    return getattr(state, "cloud_model", None), getattr(state, "intelligent_agent", None), "cloud"


def _build_chat_response(
    request: ChatRequest,
    actual_provider: ModelProvider,
    request_id: str,
    *,
    answer: str,
    sources: list[Source],
    web_sources: list[WebSource],
    error: str | None,
) -> ChatResponse:
    """Build the authoritative response shared by blocking and streaming delivery."""
    return ChatResponse(
        answer=answer or "",
        sources=sources,
        web_sources=web_sources,
        agent_mode=request.agent_mode,
        model_provider=actual_provider,
        error=error,
        trace_id=request_id if _is_observability_enabled() else None,
        thread_id=request.thread_id,
    )


def _streaming_enabled(state: Any) -> bool:
    """Return true only for the explicit reviewed streaming feature flag."""
    streaming = getattr(getattr(state, "config", None), "streaming", None)
    return getattr(streaming, "enabled", False) is True


async def _validate_stream_preconditions(
    request: ChatRequest,
    state: Any,
    memory_manager: Any,
    intelligent_agent: Any,
) -> None:
    """Reject known failures before creating a streaming response."""
    if not _streaming_enabled(state):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "streaming_disabled",
                "message": "Streaming chat is disabled.",
            },
        )
    if request.agent_mode != "intelligent":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "stream_mode_unsupported",
                "message": "Streaming is supported only for intelligent mode.",
            },
        )
    if intelligent_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Intelligent agent not available",
        )
    if memory_manager is not None and request.thread_id is not None and request.thread_action == "replace_last":
        thread = await memory_manager.get_thread(str(request.thread_id))
        if thread is None or thread.completed_turns == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot replace a turn before the thread has a completed turn",
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


_RAG_SOURCE_TOOLS = {
    "search_wine_knowledge",
    "search_wine_region_info",
    "search_grape_variety_info",
    "search_wine_term_definition",
    "search_wine_producer_info",
}
_PERSONAL_SOURCE_TOOLS = {
    "cellar": {
        "get_cellar_wines",
        "get_wine_details",
        "get_cellar_statistics",
        "get_wine_recommendations_from_profile",
        "get_pairing_for_wine",
        "get_food_pairing_wines",
        "get_wine_and_cheese_pairings",
    },
    "tasting_history": {
        "get_user_taste_profile",
        "get_tasting_history",
        "get_top_rated_wines",
        "get_wine_recommendations_from_profile",
        "compare_wine_to_profile",
    },
}
_PERSONAL_SOURCE_LABELS = {
    "cellar": "your cellar data",
    "tasting_history": "your tasting history",
    "user_provided": "information you provided",
}
_DOCUMENT_HEADER_RE = re.compile(
    r"^\[Source (?P<number>\d+) - (?P<name>.+?)"
    r"(?:, Page (?P<page>-?\d+))?(?:, Chunk [^,\]\n]+)?"
    r"(?:, Relevance: (?P<relevance>\d+(?:\.\d+)?))?\]$",
    re.MULTILINE,
)
_SOURCE_NOTE_RE = re.compile(r"^\s*(?:\*\*)?Sources:(?:\*\*)?[ \t]*", re.MULTILINE)
_NUMERIC_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")


def _source_name_key(name: str) -> str:
    """Compare document identities without filename extensions or incidental whitespace."""
    name = name.strip()
    if Path(name).suffix.casefold() in {".epub", ".pdf", ".txt", ".md", ".html"}:
        name = Path(name).stem
    return " ".join(name.casefold().split())


def _has_personal_source_data(content: str, tool_name: str, kind: str) -> bool:
    """Reject failed, empty, and status-only personal tool responses."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return False
    if isinstance(payload, list):
        if kind == "tasting_history" and tool_name == "get_wine_recommendations_from_profile":
            return any(isinstance(item, dict) and item.get("predicted_rating") is not None for item in payload)
        return bool(payload)
    if not isinstance(payload, dict) or payload.get("error") or payload.get("success") is False:
        return False
    if tool_name == "get_food_pairing_wines":
        return bool(payload.get("cellar_matches"))
    if tool_name == "get_wine_and_cheese_pairings":
        return bool(payload.get("cellar_suggestions"))
    if kind == "tasting_history" and tool_name == "compare_wine_to_profile":
        return payload.get("predicted_rating") is not None
    if "items" in payload:
        return bool(payload["items"])
    if "total_wines_rated" in payload:
        return bool(payload["total_wines_rated"])
    return any(value for key, value in payload.items() if key not in {"message", "status", "success", "query"})


def _finalize_intelligent_sources(result: dict) -> tuple[str, list[Source], list[WebSource]]:
    """Resolve the final answer's source selections against current-turn evidence.

    The terminal Sources note selects evidence; it never supplies trusted metadata.
    Unknown selections are omitted, and tool-local numeric markers are retained only
    when they map unambiguously to a selected document. Historical stored answers
    never pass through this publication boundary.
    """
    answer = result.get("final_answer", "")
    messages = list(result.get("messages", []))
    start = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), 0)
    messages = messages[start:]
    note_match = None
    for candidate in _SOURCE_NOTE_RE.finditer(answer):
        # A Sources example inside a fenced code block is answer content.
        preceding_fences = re.findall(r"^\s*(?:```|~~~)", answer[: candidate.start()], re.MULTILINE)
        if len(preceding_fences) % 2 == 0:
            note_match = candidate
    note = answer[note_match.end() :].strip() if note_match else ""
    body = answer[: note_match.start()].rstrip() if note_match else answer
    selections = {line.strip().strip("-* `").casefold() for line in re.split(r"[\n;]", note) if line.strip()}
    document_selections = {_source_name_key(line) for line in selections}
    sources: list[Source] = []
    number_targets: dict[int, set[tuple[str, int | None]]] = {}
    selected_targets: dict[tuple[str, int | None], int] = {}
    successful_tools = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.status != "error" and isinstance(message.content, str)
    ]
    for message in successful_tools:
        if message.name in _WEB_SEARCH_TOOLS:
            for number in re.findall(r"^\[(\d+)\] ", message.content, re.MULTILINE):
                number_targets.setdefault(int(number), set()).add((f"web:{message.tool_call_id}:{number}", None))
        if message.name not in _RAG_SOURCE_TOOLS:
            continue
        for number in re.findall(r"^\[Source (\d+)\]$", message.content, re.MULTILINE):
            number_targets.setdefault(int(number), set()).add((f"unknown:{message.tool_call_id}:{number}", None))
        for match in _DOCUMENT_HEADER_RE.finditer(message.content):
            name = Path(match["name"].strip()).stem
            page = int(match["page"]) if match["page"] and int(match["page"]) > 0 else None
            target = (_source_name_key(match["name"]), page)
            number_targets.setdefault(int(match["number"]), set()).add(target)
            if target[0] not in document_selections:
                continue
            relevance = float(match["relevance"]) if match["relevance"] else None
            if target in selected_targets:
                existing = sources[selected_targets[target] - 1]
                if relevance is not None and (existing.relevance is None or relevance > existing.relevance):
                    existing.relevance = relevance
                continue
            sources.append(Source(name=name, page=page, relevance=relevance))
            selected_targets[target] = len(sources)

    for kind, label in _PERSONAL_SOURCE_LABELS.items():
        if label not in selections:
            continue
        if kind == "user_provided":
            available = any(isinstance(message, HumanMessage) and bool(message.content) for message in messages)
        else:
            available = any(
                message.name in _PERSONAL_SOURCE_TOOLS[kind]
                and _has_personal_source_data(message.content, message.name, kind)
                for message in successful_tools
            )
        if available:
            sources.append(Source(name=label, kind=kind))

    selected_urls = {url.rstrip(".,;)") for url in re.findall(r"https?://[^\s<>]+", note)}
    web_sources = [
        source for source in _extract_web_sources_from_messages(successful_tools) if source.url in selected_urls
    ]

    def resolve_marker(match: re.Match) -> str:
        """Remap unambiguous tool-local markers to the public source order."""
        resolved = []
        for value in match[1].split(","):
            targets = number_targets.get(int(value), set())
            if len(targets) != 1:
                continue
            target = next(iter(targets))
            if target in selected_targets:
                resolved.append(str(selected_targets[target]))
        return f"[{', '.join(dict.fromkeys(resolved))}]" if resolved else ""

    body = _NUMERIC_CITATION_RE.sub(resolve_marker, body)
    body = re.sub(r"[ \t]+([.,;:!?])", r"\1", body)
    return body, sources, web_sources


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

        raw_source: str = str(metadata.get("source", metadata.get("filename", "Unknown")) or "Unknown")
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
    thread_id: str | None = None,
    thread_action: ThreadAction = "append",
    progress_reporter: ToolProgressReporter | None = None,
) -> tuple[str, list[Source], list[WebSource]]:
    """Await the intelligent agent and preserve the route helper result shape."""
    invoke_kwargs: dict[str, Any] = {
        "message_history": message_history,
        "trace_context": trace_context,
    }
    if thread_id is not None:
        invoke_kwargs.update(thread_id=thread_id, thread_action=thread_action)
    if progress_reporter is not None:
        invoke_kwargs["progress_reporter"] = progress_reporter
    try:
        result = await agent.ainvoke(prompt, **invoke_kwargs)
    except RuntimeError as error:
        if thread_action == "replace_last" and str(error) == (
            "Cannot replace a turn before the thread has a completed turn"
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot replace a turn before the thread has a completed turn",
            ) from error
        raise
    return _finalize_intelligent_sources(result)


def _serialize_stream_event(event: BaseModel) -> str:
    """Serialize one validated event with matching SSE and JSON type names."""
    event_type = getattr(event, "type")
    return f"event: {event_type}\ndata: {event.model_dump_json()}\n\n"


async def _execute_streaming_agent(
    *,
    request: ChatRequest,
    intelligent_agent: Any,
    actual_provider: ModelProvider,
    request_id: str,
    trace_context: dict[str, str],
    message_history: list[dict],
    active_thread_id: str | None,
    progress_reporter: ToolProgressReporter,
    conversation_service: Any = None,
) -> ChatResponse:
    """Execute and finalize one intelligent turn for streaming delivery."""
    if request.conversation_id is not None:
        return await _execute_saved_intelligent(
            request=request,
            conversation_service=conversation_service,
            intelligent_agent=intelligent_agent,
            actual_provider=actual_provider,
            request_id=request_id,
            trace_context=trace_context,
            progress_reporter=progress_reporter,
        )
    invoke_kwargs: dict[str, Any] = {
        "trace_context": trace_context,
        "progress_reporter": progress_reporter,
    }
    if active_thread_id is not None:
        invoke_kwargs.update(
            thread_id=active_thread_id,
            thread_action=request.thread_action,
        )
    answer, sources, web_sources = await _ainvoke_intelligent_agent(
        intelligent_agent,
        request.message,
        message_history,
        **invoke_kwargs,
    )
    return _build_chat_response(
        request,
        actual_provider,
        request_id,
        answer=answer,
        sources=sources,
        web_sources=web_sources,
        error=None,
    )


async def _stream_agent_events(
    *,
    request: ChatRequest,
    intelligent_agent: Any,
    actual_provider: ModelProvider,
    request_id: str,
    trace_context: dict[str, str],
    message_history: list[dict],
    active_thread_id: str | None,
    conversation_service: Any = None,
    heartbeat_seconds: float = _STREAM_HEARTBEAT_SECONDS,
) -> AsyncIterator[str]:
    """Yield bounded progress and exactly one terminal event for a connected run."""
    progress_reporter = ToolProgressReporter()
    progress_task: asyncio.Task | None = None
    execution_task: asyncio.Task | None = None
    with start_request_span(trace_context) as span:
        set_span_attributes(span, {"route": "/api/chat/stream", "agent_mode": request.agent_mode})
        if span is not None:
            execution_provenance = _resolve_request_execution_provenance(
                mode=request.agent_mode,
                model=None,
                intelligent_agent=intelligent_agent,
            )
            if execution_provenance is not None:
                set_execution_provenance_attributes(
                    span,
                    execution_provenance.to_trace_attributes(),
                )
        execution_task = asyncio.create_task(
            _execute_streaming_agent(
                request=request,
                intelligent_agent=intelligent_agent,
                actual_provider=actual_provider,
                request_id=request_id,
                trace_context=trace_context,
                message_history=message_history,
                active_thread_id=active_thread_id,
                progress_reporter=progress_reporter,
                conversation_service=conversation_service,
            ),
            name="chat-stream-execution",
        )
        terminal_emitted = False
        try:
            while not terminal_emitted:
                if progress_reporter.pending_count:
                    progress = progress_reporter.get_nowait()
                    yield _serialize_stream_event(
                        ToolProgressStreamEvent(
                            invocation_id=progress.invocation_id,
                            tool_key=progress.tool_key,
                            status=progress.status.value,
                        )
                    )
                    continue

                if execution_task.done():
                    try:
                        response = await execution_task
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        logger.error("Error in streaming chat (intelligent): %s", type(error).__name__)
                        set_span_attributes(
                            span,
                            {
                                "http_status_code": 500,
                                "error_class": type(error).__name__,
                            },
                        )
                        yield _serialize_stream_event(StreamErrorEvent())
                    else:
                        set_span_attributes(
                            span,
                            {
                                "http_status_code": 200,
                                "retrieval_enabled": request.enable_rag,
                            },
                        )
                        yield _serialize_stream_event(AgentDoneStreamEvent(response=response))
                    terminal_emitted = True
                    continue

                if progress_task is None:
                    progress_task = asyncio.create_task(
                        progress_reporter.get(),
                        name="chat-stream-progress-reader",
                    )
                done, _pending = await asyncio.wait(
                    {execution_task, progress_task},
                    timeout=heartbeat_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if progress_task in done:
                    progress = progress_task.result()
                    progress_task = None
                    yield _serialize_stream_event(
                        ToolProgressStreamEvent(
                            invocation_id=progress.invocation_id,
                            tool_key=progress.tool_key,
                            status=progress.status.value,
                        )
                    )
                elif execution_task not in done:
                    yield ": heartbeat\n\n"
        finally:
            if progress_task is not None and not progress_task.done():
                progress_task.cancel()
            if execution_task is not None and not execution_task.done():
                execution_task.cancel()
            cleanup_tasks = [task for task in (progress_task, execution_task) if task is not None]
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)


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
    """Run the shared asynchronous production RAG pipeline directly."""
    with correction_request_scope():
        result = await execute_production_rag_async(
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
    sources = [Source(name=source.name, page=source.page, relevance=source.relevance) for source in result.sources]
    return result.answer, sources, []


def _rag_only_relevance_redirect(prompt: str, cfg: Any) -> str | None:
    """Return the fixed redirect for a configured off-topic RAG-only query."""
    relevance = load_relevance_config(cfg)
    decision = evaluate_relevance(prompt, relevance)
    return RELEVANCE_REDIRECT if decision.route == "deflect" else None


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
    """Return provenance for the Cloud resource used by this request."""
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
        return "The AI service quota has been exceeded. Please try again later " "or switch to 'rag_only' mode."
    if error_type in {"ResponseError", "ConnectError", "TimeoutException"}:
        return (
            f"There was an issue with the AI service. Please try again later "
            f"or switch to 'rag_only' mode. (Error: {error_type})"
        )
    return (
        f"Error processing your request with the {agent_label}. "
        f"Please try again or switch to a different agent mode. (Error: {error_type})"
    )


def _reconcile_saved_operation(operation: ConversationOperation) -> ChatResponse:
    """Return a stored completion or expose durable non-replayable operation state."""
    if operation.status == "completed" and operation.response is not None:
        return ChatResponse.model_validate(operation.response)
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": f"operation_{operation.status}",
            "operation_id": operation.id,
            "conversation_version": operation.conversation_version,
            "error_code": operation.error_code,
        },
    )


async def _record_saved_failure(
    conversation_service: Any,
    conversation_id: str,
    operation_id: str,
    terminal_status: str,
    error_code: str,
) -> None:
    """Best-effort terminal persistence; startup recovery resolves storage uncertainty."""
    try:
        await conversation_service.repository.record_operation_failure(
            conversation_id,
            operation_id,
            terminal_status,
            error_code,
        )
    except Exception:
        logger.error("Saved operation terminal status could not be persisted", exc_info=True)


async def _execute_saved_intelligent(
    *,
    request: ChatRequest,
    conversation_service: Any,
    intelligent_agent: Any,
    actual_provider: ModelProvider,
    request_id: str,
    trace_context: dict[str, str],
    progress_reporter: ToolProgressReporter | None = None,
) -> ChatResponse:
    """Reserve, execute, and atomically publish one saved intelligent turn."""
    if request.conversation_id is None or request.operation_id is None or request.expected_version is None:
        raise RuntimeError("Saved request identity was not validated")
    if request.display_message is None:
        raise RuntimeError("Saved display message was not validated")
    conversation_id = str(request.conversation_id)
    operation_id = str(request.operation_id)
    async with conversation_service.memory_manager.lifecycle_lock(conversation_id):
        reservation = await conversation_service.reserve_operation(
            conversation_id,
            operation_id,
            request.agent_mode,
            request.thread_action,
            request.message,
            request.display_message,
            request.expected_version,
            request.model_provider,
            request.enable_rag,
            request.n_results,
        )
        if not reservation.accepted:
            return _reconcile_saved_operation(reservation.operation)

        published: ConversationOperation | None = None

        async def publish(thread: Any, _action: ThreadAction, terminal_checkpoint_id: str, result: dict) -> None:
            nonlocal published
            answer, sources, web_sources = _finalize_intelligent_sources(result)
            response = _build_chat_response(
                request,
                actual_provider,
                request_id,
                answer=answer,
                sources=sources,
                web_sources=web_sources,
                error=None,
            )
            published = await conversation_service.repository.publish_operation(
                conversation_id,
                operation_id,
                thread.active_checkpoint_id,
                terminal_checkpoint_id,
                response.model_dump(mode="json"),
            )

        try:
            await intelligent_agent.ainvoke(
                request.message,
                message_history=[],
                trace_context=trace_context,
                thread_id=conversation_id,
                thread_action=request.thread_action,
                progress_reporter=progress_reporter,
                thread_publisher=publish,
            )
        except asyncio.CancelledError:
            await _record_saved_failure(
                conversation_service, conversation_id, operation_id, "interrupted", "request_cancelled"
            )
            raise
        except Exception as error:
            await _record_saved_failure(
                conversation_service, conversation_id, operation_id, "failed", type(error).__name__
            )
            raise
        if published is None:
            await _record_saved_failure(
                conversation_service, conversation_id, operation_id, "failed", "publication_missing"
            )
            raise RuntimeError("Saved intelligent execution completed without publication")
        return _reconcile_saved_operation(published)


async def _execute_saved_rag(
    *,
    request: ChatRequest,
    conversation_service: Any,
    rag_runtime: AsyncRAGRuntimeResources,
    model: BaseChatModel | None,
    actual_provider: ModelProvider,
    request_id: str,
    trace_context: dict[str, str],
) -> ChatResponse:
    """Reserve, execute with bounded committed history, and publish one RAG turn."""
    if request.conversation_id is None or request.operation_id is None or request.expected_version is None:
        raise RuntimeError("Saved request identity was not validated")
    if request.display_message is None:
        raise RuntimeError("Saved display message was not validated")
    conversation_id = str(request.conversation_id)
    operation_id = str(request.operation_id)
    async with conversation_service.memory_manager.lifecycle_lock(conversation_id):
        reservation = await conversation_service.reserve_operation(
            conversation_id,
            operation_id,
            request.agent_mode,
            request.thread_action,
            request.message,
            request.display_message,
            request.expected_version,
            request.model_provider,
            request.enable_rag,
            request.n_results,
        )
        if not reservation.accepted:
            return _reconcile_saved_operation(reservation.operation)
        try:
            history = await conversation_service.repository.completed_history(
                conversation_id,
                reservation.operation.turn_sequence,
                conversation_service.memory_manager.config.max_prior_turns,
            )
            relevance_redirect = _rag_only_relevance_redirect(request.message, rag_runtime.config)
            if relevance_redirect is not None:
                answer, sources, web_sources = relevance_redirect, [], []
            elif model is None:
                raise RuntimeError("LLM model not available")
            else:
                answer, sources, web_sources = await _ainvoke_rag_only(
                    prompt=request.message,
                    cfg=rag_runtime.config,
                    model=model,
                    retriever=rag_runtime.retriever,
                    reranker=rag_runtime.reranker,
                    message_history=history,
                    enable_rag=request.enable_rag,
                    n_results_override=request.n_results,
                    trace_context=trace_context,
                )
            response = _build_chat_response(
                request,
                actual_provider,
                request_id,
                answer=answer,
                sources=sources,
                web_sources=web_sources,
                error=None,
            )
            published = await conversation_service.repository.publish_rag_operation(
                conversation_id,
                operation_id,
                response.model_dump(mode="json"),
            )
            return _reconcile_saved_operation(published)
        except asyncio.CancelledError:
            await _record_saved_failure(
                conversation_service, conversation_id, operation_id, "interrupted", "request_cancelled"
            )
            raise
        except Exception as error:
            await _record_saved_failure(
                conversation_service, conversation_id, operation_id, "failed", type(error).__name__
            )
            raise


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("/", response_model=ChatResponse)
async def send_message(
    http_request: Request,
    request: ChatRequest,
    rag_runtime: AsyncRAGRuntimeResources = Depends(get_async_rag_runtime),
    memory_manager=Depends(get_conversation_memory_manager),
    conversation_service=Depends(get_optional_conversation_service),
) -> ChatResponse:
    """Send a chat message and get a response from the selected agent.

    The ``agent_mode`` field selects the execution path:

    * ``intelligent`` -- LangGraph ReAct agent with tool selection (2-3 LLM calls).
    * ``rag_only`` -- Traditional RAG pipeline, no agent.

    The optional ``model_provider`` field accepts only ``cloud``.
    """
    mode = request.agent_mode
    prompt = request.message
    state = http_request.app.state
    request_id = http_request.headers.get("X-Request-Id") or str(uuid.uuid4())
    session_id = http_request.headers.get("X-Session-Id")
    trace_context = get_trace_context(request_id=request_id, session_id=session_id, agent_mode=mode)

    model, intelligent_agent, actual_provider = _select_execution_resources(state)

    message_history = [{"role": m.role, "content": m.content} for m in request.message_history]
    supplied_thread_id = str(request.thread_id) if request.thread_id is not None else None
    is_saved = request.conversation_id is not None
    if is_saved and conversation_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Saved conversations unavailable")
    if not is_saved:
        await _reject_saved_thread(supplied_thread_id, conversation_service)
    active_thread_id = supplied_thread_id if memory_manager is not None and mode == "intelligent" else None

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
            if is_saved and mode == "intelligent":
                if intelligent_agent is None:
                    raise HTTPException(status_code=503, detail="Intelligent agent not available")
                saved_response = await _execute_saved_intelligent(
                    request=request,
                    conversation_service=conversation_service,
                    intelligent_agent=intelligent_agent,
                    actual_provider=actual_provider,
                    request_id=request_id,
                    trace_context=trace_context,
                )
                set_span_attributes(span, {"http_status_code": 200, "retrieval_enabled": request.enable_rag})
                return saved_response
            if is_saved:
                saved_response = await _execute_saved_rag(
                    request=request,
                    conversation_service=conversation_service,
                    rag_runtime=rag_runtime,
                    model=model,
                    actual_provider=actual_provider,
                    request_id=request_id,
                    trace_context=trace_context,
                )
                set_span_attributes(span, {"http_status_code": 200, "retrieval_enabled": request.enable_rag})
                return saved_response
            if mode == "intelligent":
                if intelligent_agent is None:
                    raise HTTPException(status_code=503, detail="Intelligent agent not available")
                if active_thread_id is None:
                    answer, sources, web_sources = await _ainvoke_intelligent_agent(
                        intelligent_agent,
                        prompt,
                        message_history,
                        trace_context=trace_context,
                    )
                else:
                    answer, sources, web_sources = await _ainvoke_intelligent_agent(
                        intelligent_agent,
                        prompt,
                        message_history,
                        trace_context=trace_context,
                        thread_id=active_thread_id,
                        thread_action=request.thread_action,
                    )
            else:  # rag_only (default fallback)
                relevance_redirect = _rag_only_relevance_redirect(prompt, rag_runtime.config)
                if relevance_redirect is not None:
                    answer = relevance_redirect
                elif model is None:
                    raise HTTPException(
                        status_code=503,
                        detail="LLM model not available. Check startup logs for loading errors.",
                    )
                else:
                    answer, sources, web_sources = await _ainvoke_rag_only(
                        prompt=prompt,
                        cfg=rag_runtime.config,
                        model=model,
                        retriever=rag_runtime.retriever,
                        reranker=rag_runtime.reranker,
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
        except ConversationConflict as conflict:
            current = (
                {
                    "id": conflict.session.id,
                    "agent_mode": conflict.session.agent_mode,
                    "title": conflict.session.title,
                    "version": conflict.session.version,
                    "state": conflict.session.state,
                    "created_at": conflict.session.created_at.isoformat(),
                    "updated_at": conflict.session.updated_at.isoformat(),
                }
                if conflict.session is not None
                else None
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": conflict.code, "current": current},
            ) from conflict
        except Exception as e:
            if is_saved:
                logger.error("Saved chat execution failed (%s): %s", mode, type(e).__name__)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"code": "saved_operation_failed"},
                ) from e
            agent_label = {"intelligent": "intelligent agent"}.get(mode, "RAG pipeline")
            logger.error("Error in chat (%s): %s", mode, type(e).__name__)
            error = _friendly_error_message(e, agent_label)
            answer = error
            set_span_attributes(
                span,
                {
                    "http_status_code": 500,
                    "error_class": type(e).__name__,
                },
            )

    return _build_chat_response(
        request,
        actual_provider,
        request_id,
        answer=answer,
        sources=sources,
        web_sources=web_sources,
        error=error,
    )


@router.post("/stream", response_class=StreamingResponse)
async def stream_message(
    http_request: Request,
    request: ChatRequest,
    memory_manager=Depends(get_conversation_memory_manager),
    conversation_service=Depends(get_optional_conversation_service),
) -> StreamingResponse:
    """Stream safe intelligent-agent progress followed by one finalized response."""
    state = http_request.app.state
    supplied_thread_id = str(request.thread_id) if request.thread_id is not None else None
    is_saved = request.conversation_id is not None
    if is_saved and conversation_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Saved conversations unavailable")
    if not is_saved:
        await _reject_saved_thread(supplied_thread_id, conversation_service)
    _model, intelligent_agent, actual_provider = _select_execution_resources(state)
    await _validate_stream_preconditions(request, state, memory_manager, intelligent_agent)

    request_id = http_request.headers.get("X-Request-Id") or str(uuid.uuid4())
    session_id = http_request.headers.get("X-Session-Id")
    trace_context = get_trace_context(
        request_id=request_id,
        session_id=session_id,
        agent_mode=request.agent_mode,
    )
    message_history = [{"role": message.role, "content": message.content} for message in request.message_history]
    active_thread_id = supplied_thread_id if memory_manager is not None else None

    return StreamingResponse(
        _stream_agent_events(
            request=request,
            intelligent_agent=intelligent_agent,
            actual_provider=actual_provider,
            request_id=request_id,
            trace_context=trace_context,
            message_history=message_history,
            active_thread_id=active_thread_id,
            conversation_service=conversation_service,
        ),
        media_type="text/event-stream",
        headers=_STREAM_CACHE_HEADERS,
    )


@router.get("/initial-message", response_model=InitialMessageResponse)
def get_initial() -> InitialMessageResponse:
    """Return the initial welcome message for new chat sessions."""
    return _DEFAULT_INITIAL_MESSAGE


@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: uuid.UUID,
    memory_manager=Depends(get_conversation_memory_manager),
    conversation_service=Depends(get_optional_conversation_service),
) -> Response:
    """Delete one conversation thread idempotently when memory is enabled."""
    await _reject_saved_thread(str(thread_id), conversation_service)
    if memory_manager is not None:
        try:
            await memory_manager.delete_thread(str(thread_id))
        except Exception as error:
            logger.error("Conversation thread deletion failed", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Conversation thread deletion failed",
            ) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
    StreamErrorEvent,
    ToolProgressStreamEvent,
