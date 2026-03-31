"""Chat API endpoints.

Consolidates agent invocation and RAG-only query logic from
``src/ui/pages/chatbot.py`` into stateless REST endpoints.
Each request carries its own message history; no server-side session.

Note: all route handlers are synchronous (``def``). FastAPI runs them in a
thread-pool executor so the event loop remains unblocked. Migrating to async
I/O would require async database drivers and is tracked as a future improvement.
"""
import re

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.language_models import BaseChatModel

from src.api.dependencies import (
    get_intelligent_agent,
    get_keyword_agent,
    get_model,
    get_optional_model,
    get_reranker,
    get_retriever,
)
from src.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    InitialMessageResponse,
    Source,
    WebSource,
)
from src.utils import get_config, logger

router = APIRouter(prefix="/api/chat", tags=["chat"])


_WEB_SEARCH_TOOLS = {"search_web_for_wine", "search_wine_price", "search_wine_reviews"}
_SOURCE_RE = re.compile(r"Source:\s*(https?://\S+)")


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


def _extract_web_sources_from_tool_results(tool_results: dict) -> list[WebSource]:
    """Extract web sources from keyword agent ``tool_results`` dict.

    The keyword agent stores web search output as a plain text block
    under the ``"web_search"`` key rather than as LangGraph messages.

    Args:
        tool_results: The ``tool_results`` dict from a keyword agent invocation.

    Returns:
        Deduplicated list of ``WebSource`` instances.
    """
    web_text = tool_results.get("web_search", "")
    if not web_text:
        return []

    seen: set[str] = set()
    sources: list[WebSource] = []
    lines = web_text.splitlines()

    for i, line in enumerate(lines):
        m = _SOURCE_RE.search(line)
        if not m:
            continue
        url = m.group(1)
        if url in seen:
            continue
        seen.add(url)
        title = url
        if i >= 2:
            title_line = lines[i - 2].strip()
            if title_line:
                title = re.sub(r"^\[\d+] ", "", title_line)
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

        raw_source = metadata.get("source", metadata.get("filename", "Unknown"))
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


def _invoke_intelligent_agent(
    agent, prompt: str, message_history: list[dict]
) -> tuple[str, list[Source], list[WebSource]]:
    """Run the intelligent (LangGraph ReAct) agent.

    Args:
        agent: Pre-loaded ``WineAgent`` instance.
        prompt: User question.
        message_history: Prior conversation turns as list of role/content dicts.

    Returns:
        Tuple of (answer, rag_sources, web_sources).
    """
    result = agent.invoke(prompt, message_history=message_history)
    answer = result.get("final_answer", "")
    web_sources = _extract_web_sources_from_messages(result.get("messages", []))
    return answer, [], web_sources


def _invoke_keyword_agent(
    agent, prompt: str, message_history: list[dict]
) -> tuple[str, list[Source], list[WebSource]]:
    """Run the keyword-routing agent.

    Args:
        agent: Pre-loaded ``KeywordWineAgent`` instance.
        prompt: User question.
        message_history: Prior conversation turns as list of role/content dicts.

    Returns:
        Tuple of (answer, rag_sources, web_sources).
    """
    result = agent.invoke(prompt, message_history=message_history)
    answer = result.get("final_answer", "")
    web_sources = _extract_web_sources_from_tool_results(result.get("tool_results", {}))
    return answer, [], web_sources


def _invoke_rag_only(
    prompt: str,
    model: BaseChatModel,
    retriever,
    reranker,
    message_history: list[dict],
    enable_rag: bool,
    n_results_override: int | None,
) -> tuple[str, list[Source], list[WebSource]]:
    """Run the traditional RAG pipeline (no agent).

    Consolidates the ``else`` branch (~120 lines) of ``chatbot.py main()``.

    Args:
        prompt: User question.
        model: Pre-loaded LLM.
        retriever: Pre-loaded retriever (hybrid or vector-only), or None.
        reranker: Pre-loaded reranker, or None.
        message_history: Conversation history as list of dicts.
        enable_rag: Whether to perform retrieval.
        n_results_override: Optional override for number of retrieved chunks.

    Returns:
        Tuple of (answer, rag_sources, empty web_sources).
    """
    from src.agents.llm import process_user_prompt
    from src.retrieval import (
        analyze_query,
        boost_by_metadata_match,
        build_context_from_chunks,
        build_semantic_context,
        compress_context,
    )

    cfg = get_config()
    context = ""
    sources: list[Source] = []

    if enable_rag and retriever is not None:
        try:
            n_results = n_results_override or cfg.chroma.retrieval.n_results
            retrieve_count = n_results * 2 if reranker else n_results

            # Analyze query for metadata-based boosting
            query_analysis = analyze_query(prompt)

            # Retrieve documents
            retrieved_docs = retriever.retrieve(prompt, n_results=retrieve_count)

            # Metadata boosting
            enable_metadata_boost = getattr(cfg.chroma.retrieval, "enable_metadata_boost", True)
            if enable_metadata_boost and query_analysis.has_filters and retrieved_docs:
                boost_factor = getattr(cfg.chroma.retrieval, "metadata_boost_factor", 0.1)
                retrieved_docs = boost_by_metadata_match(
                    retrieved_docs, query_analysis, boost_factor=boost_factor
                )
                logger.debug(f"Applied metadata boosting for: {query_analysis.get_boost_terms()}")

            # Reranking
            if reranker and retrieved_docs:
                rerank_top_k = getattr(cfg.chroma.retrieval, "rerank_top_k", n_results)
                retrieved_docs = reranker.rerank(prompt, retrieved_docs, top_k=rerank_top_k)
                logger.debug(f"Reranked to top {rerank_top_k} documents")

            # Small-to-big context expansion
            enable_small_to_big = getattr(cfg.chroma.chunking, "enable_small_to_big", False)
            if enable_small_to_big and retrieved_docs:
                from src.chroma.hierarchical_chunks import expand_to_parent_context

                retrieved_docs = expand_to_parent_context(retrieved_docs)
                logger.debug("Expanded to parent context (small-to-big)")

            # Build context with optional deduplication
            if cfg.chroma.retrieval.use_deduplication:
                context = build_semantic_context(
                    retrieved_docs,
                    similarity_threshold=cfg.chroma.retrieval.deduplication_threshold,
                    include_metadata=True,
                    embedding_model=cfg.chroma.settings.embedder,
                )
            else:
                context = build_context_from_chunks(
                    retrieved_docs,
                    include_metadata=True,
                    include_similarity=False,
                    max_chunks=None,
                )

            # Context compression
            enable_compression = getattr(cfg.chroma.retrieval, "enable_compression", False)
            if enable_compression and context:
                max_chars = getattr(cfg.chroma.retrieval, "compression_max_chars", 8000)
                context = compress_context(context, max_chars=max_chars)

            # Format sources
            if retrieved_docs:
                sources = _format_sources(retrieved_docs)

        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")

    # Generate answer
    answer = process_user_prompt(model, prompt, context, message_history)

    # Filter to only cited sources
    if sources:
        sources = _filter_cited_sources(answer, sources)

    return answer, sources, []


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
def send_message(
    request: ChatRequest,
    model: BaseChatModel | None = Depends(get_optional_model),
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
    intelligent_agent=Depends(get_intelligent_agent),
    keyword_agent=Depends(get_keyword_agent),
) -> ChatResponse:
    """Send a chat message and get a response from the selected agent.

    The ``agent_mode`` field in the request selects the execution path:

    * ``intelligent`` -- LangGraph ReAct agent with tool selection (2-3 LLM calls).
    * ``keyword`` -- Pattern-matching router with 1 LLM call.
    * ``rag_only`` -- Traditional RAG pipeline, no agent.
    """
    mode = request.agent_mode
    prompt = request.message

    # History in standard role/content format for agents
    agent_history = [{"role": m.role, "content": m.content} for m in request.message_history]
    # History in legacy question/answer format expected by invoke_llm (RAG-only path)
    rag_history = [
        {"role": m.role, "question" if m.role == "human" else "answer": m.content}
        for m in request.message_history
    ]

    answer = ""
    sources: list[Source] = []
    web_sources: list[WebSource] = []
    error: str | None = None

    try:
        if mode == "intelligent":
            if intelligent_agent is None:
                raise HTTPException(status_code=503, detail="Intelligent agent not available")
            answer, sources, web_sources = _invoke_intelligent_agent(
                intelligent_agent, prompt, agent_history
            )

        elif mode == "keyword":
            if keyword_agent is None:
                raise HTTPException(status_code=503, detail="Keyword agent not available")
            answer, sources, web_sources = _invoke_keyword_agent(
                keyword_agent, prompt, agent_history
            )

        else:  # rag_only (default fallback)
            if model is None:
                raise HTTPException(status_code=503, detail="LLM model not available. Check startup logs for loading errors.")
            answer, sources, web_sources = _invoke_rag_only(
                prompt=prompt,
                model=model,
                retriever=retriever,
                reranker=reranker,
                message_history=rag_history,
                enable_rag=request.enable_rag,
                n_results_override=request.n_results,
            )

    except HTTPException:
        raise
    except Exception as e:
        agent_label = {"intelligent": "intelligent agent", "keyword": "keyword agent"}.get(
            mode, "RAG pipeline"
        )
        logger.error(f"Error in chat ({mode}): {e}", exc_info=True)
        error = _friendly_error_message(e, agent_label)
        answer = error

    return ChatResponse(
        answer=answer,
        sources=sources,
        web_sources=web_sources,
        agent_mode=mode,
        error=error,
    )


@router.get("/initial-message", response_model=InitialMessageResponse)
def get_initial() -> InitialMessageResponse:
    """Return the initial welcome message for new chat sessions."""
    from src.utils import get_initial_message

    messages = get_initial_message()
    msg = messages[0] if messages else {"role": "ai", "answer": "Welcome! Ask me anything about wine."}
    return InitialMessageResponse(
        role=msg.get("role", "ai"),
        content=msg.get("answer", "Welcome! Ask me anything about wine."),
    )

