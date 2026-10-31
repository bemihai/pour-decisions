"""Deterministic tests for the async chat execution boundary."""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from src.api.schemas.chat import Source, WebSource


def test_chat_post_route_is_async_and_initial_message_route_remains_sync() -> None:
    """Only the production chat dispatcher should change invocation mode."""
    from src.api.routes import chat

    assert inspect.iscoroutinefunction(chat.send_message)
    assert not inspect.iscoroutinefunction(chat.get_initial)


@pytest.mark.asyncio
async def test_intelligent_async_helper_awaits_agent_and_preserves_sources() -> None:
    """The intelligent helper should await ainvoke and keep its response tuple."""
    from src.api.routes import chat

    agent = MagicMock()
    agent.ainvoke = AsyncMock(
        return_value={
            "final_answer": "Use the cited recommendation.\n\nSources:\n- https://example.test/review",
            "messages": [
                ToolMessage(
                    content=("[1] Example Wine Review\n" "A useful result.\n" "Source: https://example.test/review"),
                    name="search_wine_reviews",
                    tool_call_id="web-call",
                )
            ],
            "guardrail_events": [{"code": "internal-only"}],
        }
    )
    history = [{"role": "human", "content": "Earlier question."}]
    trace_context = {"request_id": "agent-async", "session_id": "session-async"}

    answer, sources, web_sources = await chat._ainvoke_intelligent_agent(
        agent,
        "Recommend a wine.",
        history,
        trace_context=trace_context,
    )

    assert answer == "Use the cited recommendation."
    assert sources == []
    assert web_sources == [WebSource(title="Example Wine Review", url="https://example.test/review")]
    agent.ainvoke.assert_awaited_once_with(
        "Recommend a wine.",
        message_history=history,
        trace_context=trace_context,
    )
    agent.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_rag_only_async_helper_awaits_service_and_preserves_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RAG-only helper should await the async service with unchanged inputs."""
    from src.api.routes import chat

    service_result = SimpleNamespace(
        answer="Async answer.",
        sources=[SimpleNamespace(name="wine_book", page=42, relevance=0.91)],
    )
    from src.retrieval import get_request_correction_budget

    seen_budgets = []

    async def _execute_async(**_kwargs: object) -> object:
        seen_budgets.append(get_request_correction_budget())
        return service_result

    execute_async = AsyncMock(side_effect=_execute_async)
    monkeypatch.setattr(chat, "execute_production_rag_async", execute_async)
    config = MagicMock()
    model = MagicMock()
    retriever = MagicMock()
    reranker = MagicMock()
    history = [{"role": "human", "content": "Earlier question."}]
    trace_context = {"request_id": "rag-thread", "session_id": "session-thread"}

    result = await chat._ainvoke_rag_only(
        prompt="What is Barolo?",
        cfg=config,
        model=model,
        retriever=retriever,
        reranker=reranker,
        message_history=history,
        enable_rag=True,
        n_results_override=7,
        trace_context=trace_context,
    )

    assert result == (
        "Async answer.",
        [Source(name="wine_book", page=42, relevance=0.91)],
        [],
    )
    execute_async.assert_awaited_once_with(
        prompt="What is Barolo?",
        config=config,
        model=model,
        retriever=retriever,
        reranker=reranker,
        message_history=history,
        enable_retrieval=True,
        n_results_override=7,
        generation_enabled=True,
        trace_context=trace_context,
    )
    assert seen_budgets[0] is not None
    assert get_request_correction_budget() is None


@pytest.mark.asyncio
async def test_rag_only_cancellation_propagates_to_async_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the route helper should cancel its direct service await."""
    import asyncio

    from src.api.routes import chat

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _wait_for_cancellation(**_kwargs: object) -> object:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(chat, "execute_production_rag_async", _wait_for_cancellation)
    task = asyncio.create_task(
        chat._ainvoke_rag_only(
            prompt="What is Barolo?",
            cfg=MagicMock(),
            model=MagicMock(),
            retriever=MagicMock(),
            reranker=None,
            message_history=[],
            enable_rag=True,
            n_results_override=None,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled.is_set()


def test_intelligent_sources_select_only_current_supporting_evidence() -> None:
    """Selections are resolved against evidence, never invented or copied from older turns."""
    from langchain_core.messages import HumanMessage

    from src.api.routes.chat import _finalize_intelligent_sources

    def tool(content: str, name: str = "search_wine_knowledge", status: str = "success") -> ToolMessage:
        return ToolMessage(content=content, name=name, tool_call_id=name, status=status)

    answer, sources, web = _finalize_intelligent_sources(
        {
            "final_answer": "French regions [3].\n\nSources:\n- Marnie Old - Wine, A Tasting Course.epub\n"
            "- old.pdf\n- invented.pdf\n- your cellar data\n- your tasting history\n"
            "- information you provided\n- https://example.test/used",
            "messages": [
                tool("[Source 1 - old.pdf, Page 3]\nEarlier evidence"),
                HumanMessage(content="Tell me about France. I like dry wines."),
                tool(
                    "[Source 1 - unused.pdf, Page 4]\nUnused\n\n---\n\n"
                    "[Source 3 - Marnie Old - Wine, A Tasting Course.epub, Page -1, Chunk 8]\nFrench regions"
                ),
                tool('[{"name":"Wine"}]', "get_cellar_wines"),
                tool('{"items": [], "total": 0}', "get_tasting_history"),
                tool(
                    "[1] Used review\nDetails\nSource: https://example.test/used\n"
                    "[2] Unused review\nDetails\nSource: https://example.test/unused",
                    "search_wine_reviews",
                ),
                tool("[Source 3 - invented.pdf]\nFailed", status="error"),
            ],
        }
    )
    assert answer == "French regions [1]."
    assert sources == [
        Source(name="Marnie Old - Wine, A Tasting Course"),
        Source(name="your cellar data", kind="cellar"),
        Source(name="information you provided", kind="user_provided"),
    ]
    assert web == [WebSource(title="Used review", url="https://example.test/used")]


def test_intelligent_sources_do_not_resolve_ambiguous_tool_local_numbers() -> None:
    """Two searches can both call a different document Source 1."""
    from src.api.routes.chat import _finalize_intelligent_sources

    messages = [
        ToolMessage(
            content=f"[Source 1 - {name}.pdf, Page 2]\nEvidence", name="search_wine_knowledge", tool_call_id=name
        )
        for name in ["A", "B"]
    ]
    answer, sources, _ = _finalize_intelligent_sources(
        {
            "final_answer": "Answer [1].\n\nSources:\n- A.pdf\n- B.pdf",
            "messages": messages,
        }
    )
    assert answer == "Answer."
    assert [source.name for source in sources] == ["A", "B"]


@pytest.mark.parametrize(
    "content", ["[]", "{}", '{"error":"failed"}', '{"items":[],"total":0}', "[cellar_tool_failed] Unavailable"]
)
def test_empty_or_failed_tools_cannot_support_a_personal_source(content: str) -> None:
    """A generated label cannot turn absent data into source evidence."""
    from src.api.routes.chat import _finalize_intelligent_sources

    answer, sources, web = _finalize_intelligent_sources(
        {
            "final_answer": "Cannot verify.\n\nSources:\n- your cellar data",
            "messages": [ToolMessage(content=content, name="get_cellar_wines", tool_call_id="cellar")],
        }
    )
    assert (answer, sources, web) == ("Cannot verify.", [], [])


def test_sources_without_selection_are_not_exposed_and_legacy_kind_defaults() -> None:
    """Retrieving a source does not establish that the final answer used it."""
    from src.api.routes.chat import _finalize_intelligent_sources

    assert Source.model_validate({"name": "Old book"}).kind == "document"
    result = _finalize_intelligent_sources(
        {
            "final_answer": "Hello.",
            "messages": [
                ToolMessage(content="[Source 1 - unused.pdf]\nUnused", name="search_wine_knowledge", tool_call_id="r")
            ],
        }
    )
    assert result == ("Hello.", [], [])


def test_source_pages_and_relevance_are_taken_from_tool_headers() -> None:
    """Metadata is evidence-derived; duplicate documents retain distinct valid pages."""
    from src.api.routes.chat import _finalize_intelligent_sources

    _, sources, _ = _finalize_intelligent_sources(
        {
            "final_answer": "Answer.\n\nSources:\n- Book.pdf",
            "messages": [
                ToolMessage(
                    content="[Source 1 - Book.pdf, Page 4, Chunk 2, Relevance: 0.85]\nFirst.\n"
                    "[Source 2 - Book.pdf, Page 8]\nSecond.",
                    name="search_wine_knowledge",
                    tool_call_id="rag",
                )
            ],
        }
    )
    assert [(source.page, source.relevance) for source in sources] == [(4, 0.85), (8, None)]


def test_source_examples_in_code_are_not_consumed_as_attribution() -> None:
    """A quoted Sources example remains readable answer content."""
    from src.api.routes.chat import _finalize_intelligent_sources

    content = "Example:\n```text\nSources: A.pdf\n```"
    assert _finalize_intelligent_sources({"final_answer": content, "messages": []}) == (content, [], [])


@pytest.mark.parametrize(
    ("tool_name", "content", "label"),
    [
        ("get_food_pairing_wines", '{"food":"fish","recommended_varietals":["Chardonnay"]}', "your cellar data"),
        ("get_wine_and_cheese_pairings", '{"cheese":"Brie","classic_pairings":["Chardonnay"]}', "your cellar data"),
        ("get_wine_recommendations_from_profile", '[{"name":"Wine","predicted_rating":null}]', "your tasting history"),
    ],
)
def test_general_or_declared_preference_results_do_not_imply_personal_history(
    tool_name: str, content: str, label: str
) -> None:
    """Successful tools must actually contain the selected personal evidence kind."""
    from src.api.routes.chat import _finalize_intelligent_sources

    _, sources, _ = _finalize_intelligent_sources(
        {
            "final_answer": f"Answer.\n\nSources:\n- {label}",
            "messages": [ToolMessage(content=content, name=tool_name, tool_call_id="tool")],
        }
    )
    assert sources == []


@pytest.mark.parametrize(
    ("other_tool", "other_content"),
    [
        ("search_wine_reviews", "[1] Review\nEvidence\nSource: https://example.test/review"),
        ("search_wine_region_info", "[Source 1]\nUnnamed evidence"),
    ],
)
def test_numeric_citations_do_not_confuse_documents_with_web_or_unnamed_evidence(
    other_tool: str, other_content: str
) -> None:
    """Local number collisions must not turn another tool's citation into a book citation."""
    from src.api.routes.chat import _finalize_intelligent_sources

    answer, sources, _ = _finalize_intelligent_sources(
        {
            "final_answer": "Answer [1].\n\nSources:\n- Book.pdf\n- https://example.test/review",
            "messages": [
                ToolMessage(
                    content="[Source 1 - Book.pdf]\nBook evidence", name="search_wine_knowledge", tool_call_id="book"
                ),
                ToolMessage(content=other_content, name=other_tool, tool_call_id="other"),
            ],
        }
    )
    assert answer == "Answer."
    assert sources == [Source(name="Book")]
