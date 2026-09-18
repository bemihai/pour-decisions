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
            "final_answer": "Use the cited recommendation.",
            "messages": [
                ToolMessage(
                    content=(
                        "[1] Example Wine Review\n"
                        "A useful result.\n"
                        "Source: https://example.test/review"
                    ),
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
    assert web_sources == [
        WebSource(title="Example Wine Review", url="https://example.test/review")
    ]
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
    execute_async = AsyncMock(return_value=service_result)
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
