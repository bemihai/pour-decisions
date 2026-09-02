"""Deterministic tests for the async chat execution boundary."""

import inspect
import threading
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
async def test_rag_only_async_bridge_preserves_arguments_and_leaves_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The temporary RAG bridge should delegate unchanged work to a thread."""
    from src.api.routes import chat

    event_loop_thread = threading.get_ident()
    worker_threads: list[int] = []
    captured_arguments: dict[str, object] = {}
    expected_result = (
        "Threaded answer.",
        [Source(name="wine_book", page=42, relevance=0.91)],
        [],
    )

    def _capture_rag_only(**kwargs: object) -> tuple[str, list[Source], list[WebSource]]:
        """Capture the synchronous helper invocation from its worker thread."""
        worker_threads.append(threading.get_ident())
        captured_arguments.update(kwargs)
        return expected_result

    monkeypatch.setattr(chat, "_invoke_rag_only", _capture_rag_only)
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

    assert result == expected_result
    assert worker_threads and worker_threads[0] != event_loop_thread
    assert captured_arguments == {
        "prompt": "What is Barolo?",
        "cfg": config,
        "model": model,
        "retriever": retriever,
        "reranker": reranker,
        "message_history": history,
        "enable_rag": True,
        "n_results_override": 7,
        "trace_context": trace_context,
    }
