"""Saved JSON, SSE, operation reconciliation, and bounded RAG history tests."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from src.agents.conversation_service import ConversationService
from src.agents.memory import ConversationMemoryManager, SessionMemoryConfig
from src.database.migrations.create_chat_sessions import run_migration

CONVERSATION_ID = "123e4567-e89b-12d3-a456-426614174000"
SECOND_CONVERSATION_ID = "123e4567-e89b-12d3-a456-426614174001"
OPERATION_ID = "223e4567-e89b-12d3-a456-426614174000"
SECOND_OPERATION_ID = "223e4567-e89b-12d3-a456-426614174001"
THIRD_OPERATION_ID = "223e4567-e89b-12d3-a456-426614174002"


@pytest.fixture()
async def saved_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Yield a client and real saved-conversation resources with provider I/O mocked."""
    from src.api import main
    from src.api.routes import chat

    path = tmp_path / "saved-chat.db"
    path.touch()
    run_migration(path)
    manager = await ConversationMemoryManager.open(SessionMemoryConfig(enabled=True, db_path=str(path)))
    assert manager is not None
    service = await ConversationService.open(manager)
    config = OmegaConf.create({"streaming": {"enabled": True}})
    agent = MagicMock()
    model = MagicMock()
    main.app.state.config = config
    main.app.state.cloud_model = model
    main.app.state.intelligent_agent = agent
    main.app.state.conversation_memory_manager = manager
    main.app.state.conversation_service = service
    main.app.state.retriever = None
    main.app.state.reranker = None
    main.app.state.async_rag_runtime = SimpleNamespace(config=config, retriever=None, reranker=None)
    rag_invoke = AsyncMock(return_value=("RAG answer", [], []))
    monkeypatch.setattr(chat, "_ainvoke_rag_only", rag_invoke)
    try:
        yield TestClient(main.app), service, manager, agent, rag_invoke
    finally:
        main.app.state.conversation_service = None
        main.app.state.conversation_memory_manager = None
        await service.close()
        await manager.close()


def _saved_payload(
    *,
    conversation_id: str = CONVERSATION_ID,
    operation_id: str = OPERATION_ID,
    mode: str = "rag_only",
    message: str = "Which wine should I open?",
    version: int = 1,
) -> dict:
    """Build one valid saved chat request."""
    return {
        "message": message,
        "display_message": message,
        "agent_mode": mode,
        "conversation_id": conversation_id,
        "operation_id": operation_id,
        "expected_version": version,
    }


async def test_saved_rag_json_reconciles_without_replay_and_uses_committed_history(saved_chat) -> None:
    """RAG-only stores one response and derives later history from completed turns."""
    client, service, _manager, _agent, rag_invoke = saved_chat
    await service.create(CONVERSATION_ID, "rag_only", "Which wine should I open?")

    first = client.post("/api/chat/", json=_saved_payload())
    duplicate = client.post("/api/chat/", json=_saved_payload())
    assert first.status_code == duplicate.status_code == 200
    assert first.json() == duplicate.json()
    assert first.json()["conversation_version"] == 3
    assert first.json()["turn_sequence"] == 1
    assert rag_invoke.await_count == 1

    operation = client.get(f"/api/conversations/{CONVERSATION_ID}/operations/{OPERATION_ID}")
    assert operation.status_code == 200
    assert operation.json()["status"] == "completed"
    assert operation.json()["response"] == first.json()

    second = client.post(
        "/api/chat/",
        json=_saved_payload(
            operation_id=SECOND_OPERATION_ID,
            message="What about tomorrow?",
            version=3,
        ),
    )
    assert second.status_code == 200
    assert rag_invoke.await_count == 2
    assert rag_invoke.await_args.kwargs["message_history"] == [
        {"role": "human", "content": "Which wine should I open?"},
        {"role": "ai", "content": "RAG answer"},
    ]

    replaced = client.post(
        "/api/chat/",
        json={
            **_saved_payload(
                operation_id=THIRD_OPERATION_ID,
                message="What about tomorrow?",
                version=5,
            ),
            "thread_action": "replace_last",
        },
    )
    assert replaced.status_code == 200
    assert replaced.json()["turn_sequence"] == 2
    assert rag_invoke.await_args.kwargs["message_history"] == [
        {"role": "human", "content": "Which wine should I open?"},
        {"role": "ai", "content": "RAG answer"},
    ]


async def test_saved_intelligent_json_and_sse_share_stored_completion(saved_chat) -> None:
    """SSE reconciliation returns the exact committed JSON response without execution."""
    client, service, manager, agent, _rag_invoke = saved_chat
    await service.create(SECOND_CONVERSATION_ID, "intelligent", "Compare these wines")

    async def invoke(query: str, **kwargs) -> dict:
        thread = await manager._get_or_create_thread(SECOND_CONVERSATION_ID)
        result = {"final_answer": f"Answer: {query}", "messages": []}
        await kwargs["thread_publisher"](
            thread,
            kwargs["thread_action"],
            f"{thread.active_checkpoint_id}-terminal",
            result,
        )
        return result

    agent.ainvoke = AsyncMock(side_effect=invoke)
    payload = _saved_payload(
        conversation_id=SECOND_CONVERSATION_ID,
        mode="intelligent",
        message="Compare these wines",
    )
    json_response = client.post("/api/chat/", json=payload)
    stream_response = client.post("/api/chat/stream", json=payload)
    assert json_response.status_code == stream_response.status_code == 200
    assert agent.ainvoke.await_count == 1
    assert "event: agent_done" in stream_response.text
    event_data = next(
        line.removeprefix("data: ") for line in stream_response.text.splitlines() if line.startswith("data: ")
    )
    assert json.loads(event_data)["response"] == json_response.json()


async def test_saved_failure_is_terminal_and_same_operation_cannot_replay(saved_chat) -> None:
    """An ordinary execution failure is queryable and requires a new operation ID."""
    client, service, _manager, agent, _rag_invoke = saved_chat
    await service.create(SECOND_CONVERSATION_ID, "intelligent", "Fail once")
    agent.ainvoke = AsyncMock(side_effect=RuntimeError("provider failed"))
    payload = _saved_payload(
        conversation_id=SECOND_CONVERSATION_ID,
        mode="intelligent",
        message="Fail once",
    )
    failed = client.post("/api/chat/", json=payload)
    repeated = client.post("/api/chat/", json=payload)
    assert failed.status_code == 500
    assert repeated.status_code == 409
    assert repeated.json()["detail"]["code"] == "operation_failed"
    assert agent.ainvoke.await_count == 1
    operation = client.get(f"/api/conversations/{SECOND_CONVERSATION_ID}/operations/{OPERATION_ID}")
    assert operation.json()["status"] == "failed"
    assert operation.json()["conversation_version"] == 3


async def test_saved_cancellation_records_interrupted_without_publication(saved_chat) -> None:
    """Cancellation advances only terminal operation state and leaves the turn unpublished."""
    from src.api.routes.chat import _execute_saved_intelligent
    from src.api.schemas.chat import ChatRequest

    _client, service, _manager, agent, _rag_invoke = saved_chat
    await service.create(SECOND_CONVERSATION_ID, "intelligent", "Cancel")
    agent.ainvoke = AsyncMock(side_effect=asyncio.CancelledError)
    request = ChatRequest.model_validate(
        _saved_payload(
            conversation_id=SECOND_CONVERSATION_ID,
            mode="intelligent",
            message="Cancel",
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await _execute_saved_intelligent(
            request=request,
            conversation_service=service,
            intelligent_agent=agent,
            actual_provider="cloud",
            request_id="request",
            trace_context={},
        )
    operation = await service.get_operation(SECOND_CONVERSATION_ID, OPERATION_ID)
    turn = (await service.list_turns(SECOND_CONVERSATION_ID, 10, None)).items[0]
    assert operation.status == "interrupted"
    assert operation.conversation_version == 3
    assert turn.status == "interrupted"
    assert turn.answer is None


@pytest.mark.parametrize(
    "updates",
    [
        {"operation_id": None},
        {"display_message": None},
        {"message_history": [{"role": "human", "content": "client history"}]},
        {"thread_id": "123e4567-e89b-12d3-a456-426614174099"},
    ],
)
async def test_saved_request_contract_rejects_partial_or_ambiguous_input(saved_chat, updates: dict) -> None:
    """Saved identity, display text, history, and thread ownership are unambiguous."""
    client, _service, _manager, _agent, _rag_invoke = saved_chat
    payload = {**_saved_payload(), **updates}
    response = client.post("/api/chat/", json=payload)
    assert response.status_code == 422


async def test_intelligent_source_kinds_survive_regeneration_stream_and_reload(saved_chat) -> None:
    """One finalized source representation is published, reconciled, and restored."""
    from langchain_core.messages import HumanMessage, ToolMessage

    client, service, manager, agent, _rag_invoke = saved_chat
    await service.create(CONVERSATION_ID, "intelligent", "French regions")

    async def invoke(query: str, **kwargs) -> dict:
        thread = await manager._get_or_create_thread(CONVERSATION_ID)
        result = {
            "final_answer": "Bordeaux [1].\n\nSources:\n- Wine.epub\n- your tasting history\n- information you provided",
            "messages": [
                HumanMessage(content=query),
                ToolMessage(
                    content="[Source 1 - Wine.epub, Page -1]\nBordeaux is in France.",
                    name="search_wine_region_info",
                    tool_call_id="rag",
                ),
                ToolMessage(
                    content='{"items":[{"name":"Bordeaux"}],"total":1}',
                    name="get_tasting_history",
                    tool_call_id="history",
                ),
            ],
        }
        await kwargs["thread_publisher"](thread, kwargs["thread_action"], f"{thread.active_checkpoint_id}-next", result)
        return result

    agent.ainvoke = AsyncMock(side_effect=invoke)
    initial = client.post("/api/chat/", json=_saved_payload(mode="intelligent"))
    assert initial.status_code == 200
    expected = initial.json()
    assert expected["answer"] == "Bordeaux [1]."
    assert [source["kind"] for source in expected["sources"]] == ["document", "tasting_history", "user_provided"]
    assert expected["sources"][0]["page"] is None
    regenerated = client.post(
        "/api/chat/stream",
        json={
            **_saved_payload(mode="intelligent", operation_id=SECOND_OPERATION_ID, version=3),
            "thread_action": "replace_last",
        },
    )
    assert regenerated.status_code == 200
    terminal = json.loads(
        next(line[6:] for line in regenerated.text.splitlines() if line.startswith("data: ") and '"agent_done"' in line)
    )["response"]
    assert terminal["sources"] == expected["sources"]
    assert terminal["answer"] == expected["answer"]
    restored = (await service.list_turns(CONVERSATION_ID, 10, None)).items[0]
    assert restored.sources == expected["sources"]
    assert restored.answer == expected["answer"]
    operation = client.get(f"/api/conversations/{CONVERSATION_ID}/operations/{SECOND_OPERATION_ID}").json()
    assert operation["response"] == terminal
    assert agent.ainvoke.await_count == 2
