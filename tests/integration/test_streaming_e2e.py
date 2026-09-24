"""Socket-backed incremental delivery and cleanup tests for M07 streaming."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from pathlib import Path
import socket
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
import httpx
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from omegaconf import OmegaConf
import pytest
import uvicorn

from src.agents.intelligent.agent import WineAgent
from src.agents.prompt_registry import RenderedPrompt
from src.agents.tools.registry import (
    CostClass,
    LatencyClass,
    ToolCategory,
    ToolDefinition,
    ToolMetadata,
    ToolRegistry,
    ToolSelectionSnapshot,
    ToolTier,
)
from src.api.routes.chat import router as chat_router


def _tool_definition(tool_instance: BaseTool) -> ToolDefinition:
    """Attach reviewed metadata to one deterministic integration tool."""
    return ToolDefinition(
        tool=tool_instance,
        metadata=ToolMetadata(
            name=tool_instance.name,
            category=ToolCategory.CELLAR,
            tier=ToolTier.CORE,
            cost_class=CostClass.FREE,
            latency_class=LatencyClass.FAST,
            idempotent=True,
            capability="Exercise incremental SSE delivery.",
        ),
    )


def _registry(
    monkeypatch: pytest.MonkeyPatch,
    definitions: tuple[ToolDefinition, ...],
) -> ToolRegistry:
    """Return an isolated immutable tool snapshot and deterministic prompt."""
    monkeypatch.setattr(
        "src.agents.intelligent.agent.render_intelligent_agent_system_prompt",
        lambda _snapshot: RenderedPrompt(
            name="intelligent_agent_system",
            content="Streaming integration test prompt.",
            source_hash="sha256:stream-source",
            rendered_hash="sha256:stream-rendered",
            label="",
        ),
    )
    registry = MagicMock(spec=ToolRegistry)
    registry.select.return_value = ToolSelectionSnapshot(definitions=definitions, readiness=())
    return registry


def _app(agent: object, memory_manager: object | None = None) -> FastAPI:
    """Build a socket-served API with no startup or external resources."""
    app = FastAPI()
    app.include_router(chat_router)
    model = MagicMock()
    app.state.config = OmegaConf.create({"streaming": {"enabled": True}})
    app.state.local_model = None
    app.state.cloud_model = model
    app.state.model = model
    app.state.local_intelligent_agent = None
    app.state.cloud_intelligent_agent = agent
    app.state.intelligent_agent = agent
    app.state.conversation_memory_manager = memory_manager
    return app


@asynccontextmanager
async def _serve(app: FastAPI) -> AsyncIterator[str]:
    """Serve one app on an ephemeral real TCP socket and stop it cleanly."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = listener.getsockname()[1]
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        lifespan="off",
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(
        server.serve(sockets=[listener]),
        name="streaming-test-server",
    )
    try:
        for _attempt in range(200):
            if server.started:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.005)
        else:
            raise RuntimeError("Socket-backed test server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()


async def _read_event(lines: AsyncIterator[str]) -> tuple[str, dict]:
    """Read one complete non-comment SSE frame from a real response stream."""
    event_name: str | None = None
    data: str | None = None
    while True:
        line = await anext(lines)
        if line.startswith(":"):
            continue
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data = line.removeprefix("data: ")
        elif line == "" and event_name is not None and data is not None:
            return event_name, json.loads(data)


async def _wait_for_no_stream_tasks() -> None:
    """Wait until every request-owned streaming task has terminated."""
    for _attempt in range(200):
        remaining = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and task.get_name().startswith("chat-stream-")
        ]
        if not remaining:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"Leaked request-owned streaming tasks: {remaining!r}")


@pytest.mark.asyncio
async def test_socket_delivers_progress_while_real_agent_tool_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A useful progress frame must cross TCP before the controlled tool completes."""
    tool_entered = asyncio.Event()
    release_tool = asyncio.Event()

    @tool
    async def held_cellar_tool(value: str) -> str:
        """Hold execution until the socket client observes progress."""
        tool_entered.set()
        await release_tool.wait()
        return f"private-result-{value}"

    definition = _tool_definition(held_cellar_tool)
    planned = AIMessage(
        content="",
        tool_calls=[
            {
                "name": held_cellar_tool.name,
                "args": {"value": "private-input"},
                "id": "private-call-id",
            }
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(
        side_effect=[planned, AIMessage(content="Finalized answer")]
    )
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    agent = WineAgent(
        llm=model,
        tool_registry=_registry(monkeypatch, (definition,)),
    )

    async with _serve(_app(agent)) as base_url:
        async with httpx.AsyncClient(timeout=5) as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/chat/stream",
                json={"message": "Use the held tool"},
            ) as response:
                lines = response.aiter_lines()
                started_name, started = await asyncio.wait_for(_read_event(lines), timeout=2)

                assert response.status_code == 200
                assert started_name == "tool_progress"
                assert started["status"] == "started"
                assert tool_entered.is_set()
                assert not release_tool.is_set()
                assert "private" not in json.dumps(started)

                release_tool.set()
                received = [(started_name, started)]
                while received[-1][0] not in {"agent_done", "stream_error"}:
                    received.append(await asyncio.wait_for(_read_event(lines), timeout=2))
                with pytest.raises(StopAsyncIteration):
                    await asyncio.wait_for(anext(lines), timeout=2)

    assert [name for name, _payload in received] == [
        "tool_progress",
        "tool_progress",
        "agent_done",
    ]
    assert received[1][1]["status"] == "completed"
    assert received[-1][1]["response"]["answer"] == "Finalized answer"
    assert bound_model.ainvoke.await_count == 2
    assert sum(name in {"agent_done", "stream_error"} for name, _payload in received) == 1
    assert "private-input" not in json.dumps(received)
    assert "private-result" not in json.dumps(received)
    assert "private-call-id" not in json.dumps(received)
    await _wait_for_no_stream_tasks()


@pytest.mark.asyncio
async def test_socket_disconnect_cancels_real_agent_and_leaves_no_owned_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Disconnect must cancel the tool and release the thread lock for recovery."""
    from src.agents.memory import ConversationMemoryManager, SessionMemoryConfig

    tool_entered = asyncio.Event()
    tool_cancelled = asyncio.Event()
    tool_calls = 0

    @tool
    async def cancellable_cellar_tool(value: str) -> str:
        """Wait until transport disconnect cancels the real agent task."""
        nonlocal tool_calls
        tool_calls += 1
        if tool_calls == 1:
            tool_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                tool_cancelled.set()
                raise
        return f"recovered-{value}"

    definition = _tool_definition(cancellable_cellar_tool)
    planned = AIMessage(
        content="",
        tool_calls=[
            {
                "name": cancellable_cellar_tool.name,
                "args": {"value": "private-input"},
                "id": "private-call-id",
            }
        ],
    )
    recovery_planned = AIMessage(
        content="",
        tool_calls=[
            {
                "name": cancellable_cellar_tool.name,
                "args": {"value": "second-call"},
                "id": "recovery-call-id",
            }
        ],
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(
        side_effect=[planned, recovery_planned, AIMessage(content="Recovered after disconnect")]
    )
    model = MagicMock()
    model.bind_tools.return_value = bound_model
    manager = await ConversationMemoryManager.open(
        SessionMemoryConfig(
            enabled=True,
            db_path=str(tmp_path / "stream-disconnect-memory.db"),
        )
    )
    assert manager is not None
    agent = WineAgent(
        llm=model,
        tool_registry=_registry(monkeypatch, (definition,)),
        memory_manager=manager,
    )
    thread_id = "123e4567-e89b-12d3-a456-426614174099"
    try:
        async with _serve(_app(agent, manager)) as base_url:
            async with httpx.AsyncClient(timeout=5) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/api/chat/stream",
                    json={"message": "Wait for disconnect", "thread_id": thread_id},
                ) as response:
                    name, payload = await asyncio.wait_for(
                        _read_event(response.aiter_lines()),
                        timeout=2,
                    )
                    assert name == "tool_progress"
                    assert payload["status"] == "started"
                    assert tool_entered.is_set()

            await asyncio.wait_for(tool_cancelled.wait(), timeout=2)
            await _wait_for_no_stream_tasks()
            cancelled_thread = await manager.get_thread(thread_id)
            assert cancelled_thread is not None
            assert cancelled_thread.completed_turns == 0

            async with httpx.AsyncClient(timeout=5) as client:
                recovery = await client.post(
                    f"{base_url}/api/chat/stream",
                    json={"message": "Recover", "thread_id": thread_id},
                )
            assert "Recovered after disconnect" in recovery.text
            recovered_thread = await manager.get_thread(thread_id)
            assert recovered_thread is not None
            assert recovered_thread.completed_turns == 1
            assert tool_calls == 2
            await _wait_for_no_stream_tasks()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_concurrent_socket_streams_keep_request_local_identity_and_terminal() -> None:
    """Concurrent clients must own independent progress channels and terminal responses."""
    entered = {"first": asyncio.Event(), "second": asyncio.Event()}
    release = asyncio.Event()

    class ConcurrentAgent:
        """Minimal agent double for independent request-owned transport state."""

        async def ainvoke(self, prompt: str, **kwargs: object) -> dict:
            reporter = kwargs["progress_reporter"]
            invocation_id = reporter.start("query_cellar")
            entered[prompt].set()
            await release.wait()
            from src.agents.guardrails import ToolProgressStatus

            reporter.finish(invocation_id, "query_cellar", ToolProgressStatus.COMPLETED)
            return {"final_answer": f"answer-{prompt}", "messages": []}

    async def collect(client: httpx.AsyncClient, base_url: str, prompt: str) -> list[tuple[str, dict]]:
        async with client.stream(
            "POST",
            f"{base_url}/api/chat/stream",
            json={"message": prompt},
        ) as response:
            lines = response.aiter_lines()
            events: list[tuple[str, dict]] = []
            while not events or events[-1][0] not in {"agent_done", "stream_error"}:
                events.append(await _read_event(lines))
            return events

    async with _serve(_app(ConcurrentAgent())) as base_url:
        async with httpx.AsyncClient(timeout=5) as client:
            first = asyncio.create_task(collect(client, base_url, "first"))
            second = asyncio.create_task(collect(client, base_url, "second"))
            await asyncio.wait_for(
                asyncio.gather(entered["first"].wait(), entered["second"].wait()),
                timeout=2,
            )
            release.set()
            first_events, second_events = await asyncio.gather(first, second)

        await _wait_for_no_stream_tasks()

    for prompt, events in (("first", first_events), ("second", second_events)):
        assert events[0][1]["invocation_id"] == 1
        assert events[0][1]["status"] == "started"
        assert events[-1][0] == "agent_done"
        assert events[-1][1]["response"]["answer"] == f"answer-{prompt}"
        assert sum(name in {"agent_done", "stream_error"} for name, _payload in events) == 1
