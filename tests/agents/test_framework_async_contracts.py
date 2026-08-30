"""Public LangChain and LangGraph API contracts required by M6A."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


class ValueState(TypedDict):
    """Minimal state used to prove paired runnable dispatch."""

    value: int
    execution_path: str


def _increment_sync(state: ValueState) -> dict[str, int | str]:
    """Return the synchronous result for the paired runnable probe."""
    return {"value": state["value"] + 1, "execution_path": "sync"}


async def _increment_async(state: ValueState) -> dict[str, int | str]:
    """Return the asynchronous result for the paired runnable probe."""
    await asyncio.sleep(0)
    return {"value": state["value"] + 1, "execution_path": "async"}


def _build_paired_runnable_graph() -> Runnable[ValueState, ValueState]:
    """Compile one graph with paired sync and async public callables."""
    workflow = StateGraph(ValueState)
    workflow.add_node("increment", RunnableLambda(_increment_sync, afunc=_increment_async))
    workflow.add_edge(START, "increment")
    workflow.add_edge("increment", END)
    return workflow.compile()


@pytest.mark.asyncio
async def test_compiled_graph_dispatches_paired_runnable_by_invocation_mode() -> None:
    """One compiled graph must support both invoke and ainvoke."""
    graph = _build_paired_runnable_graph()

    sync_result = graph.invoke({"value": 1, "execution_path": "unset"})
    async_result = await graph.ainvoke({"value": 1, "execution_path": "unset"})

    assert sync_result == {"value": 2, "execution_path": "sync"}
    assert async_result == {"value": 2, "execution_path": "async"}


@pytest.mark.asyncio
async def test_tool_node_uses_public_sync_and_async_wrappers() -> None:
    """ToolNode must dispatch sync and coroutine tools through the matching hook."""
    sync_wrapper_calls: list[str] = []
    async_wrapper_calls: list[str] = []
    sync_tool_threads: list[int] = []
    async_tool_threads: list[int] = []

    @tool
    def sync_echo(value: str) -> str:
        """Return a value from a synchronous tool."""
        sync_tool_threads.append(threading.get_ident())
        return f"sync:{value}"

    @tool
    async def async_echo(value: str) -> str:
        """Return a value from an asynchronous tool."""
        async_tool_threads.append(threading.get_ident())
        await asyncio.sleep(0)
        return f"async:{value}"

    def wrap_tool_call(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Record and delegate a synchronous tool call."""
        sync_wrapper_calls.append(request.tool_call["name"])
        return handler(request)

    async def awrap_tool_call(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Record and delegate an asynchronous tool call."""
        async_wrapper_calls.append(request.tool_call["name"])
        return await handler(request)

    workflow = StateGraph(MessagesState)
    workflow.add_node(
        "tools",
        ToolNode(
            [sync_echo, async_echo],
            wrap_tool_call=wrap_tool_call,
            awrap_tool_call=awrap_tool_call,
        ),
    )
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    graph = workflow.compile()

    sync_result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "sync_echo", "args": {"value": "one"}, "id": "sync-1"}],
                )
            ]
        }
    )
    assert sync_wrapper_calls == ["sync_echo"]
    assert sync_result["messages"][-1].content == "sync:one"

    event_loop_thread = threading.get_ident()
    async_result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "sync_echo", "args": {"value": "two"}, "id": "sync-2"},
                        {"name": "async_echo", "args": {"value": "three"}, "id": "async-1"},
                    ],
                )
            ]
        }
    )

    assert async_wrapper_calls == ["sync_echo", "async_echo"]
    assert [message.content for message in async_result["messages"][-2:]] == ["sync:two", "async:three"]
    assert sync_tool_threads[-1] != event_loop_thread
    assert async_tool_threads == [event_loop_thread]
