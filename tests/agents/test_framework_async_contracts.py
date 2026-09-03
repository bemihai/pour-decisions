"""Public LangChain and LangGraph API contracts required by M6A."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import InjectedToolCallId, tool
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


def _build_tool_graph(tool_node: ToolNode) -> Runnable:
    """Compile a ToolNode with the runtime injection used by production graphs."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
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


@pytest.mark.asyncio
async def test_tool_node_async_handler_can_repeat_one_request() -> None:
    """The public async handler must support a bounded retry of the same call."""
    attempts: list[str] = []

    @tool
    async def succeeds_on_retry(value: str) -> str:
        """Fail once before returning the supplied value."""
        attempts.append(value)
        if len(attempts) == 1:
            raise RuntimeError("synthetic transient failure")
        return f"retried:{value}"

    async def retry_once(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Repeat the same framework request after one synthetic failure."""
        try:
            return await handler(request)
        except RuntimeError:
            return await handler(request)

    graph = _build_tool_graph(
        ToolNode(
            [succeeds_on_retry],
            handle_tool_errors=False,
            awrap_tool_call=retry_once,
        )
    )
    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": succeeds_on_retry.name,
                            "args": {"value": "wine"},
                            "id": "retry-call",
                        }
                    ],
                )
            ]
        }
    )

    assert attempts == ["wine", "wine"]
    message = result["messages"][-1]
    assert message.content == "retried:wine"
    assert message.name == succeeds_on_retry.name
    assert message.tool_call_id == "retry-call"


@pytest.mark.asyncio
async def test_tool_node_invalid_call_stays_framework_owned() -> None:
    """Unknown calls should pass through the wrapper to LangGraph validation."""
    wrapper_calls: list[str] = []

    @tool
    async def known_tool(value: str) -> str:
        """Return a known value."""
        return value

    async def delegate(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Record and delegate one request without adding fallback metadata."""
        wrapper_calls.append(request.tool_call["name"])
        return await handler(request)

    graph = _build_tool_graph(ToolNode([known_tool], awrap_tool_call=delegate))
    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "unknown_tool", "args": {}, "id": "invalid-call"}
                    ],
                )
            ]
        }
    )

    message = result["messages"][-1]
    assert wrapper_calls == ["unknown_tool"]
    assert message.name == "unknown_tool"
    assert message.tool_call_id == "invalid-call"
    assert message.status == "error"
    assert "unknown_tool" in str(message.content)
    assert known_tool.name in str(message.content)


@pytest.mark.asyncio
async def test_tool_node_async_passthrough_propagates_unhandled_failure() -> None:
    """Disabling framework error handling should preserve raised tool failures."""

    @tool
    async def failing_tool() -> str:
        """Raise one unhandled synthetic failure."""
        raise RuntimeError("synthetic framework failure")

    async def delegate(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Delegate without handling the tool exception."""
        return await handler(request)

    graph = _build_tool_graph(
        ToolNode(
            [failing_tool],
            handle_tool_errors=False,
            awrap_tool_call=delegate,
        )
    )

    with pytest.raises(RuntimeError, match="synthetic framework failure"):
        await graph.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": failing_tool.name, "args": {}, "id": "failure-call"}
                        ],
                    )
                ]
            }
        )


@pytest.mark.asyncio
async def test_tool_node_mixed_batch_preserves_output_and_command_order() -> None:
    """Mixed async outputs should retain input order and matching call IDs."""

    @tool
    async def successful_tool(value: str) -> str:
        """Return a successful result."""
        return f"success:{value}"

    @tool
    async def failing_tool() -> str:
        """Raise a failure handled by the wrapper."""
        raise RuntimeError("synthetic mixed-batch failure")

    @tool
    async def command_tool(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Return a state command with the required matching tool message."""
        return Command(
            update={
                "messages": [
                    ToolMessage(content="command:ok", tool_call_id=tool_call_id)
                ]
            }
        )

    async def handle_failure(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Convert only the synthetic failure into a stable test result."""
        try:
            return await handler(request)
        except RuntimeError:
            return ToolMessage(
                content="safe:error",
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )

    graph = _build_tool_graph(
        ToolNode(
            [successful_tool, failing_tool, command_tool],
            handle_tool_errors=False,
            awrap_tool_call=handle_failure,
        )
    )
    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": successful_tool.name,
                            "args": {"value": "one"},
                            "id": "success-call",
                        },
                        {"name": failing_tool.name, "args": {}, "id": "failure-call"},
                        {"name": command_tool.name, "args": {}, "id": "command-call"},
                    ],
                )
            ]
        }
    )

    success_message, failure_message, command_message = result["messages"][-3:]
    assert success_message.content == "success:one"
    assert success_message.tool_call_id == "success-call"
    assert failure_message.content == "safe:error"
    assert failure_message.tool_call_id == "failure-call"
    assert command_message.content == "command:ok"
    assert command_message.tool_call_id == "command-call"
