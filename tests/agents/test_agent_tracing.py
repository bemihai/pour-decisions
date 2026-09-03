"""Unit tests for agent trace-context propagation into LangGraph invoke config."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator, cast

import pytest
from langchain_core.messages import AIMessage

from src.agents.guardrails import (
    CALL_BUDGET_EVENT_CODE,
    TOOL_EXECUTION_REPORT_CONFIG_KEY,
    CallBudgetConfig,
    SensitiveOutputSanitizer,
    ToolExecutionEvent,
    ToolExecutionEventCode,
    ToolExecutionReport,
)
from src.agents.intelligent.agent import WineAgent


@dataclass
class _GraphInvokeRecorder:
    """Capture LangGraph invoke payload and config for assertions."""

    captured_payload: dict | None = None
    captured_config: dict | None = None

    def invoke(self, payload: dict, config: dict | None = None) -> dict:
        """Record invocation args and return a minimal valid agent response.

        Args:
            payload: State payload sent by the agent.
            config: Optional runnable config object.

        Returns:
            Minimal response dictionary expected by agent wrappers.
        """
        self.captured_payload = payload
        self.captured_config = config
        return {
            "messages": [AIMessage(content="ok")],
            "query_type": "knowledge",
            "tool_results": {},
        }

    async def ainvoke(self, payload: dict, config: dict | None = None) -> dict:
        """Record async invocation args and return a minimal agent response."""
        self.captured_payload = payload
        self.captured_config = config
        return {
            "messages": [AIMessage(content="ok")],
            "query_type": "knowledge",
            "tool_results": {},
        }


@dataclass
class _GraphStreamRecorder:
    """Capture LangGraph stream payload and config for assertions."""

    captured_payload: dict | None = None
    captured_config: dict | None = None

    def stream(
        self,
        payload: dict,
        config: dict | None = None,
        stream_mode: str | None = None,
    ) -> Iterator[dict]:
        """Record streaming arguments and yield one state value."""
        self.captured_payload = payload
        self.captured_config = config
        yield payload


def test_wine_agent_invoke_passes_trace_context_metadata() -> None:
    """WineAgent.invoke should pass trace metadata via RunnableConfig."""
    recorder = _GraphInvokeRecorder()

    agent = cast(WineAgent, object.__new__(WineAgent))
    agent.verbose = False
    agent.agent = recorder
    agent.call_budget = CallBudgetConfig(max_graph_steps_per_query=17)
    agent.output_sanitizer = SensitiveOutputSanitizer(environment={})

    result = agent.invoke(
        "What wines do I have?",
        message_history=[{"role": "human", "content": "show my cellar"}],
        trace_context={"request_id": "req-123", "agent_mode": "intelligent"},
    )

    assert result["final_answer"] == "ok"
    assert recorder.captured_config is not None
    assert recorder.captured_config.get("metadata", {}).get("request_id") == "req-123"
    assert recorder.captured_config.get("recursion_limit") == 17
    assert "configurable" not in recorder.captured_config
    assert recorder.captured_payload is not None
    assert recorder.captured_payload["llm_call_count"] == 0
    assert recorder.captured_payload["tool_call_history"] == []
    assert recorder.captured_payload["guardrail_events"] == []


@pytest.mark.asyncio
async def test_wine_agent_ainvoke_passes_trace_context_metadata() -> None:
    """WineAgent.ainvoke should share payload and RunnableConfig construction."""
    recorder = _GraphInvokeRecorder()

    agent = cast(WineAgent, object.__new__(WineAgent))
    agent.verbose = False
    agent.agent = recorder
    agent.call_budget = CallBudgetConfig(max_graph_steps_per_query=17)
    agent.output_sanitizer = SensitiveOutputSanitizer(environment={})

    result = await agent.ainvoke(
        "What wines do I have?",
        message_history=[{"role": "human", "content": "show my cellar"}],
        trace_context={"request_id": "req-async-123", "agent_mode": "intelligent"},
    )

    assert result["final_answer"] == "ok"
    assert recorder.captured_config is not None
    assert recorder.captured_config.get("metadata", {}).get("request_id") == "req-async-123"
    assert recorder.captured_config.get("recursion_limit") == 17
    report = recorder.captured_config.get("configurable", {}).get(
        TOOL_EXECUTION_REPORT_CONFIG_KEY
    )
    assert isinstance(report, ToolExecutionReport)
    assert recorder.captured_payload is not None
    assert recorder.captured_payload["llm_call_count"] == 0
    assert recorder.captured_payload["tool_call_history"] == []
    assert recorder.captured_payload["guardrail_events"] == []


@pytest.mark.asyncio
async def test_wine_agent_ainvoke_merges_request_report_after_state_events() -> None:
    """Async finalization should merge detached report events in stable order."""
    recorder = _GraphInvokeRecorder()

    async def record_event(payload: dict, config: dict | None = None) -> dict:
        assert config is not None
        report = config["configurable"][TOOL_EXECUTION_REPORT_CONFIG_KEY]
        report.append(
            ToolExecutionEvent(
                code=ToolExecutionEventCode.TERMINAL_FAILURE,
                tool_name="get_cellar_wines",
            )
        )
        return {
            "messages": [AIMessage(content="ok")],
            "guardrail_events": [{"code": CALL_BUDGET_EVENT_CODE}],
        }

    recorder.ainvoke = record_event  # type: ignore[method-assign]
    agent = cast(WineAgent, object.__new__(WineAgent))
    agent.verbose = False
    agent.agent = recorder
    agent.call_budget = CallBudgetConfig(max_graph_steps_per_query=17)
    agent.output_sanitizer = SensitiveOutputSanitizer(environment={})

    result = await agent.ainvoke("What wines do I have?")

    assert result["guardrail_events"] == [
        {"code": CALL_BUDGET_EVENT_CODE},
        {"code": "tool_terminal_failure", "tool_name": "get_cellar_wines"},
    ]


@pytest.mark.asyncio
async def test_concurrent_ainvoke_requests_keep_reports_isolated() -> None:
    """Concurrent requests should neither reuse reports nor exchange events."""
    reports: list[ToolExecutionReport] = []
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def invoke_graph(payload: dict, config: dict | None = None) -> dict:
        assert config is not None
        report = config["configurable"][TOOL_EXECUTION_REPORT_CONFIG_KEY]
        reports.append(report)
        query = payload["messages"][-1].content
        tool_name = "first_tool" if query == "first" else "second_tool"
        report.append(
            ToolExecutionEvent(
                code=ToolExecutionEventCode.TERMINAL_FAILURE,
                tool_name=tool_name,
            )
        )
        if len(reports) == 2:
            both_started.set()
        await release.wait()
        return {"messages": [AIMessage(content=tool_name)], "guardrail_events": []}

    agent = cast(WineAgent, object.__new__(WineAgent))
    agent.verbose = False
    agent.agent = SimpleNamespace(ainvoke=invoke_graph)
    agent.call_budget = CallBudgetConfig(max_graph_steps_per_query=17)
    agent.output_sanitizer = SensitiveOutputSanitizer(environment={})

    first = asyncio.create_task(agent.ainvoke("first"))
    second = asyncio.create_task(agent.ainvoke("second"))
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert reports[0] is not reports[1]
    assert first_result["guardrail_events"] == [
        {"code": "tool_terminal_failure", "tool_name": "first_tool"}
    ]
    assert second_result["guardrail_events"] == [
        {"code": "tool_terminal_failure", "tool_name": "second_tool"}
    ]


def test_wine_agent_stream_passes_limit_and_initializes_state() -> None:
    """WineAgent.stream should pass the graph limit and complete initial state."""
    recorder = _GraphStreamRecorder()

    agent = cast(WineAgent, object.__new__(WineAgent))
    agent.agent = recorder
    agent.call_budget = CallBudgetConfig(max_graph_steps_per_query=19)

    chunks = list(agent.stream("What is tannin?"))

    assert len(chunks) == 1
    assert recorder.captured_config is not None
    assert recorder.captured_config.get("recursion_limit") == 19
    assert recorder.captured_payload == {
        "messages": [("user", "What is tannin?")],
        "llm_call_count": 0,
        "tool_call_history": [],
        "guardrail_events": [],
    }


def test_wine_agent_invoke_emits_guardrail_summary_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invoke should attach bounded guardrail outcomes to the active request span."""
    recorder = _GraphInvokeRecorder()
    captured_attributes: list[dict[str, str | int | bool]] = []
    monkeypatch.setattr(
        "src.agents.intelligent.agent.set_current_span_attributes",
        captured_attributes.append,
    )

    agent = cast(WineAgent, object.__new__(WineAgent))
    agent.verbose = False
    agent.agent = recorder
    agent.call_budget = CallBudgetConfig(max_graph_steps_per_query=23)
    agent.output_sanitizer = SensitiveOutputSanitizer(environment={})
    recorder.invoke = lambda _payload, config=None: {
        "messages": [AIMessage(content="Safe answer.")],
        "llm_call_count": 3,
        "guardrail_events": [{"code": CALL_BUDGET_EVENT_CODE, "attempted_call": 4}],
    }

    result = agent.invoke("What is tannin?")

    assert result["final_answer"] == "Safe answer."
    assert captured_attributes == [
        {
            "guardrail.call_budget.triggered": True,
            "guardrail.llm_calls": 3,
            "guardrail.graph_limit": 23,
            "guardrail.loop.triggered": False,
            "guardrail.relevance.triggered": False,
            "guardrail.tool_error.count": 0,
            "guardrail.output_redaction.count": 0,
        }
    ]
    assert "attempted_call" not in captured_attributes[0]
