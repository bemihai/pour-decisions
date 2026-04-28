"""Unit tests for trace-context propagation in LLM helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
import pytest

from src.agents import llm


class _FakeChain:
    """Simple runnable chain that records invoke arguments."""

    def __init__(self) -> None:
        self.captured_payload: dict | None = None
        self.captured_config: dict | None = None

    def invoke(self, payload: dict, config: dict | None = None) -> SimpleNamespace:
        """Record invoke arguments and return content payload.

        Args:
            payload: Invocation payload.
            config: Optional runnable configuration.

        Returns:
            Object exposing ``content`` to mirror LangChain response shape.
        """
        self.captured_payload = payload
        self.captured_config = config
        return SimpleNamespace(content="ok")


class _FakePrompt:
    """Prompt object that composes into a predefined fake chain."""

    def __init__(self, chain: _FakeChain) -> None:
        self.chain = chain

    def __or__(self, model: object) -> _FakeChain:
        """Return the fake chain for prompt | model composition."""
        _ = model
        return self.chain


def test_invoke_llm_forwards_trace_context_to_chain_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """invoke_llm should pass trace_context into LangChain invoke metadata."""
    fake_chain = _FakeChain()

    monkeypatch.setattr(llm.ChatPromptTemplate, "from_messages", lambda _messages: _FakePrompt(fake_chain))

    answer = llm.invoke_llm(
        question="What is Barolo?",
        context="Barolo context",
        model=MagicMock(spec=BaseChatModel),
        message_history=[],
        trace_context={"request_id": "req-llm", "agent_mode": "rag_only"},
    )

    assert answer == "ok"
    assert fake_chain.captured_config is not None
    assert fake_chain.captured_config.get("metadata", {}).get("request_id") == "req-llm"


def test_process_user_prompt_forwards_trace_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """process_user_prompt should propagate trace_context to invoke_llm."""
    captured_trace_context: dict[str, str] = {}

    def _fake_invoke_llm(
        question: str,
        context: str,
        model,
        message_history: list,
        trace_context: dict[str, str] | None = None,
    ) -> str:
        _ = question
        _ = context
        _ = model
        _ = message_history
        if trace_context:
            captured_trace_context.update(trace_context)
        return "response"

    monkeypatch.setattr(llm, "invoke_llm", _fake_invoke_llm)

    answer = llm.process_user_prompt(
        model=MagicMock(spec=BaseChatModel),
        prompt="Question",
        context="Context",
        message_history=[],
        trace_context={"request_id": "req-forward", "agent_mode": "rag_only"},
    )

    assert answer == "response"
    assert captured_trace_context == {"request_id": "req-forward", "agent_mode": "rag_only"}


