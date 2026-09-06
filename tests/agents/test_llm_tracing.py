"""Unit tests for trace-context propagation in LLM helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from src.agents import llm
from src.agents.prompt_registry import get_prompt_registry


def test_invoke_llm_forwards_trace_context_to_chain_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """invoke_llm should pass trace_context into model.invoke metadata config."""
    fake_model = MagicMock(spec=BaseChatModel)
    fake_model.invoke.return_value = SimpleNamespace(content="ok")

    trace_context = {"request_id": "req-llm", "agent_mode": "rag_only"}
    answer = llm.invoke_llm(
        question="What is Barolo?",
        context="Barolo context",
        model=fake_model,
        message_history=[],
        trace_context=trace_context,
    )

    assert answer == "ok"
    fake_model.invoke.assert_called_once()
    _, kwargs = fake_model.invoke.call_args
    config = kwargs.get("config") or (fake_model.invoke.call_args.args[1] if len(fake_model.invoke.call_args.args) > 1 else None)
    assert config is not None
    metadata = config.get("metadata", {})
    assert metadata.get("request_id") == "req-llm"
    assert metadata.get("pour_decisions.execution.mode") == "rag"
    assert (
        metadata.get("pour_decisions.model.generation.model_class")
        == "unittest.mock.MagicMock"
    )
    assert "pour_decisions.model.planning.model_class" not in metadata
    assert set(
        key for key in metadata if key.startswith("pour_decisions.prompt.")
    ) == {
        "pour_decisions.prompt.bundle_hash",
        "pour_decisions.prompt.rag_only_system.source_hash",
        "pour_decisions.prompt.rag_only_user.source_hash",
    }
    assert trace_context == {"request_id": "req-llm", "agent_mode": "rag_only"}


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


def test_invoke_llm_uses_role_content_message_history() -> None:
    """invoke_llm should translate role/content history into LangChain messages."""
    fake_model = MagicMock(spec=BaseChatModel)
    fake_model.invoke.return_value = SimpleNamespace(content="ok")

    llm.invoke_llm(
        question="What about decanting?",
        context="Barolo context",
        model=fake_model,
        message_history=[
            {"role": "human", "content": "Tell me about Barolo"},
            {"role": "ai", "content": "Barolo is a Nebbiolo wine."},
        ],
    )

    invoke_messages = fake_model.invoke.call_args.args[0]
    assert isinstance(invoke_messages[1], HumanMessage)
    assert invoke_messages[1].content == "Tell me about Barolo"
    assert isinstance(invoke_messages[2], AIMessage)
    assert invoke_messages[2].content == "Barolo is a Nebbiolo wine."


def test_invoke_llm_uses_registered_prompts_without_content_changes() -> None:
    """RAG invocation should preserve the effective checked-in prompt strings."""
    fake_model = MagicMock(spec=BaseChatModel)
    fake_model.invoke.return_value = SimpleNamespace(content="ok")
    registry = get_prompt_registry()
    context = "Structured context"
    question = "Which wine?"

    llm.invoke_llm(
        question=question,
        context=context,
        model=fake_model,
        message_history=[],
    )

    invoke_messages = fake_model.invoke.call_args.args[0]
    expected_system = registry.get("rag_only_system").source.strip()
    expected_user = (
        registry.get("rag_only_user")
        .source.strip()
        .replace("{context}", context)
        .replace("{question}", question)
    )
    assert invoke_messages == [
        SystemMessage(content=expected_system),
        HumanMessage(content=expected_user),
    ]


def test_invoke_llm_preserves_literal_braces_in_context_and_question() -> None:
    """Explicit token replacement should not interpret braces in request data."""
    fake_model = MagicMock(spec=BaseChatModel)
    fake_model.invoke.return_value = SimpleNamespace(content="ok")
    context = 'Metadata: {"region": "Barolo"}'
    question = "Compare {2019} and {2020}."

    llm.invoke_llm(
        question=question,
        context=context,
        model=fake_model,
        message_history=[],
    )

    user_message = fake_model.invoke.call_args.args[0][-1]
    assert context in user_message.content
    assert question in user_message.content


def test_invoke_llm_propagates_prompt_registry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry failure should stop invocation instead of using fallback text."""
    fake_model = MagicMock(spec=BaseChatModel)
    monkeypatch.setattr(
        llm,
        "get_prompt_registry",
        MagicMock(side_effect=FileNotFoundError("missing versions.yml")),
    )

    with pytest.raises(FileNotFoundError, match="missing versions.yml"):
        llm.invoke_llm("Question", "Context", fake_model, [])

    fake_model.invoke.assert_not_called()
