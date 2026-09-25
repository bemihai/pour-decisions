"""Parity tests for native async RAG LLM helpers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
import pytest

from src.agents import llm


@pytest.mark.asyncio
async def test_ainvoke_llm_matches_sync_messages_output_and_provenance() -> None:
    """Sync and async calls should receive identical input and trace metadata."""
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = SimpleNamespace(content="same answer")
    model.ainvoke.return_value = SimpleNamespace(content="same answer")
    history = [
        {"role": "human", "content": "Earlier question"},
        {"role": "ai", "content": "Earlier answer"},
    ]
    trace_context = {"request_id": "m06b-phase1", "agent_mode": "rag_only"}

    sync_answer = llm.invoke_llm(
        "What is Barolo?",
        "Barolo context",
        model,
        history,
        trace_context=trace_context,
    )
    async_answer = await llm.ainvoke_llm(
        "What is Barolo?",
        "Barolo context",
        model,
        history,
        trace_context=trace_context,
    )

    assert sync_answer == async_answer == "same answer"
    assert model.invoke.call_args.args[0] == model.ainvoke.call_args.args[0]
    assert model.invoke.call_args.kwargs["config"] == model.ainvoke.call_args.kwargs["config"]
    assert model.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_ainvoke_llm_uses_native_async_call_without_sync_fallback() -> None:
    """The async helper should never enter the model's synchronous method."""
    model = MagicMock(spec=BaseChatModel)
    model.invoke.side_effect = AssertionError("sync invoke must not run")
    model.ainvoke.return_value = SimpleNamespace(content="async answer")

    answer = await llm.ainvoke_llm("Question", "Context", model, [])

    assert answer == "async answer"
    model.invoke.assert_not_called()
    model.ainvoke.assert_awaited_once()
    assert model.ainvoke.call_args.kwargs == {}


@pytest.mark.asyncio
async def test_ainvoke_llm_preserves_callbacks_and_trace_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async invocation should pass the same callbacks and M5 metadata as sync."""
    model = MagicMock(spec=BaseChatModel)
    model.invoke.return_value = SimpleNamespace(content="sync")
    model.ainvoke.return_value = SimpleNamespace(content="async")
    callback = MagicMock()
    monkeypatch.setattr(llm, "get_tracing_callbacks", lambda: [callback])

    llm.invoke_llm("Question", "Context", model, [], trace_context={"request_id": "same"})
    await llm.ainvoke_llm("Question", "Context", model, [], trace_context={"request_id": "same"})

    sync_config = model.invoke.call_args.kwargs["config"]
    async_config = model.ainvoke.call_args.kwargs["config"]
    assert async_config == sync_config
    assert async_config["callbacks"] == [callback]
    assert async_config["metadata"]["request_id"] == "same"
    assert async_config["metadata"]["pour_decisions.execution.mode"] == "rag"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_output", "expected"),
    [
        (SimpleNamespace(content=["structured", "content"]), "['structured', 'content']"),
        ({"content": "dictionary content"}, "dictionary content"),
        (42, "42"),
    ],
)
async def test_ainvoke_llm_matches_sync_output_coercion(model_output: object, expected: str) -> None:
    """Async provider outputs should use the shared synchronous coercion rules."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke.return_value = model_output

    assert await llm.ainvoke_llm("Question", "Context", model, []) == expected


@pytest.mark.asyncio
async def test_ainvoke_llm_wraps_model_failure_with_original_cause() -> None:
    """Async model failures should retain the synchronous typed error boundary."""
    model = MagicMock(spec=BaseChatModel)
    failure = RuntimeError("provider unavailable")
    model.ainvoke.side_effect = failure

    with pytest.raises(llm.ModelInternalError, match="Model internal error") as exc_info:
        await llm.ainvoke_llm("Question", "Context", model, [])

    assert exc_info.value.__cause__ is failure


@pytest.mark.asyncio
async def test_async_provider_failure_keeps_synthetic_secret_out_of_public_error_and_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider exception may retain its cause without publishing its header."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke.side_effect = RuntimeError("Authorization: Bearer synthetic-cloud-secret")

    with caplog.at_level("ERROR"):
        answer = await llm.process_user_prompt_async(model, "Question", "Context", [])

    assert answer == llm.ModelInternalError().default_message
    assert "synthetic-cloud-secret" not in caplog.text


@pytest.mark.asyncio
async def test_ainvoke_llm_propagates_cancellation() -> None:
    """Native cancellation should remain cancellation rather than a model error."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await llm.ainvoke_llm("Question", "Context", model, [])


@pytest.mark.asyncio
async def test_process_user_prompt_async_preserves_fail_soft_answer() -> None:
    """The async convenience wrapper should return the established safe answer."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke.side_effect = RuntimeError("provider unavailable")

    answer = await llm.process_user_prompt_async(model, "Question", "Context", [])

    assert answer == llm.ModelInternalError().default_message


@pytest.mark.asyncio
async def test_process_user_prompt_async_forwards_trace_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async wrapper should forward trace metadata without mutation."""
    async_invoke = MagicMock()
    async_invoke.side_effect = None

    async def _invoke(
        question: str,
        context: str,
        model: BaseChatModel,
        message_history: list,
        trace_context: dict[str, str] | None = None,
    ) -> str:
        async_invoke(question, context, model, message_history, trace_context=trace_context)
        return "answer"

    monkeypatch.setattr(llm, "ainvoke_llm", _invoke)
    model = MagicMock(spec=BaseChatModel)
    trace_context = {"request_id": "forwarded", "agent_mode": "rag_only"}

    answer = await llm.process_user_prompt_async(
        model,
        "Question",
        "Context",
        [],
        trace_context=trace_context,
    )

    assert answer == "answer"
    async_invoke.assert_called_once_with(
        "Question",
        "Context",
        model,
        [],
        trace_context=trace_context,
    )


@pytest.mark.asyncio
async def test_ainvoke_llm_propagates_prompt_registry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt registry failures remain outside the model error translation boundary."""
    model = MagicMock(spec=BaseChatModel)
    monkeypatch.setattr(
        llm,
        "get_prompt_registry",
        MagicMock(side_effect=FileNotFoundError("missing versions.yml")),
    )

    with pytest.raises(FileNotFoundError, match="missing versions.yml"):
        await llm.ainvoke_llm("Question", "Context", model, [])

    model.ainvoke.assert_not_awaited()
