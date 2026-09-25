"""Offline tests for the single direct Ollama Cloud model loader."""

from typing import AsyncIterator
from unittest.mock import patch

import pytest
from langchain_ollama import ChatOllama
from ollama import AsyncClient, Client

from src.agents.llm import load_base_model


@pytest.fixture(autouse=True)
def synthetic_cloud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep loader tests independent of real credentials and dotenv files."""
    monkeypatch.setenv("OLLAMA_API_KEY", "synthetic-cloud-key")
    monkeypatch.setattr("src.agents.llm.load_env", lambda: None)


def test_loads_direct_cloud_model_with_both_clients() -> None:
    """One ChatOllama instance owns native sync and async SDK transports."""
    model = load_base_model("ollama", "gemma4:31b")

    assert isinstance(model, ChatOllama)
    assert isinstance(model._client, Client)
    assert isinstance(model._async_client, AsyncClient)
    assert model.base_url == "https://ollama.com"
    assert model.client_kwargs == {"timeout": 60}
    assert model.temperature == 1.0
    assert model.top_p == 0.95
    assert model.top_k == 64
    assert model.num_predict is None


@pytest.mark.asyncio
async def test_sync_and_async_calls_use_their_own_sdk_transports() -> None:
    """Both invocation modes work with injected responses and no network."""
    model = load_base_model("ollama", "gemma4:31b")
    response = {"model": "gemma4:31b", "message": {"role": "assistant", "content": "Barolo"}, "done": True}

    def sync_chat(_client: Client, **_kwargs: object) -> object:
        return iter([response])

    async def async_chat(_client: AsyncClient, **_kwargs: object) -> object:
        async def chunks() -> AsyncIterator[dict[str, object]]:
            yield response

        return chunks()

    with patch.object(Client, "chat", sync_chat), patch.object(AsyncClient, "chat", async_chat):
        sync_reply = model.invoke("Name a wine")
        async_reply = await model.ainvoke("Name a wine")

    assert sync_reply.content == async_reply.content == "Barolo"


def test_provider_failure_has_no_hidden_retry_or_fallback() -> None:
    """A failed SDK call is attempted once and propagates to the caller."""
    model = load_base_model("ollama", "gemma4:31b")
    with patch.object(Client, "chat", side_effect=RuntimeError("cloud unavailable")) as chat:
        with pytest.raises(RuntimeError, match="cloud unavailable"):
            model.invoke("Name a wine")

    chat.assert_called_once()


def test_judge_controls_and_timeout_are_forwarded() -> None:
    """Judge settings stay explicit and separate from application sampling."""
    with patch("src.agents.llm.ChatOllama") as constructor:
        load_base_model("ollama", "gemma4:31b", temperature=0.0, reasoning=False, num_predict=2048, timeout=120)

    options = constructor.call_args.kwargs
    assert options["model"] == "gemma4:31b"
    assert options["base_url"] == "https://ollama.com"
    assert options["client_kwargs"] == {"timeout": 120}
    assert options["temperature"] == 0.0
    assert options["reasoning"] is False
    assert options["num_predict"] == 2048


@pytest.mark.parametrize(
    ("provider", "options"),
    [
        ("google", {}),
        ("openai", {}),
        ("ollama", {"base_url": "http://localhost:11434"}),
        ("ollama", {"base_url": "https://other.example"}),
        ("ollama", {"client_kwargs": {"headers": {"Authorization": "Bearer synthetic-cloud-key"}}}),
    ],
)
def test_unsupported_settings_fail_before_client_construction(provider: str, options: dict[str, object]) -> None:
    """Unsupported providers and transport overrides never construct a model."""
    with patch("src.agents.llm.ChatOllama") as constructor:
        with pytest.raises(ValueError) as error:
            load_base_model(provider, "gemma4:31b", **options)

    constructor.assert_not_called()
    assert "synthetic-cloud-key" not in str(error.value)


def test_missing_key_fails_before_client_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credential failure is explicit and does not touch the transport."""
    monkeypatch.delenv("OLLAMA_API_KEY")

    with patch("src.agents.llm.ChatOllama") as constructor:
        with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
            load_base_model("ollama", "gemma4:31b")

    constructor.assert_not_called()
