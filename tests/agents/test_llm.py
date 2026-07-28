"""Unit tests for src/agents/llm.py.

All LLM constructors are mocked so no network calls or Ollama server are needed.
Tests cover:
- load_base_model: correct class returned per provider, correct kwargs forwarded,
  ValueError on unknown provider.
- load_model_with_fallback: primary success, primary failure + fallback success,
  both fail -> RuntimeError, no fallback configured -> re-raises original error.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from src.agents.llm import load_base_model, load_model_with_fallback


# ---------------------------------------------------------------------------
# load_base_model -- ollama
# ---------------------------------------------------------------------------

class TestLoadBaseModelOllama:
    """load_base_model with provider='ollama'."""

    def test_returns_chat_ollama_instance(self, mocker):
        """Returns the ChatOllama instance created by the constructor."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        result = load_base_model("ollama", "gemma4:e2b")
        assert result is mock_cls.return_value

    def test_uses_correct_model_name(self, mocker):
        """Passes the model name through to ChatOllama."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("ollama", "gemma4:e2b")
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "gemma4:e2b"

    def test_default_base_url(self, mocker):
        """Uses localhost:11434 when base_url is not supplied."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("ollama", "gemma4:e2b")
        _, kwargs = mock_cls.call_args
        assert kwargs["base_url"] == "http://localhost:11434"

    def test_custom_base_url(self, mocker):
        """Accepts a custom base_url via kwargs."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("ollama", "gemma4:e2b", base_url="http://remote:11434")
        _, kwargs = mock_cls.call_args
        assert kwargs["base_url"] == "http://remote:11434"

    def test_google_recommended_sampling_params(self, mocker):
        """Applies temperature=1.0, top_p=0.95, top_k=64 as recommended by Google for Gemma 4."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("ollama", "gemma4:e2b")
        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 1.0
        assert kwargs["top_p"] == 0.95
        assert kwargs["top_k"] == 64

    def test_explicit_temperature_overrides_family_default(self, mocker):
        """An evaluator can request deterministic sampling independently."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")

        load_base_model("ollama", "gemma4:cloud", temperature=0.0)

        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 0.0

    def test_non_gemma4_uses_standard_sampling_params(self, mocker):
        """Applies only the standard temperature for non-Gemma-4 Ollama models."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("ollama", "gemma3:4b")
        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 0.7
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs

    def test_num_predict_not_set(self, mocker):
        """num_predict must never be set -- it breaks Gemma 4's reasoning pass."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("ollama", "gemma4:e2b")
        _, kwargs = mock_cls.call_args
        assert "num_predict" not in kwargs

    def test_provider_case_insensitive(self, mocker):
        """Provider matching is case-insensitive."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("Ollama", "gemma4:e2b")
        mock_cls.assert_called_once()

    def test_extra_kwargs_forwarded(self, mocker):
        """Extra kwargs (e.g. timeout) are passed through to ChatOllama."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")
        load_base_model("ollama", "gemma4:e2b", timeout=120)
        _, kwargs = mock_cls.call_args
        assert kwargs["timeout"] == 120

    def test_judge_control_kwargs_forwarded(self, mocker):
        """Reasoning and output limits are forwarded to the Ollama judge."""
        mock_cls = mocker.patch("src.agents.llm.ChatOllama")

        load_base_model(
            "ollama",
            "gpt-oss:20b-cloud",
            reasoning=False,
            num_predict=2048,
        )

        _, kwargs = mock_cls.call_args
        assert kwargs["reasoning"] is False
        assert kwargs["num_predict"] == 2048


# ---------------------------------------------------------------------------
# load_base_model -- google
# ---------------------------------------------------------------------------

class TestLoadBaseModelGoogle:
    """load_base_model with provider='google'."""

    def test_returns_chat_google_instance(self, mocker):
        """Returns the ChatGoogleGenerativeAI instance."""
        mock_cls = mocker.patch("src.agents.llm.ChatGoogleGenerativeAI")
        result = load_base_model("google", "gemini-2.5-flash")
        assert result is mock_cls.return_value

    def test_uses_correct_model_name(self, mocker):
        """Passes the model name through to ChatGoogleGenerativeAI."""
        mock_cls = mocker.patch("src.agents.llm.ChatGoogleGenerativeAI")
        load_base_model("google", "gemini-2.5-flash")
        _, kwargs = mock_cls.call_args
        assert kwargs["model"] == "gemini-2.5-flash"

    def test_temperature_zero(self, mocker):
        """Google model is initialised with temperature=0.0 (deterministic)."""
        mock_cls = mocker.patch("src.agents.llm.ChatGoogleGenerativeAI")
        load_base_model("google", "gemini-2.5-flash")
        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 0.0

    def test_provider_case_insensitive(self, mocker):
        """Provider matching is case-insensitive."""
        mock_cls = mocker.patch("src.agents.llm.ChatGoogleGenerativeAI")
        load_base_model("Google", "gemini-2.5-flash")
        mock_cls.assert_called_once()

    def test_extra_kwargs_forwarded(self, mocker):
        """Extra kwargs are passed through to ChatGoogleGenerativeAI."""
        mock_cls = mocker.patch("src.agents.llm.ChatGoogleGenerativeAI")
        load_base_model("google", "gemini-2.5-flash", convert_system_message_to_human=True)
        _, kwargs = mock_cls.call_args
        assert kwargs["convert_system_message_to_human"] is True


# ---------------------------------------------------------------------------
# load_base_model -- unknown provider
# ---------------------------------------------------------------------------

class TestLoadBaseModelUnknown:
    """load_base_model raises ValueError for unsupported providers."""

    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported model provider: openai"):
            load_base_model("openai", "gpt-4o")

    def test_error_message_includes_provider(self):
        with pytest.raises(ValueError, match="bedrock"):
            load_base_model("bedrock", "some-model")


# ---------------------------------------------------------------------------
# load_model_with_fallback
# ---------------------------------------------------------------------------

class TestLoadModelWithFallback:
    """load_model_with_fallback loading and fallback behaviour."""

    def test_returns_primary_on_success(self, mocker):
        """Returns the primary model when it loads successfully."""
        mock_primary = MagicMock()
        mocker.patch("src.agents.llm.load_base_model", return_value=mock_primary)
        result = load_model_with_fallback("ollama", "gemma4:e2b", "google", "gemini-2.5-flash")
        assert result is mock_primary

    def test_primary_called_with_correct_args(self, mocker):
        """load_base_model is called with the primary provider and name."""
        mock_load = mocker.patch("src.agents.llm.load_base_model", return_value=MagicMock())
        load_model_with_fallback("ollama", "gemma4:e2b", "google", "gemini-2.5-flash")
        mock_load.assert_called_once_with("ollama", "gemma4:e2b")

    def test_falls_back_when_primary_fails(self, mocker):
        """Returns the fallback model when the primary raises."""
        mock_fallback = MagicMock()
        mock_load = mocker.patch(
            "src.agents.llm.load_base_model",
            side_effect=[ConnectionRefusedError("Ollama offline"), mock_fallback],
        )
        result = load_model_with_fallback("ollama", "gemma4:e2b", "google", "gemini-2.5-flash")
        assert result is mock_fallback
        assert mock_load.call_count == 2
        mock_load.assert_any_call("google", "gemini-2.5-flash")

    def test_raises_runtime_error_when_both_fail(self, mocker):
        """Raises RuntimeError when both primary and fallback fail."""
        mocker.patch(
            "src.agents.llm.load_base_model",
            side_effect=[
                ConnectionRefusedError("Ollama offline"),
                Exception("Invalid API key"),
            ],
        )
        with pytest.raises(RuntimeError, match="Both primary.*and fallback.*failed"):
            load_model_with_fallback("ollama", "gemma4:e2b", "google", "gemini-2.5-flash")

    def test_raises_original_error_when_no_fallback(self, mocker):
        """Re-raises the primary error when no fallback provider is given."""
        original = ConnectionRefusedError("Ollama offline")
        mocker.patch("src.agents.llm.load_base_model", side_effect=original)
        with pytest.raises(ConnectionRefusedError, match="Ollama offline"):
            load_model_with_fallback("ollama", "gemma4:e2b")

    def test_no_fallback_none_values(self, mocker):
        """Explicit None fallback args also re-raise the primary error."""
        original = ConnectionRefusedError("Ollama offline")
        mocker.patch("src.agents.llm.load_base_model", side_effect=original)
        with pytest.raises(ConnectionRefusedError):
            load_model_with_fallback("ollama", "gemma4:e2b", None, None)

    def test_runtime_error_chains_original(self, mocker):
        """RuntimeError.__cause__ is the fallback exception for full traceability."""
        fallback_exc = Exception("Invalid API key")
        mocker.patch(
            "src.agents.llm.load_base_model",
            side_effect=[ConnectionRefusedError("Ollama offline"), fallback_exc],
        )
        with pytest.raises(RuntimeError) as exc_info:
            load_model_with_fallback("ollama", "gemma4:e2b", "google", "gemini-2.5-flash")
        assert exc_info.value.__cause__ is fallback_exc
