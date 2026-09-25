"""Offline validation of the direct Ollama Cloud connection contract."""

import pytest

from src.agents.llm import validate_cloud_model_config


def test_valid_cloud_connection() -> None:
    """The approved provider, endpoint, and bounded timeout are accepted."""
    validate_cloud_model_config("ollama", "gemma4:31b", "https://ollama.com", 60, api_key="synthetic-key")


@pytest.mark.parametrize(
    ("provider", "model", "url", "timeout", "key"),
    [
        ("google", "gemini-2.5-flash", "https://ollama.com", 60, "synthetic-key"),
        ("ollama", "gemma4:31b", "http://localhost:11434", 60, "synthetic-key"),
        ("ollama", "gemma4:31b", "https://ollama.com/other", 60, "synthetic-key"),
        ("ollama", "", "https://ollama.com", 60, "synthetic-key"),
        ("ollama", "gemma4:31b", "https://ollama.com", 0, "synthetic-key"),
        ("ollama", "gemma4:31b", "https://ollama.com", 60, ""),
    ],
)
def test_rejects_invalid_connection_without_disclosing_key(
    provider: str, model: str, url: str, timeout: float, key: str
) -> None:
    """Unsupported settings fail without echoing a supplied credential."""
    with pytest.raises(ValueError) as error:
        validate_cloud_model_config(provider, model, url, timeout, api_key=key)

    assert "synthetic-key" not in str(error.value)
