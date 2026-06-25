"""Unit tests for DescriptionService web-search configuration behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agents import description_service as module


class _FakeModel:
    """Minimal model stub for DescriptionService initialization tests."""

    def with_structured_output(self, _schema):
        """Return self to satisfy structured model setup."""
        return self


def _base_config() -> dict:
    """Build minimal config dict required by DescriptionService.__init__."""
    return {
        "model": {"provider": "google", "name": "gemini-2.5-flash"},
        "web_search": {"tavily": {"api_key_env": "TAVILY_API_KEY"}},
        "description_generation": {
            "enable_web_search": True,
            "max_context_chunks": 3,
            "min_relevance_score": 0.4,
        },
    }


@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch heavy dependencies to keep init tests fast and isolated."""
    monkeypatch.setattr(module, "load_base_model", lambda *_args, **_kwargs: _FakeModel())
    monkeypatch.setattr(module.DescriptionService, "_load_prompt", lambda *_args, **_kwargs: "prompt")


def test_description_service_web_search_uses_configured_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured api_key_env should be honored when enabling web search."""
    cfg = _base_config()
    cfg["web_search"]["tavily"]["api_key_env"] = "CUSTOM_TAVILY_KEY"

    monkeypatch.setenv("CUSTOM_TAVILY_KEY", "configured")

    service = module.DescriptionService(
        model=MagicMock(),
        use_web_search=True,
        config=cfg,
        wine_repo=SimpleNamespace(),
        producer_repo=SimpleNamespace(),
    )

    assert service.use_web_search is True
    assert service.web_search_api_key_env == "CUSTOM_TAVILY_KEY"


def test_description_service_web_search_disabled_by_config_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global config switch should disable web search even if key exists."""
    cfg = _base_config()
    cfg["description_generation"]["enable_web_search"] = False

    monkeypatch.setenv("TAVILY_API_KEY", "configured")

    service = module.DescriptionService(
        model=MagicMock(),
        use_web_search=True,
        config=cfg,
        wine_repo=SimpleNamespace(),
        producer_repo=SimpleNamespace(),
    )

    assert service.use_web_search is False
    assert service._web_search_available is False


def test_description_service_logs_web_search_status(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Initialization should log effective web-search status and api-key presence."""
    cfg = _base_config()
    monkeypatch.setenv("TAVILY_API_KEY", "configured")

    with caplog.at_level("INFO"):
        service = module.DescriptionService(
            model=MagicMock(),
            use_web_search=True,
            config=cfg,
            wine_repo=SimpleNamespace(),
            producer_repo=SimpleNamespace(),
        )

    assert service.use_web_search is True
    assert "DescriptionService web-search status:" in caplog.text
    assert "effective_use_web_search=True" in caplog.text

