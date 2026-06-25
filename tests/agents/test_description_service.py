"""Unit tests for Phase 5: Description service model selection.

Tests cover:
- DescriptionService auto-selects the cloud/fallback model by default when no model
  is provided and ``description_generation.use_cloud_model: true`` (default).
- DescriptionService falls back to the primary model when
  ``description_generation.use_cloud_model: false``.
- DescriptionService always uses an explicitly provided model, ignoring the config.
- ``_invoke_structured`` gracefully falls back to plain invoke when structured
  output raises an exception.
- ``get_description_model`` dependency returns cloud model when available, otherwise
  falls back to the default model.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    primary_provider: str = "ollama",
    primary_name: str = "gemma4:e2b",
    fallback_provider: str = "google",
    fallback_name: str = "gemini-2.5-flash",
    use_cloud_model: bool = True,
    max_context_chunks: int = 2,
    min_relevance_score: float = 0.4,
) -> SimpleNamespace:
    """Build a lightweight config object that mirrors the OmegaConf structure."""
    return SimpleNamespace(
        model=SimpleNamespace(
            provider=primary_provider,
            name=primary_name,
            fallback_provider=fallback_provider,
            fallback_name=fallback_name,
        ),
        description_generation=SimpleNamespace(
            use_cloud_model=use_cloud_model,
            max_context_chunks=max_context_chunks,
            min_relevance_score=min_relevance_score,
        ),
    )


def _make_service(model=None, config=None, mock_load=None):
    """Instantiate DescriptionService with mocked I/O dependencies."""
    wine_repo = MagicMock()
    producer_repo = MagicMock()
    with patch("src.agents.description_service.DescriptionService._load_prompt", return_value="prompt {wine_name}"):
        if mock_load is not None:
            with patch("src.agents.description_service.load_base_model", mock_load):
                from src.agents.description_service import DescriptionService
                return DescriptionService(
                    model=model,
                    config=config,
                    wine_repo=wine_repo,
                    producer_repo=producer_repo,
                )
        else:
            from src.agents.description_service import DescriptionService
            return DescriptionService(
                model=model,
                config=config,
                wine_repo=wine_repo,
                producer_repo=producer_repo,
            )


# ---------------------------------------------------------------------------
# Model selection -- auto-load path (model is None)
# ---------------------------------------------------------------------------

class TestDescriptionServiceModelSelection:
    """DescriptionService auto-selects the correct model based on config."""

    def test_cloud_model_loaded_by_default(self):
        """When model is None and use_cloud_model=True, the fallback/cloud provider is loaded."""
        cfg = _make_config(use_cloud_model=True, fallback_provider="google", fallback_name="gemini-2.5-flash")
        mock_load = MagicMock(return_value=MagicMock())

        service = _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_called_once_with("google", "gemini-2.5-flash")

    def test_primary_model_loaded_when_use_cloud_false(self):
        """When model is None and use_cloud_model=False, the primary provider is loaded."""
        cfg = _make_config(use_cloud_model=False, primary_provider="ollama", primary_name="gemma4:e2b")
        mock_load = MagicMock(return_value=MagicMock())

        service = _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_called_once_with("ollama", "gemma4:e2b")

    def test_explicit_model_bypasses_config(self):
        """When an explicit model is passed, load_base_model is never called."""
        cfg = _make_config(use_cloud_model=True)
        explicit_model = MagicMock()
        mock_load = MagicMock()

        service = _make_service(model=explicit_model, config=cfg, mock_load=mock_load)

        mock_load.assert_not_called()
        assert service.model is explicit_model

    def test_service_stores_cloud_model(self):
        """service.model is the model returned by load_base_model for cloud."""
        cfg = _make_config(use_cloud_model=True)
        cloud_model = MagicMock(name="cloud_model")
        mock_load = MagicMock(return_value=cloud_model)

        service = _make_service(config=cfg, mock_load=mock_load)

        assert service.model is cloud_model

    def test_missing_description_generation_section_defaults_to_cloud(self):
        """When the description_generation config section is absent, cloud is used."""
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="ollama",
                name="gemma4:e2b",
                fallback_provider="google",
                fallback_name="gemini-2.5-flash",
            ),
            # No description_generation attribute
        )
        mock_load = MagicMock(return_value=MagicMock())

        service = _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_called_once_with("google", "gemini-2.5-flash")

    def test_missing_model_section_uses_hardcoded_defaults(self):
        """When the model config section is absent, safe hardcoded defaults are used."""
        cfg = SimpleNamespace(
            description_generation=SimpleNamespace(use_cloud_model=True),
            # No model attribute
        )
        mock_load = MagicMock(return_value=MagicMock())

        service = _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_called_once_with("google", "gemini-2.5-flash")


# ---------------------------------------------------------------------------
# Structured-output fallback (_invoke_structured)
# ---------------------------------------------------------------------------

class TestInvokeStructuredFallback:
    """_invoke_structured gracefully handles structured-output failures."""

    def _make_service_with_explicit_model(self, model):
        """Create service with an explicit model (avoids load_base_model calls)."""
        cfg = _make_config()
        return _make_service(model=model, config=cfg)

    def test_returns_wine_analysis_on_success(self):
        """Returns a WineAnalysis instance when structured output succeeds."""
        from src.agents.description_service import WineAnalysis

        mock_model = MagicMock()
        expected = WineAnalysis(
            description="Rich and complex Barolo.",
            drink_from_year=2025,
            drink_to_year=2035,
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = expected
        mock_model.with_structured_output.return_value = mock_structured

        service = self._make_service_with_explicit_model(mock_model)
        result = service._invoke_structured("some prompt")

        assert result is expected

    def test_falls_back_to_plain_invoke_on_structured_error(self):
        """When structured output raises, falls back to plain model.invoke()."""
        from src.agents.description_service import WineAnalysis

        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("OutputParserException")
        mock_model.with_structured_output.return_value = mock_structured

        plain_response = MagicMock()
        plain_response.content = "A lovely Burgundy with earthy notes and good structure."
        mock_model.invoke.return_value = plain_response

        service = self._make_service_with_explicit_model(mock_model)
        result = service._invoke_structured("some prompt")

        assert isinstance(result, WineAnalysis)
        assert "Burgundy" in result.description
        assert result.drink_from_year is None
        assert result.drink_to_year is None

    def test_returns_none_when_plain_fallback_response_too_short(self):
        """Returns None when the plain fallback produces a very short response."""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("fail")
        mock_model.with_structured_output.return_value = mock_structured

        short_response = MagicMock()
        short_response.content = "ok"  # fewer than 20 chars
        mock_model.invoke.return_value = short_response

        service = self._make_service_with_explicit_model(mock_model)
        result = service._invoke_structured("some prompt")

        assert result is None

    def test_returns_none_when_both_invocations_fail(self):
        """Returns None when both structured and plain invocations raise."""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("structured fail")
        mock_model.with_structured_output.return_value = mock_structured
        mock_model.invoke.side_effect = Exception("plain fail")

        service = self._make_service_with_explicit_model(mock_model)
        result = service._invoke_structured("some prompt")

        assert result is None

    def test_structured_model_built_from_service_model(self):
        """service._structured_model is built by calling with_structured_output on service.model."""
        from src.agents.description_service import WineAnalysis

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = MagicMock()

        service = self._make_service_with_explicit_model(mock_model)

        mock_model.with_structured_output.assert_called_once_with(WineAnalysis)


# ---------------------------------------------------------------------------
# get_description_model dependency
# ---------------------------------------------------------------------------

class TestGetDescriptionModelDependency:
    """get_description_model returns the cloud model when available."""

    def _make_request(self, cloud_model=None, local_model=None, default_model=None):
        """Build a fake FastAPI Request with app.state populated."""
        state = SimpleNamespace(
            cloud_model=cloud_model,
            local_model=local_model,
            model=default_model,
        )
        app = SimpleNamespace(state=state)
        return SimpleNamespace(app=app)

    def test_returns_cloud_model_when_available(self):
        """Cloud model is returned when it is loaded in app.state."""
        from src.api.dependencies import get_description_model

        cloud = MagicMock(name="cloud_model")
        request = self._make_request(cloud_model=cloud, default_model=MagicMock())

        result = get_description_model(request)

        assert result is cloud

    def test_falls_back_to_default_model_when_cloud_absent(self):
        """Falls back to app.state.model when cloud model is not loaded."""
        from src.api.dependencies import get_description_model

        default = MagicMock(name="default_model")
        request = self._make_request(cloud_model=None, default_model=default)

        result = get_description_model(request)

        assert result is default

    def test_returns_none_when_no_models_available(self):
        """Returns None when neither cloud nor default model is loaded."""
        from src.api.dependencies import get_description_model

        request = self._make_request(cloud_model=None, default_model=None)

        result = get_description_model(request)

        assert result is None
