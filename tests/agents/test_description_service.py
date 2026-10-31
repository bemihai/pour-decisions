"""Unit tests for description model selection and structured output."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    primary_provider: str = "ollama",
    primary_name: str = "gemma4:31b",
    max_context_chunks: int = 2,
    min_relevance_score: float = 0.4,
) -> SimpleNamespace:
    """Build a lightweight config object that mirrors the OmegaConf structure."""
    return SimpleNamespace(
        model=SimpleNamespace(
            provider=primary_provider,
            name=primary_name,
            base_url="https://ollama.com",
            timeout_seconds=60,
        ),
        description_generation=SimpleNamespace(
            max_context_chunks=max_context_chunks,
            min_relevance_score=min_relevance_score,
        ),
    )


def _make_service(model=None, config=None, mock_load=None):
    """Instantiate DescriptionService with mocked I/O dependencies."""
    wine_repo = MagicMock()
    producer_repo = MagicMock()
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


def _make_async_service(retriever=None, reranker=None):
    """Build a real service with async I/O boundaries replaced by mocks."""
    from src.agents.description_service import DescriptionService

    model = MagicMock()
    structured_model = MagicMock()
    model.with_structured_output.return_value = structured_model
    wine_repo = MagicMock()
    wine_repo.update.return_value = True
    producer_repo = MagicMock()
    producer_repo.update.return_value = True
    service = DescriptionService(
        model=model,
        retriever=retriever,
        reranker=reranker,
        use_rag_context=True,
        use_web_search=False,
        config=_make_config(),
        wine_repo=wine_repo,
        producer_repo=producer_repo,
    )
    return service, model, structured_model, wine_repo, producer_repo


def _wine() -> SimpleNamespace:
    """Return a wine object suitable for description service tests."""
    return SimpleNamespace(
        id=7,
        description=None,
        wine_name="Barolo",
        producer_name="Producer",
        vintage=2019,
        wine_type="Red",
        varietal="Nebbiolo",
        region_name="Piedmont",
        country="Italy",
        appellation="Barolo DOCG",
        designation="Barolo",
        vineyard=None,
        drink_window_source=None,
    )


def _producer() -> SimpleNamespace:
    """Return a producer object suitable for description service tests."""
    return SimpleNamespace(id=11, description=None, name="Producer", country="Italy", region="Piedmont")


# ---------------------------------------------------------------------------
# Model selection -- auto-load path (model is None)
# ---------------------------------------------------------------------------


class TestDescriptionServiceModelSelection:
    """DescriptionService auto-selects the correct model based on config."""

    def test_cloud_model_loaded_by_default(self):
        """Descriptions use the application Cloud model directly."""
        cfg = _make_config()
        mock_load = MagicMock(return_value=MagicMock())

        service = _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_called_once_with("ollama", "gemma4:31b", base_url="https://ollama.com", timeout=60.0)

    def test_local_description_mode_is_rejected(self):
        """The old local description switch cannot bypass Cloud policy."""
        cfg = _make_config()
        cfg.description_generation.use_cloud_model = False
        mock_load = MagicMock(return_value=MagicMock())

        with pytest.raises(ValueError, match="Legacy description model selection"):
            _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_not_called()

    def test_removed_description_switch_is_rejected_when_null(self):
        """A stale null switch still counts as unsupported configuration."""
        cfg = _make_config()
        cfg.description_generation.use_cloud_model = None
        mock_load = MagicMock(return_value=MagicMock())

        with pytest.raises(ValueError, match="Legacy description model selection"):
            _make_service(config=cfg, mock_load=mock_load)
        mock_load.assert_not_called()

    def test_explicit_model_bypasses_config(self):
        """When an explicit model is passed, load_base_model is never called."""
        cfg = _make_config()
        explicit_model = MagicMock()
        mock_load = MagicMock()

        service = _make_service(model=explicit_model, config=cfg, mock_load=mock_load)

        mock_load.assert_not_called()
        assert service.model is explicit_model

    def test_service_stores_cloud_model(self):
        """service.model is the model returned by load_base_model for cloud."""
        cfg = _make_config()
        cloud_model = MagicMock(name="cloud_model")
        mock_load = MagicMock(return_value=cloud_model)

        service = _make_service(config=cfg, mock_load=mock_load)

        assert service.model is cloud_model

    def test_missing_description_generation_section_defaults_to_cloud(self):
        """When the description_generation config section is absent, cloud is used."""
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                provider="ollama",
                name="gemma4:31b",
            ),
            # No description_generation attribute
        )
        mock_load = MagicMock(return_value=MagicMock())

        service = _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_called_once_with("ollama", "gemma4:31b", base_url="https://ollama.com", timeout=60.0)

    def test_missing_model_section_fails_explicitly(self):
        """A missing model config cannot silently choose a provider."""
        cfg = SimpleNamespace(
            description_generation=SimpleNamespace(),
            # No model attribute
        )
        mock_load = MagicMock(return_value=MagicMock())

        with pytest.raises(ValueError, match="Cloud model configuration"):
            _make_service(config=cfg, mock_load=mock_load)

        mock_load.assert_not_called()


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

        assert result == expected

    def test_structured_error_does_not_issue_a_plain_model_call(self, caplog):
        """A provider or parser error cannot trigger another billable request."""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("Bearer synthetic-cloud-secret")
        mock_model.with_structured_output.return_value = mock_structured

        service = self._make_service_with_explicit_model(mock_model)
        with caplog.at_level("WARNING"):
            result = service._invoke_structured("some prompt")

        assert result is None
        mock_model.invoke.assert_not_called()
        assert "synthetic-cloud-secret" not in caplog.text

    def test_returns_none_for_unvalidated_structured_result(self):
        """Only Pydantic-validated wine analysis may reach persistence."""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"description": "unvalidated"}
        mock_model.with_structured_output.return_value = mock_structured

        service = self._make_service_with_explicit_model(mock_model)
        result = service._invoke_structured("some prompt")

        assert result is None

    def test_rejects_empty_description_even_in_wine_analysis(self):
        """A model-constructed object cannot bypass the persistence guard."""
        from src.agents.description_service import WineAnalysis

        model = MagicMock()
        model.with_structured_output.return_value.invoke.return_value = WineAnalysis.model_construct(description="  ")
        service = self._make_service_with_explicit_model(model)

        assert service._invoke_structured("some prompt") is None

    def test_structured_failure_does_not_use_plain_fallback(self):
        """A second provider request is never used after structured failure."""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("structured fail")
        mock_model.with_structured_output.return_value = mock_structured
        mock_model.invoke.side_effect = Exception("plain fail")

        service = self._make_service_with_explicit_model(mock_model)
        result = service._invoke_structured("some prompt")

        assert result is None
        mock_model.invoke.assert_not_called()

    def test_structured_model_built_from_service_model(self):
        """service._structured_model is built by calling with_structured_output on service.model."""
        from src.agents.description_service import WineAnalysis

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = MagicMock()

        service = self._make_service_with_explicit_model(mock_model)

        mock_model.with_structured_output.assert_called_once_with(WineAnalysis, method="function_calling")


# ---------------------------------------------------------------------------
# get_description_model dependency
# ---------------------------------------------------------------------------


class TestGetDescriptionModelDependency:
    """get_description_model returns the cloud model when available."""

    def _make_request(self, cloud_model=None):
        """Build a fake FastAPI Request with app.state populated."""
        state = SimpleNamespace(cloud_model=cloud_model)
        app = SimpleNamespace(state=state)
        return SimpleNamespace(app=app)

    def test_returns_cloud_model_when_available(self):
        """Cloud model is returned when it is loaded in app.state."""
        from src.api.dependencies import get_description_model

        cloud = MagicMock(name="cloud_model")
        request = self._make_request(cloud_model=cloud)

        result = get_description_model(request)

        assert result is cloud

    def test_returns_none_when_no_models_available(self):
        """Returns None when the Cloud model is unavailable."""
        from src.api.dependencies import get_description_model

        request = self._make_request(cloud_model=None)

        result = get_description_model(request)

        assert result is None


class TestDescriptionPersistence:
    """Only validated Cloud output can change stored descriptions."""

    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_producer_paragraph_survives_generation_and_persistence(self, async_mode: bool) -> None:
        """Both generation paths retain a full paragraph beyond the old three-sentence cutoff."""
        paragraph = (
            "Test Estate grows its vines on exposed hillside parcels above the village, where its documented "
            "vineyard work is adapted to each site's soil and aspect. "
            "The estate's stated philosophy puts vineyard observation at the centre of decisions about harvest, "
            "with each parcel picked separately. "
            "In the cellar, separate fermentations preserve those differences before the final blending decisions "
            "are made, according to the estate's own account. "
            "Its distinctive approach is the continuity between these vineyard decisions and the way individual "
            "lots are handled through maturation, giving a clear picture of how the producer works."
        )
        assert len(paragraph) > 500
        retriever = MagicMock()
        evidence = [{"document": paragraph, "metadata": {}, "distance": 0.1}]
        retriever.retrieve.return_value = evidence
        retriever.aretrieve = AsyncMock(return_value=evidence)
        service, model, _structured, _wine_repo, producer_repo = _make_async_service(retriever)
        model.invoke.return_value = AIMessage(content=paragraph)
        model.ainvoke = AsyncMock(return_value=AIMessage(content=paragraph))
        producer = _producer()
        producer.name = "Test Estate"

        result = (
            await service.aget_producer_description(producer)
            if async_mode
            else service.get_producer_description(producer)
        )

        assert result == paragraph
        assert producer_repo.update.call_args.args[0].description == paragraph

    @pytest.mark.parametrize("structured_result", [None, {"description": "unvalidated"}])
    def test_invalid_wine_output_preserves_existing_description(self, structured_result):
        model = MagicMock()
        model.with_structured_output.return_value.invoke.return_value = structured_result
        service = _make_service(model=model, config=_make_config())
        service._build_context_section = lambda _query, _wine: "Producer Barolo evidence"
        wine = SimpleNamespace(
            id=7,
            description="Existing description",
            wine_name="Barolo",
            producer_name="Producer",
            vintage=2019,
            wine_type="Red",
            varietal="Nebbiolo",
            region_name="Piedmont",
            country="Italy",
            appellation="Barolo DOCG",
        )

        assert service.get_wine_description(wine, force_regenerate=True) is None
        assert wine.description == "Existing description"
        service.wine_repo.update.assert_not_called()
        model.invoke.assert_not_called()

    def test_failed_wine_update_restores_supplied_object(self):
        from src.agents.description_service import WineAnalysis

        model = MagicMock()
        model.with_structured_output.return_value.invoke.return_value = WineAnalysis(description="New description")
        service = _make_service(model=model, config=_make_config())
        service._build_context_section = lambda _query, _wine: "Producer Barolo evidence"
        service.wine_repo.update.return_value = False
        wine = SimpleNamespace(
            id=7,
            description="Existing description",
            wine_name="Barolo",
            producer_name="Producer",
            vintage=2019,
            wine_type="Red",
            varietal="Nebbiolo",
            region_name="Piedmont",
            country="Italy",
            appellation="Barolo DOCG",
        )

        assert service.get_wine_description(wine, force_regenerate=True) is None
        assert wine.description == "Existing description"
        service.wine_repo.update_drinking_window.assert_not_called()

    def test_failed_producer_update_restores_supplied_object(self):
        service = _make_service(model=MagicMock(), config=_make_config())
        service.use_rag_context = False
        service._build_producer_context_section = lambda _query, _producer: "Producer evidence"
        service._generate_with_llm = lambda _prompt: "Long enough generated producer description."
        service.producer_repo.update.return_value = False
        producer = SimpleNamespace(id=11, description=None, name="Producer", country="Italy", region="Piedmont")

        assert service.get_producer_description(producer) is None
        assert producer.description is None


class TestAsyncDescriptionService:
    """Async descriptions use native retrieval and preserve sync behavior."""

    @pytest.mark.parametrize("retriever_kind", ["vector", "hybrid"])
    async def test_wine_description_awaits_async_retrieval_with_context(self, retriever_kind: str) -> None:
        """Vector and hybrid retrievers both provide grounded wine context."""
        from src.agents.description_service import WineAnalysis

        results = [
            {
                "document": f"{retriever_kind} Producer Barolo evidence",
                "metadata": {"source": "book.pdf", "page_number": 42},
                "distance": 0.2,
            }
        ]
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=results)
        reranker = MagicMock()
        reranker.arerank = AsyncMock(return_value=results)
        service, _model, structured, wine_repo, _producer_repo = _make_async_service(retriever, reranker)
        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="Generated wine description."))

        result = await service.aget_wine_description(_wine())

        assert result == "Generated wine description."
        retriever.aretrieve.assert_awaited_once()
        retriever.retrieve.assert_not_called()
        reranker.arerank.assert_awaited_once()
        reranker.rerank.assert_not_called()
        assert f"{retriever_kind} Producer Barolo evidence" in structured.ainvoke.await_args.args[0]
        wine_repo.update.assert_called_once()

    async def test_producer_description_awaits_async_retrieval_and_model(self) -> None:
        """Producer generation awaits retrieval, reranking, and plain-text model invocation."""
        results = [{"document": "Producer evidence", "metadata": {}, "distance": 0.1}]
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=results)
        reranker = MagicMock()
        reranker.arerank = AsyncMock(return_value=results)
        service, model, _structured, _wine_repo, producer_repo = _make_async_service(retriever, reranker)
        model.ainvoke = AsyncMock(return_value=AIMessage(content="Generated producer description."))

        result = await service.aget_producer_description(_producer())

        assert result == "Generated producer description."
        retriever.aretrieve.assert_awaited_once()
        retriever.retrieve.assert_not_called()
        reranker.arerank.assert_awaited_once()
        assert "Producer evidence" in model.ainvoke.await_args.args[0]
        producer_repo.update.assert_called_once()

    async def test_generic_rag_falls_back_to_one_qualifying_producer_web_search(self) -> None:
        """Generic RAG is rejected before one cached producer-search attempt."""
        results = [{"document": "Piedmont is a historic Italian wine region.", "metadata": {}, "distance": 0.1}]
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=results)
        service, model, _structured, _wine_repo, producer_repo = _make_async_service(retriever)
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        service._web_search_engine.search.return_value = [
            {
                "title": "Producer estate profile",
                "snippet": "Producer makes traditionally styled wines in Piedmont.",
                "url": "https://example.test/producer",
            }
        ]
        model.ainvoke = AsyncMock(return_value=AIMessage(content="Producer makes traditionally styled wines."))

        result = await service.aget_producer_description(_producer())

        assert result == "Producer makes traditionally styled wines."
        service._web_search_engine.search.assert_called_once_with(
            "Producer Piedmont Italy winery", search_type="producer", max_results=3
        )
        prompt = model.ainvoke.await_args.args[0]
        assert "Piedmont is a historic Italian wine region." not in prompt
        assert "Producer makes traditionally styled wines in Piedmont." in prompt
        producer_repo.update.assert_called_once()

    async def test_qualifying_rag_prevents_producer_web_search(self) -> None:
        """Crama Bauer uses only producer-specific RAG without spending a web request."""
        results = [
            {"document": "Romania has a long history of winemaking.", "metadata": {}, "distance": 0.1},
            {"document": "Crama Bauer farms small vineyard parcels.", "metadata": {}, "distance": 0.1},
        ]
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=results)
        service, model, _structured, _wine_repo, _producer_repo = _make_async_service(retriever)
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="Crama Bauer farms small vineyard parcels."))
        producer = SimpleNamespace(
            id=11,
            description=None,
            name="Crama Bauer",
            country="Romania",
            region="Dragasani",
        )

        assert await service.aget_producer_description(producer) == "Crama Bauer farms small vineyard parcels."

        service._web_search_engine.search.assert_not_called()
        prompt = model.ainvoke.await_args.args[0]
        assert "Crama Bauer farms small vineyard parcels." in prompt
        assert "Romania has a long history of winemaking." not in prompt

    async def test_irrelevant_web_results_skip_model_and_persistence(self) -> None:
        """No qualifying evidence fails safely before model or repository I/O."""
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=[])
        service, model, _structured, _wine_repo, producer_repo = _make_async_service(retriever)
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        service._web_search_engine.search.return_value = [
            {
                "title": "Italian wine history",
                "snippet": "Piedmont has a long tradition of viticulture.",
                "url": "https://example.test/region",
            }
        ]

        assert await service.aget_producer_description(_producer()) is None

        model.ainvoke.assert_not_called()
        producer_repo.update.assert_not_called()

    @pytest.mark.parametrize(
        ("producer_name", "evidence"),
        [
            ("Crama Bauer", "CRAMA-BAUER focuses on small vineyard parcels."),
            ("Chateau d'Été", "Château d’Ete is owned by the founding family."),
        ],
    )
    def test_producer_matching_folds_case_accents_and_punctuation(self, producer_name: str, evidence: str) -> None:
        """Strict matching tolerates only the approved normalization differences."""
        from src.agents.description_service import DescriptionService

        assert DescriptionService._contains_normalized_producer_name(evidence, producer_name)

    def test_partial_producer_name_does_not_qualify(self) -> None:
        """Aliases and partial matches remain unsupported."""
        from src.agents.description_service import DescriptionService

        assert not DescriptionService._contains_normalized_producer_name("Bauer makes Romanian wine.", "Crama Bauer")
        assert not DescriptionService._contains_normalized_producer_name("Cart Winery makes red wine.", "Art")

    def test_sync_producer_path_uses_shared_evidence_gate(self) -> None:
        """Batch-style sync generation also avoids web search for qualifying RAG."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"document": "Producer owns hillside vineyards.", "metadata": {}, "distance": 0.1}
        ]
        service = _make_service(model=MagicMock(), config=_make_config())
        service.retriever = retriever
        service.use_rag_context = True
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        service.model.invoke.return_value = AIMessage(content="Producer owns hillside vineyards.")
        service.producer_repo.update.return_value = True

        assert service.get_producer_description(_producer()) == "Producer owns hillside vineyards."

        service._web_search_engine.search.assert_not_called()

    def test_producer_web_search_failure_skips_model_and_persistence(self) -> None:
        """A producer-search failure cannot fall through to model prior knowledge."""
        service = _make_service(model=MagicMock(), config=_make_config())
        service.use_rag_context = False
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        service._web_search_engine.search.side_effect = RuntimeError("provider unavailable")

        assert service.get_producer_description(_producer()) is None

        service.model.invoke.assert_not_called()
        service.producer_repo.update.assert_not_called()

    def test_wine_query_includes_all_available_identity_fields(self) -> None:
        """RAG and web searches receive the complete available wine identity."""
        service = _make_service(model=MagicMock(), config=_make_config())
        wine = _wine()
        wine.vineyard = "Cannubi"

        assert service._build_wine_search_query(wine) == (
            "Barolo Producer 2019 Nebbiolo Barolo Cannubi Barolo DOCG Piedmont Italy"
        )

    @pytest.mark.parametrize(
        ("producer_name", "wine_name", "evidence"),
        [
            ("Crama Bauer", "Știrbey Cuvée", "CRAMA-BAUER Știrbey-Cuvée is matured in oak."),
            ("Château d'Été", "Les Roches", "Chateau d’Ete Les Roches comes from one parcel."),
        ],
    )
    def test_wine_matching_folds_case_accents_and_punctuation(
        self, producer_name: str, wine_name: str, evidence: str
    ) -> None:
        """Wine evidence matching applies the shared strict normalization."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = producer_name
        wine.wine_name = wine_name

        assert DescriptionService._is_wine_specific_evidence(evidence, wine)

    def test_wine_evidence_requires_full_producer_and_wine_identity(self) -> None:
        """Generic, split, and partial identity matches do not qualify."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "Crama Bauer"
        wine.wine_name = "Sauvignonasse Reserve"
        wine.designation = None

        assert not DescriptionService._is_wine_specific_evidence("Crama Bauer makes Romanian wine.", wine)
        assert not DescriptionService._is_wine_specific_evidence("Bauer Sauvignonasse Reserve is fresh.", wine)
        assert not DescriptionService._is_wine_specific_evidence("Crama Bauer Sauvignonasse is fresh.", wine)

    def test_wine_identity_allows_intervening_subtitle_with_varietal(self) -> None:
        """A source subtitle may split a complete cuvee identity without becoming a partial match."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "Crama DeMatei"
        wine.wine_name = "Crama DeMatei Merlot Prince Matei Rezerva"
        wine.designation = "Prince Matei Rezerva"
        wine.varietal = "Merlot"
        evidence = "Crama DeMatei | Prince Matei Selectii Anul I Merlot Rezerva — WineStatistics"

        assert DescriptionService._is_wine_specific_evidence(evidence, wine)

    def test_intervening_subtitle_match_still_requires_varietal(self) -> None:
        """Relaxed token placement cannot qualify an adjacent cuvee without the stored varietal."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "Crama DeMatei"
        wine.wine_name = "Crama DeMatei Merlot Prince Matei Rezerva"
        wine.designation = "Prince Matei Rezerva"
        wine.varietal = "Merlot"
        evidence = "Crama DeMatei | Prince Matei Selectii Anul I Rezerva"

        assert not DescriptionService._is_wine_specific_evidence(evidence, wine)

    @pytest.mark.parametrize(
        "evidence",
        [
            "Lapostolle Cuvée Alexandre Cabernet Sauvignon | Winebow",
            "2022 Lapostolle 'Cuvee Alexandre' Apalta Cabernet Sauvignon | The Wine Concierge",
            "2022 Lapostolle Cabernet Sauvignon Cuvée Alexandre, Chile, Apalta | IntoWine",
        ],
    )
    def test_shared_designation_does_not_qualify_conflicting_varietal(self, evidence: str) -> None:
        """A shared cuvee designation cannot substitute another varietal's evidence."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "Lapostolle"
        wine.wine_name = "Lapostolle Cabernet Franc Cuvée Alexandre"
        wine.designation = "Cuvée Alexandre"
        wine.varietal = "Cabernet Franc"

        assert not DescriptionService._is_wine_specific_evidence(evidence, wine)

    @pytest.mark.parametrize(
        "evidence",
        [
            "Lapostolle Cuvée Alexandre is 94% Cabernet Sauvignon and 6% Cabernet Franc.",
            "Lapostolle Cuvée Alexandre is Cabernet Sauvignon with Cabernet Franc 6%.",
        ],
    )
    def test_minor_blend_component_does_not_establish_varietal_identity(self, evidence: str) -> None:
        """A percentage blend mention cannot qualify a shared designation."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "Lapostolle"
        wine.wine_name = "Lapostolle Cabernet Franc Cuvée Alexandre"
        wine.designation = "Cuvée Alexandre"
        wine.varietal = "Cabernet Franc"

        assert not DescriptionService._is_wine_specific_evidence(evidence, wine)

    def test_reordered_complete_identity_qualifies_matching_varietal(self) -> None:
        """A reordered complete name remains valid when the varietal is the product identity."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "Lapostolle"
        wine.wine_name = "Lapostolle Cabernet Franc Cuvée Alexandre"
        wine.designation = "Cuvée Alexandre"
        wine.varietal = "Cabernet Franc"

        assert DescriptionService._is_wine_specific_evidence(
            "Lapostolle Cuvée Alexandre 2022 Cabernet Franc from Apalta.", wine
        )

    def test_missing_varietal_requires_complete_wine_identity(self) -> None:
        """Without varietal metadata, a designation alone cannot identify the wine."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "Schloss Johannisberg"
        wine.wine_name = "Schloss Johannisberg Riesling Gelblack feinherb"
        wine.designation = "Gelblack feinherb"
        wine.varietal = None

        assert not DescriptionService._is_wine_specific_evidence(
            "Schloss Johannisberg Gelblack feinherb is available now.", wine
        )
        assert DescriptionService._is_wine_specific_evidence(
            "Schloss Johannisberg Gelblack feinherb Riesling is available now.", wine
        )

    def test_full_wine_name_allows_reordered_classification_without_varietal_label(self) -> None:
        """A complete, specific wine name is sufficient when a source reorders classification words."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "CastellRoig"
        wine.wine_name = "CastellRoig Corpinnat Sabaté I Coca Reserva Brut"
        wine.designation = "Sabaté I Coca Reserva Brut"
        wine.varietal = "Macabeo-Xarello-Parellada Blend"
        evidence = "Castellroig Sabate i Coca Brut Reserva Corpinnat"

        assert DescriptionService._is_wine_specific_evidence(evidence, wine)

    def test_full_wine_name_tokens_scattered_through_navigation_do_not_qualify(self) -> None:
        """A long page menu cannot assemble an identity from unrelated distant labels."""
        from src.agents.description_service import DescriptionService

        wine = _wine()
        wine.producer_name = "CastellRoig"
        wine.wine_name = "CastellRoig Corpinnat Sabaté I Coca Reserva Brut"
        wine.designation = "Sabaté I Coca Reserva Brut"
        wine.varietal = "Macabeo-Xarello-Parellada Blend"
        evidence = (
            "CastellRoig Sabaté I Coca producer overview with unrelated navigation links for visits experiences "
            "booking locations shop wines Reserva Familiar Brut Corpinnat"
        )

        assert not DescriptionService._is_wine_specific_evidence(evidence, wine)

    async def test_generic_wine_rag_falls_back_to_one_qualifying_web_search(self) -> None:
        """Generic RAG is rejected before one filtered review search."""
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(
            return_value=[{"document": "Barolo is made in Piedmont.", "metadata": {}, "distance": 0.1}]
        )
        service, _model, structured, wine_repo, _producer_repo = _make_async_service(retriever)
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        service._web_search_engine.search.return_value = [
            {"title": "Producer Barolo review", "snippet": "Producer Barolo is matured in cask."},
            {"title": "Producer profile", "snippet": "Barolo is a Piedmont appellation."},
        ]
        from src.agents.description_service import WineAnalysis

        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="Grounded Barolo description."))

        assert await service.aget_wine_description(_wine()) == "Grounded Barolo description."

        service._web_search_engine.search.assert_called_once_with(
            "Barolo Producer 2019 Nebbiolo Barolo Barolo DOCG Piedmont Italy tasting review drinking window",
            search_type="review",
            max_results=3,
        )
        prompt = structured.ainvoke.await_args.args[0]
        assert "Producer Barolo is matured in cask." in prompt
        assert "Barolo is a Piedmont appellation." not in prompt
        wine_repo.update.assert_called_once()

    async def test_qualifying_wine_rag_skips_web_and_allows_supplementary_context(self) -> None:
        """One specific source qualifies the retrieved context without another paid search."""
        results = [
            {"document": "Barolo is made from Nebbiolo.", "metadata": {}, "distance": 0.1},
            {"document": "Producer Barolo is matured in large casks.", "metadata": {}, "distance": 0.1},
        ]
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=results)
        service, _model, structured, _wine_repo, _producer_repo = _make_async_service(retriever)
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        from src.agents.description_service import WineAnalysis

        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="Grounded Barolo description."))

        assert await service.aget_wine_description(_wine()) == "Grounded Barolo description."
        service._web_search_engine.search.assert_not_called()
        prompt = structured.ainvoke.await_args.args[0]
        assert "Producer Barolo is matured in large casks." in prompt
        assert "Barolo is made from Nebbiolo." in prompt

    async def test_conflicting_varietal_results_use_general_knowledge(self) -> None:
        """Rejected identities never reach the general-knowledge prompt."""
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=[])
        service, _model, structured, wine_repo, _producer_repo = _make_async_service(retriever)
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        service._web_search_engine.search.return_value = [
            {"title": "Lapostolle Cuvée Alexandre Cabernet Sauvignon | Winebow", "snippet": ""},
            {
                "title": "2022 Lapostolle 'Cuvee Alexandre' Apalta Cabernet Sauvignon | The Wine Concierge",
                "snippet": "",
            },
            {
                "title": "2022 Lapostolle Cabernet Sauvignon Cuvée Alexandre, Chile, Apalta | IntoWine",
                "snippet": "",
            },
        ]
        from src.agents.description_service import WineAnalysis

        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="General-knowledge description."))
        wine = _wine()
        wine.producer_name = "Lapostolle"
        wine.wine_name = "Lapostolle Cabernet Franc Cuvée Alexandre"
        wine.designation = "Cuvée Alexandre"
        wine.varietal = "Cabernet Franc"

        assert await service.aget_wine_description(wine) == "General-knowledge description."
        assert "Cabernet Sauvignon" not in structured.ainvoke.await_args.args[0]
        service._web_search_engine.search.assert_called_once()
        structured.ainvoke.assert_awaited_once()
        assert "Use general-knowledge fallback." in structured.ainvoke.await_args.args[0]
        wine_repo.update.assert_called_once()

    async def test_irrelevant_wine_evidence_uses_general_knowledge(self) -> None:
        """Rejected web results are excluded while one fallback description is saved."""
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=[])
        service, _model, structured, wine_repo, _producer_repo = _make_async_service(retriever)
        service.use_web_search = True
        service._web_search_available = True
        service._web_search_engine = MagicMock()
        service._web_search_engine.search.return_value = [
            {"title": "Producer profile", "snippet": "Barolo is made in Piedmont."}
        ]
        from src.agents.description_service import WineAnalysis

        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="General-knowledge description."))

        assert await service.aget_wine_description(_wine()) == "General-knowledge description."
        structured.ainvoke.assert_awaited_once()
        assert "Use general-knowledge fallback." in structured.ainvoke.await_args.args[0]
        wine_repo.update.assert_called_once()

    async def test_disabled_rag_skips_retrieval(self) -> None:
        """Disabling RAG bypasses both async and sync retrieval methods."""
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock()
        service, _model, structured, _wine_repo, _producer_repo = _make_async_service(retriever)
        service.use_rag_context = False
        from src.agents.description_service import WineAnalysis

        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="General-knowledge description."))

        assert await service.aget_wine_description(_wine()) == "General-knowledge description."

        retriever.aretrieve.assert_not_awaited()
        retriever.retrieve.assert_not_called()
        structured.ainvoke.assert_awaited_once()
        assert "Use general-knowledge fallback." in structured.ainvoke.await_args.args[0]

    @pytest.mark.parametrize("failure_stage", ["retrieval", "reranking"])
    async def test_context_failure_uses_general_knowledge(self, failure_stage: str) -> None:
        """Technical evidence failures allow one general-knowledge generation."""
        results = [{"document": "Unavailable evidence", "metadata": {}, "distance": 0.1}]
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(
            side_effect=RuntimeError("retrieval failed") if failure_stage == "retrieval" else None,
            return_value=results,
        )
        reranker = MagicMock()
        reranker.arerank = AsyncMock(
            side_effect=RuntimeError("reranking failed") if failure_stage == "reranking" else None,
            return_value=results,
        )
        service, _model, structured, _wine_repo, _producer_repo = _make_async_service(retriever, reranker)
        from src.agents.description_service import WineAnalysis

        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="General-knowledge description."))

        result = await service.aget_wine_description(_wine())

        assert result == "General-knowledge description."
        structured.ainvoke.assert_awaited_once()
        assert "Use general-knowledge fallback." in structured.ainvoke.await_args.args[0]

    async def test_retrieval_cancellation_propagates_without_model_call(self) -> None:
        """Cancellation is not converted into fail-soft generation."""
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(side_effect=asyncio.CancelledError)
        service, _model, structured, _wine_repo, _producer_repo = _make_async_service(retriever)
        structured.ainvoke = AsyncMock()

        with pytest.raises(asyncio.CancelledError):
            await service.aget_wine_description(_wine())

        structured.ainvoke.assert_not_awaited()

    async def test_model_failure_does_not_retry_or_persist(self) -> None:
        """A failed structured call returns failure after exactly one model attempt."""
        service, _model, structured, wine_repo, _producer_repo = _make_async_service()
        service._abuild_context_section = AsyncMock(return_value="Producer Barolo evidence")
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("provider failed"))

        assert await service.aget_wine_description(_wine()) is None

        structured.ainvoke.assert_awaited_once()
        wine_repo.update.assert_not_called()

    async def test_persistence_failure_does_not_retry_model_and_restores_wine(self) -> None:
        """Persistence failure preserves the object and does not replay generation."""
        from src.agents.description_service import WineAnalysis

        service, _model, structured, wine_repo, _producer_repo = _make_async_service()
        service._abuild_context_section = AsyncMock(return_value="Producer Barolo evidence")
        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="Generated wine description."))
        wine_repo.update.return_value = False
        wine = _wine()

        assert await service.aget_wine_description(wine) is None

        assert wine.description is None
        structured.ainvoke.assert_awaited_once()
        wine_repo.update.assert_called_once()


class TestDescriptionPromptRegistryIntegration:
    """Description prompt templates should come from the immutable registry."""

    def test_service_retains_registered_prompt_sources(self) -> None:
        """Construction should retain both registered records and exact source text."""
        from src.agents.prompt_registry import get_prompt_registry

        service = _make_service(model=MagicMock(), config=_make_config())
        registry = get_prompt_registry()

        assert service.wine_prompt_record is registry.get("wine_description")
        assert service.producer_prompt_record is registry.get("producer_description")
        assert service._wine_prompt_template == registry.get("wine_description").source
        assert service._producer_prompt_template == registry.get("producer_description").source
        assert service.wine_execution_provenance.mode == "description_wine"
        assert [prompt.name for prompt in service.wine_execution_provenance.prompts] == ["wine_description"]
        assert service.producer_execution_provenance.mode == "description_producer"
        assert [prompt.name for prompt in service.producer_execution_provenance.prompts] == ["producer_description"]

    def test_wine_prompt_formatting_is_byte_identical(self) -> None:
        """Wine generation should format the registered template without normalization."""
        from src.agents.description_service import WineAnalysis

        service = _make_service(model=MagicMock(), config=_make_config())
        service._build_wine_search_query = lambda _wine: "query"
        service._build_context_section = lambda _query, _wine: "\nWine context"
        captured: dict[str, str] = {}

        def _capture(prompt: str) -> WineAnalysis:
            captured["prompt"] = prompt
            return WineAnalysis(description="Generated wine description.")

        service._invoke_structured = _capture
        wine = SimpleNamespace(
            id=7,
            description=None,
            wine_name="Barolo",
            producer_name="Producer",
            vintage=2019,
            wine_type="Red",
            varietal="Nebbiolo",
            region_name="Piedmont",
            country="Italy",
            appellation="Barolo DOCG",
            designation=None,
            vineyard=None,
        )
        expected = service.wine_prompt_record.source.format(
            wine_name="Barolo",
            producer_name="Producer",
            vintage=2019,
            wine_type="Red",
            varietal="Nebbiolo",
            region="Piedmont",
            country="Italy",
            appellation="Barolo DOCG",
            designation="N/A",
            vineyard="N/A",
            context_section="\nWine context",
        )

        service.get_wine_description(wine)

        assert captured["prompt"] == expected

    def test_producer_prompt_formatting_is_byte_identical(self) -> None:
        """Producer generation should format the registered template without normalization."""
        service = _make_service(model=MagicMock(), config=_make_config())
        service.use_rag_context = False
        service._build_producer_search_query = lambda _producer: "query"
        service._build_producer_context_section = lambda _query, _producer: "Producer evidence"
        captured: dict[str, str] = {}

        def _capture(prompt: str) -> str:
            captured["prompt"] = prompt
            return "Generated producer description."

        service._generate_with_llm = _capture
        producer = SimpleNamespace(
            id=11,
            description=None,
            name="Producer",
            country="Italy",
            region="Piedmont",
        )
        expected = service.producer_prompt_record.source.format(
            producer_name="Producer",
            country="Italy",
            region="Piedmont",
            context_section="Producer evidence",
        )

        service.get_producer_description(producer)

        assert captured["prompt"] == expected

    def test_registry_failure_stops_service_before_model_setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid prompt assets should fail construction before model preparation."""
        from src.agents import description_service as module

        model = MagicMock()
        monkeypatch.setattr(
            module,
            "get_prompt_registry",
            MagicMock(side_effect=FileNotFoundError("missing prompt manifest")),
        )

        with pytest.raises(FileNotFoundError, match="missing prompt manifest"):
            module.DescriptionService(
                model=model,
                config=_make_config(),
                wine_repo=MagicMock(),
                producer_repo=MagicMock(),
            )

        model.with_structured_output.assert_not_called()


class TestWineGeneralKnowledgeFallback:
    """Wine evidence failures allow one generation while preserving stored data."""

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("stage", ["absent", "rejected", "disabled", "retrieval", "reranking", "search"])
    async def test_fallback_after_evidence_unavailable(self, async_mode: bool, stage: str) -> None:
        """Sync and async paths discard evidence failures without extra requests."""
        from src.agents.description_service import WineAnalysis

        evidence = [{"document": "Other Producer Other Cuvee", "distance": 0.1}]
        retriever = MagicMock()
        retriever.retrieve.return_value = [] if stage == "absent" else evidence
        retriever.aretrieve = AsyncMock(return_value=retriever.retrieve.return_value)
        reranker = MagicMock()
        reranker.rerank.return_value = retriever.retrieve.return_value
        reranker.arerank = AsyncMock(return_value=reranker.rerank.return_value)
        if stage == "retrieval":
            retriever.retrieve.side_effect = RuntimeError("retrieval unavailable")
            retriever.aretrieve.side_effect = RuntimeError("retrieval unavailable")
        if stage == "reranking":
            reranker.rerank.side_effect = RuntimeError("reranking unavailable")
            reranker.arerank.side_effect = RuntimeError("reranking unavailable")
        service, model, structured, repository, _ = _make_async_service(retriever, reranker)
        service.use_rag_context = stage != "disabled"
        service.use_web_search = stage != "disabled"
        service._web_search_engine = MagicMock()
        service._web_search_engine.search.return_value = []
        if stage == "search":
            service._web_search_engine.search.side_effect = RuntimeError("search unavailable")
        analysis = WineAnalysis(
            description="Based on general knowledge, this style may be fruit-led.",
            drink_from_year=2025,
            drink_to_year=2035,
        )
        structured.invoke.return_value = analysis
        structured.ainvoke = AsyncMock(return_value=analysis)
        wine = _wine()
        wine.drink_from_year, wine.drink_to_year, wine.drink_window_source = 2026, 2027, "cellar_tracker"
        with patch.object(service, "_persist_wine_analysis", wraps=service._persist_wine_analysis) as persist:
            result = await service.aget_wine_description(wine) if async_mode else service.get_wine_description(wine)
        assert result == analysis.description
        generated = persist.call_args.args[1]
        assert generated.drink_from_year is None and generated.drink_to_year is None
        assert (wine.drink_from_year, wine.drink_to_year, wine.drink_window_source) == (2026, 2027, "cellar_tracker")
        repository.update.assert_called_once_with(wine)
        repository.update_drinking_window.assert_not_called()
        invocation = structured.ainvoke if async_mode else structured.invoke
        invocation.assert_called_once()
        prompt = invocation.call_args.args[0]
        assert "Other Producer Other Cuvee" not in prompt
        assert "Use general-knowledge fallback." in prompt
        assert "Explicitly state in the prose" in prompt
        assert "qualified wording" in prompt
        assert "Do not invent wine-specific" in prompt
        model.invoke.assert_not_called()
        model.ainvoke.assert_not_called()
        assert service._web_search_engine.search.call_count == (0 if stage == "disabled" else 1)
        assert retriever.retrieve.call_count + retriever.aretrieve.call_count == (0 if stage == "disabled" else 1)

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("failure", ["model", "validation", "save_false", "save_exception"])
    async def test_fallback_failure_preserves_existing_description(self, async_mode: bool, failure: str) -> None:
        """Fallback does not hide model, validation, or persistence failures."""
        from src.agents.description_service import WineAnalysis

        service, model, structured, repository, _ = _make_async_service()
        structured.invoke.return_value = WineAnalysis(description="New description")
        structured.ainvoke = AsyncMock(return_value=structured.invoke.return_value)
        if failure == "model":
            structured.invoke.side_effect = RuntimeError("model unavailable")
            structured.ainvoke.side_effect = RuntimeError("model unavailable")
        elif failure == "validation":
            structured.invoke.return_value = WineAnalysis(description="   ")
            structured.ainvoke.return_value = structured.invoke.return_value
        elif failure == "save_false":
            repository.update.return_value = False
        else:
            repository.update.side_effect = RuntimeError("database unavailable")
        wine = _wine()
        wine.description = "Previous description"
        result = (
            await service.aget_wine_description(wine, force_regenerate=True)
            if async_mode
            else service.get_wine_description(wine, force_regenerate=True)
        )
        assert result is None
        assert wine.description == "Previous description"
        invocation = structured.ainvoke if async_mode else structured.invoke
        invocation.assert_called_once()
        assert repository.update.call_count == (1 if failure.startswith("save") else 0)
        repository.update_drinking_window.assert_not_called()
        model.invoke.assert_not_called()
        model.ainvoke.assert_not_called()

    @pytest.mark.parametrize("stage", ["reranking", "search", "model"])
    async def test_cancellation_is_not_a_fallback_trigger(self, stage: str) -> None:
        """Cancellation at every asynchronous evidence/model boundary propagates."""
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(return_value=[{"document": "Other wine", "distance": 0.1}])
        reranker = MagicMock()
        reranker.arerank = AsyncMock(return_value=[])
        service, _, structured, repository, _ = _make_async_service(retriever, reranker)
        structured.ainvoke = AsyncMock(side_effect=asyncio.CancelledError)
        if stage == "reranking":
            reranker.arerank.side_effect = asyncio.CancelledError
        elif stage == "search":
            service.use_web_search = True
            service._web_search_engine = MagicMock()
            service._web_search_engine.search.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await service.aget_wine_description(_wine())
        assert structured.ainvoke.call_count == (1 if stage == "model" else 0)
        repository.update.assert_not_called()
