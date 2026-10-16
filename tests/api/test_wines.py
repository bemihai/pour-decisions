"""Tests for wine detail API endpoints.

Uses FastAPI TestClient with patched repositories to avoid hitting the real
SQLite database. Focused route tests mock the service, while async-boundary
tests use the real DescriptionService with mocked external I/O.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.schemas.wines import DescriptionResponse, WineDetailResponse
from src.database.models import Bottle, Producer, Region, Wine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Create a FastAPI TestClient with pre-populated app state."""
    from src.api.main import app

    # Populate state that lifespan would normally set
    app.state.model = MagicMock()
    app.state.cloud_model = None
    app.state.intelligent_agent = None
    app.state.tool_registry = None
    app.state.retriever = None
    app.state.reranker = None

    return TestClient(app)


def _make_wine(**overrides) -> Wine:
    """Build a minimal Wine model with sensible defaults."""
    defaults = dict(
        id=1,
        source="cellar_tracker",
        external_id="CT-123",
        wine_name="Test Cuvee",
        vintage=2020,
        wine_type="Red",
        varietal="Pinot Noir",
        designation=None,
        appellation="Gevrey-Chambertin",
        vineyard=None,
        bottle_size="750ml",
        producer_id=10,
        producer_name="Domaine Test",
        region_id=5,
        region_name="Burgundy",
        country="France",
        personal_rating=88,
        community_rating=4.2,
        tasting_notes="Cherry and earth notes.",
        last_tasted_date=None,
        drink_from_year=2023,
        drink_to_year=2030,
        drink_index=0.75,
        drink_window_source="heuristic",
        description="A fine Burgundy Pinot Noir.",
        q_purchased=3,
        q_quantity=2,
        q_consumed=1,
    )
    defaults.update(overrides)
    return Wine(**defaults)  # ty:ignore[invalid-argument-type]


def _make_bottle(**overrides) -> Bottle:
    """Build a minimal Bottle model with sensible defaults."""
    defaults = dict(
        id=100,
        wine_id=1,
        source="cellar_tracker",
        quantity=1,
        status="in_cellar",
        location="Cellar",
        bin="A1",
        purchase_price=25.0,
        currency="EUR",
    )
    defaults.update(overrides)
    return Bottle(**defaults)  # ty:ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# GET /api/wines/{wine_id}
# ---------------------------------------------------------------------------


class TestGetWineDetail:

    @patch("src.api.routes.wines.RegionRepository")
    @patch("src.api.routes.wines.ProducerRepository")
    @patch("src.api.routes.wines.BottleRepository")
    @patch("src.api.routes.wines.WineRepository")
    def test_returns_full_detail(self, mock_wine_cls, mock_bottle_cls, mock_producer_cls, mock_region_cls, client):
        wine = _make_wine()
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        bottle_repo = MagicMock()
        mock_bottle_cls.return_value = bottle_repo
        bottle_repo.get_by_wine.return_value = [_make_bottle(), _make_bottle(id=101, bin="A2")]
        bottle_repo.get_owned_quantity.return_value = 2

        producer = Producer(id=10, name="Domaine Test", country="France", description="Historic estate.")
        producer_repo = MagicMock()
        mock_producer_cls.return_value = producer_repo
        producer_repo.get_by_id.return_value = producer

        region = Region(id=5, name="Burgundy", country="France", description="Classic limestone slopes.")
        region_repo = MagicMock()
        mock_region_cls.return_value = region_repo
        region_repo.get_by_id.return_value = region

        resp = client.get("/api/wines/1")

        assert resp.status_code == 200
        body = WineDetailResponse(**resp.json())
        assert body.id == 1
        assert body.wine_name == "Test Cuvee"
        assert body.vintage == 2020
        assert body.varietal == "Pinot Noir"
        assert body.producer_name == "Domaine Test"
        assert body.country == "France"
        assert body.personal_rating == 88
        assert body.description == "A fine Burgundy Pinot Noir."
        assert body.producer_description == "Historic estate."
        assert body.region_description == "Classic limestone slopes."
        assert len(body.bottles) == 2
        assert body.owned_quantity == 2
        assert body.bottles[0].location == "Cellar"

    @patch("src.api.routes.wines.RegionRepository")
    @patch("src.api.routes.wines.ProducerRepository")
    @patch("src.api.routes.wines.BottleRepository")
    @patch("src.api.routes.wines.WineRepository")
    def test_detail_returns_same_canonical_readiness_as_cellar_views(
        self,
        mock_wine_cls,
        mock_bottle_cls,
        mock_producer_cls,
        mock_region_cls,
        client,
    ):
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = _make_wine(
            wine_name="Crama Bauer Cabernet Franc Clàsic 2019",
            drink_from_year=2025,
            drink_to_year=2027,
            drink_index=0.375,
            drink_window_source="cellar_tracker",
            producer_id=None,
            region_id=None,
        )
        mock_bottle_cls.return_value.get_by_wine.return_value = [_make_bottle()]
        mock_bottle_cls.return_value.get_owned_quantity.return_value = 1
        mock_producer_cls.return_value.get_by_id.return_value = None
        mock_region_cls.return_value.get_by_id.return_value = None

        with patch("src.agents.drinking_window_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.year = 2026
            resp = client.get("/api/wines/1")

        assert resp.status_code == 200
        body = WineDetailResponse(**resp.json())
        assert body.drink_index == 100.0
        assert body.drink_status == "ready"

    @patch("src.api.routes.wines.WineRepository")
    def test_wine_not_found_returns_404(self, mock_wine_cls, client):
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = None

        resp = client.get("/api/wines/999")

        assert resp.status_code == 404
        assert "999" in resp.json()["detail"]

    @patch("src.api.routes.wines.RegionRepository")
    @patch("src.api.routes.wines.ProducerRepository")
    @patch("src.api.routes.wines.BottleRepository")
    @patch("src.api.routes.wines.WineRepository")
    def test_wine_without_bottles(self, mock_wine_cls, mock_bottle_cls, mock_producer_cls, mock_region_cls, client):
        wine = _make_wine(producer_id=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        bottle_repo = MagicMock()
        mock_bottle_cls.return_value = bottle_repo
        bottle_repo.get_by_wine.return_value = []
        bottle_repo.get_owned_quantity.return_value = 0
        mock_region_cls.return_value.get_by_id.return_value = None

        resp = client.get("/api/wines/1")

        assert resp.status_code == 200
        body = WineDetailResponse(**resp.json())
        assert body.bottles == []
        assert body.owned_quantity == 0
        assert body.producer_description is None

    @patch("src.api.routes.wines.RegionRepository")
    @patch("src.api.routes.wines.ProducerRepository")
    @patch("src.api.routes.wines.BottleRepository")
    @patch("src.api.routes.wines.WineRepository")
    def test_wine_without_description(self, mock_wine_cls, mock_bottle_cls, mock_producer_cls, mock_region_cls, client):
        wine = _make_wine(description=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        bottle_repo = MagicMock()
        mock_bottle_cls.return_value = bottle_repo
        bottle_repo.get_by_wine.return_value = []
        bottle_repo.get_owned_quantity.return_value = 0

        producer_repo = MagicMock()
        mock_producer_cls.return_value = producer_repo
        producer_repo.get_by_id.return_value = None
        mock_region_cls.return_value.get_by_id.return_value = None

        resp = client.get("/api/wines/1")

        body = WineDetailResponse(**resp.json())
        assert body.description is None
        assert body.producer_description is None


# ---------------------------------------------------------------------------
# POST /api/wines/{wine_id}/description
# ---------------------------------------------------------------------------


class TestGenerateDescription:

    @patch("src.api.routes.wines.WineRepository")
    def test_wine_not_found_returns_404(self, mock_wine_cls, client):
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = None

        resp = client.post("/api/wines/999/description")

        assert resp.status_code == 404

    @patch("src.agents.description_service.DescriptionService")
    @patch("src.api.routes.wines.WineRepository")
    def test_successful_generation(self, mock_wine_cls, mock_desc_cls, client):
        wine = _make_wine(description=None)
        updated_wine = _make_wine(description="Generated desc.", drink_from_year=2024, drink_to_year=2032)

        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.side_effect = [wine, updated_wine]

        mock_service = MagicMock()
        mock_desc_cls.return_value = mock_service
        mock_service.aget_wine_description = AsyncMock(return_value="Generated desc.")

        resp = client.post("/api/wines/1/description")

        assert resp.status_code == 200
        body = DescriptionResponse(**resp.json())
        assert body.success is True
        assert body.description == "Generated desc."
        assert body.drink_from_year == 2024
        assert body.drink_to_year == 2032

    @patch("src.agents.description_service.DescriptionService")
    @patch("src.api.routes.wines.WineRepository")
    def test_generation_returns_none(self, mock_wine_cls, mock_desc_cls, client):
        wine = _make_wine(description=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        mock_service = MagicMock()
        mock_desc_cls.return_value = mock_service
        mock_service.aget_wine_description = AsyncMock(return_value=None)

        resp = client.post("/api/wines/1/description")

        # None description now raises 502 Bad Gateway
        assert resp.status_code == 502

    @patch("src.agents.description_service.DescriptionService")
    @patch("src.api.routes.wines.WineRepository")
    def test_generation_exception_handled(self, mock_wine_cls, mock_desc_cls, client):
        wine = _make_wine(description=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        mock_desc_cls.side_effect = RuntimeError("Authorization: Bearer synthetic-cloud-secret")

        resp = client.post("/api/wines/1/description")

        # Exceptions now raise 502 Bad Gateway
        assert resp.status_code == 502
        assert resp.json()["detail"] == "Description generation failed"
        assert "synthetic-cloud-secret" not in resp.text

    @patch("src.agents.description_service.DescriptionService")
    @patch("src.api.routes.wines.WineRepository")
    def test_request_body_flags_passed(self, mock_wine_cls, mock_desc_cls, client):
        wine = _make_wine(description=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        mock_service = MagicMock()
        mock_desc_cls.return_value = mock_service
        mock_service.aget_wine_description = AsyncMock(return_value="Desc")

        resp = client.post(
            "/api/wines/1/description",
            json={"use_rag_context": False, "use_web_search": True},
        )

        assert resp.status_code == 200
        # Verify the service was created with the right flags
        call_kwargs = mock_desc_cls.call_args.kwargs
        assert call_kwargs["use_rag_context"] is False
        assert call_kwargs["use_web_search"] is True
        assert call_kwargs["wine_repo"] is wine_repo
        assert "producer_repo" in call_kwargs

    @patch("src.agents.description_service.DescriptionService")
    @patch("src.api.routes.wines.WineRepository")
    def test_no_body_defaults_to_rag_true(self, mock_wine_cls, mock_desc_cls, client):
        wine = _make_wine(description=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        mock_service = MagicMock()
        mock_desc_cls.return_value = mock_service
        mock_service.aget_wine_description = AsyncMock(return_value="Desc")

        resp = client.post("/api/wines/1/description")

        call_kwargs = mock_desc_cls.call_args.kwargs
        assert call_kwargs["use_rag_context"] is True
        assert call_kwargs["use_web_search"] is True
        assert call_kwargs["wine_repo"] is wine_repo
        assert "producer_repo" in call_kwargs

    @patch("src.agents.description_service.DescriptionService")
    @patch("src.api.routes.wines.WineRepository")
    def test_empty_body_keeps_web_search_enabled(self, mock_wine_cls, mock_desc_cls, client):
        wine = _make_wine(description=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.return_value = wine

        mock_service = MagicMock()
        mock_desc_cls.return_value = mock_service
        mock_service.aget_wine_description = AsyncMock(return_value="Desc")

        resp = client.post("/api/wines/1/description", json={})

        assert resp.status_code == 200
        call_kwargs = mock_desc_cls.call_args.kwargs
        assert call_kwargs["use_rag_context"] is True
        assert call_kwargs["use_web_search"] is True
        assert call_kwargs["wine_repo"] is wine_repo
        assert "producer_repo" in call_kwargs

    @patch("src.agents.description_service.DescriptionService")
    @patch("src.api.routes.wines.WineRepository")
    def test_description_route_logs_web_search_flags(self, mock_wine_cls, mock_desc_cls, client, caplog):
        wine = _make_wine(description=None)
        wine_repo = MagicMock()
        mock_wine_cls.return_value = wine_repo
        wine_repo.get_by_id.side_effect = [wine, wine]

        mock_service = MagicMock()
        mock_service.use_web_search = True
        mock_service.aget_wine_description = AsyncMock(return_value="Desc")
        mock_desc_cls.return_value = mock_service

        with caplog.at_level("INFO"):
            resp = client.post("/api/wines/1/description", json={"use_rag_context": True, "use_web_search": True})

        assert resp.status_code == 200
        assert "Description request received: route=/api/wines/1/description" in caplog.text
        assert (
            "Description service initialized: route=/api/wines/1/description effective_use_web_search=True"
            in caplog.text
        )


class TestAsyncDescriptionBoundary:
    """Routes exercise the real async DescriptionService integration boundary."""

    @patch("src.api.routes.wines.ProducerRepository")
    @patch("src.api.routes.wines.WineRepository")
    def test_wine_route_awaits_lifespan_retriever(
        self,
        mock_wine_cls,
        mock_producer_cls,
        client,
    ) -> None:
        """The wine route uses async retrieval, reranking, model I/O, and persistence."""
        from src.agents.description_service import WineAnalysis
        from src.api.main import app

        wine = _make_wine(description=None, drink_from_year=None, drink_to_year=None)
        wine_repo = MagicMock()
        wine_repo.get_by_id.side_effect = [wine, wine]
        wine_repo.update.return_value = True
        mock_wine_cls.return_value = wine_repo
        mock_producer_cls.return_value = MagicMock()

        model = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=WineAnalysis(description="Grounded wine description."))
        model.with_structured_output.return_value = structured
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(
            return_value=[{"document": "Lifespan wine evidence", "metadata": {}, "distance": 0.1}]
        )
        reranker = MagicMock()
        reranker.arerank = AsyncMock(side_effect=lambda _query, documents, top_k: documents[:top_k])
        app.state.cloud_model = model
        app.state.retriever = retriever
        app.state.reranker = reranker

        response = client.post(
            "/api/wines/1/description",
            json={"use_rag_context": True, "use_web_search": False},
        )

        assert response.status_code == 200
        assert response.json()["description"] == "Grounded wine description."
        retriever.aretrieve.assert_awaited_once()
        retriever.retrieve.assert_not_called()
        reranker.arerank.assert_awaited_once()
        structured.ainvoke.assert_awaited_once()
        assert "Lifespan wine evidence" in structured.ainvoke.await_args.args[0]

    @patch("src.api.routes.wines.ProducerRepository")
    @patch("src.api.routes.wines.WineRepository")
    def test_producer_route_awaits_lifespan_retriever(
        self,
        mock_wine_cls,
        mock_producer_cls,
        client,
    ) -> None:
        """The producer route awaits async retrieval and plain-text generation."""
        from langchain_core.messages import AIMessage

        from src.api.main import app

        wine_repo = MagicMock()
        wine_repo.get_by_id.return_value = _make_wine(description=None)
        mock_wine_cls.return_value = wine_repo
        producer_repo = MagicMock()
        producer_repo.get_by_id.return_value = Producer(
            id=10,
            name="Domaine Test",
            country="France",
            region="Burgundy",
            description="Cached description.",
        )
        producer_repo.update.return_value = True
        mock_producer_cls.return_value = producer_repo

        model = MagicMock()
        model.with_structured_output.return_value = MagicMock()
        model.ainvoke = AsyncMock(return_value=AIMessage(content="Grounded producer description."))
        retriever = MagicMock()
        retriever.aretrieve = AsyncMock(
            return_value=[{"document": "Lifespan producer evidence", "metadata": {}, "distance": 0.1}]
        )
        reranker = MagicMock()
        reranker.arerank = AsyncMock(side_effect=lambda _query, documents, top_k: documents[:top_k])
        app.state.cloud_model = model
        app.state.retriever = retriever
        app.state.reranker = reranker

        response = client.post(
            "/api/wines/1/producer-description",
            json={"use_rag_context": True, "use_web_search": False},
        )

        assert response.status_code == 200
        assert response.json()["description"] == "Grounded producer description."
        retriever.aretrieve.assert_awaited_once()
        retriever.retrieve.assert_not_called()
        model.ainvoke.assert_awaited_once()
        assert "Lifespan producer evidence" in model.ainvoke.await_args.args[0]

    @patch("src.api.routes.wines.ProducerRepository")
    @patch("src.api.routes.wines.WineRepository")
    def test_model_failure_returns_redacted_502_without_retry(
        self,
        mock_wine_cls,
        mock_producer_cls,
        client,
    ) -> None:
        """A provider failure is redacted and never triggers a second model call."""
        from src.api.main import app

        wine_repo = MagicMock()
        wine_repo.get_by_id.return_value = _make_wine(description=None)
        mock_wine_cls.return_value = wine_repo
        mock_producer_cls.return_value = MagicMock()
        model = MagicMock()
        structured = MagicMock()
        structured.ainvoke = AsyncMock(side_effect=RuntimeError("Bearer synthetic-cloud-secret"))
        model.with_structured_output.return_value = structured
        app.state.cloud_model = model
        app.state.retriever = None
        app.state.reranker = None

        response = client.post(
            "/api/wines/1/description",
            json={"use_rag_context": False, "use_web_search": False},
        )

        assert response.status_code == 502
        assert response.json()["detail"] == "LLM failed to generate a description"
        assert "synthetic-cloud-secret" not in response.text
        structured.ainvoke.assert_awaited_once()
        wine_repo.update.assert_not_called()
