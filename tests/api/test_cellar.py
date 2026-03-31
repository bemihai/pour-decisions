"""Tests for cellar API endpoints.

Uses FastAPI TestClient with patched repositories to avoid
hitting the real SQLite database.
"""
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.schemas.cellar import (
    CellarStatsResponse,
    ChartDataResponse,
    FilterOptions,
    InventoryResponse,
    SyncResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Create a FastAPI TestClient with the app."""
    from src.api.main import app
    return TestClient(app)


def _make_inventory_row(**overrides) -> dict:
    """Build a minimal inventory row dict with sensible defaults."""
    row = {
        "wine_id": 1,
        "wine_name": "Test Cuvee",
        "producer_name": "Domaine Test",
        "vintage": 2020,
        "wine_type": "Red",
        "varietal": "Pinot Noir",
        "country": "France",
        "region_name": "Burgundy",
        "quantity": 2,
        "personal_rating": 88,
        "community_rating": 4.2,
        "drink_index": None,
        "drink_from_year": 2023,
        "drink_to_year": 2030,
        "drink_window_source": "heuristic",
        "location": "Cellar",
        "bin": "A1",
        "purchase_price": 25.0,
        "currency": "EUR",
        "description": None,
        "producer_description": None,
        "producer_id": 10,
        "bottle_note": None,
        "last_tasted_date": None,
        "like_votes": 0,
        "like_percentage": None,
        "q_purchased": 3,
        "q_consumed": 1,
        "q_quantity": 2,
        "created_at": "2024-06-01 12:00:00",
        "status": "in_cellar",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# GET /api/cellar/inventory
# ---------------------------------------------------------------------------

def _make_filter_options(**overrides) -> dict:
    """Build minimal filter options dict for mocking get_inventory_filter_options."""
    opts = {
        "wine_types": ["Red", "White"],
        "countries": ["France", "Italy"],
        "locations": ["Cellar"],
        "producers": ["Domaine Test"],
        "min_vintage": 2015,
        "max_vintage": 2022,
    }
    opts.update(overrides)
    return opts


# ---------------------------------------------------------------------------
# GET /api/cellar/inventory
# ---------------------------------------------------------------------------

class TestGetInventory:
    """Tests for the inventory endpoint."""

    @patch("src.api.routes.cellar.BottleRepository")
    def test_returns_grouped_inventory(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_inventory.return_value = [
            _make_inventory_row(wine_id=1, quantity=1),
            _make_inventory_row(wine_id=1, quantity=2),
            _make_inventory_row(wine_id=2, wine_name="Other Wine", quantity=3),
        ]
        repo.get_inventory_filter_options.return_value = _make_filter_options()

        resp = client.get("/api/cellar/inventory")

        assert resp.status_code == 200
        body = InventoryResponse(**resp.json())
        assert body.total_wines == 2
        # wine_id=1 should have quantity 3 (1+2)
        item_1 = next(i for i in body.items if i.wine_id == 1)
        assert item_1.quantity == 3
        assert body.total_bottles == 6  # 3 + 3

    @patch("src.api.routes.cellar.BottleRepository")
    def test_filter_by_wine_type(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        # Repository does SQL filtering — mock returns only the matching row
        repo.get_inventory.return_value = [
            _make_inventory_row(wine_id=1, wine_type="Red"),
        ]
        repo.get_inventory_filter_options.return_value = _make_filter_options()

        resp = client.get("/api/cellar/inventory", params={"wine_type": "Red"})

        assert resp.status_code == 200
        body = InventoryResponse(**resp.json())
        assert body.total_wines == 1
        assert body.items[0].wine_type == "Red"
        # Verify the filter was forwarded to the repository
        repo.get_inventory.assert_called_once()
        call_kwargs = repo.get_inventory.call_args.kwargs
        assert call_kwargs.get("wine_type") == "Red"

    @patch("src.api.routes.cellar.BottleRepository")
    def test_filter_by_search(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        # Repository does SQL filtering — mock returns only the matching row
        repo.get_inventory.return_value = [
            _make_inventory_row(wine_id=1, wine_name="Grand Cru"),
        ]
        repo.get_inventory_filter_options.return_value = _make_filter_options()

        resp = client.get("/api/cellar/inventory", params={"search": "grand"})

        assert resp.status_code == 200
        body = InventoryResponse(**resp.json())
        assert body.total_wines == 1
        assert "Grand" in body.items[0].wine_name
        call_kwargs = repo.get_inventory.call_args.kwargs
        assert call_kwargs.get("search") == "grand"

    @patch("src.api.routes.cellar.BottleRepository")
    def test_filter_by_rating(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        # Repository does SQL filtering — mock returns only the matching row
        repo.get_inventory.return_value = [
            _make_inventory_row(wine_id=1, personal_rating=92),
        ]
        repo.get_inventory_filter_options.return_value = _make_filter_options()

        resp = client.get("/api/cellar/inventory", params={"rating_filter": "90+"})

        assert resp.status_code == 200
        body = InventoryResponse(**resp.json())
        assert body.total_wines == 1
        assert body.items[0].personal_rating >= 90
        call_kwargs = repo.get_inventory.call_args.kwargs
        assert call_kwargs.get("rating_filter") == "90+"

    @patch("src.api.routes.cellar.BottleRepository")
    def test_sort_by_vintage_desc(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_inventory.return_value = [
            _make_inventory_row(wine_id=1, vintage=2015),
            _make_inventory_row(wine_id=2, vintage=2022, wine_name="New"),
        ]
        repo.get_inventory_filter_options.return_value = _make_filter_options()

        resp = client.get("/api/cellar/inventory", params={"sort_by": "vintage_desc"})

        assert resp.status_code == 200
        body = InventoryResponse(**resp.json())
        assert body.items[0].vintage == 2022

    @patch("src.api.routes.cellar.BottleRepository")
    def test_filter_options_always_from_full_set(self, mock_repo_cls, client):
        """Filter options come from get_inventory_filter_options (full set), not the filtered result."""
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        # Filtered result has only Red
        repo.get_inventory.return_value = [
            _make_inventory_row(wine_id=1, wine_type="Red"),
        ]
        # But filter options reflect the full inventory (both types)
        repo.get_inventory_filter_options.return_value = _make_filter_options(
            wine_types=["Red", "White"]
        )

        resp = client.get("/api/cellar/inventory", params={"wine_type": "Red"})

        body = InventoryResponse(**resp.json())
        assert "White" in body.filter_options.wine_types
        assert "Red" in body.filter_options.wine_types

    @patch("src.api.routes.cellar.BottleRepository")
    def test_empty_cellar(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_inventory.return_value = []
        repo.get_inventory_filter_options.return_value = _make_filter_options(
            wine_types=[], countries=[], locations=[], producers=[],
            min_vintage=2000, max_vintage=2025,
        )

        resp = client.get("/api/cellar/inventory")

        assert resp.status_code == 200
        body = InventoryResponse(**resp.json())
        assert body.total_wines == 0
        assert body.total_bottles == 0
        assert body.items == []


# ---------------------------------------------------------------------------
# GET /api/cellar/filters
# ---------------------------------------------------------------------------

class TestGetFilters:

    @patch("src.api.routes.cellar.BottleRepository")
    def test_returns_filter_options(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_inventory_filter_options.return_value = {
            "wine_types": ["Red", "White"],
            "countries": ["France", "Italy"],
            "locations": ["Cellar"],
            "producers": ["Domaine Test"],
            "min_vintage": 2018,
            "max_vintage": 2021,
        }

        resp = client.get("/api/cellar/filters")

        assert resp.status_code == 200
        opts = FilterOptions(**resp.json())
        assert "Red" in opts.wine_types
        assert "White" in opts.wine_types
        assert opts.min_vintage == 2018
        assert opts.max_vintage == 2021


# ---------------------------------------------------------------------------
# GET /api/cellar/stats
# ---------------------------------------------------------------------------

class TestGetStats:

    @patch("src.api.routes.cellar.StatsRepository")
    def test_returns_combined_stats(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_cellar_overview.return_value = {
            "total_bottles": 42,
            "unique_wines": 18,
            "by_type": [{"wine_type": "Red", "unique_wines": 12, "bottles": 30}],
            "by_country": [{"country": "France", "unique_wines": 10, "bottles": 25}],
        }
        repo.get_drinking_window_stats.return_value = {
            "ready_to_drink": 20,
            "to_hold": 15,
            "unknown": 7,
        }
        repo.get_cellar_value.return_value = {
            "by_currency": [{"currency": "EUR", "total_value": 1200, "wines_with_price": 30}],
            "bottles_without_price": 12,
        }

        resp = client.get("/api/cellar/stats")

        assert resp.status_code == 200
        body = CellarStatsResponse(**resp.json())
        assert body.overview.total_bottles == 42
        assert body.drinking_stats.ready_to_drink == 20
        assert body.value_stats.bottles_without_price == 12


# ---------------------------------------------------------------------------
# GET /api/cellar/charts
# ---------------------------------------------------------------------------

class TestGetCharts:

    @patch("src.api.routes.cellar.StatsRepository")
    def test_returns_chart_data(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_cellar_overview.return_value = {
            "total_bottles": 42,
            "unique_wines": 18,
            "by_type": [{"wine_type": "Red", "bottles": 30}],
            "by_country": [{"country": "France", "bottles": 25}],
        }
        repo.get_varietal_distribution.return_value = [{"varietal": "Pinot Noir", "bottles": 10}]
        repo.get_region_distribution.return_value = [{"region": "Burgundy", "bottles": 8}]
        repo.get_drinking_window_wines.return_value = {"ready_now": [], "drink_soon": [], "for_aging": []}
        repo.get_cellar_size_over_time.return_value = []
        repo.get_top_rated_wines.return_value = []

        resp = client.get("/api/cellar/charts")

        assert resp.status_code == 200
        body = ChartDataResponse(**resp.json())
        assert len(body.wine_type_distribution) == 1
        assert body.wine_type_distribution[0]["wine_type"] == "Red"


# ---------------------------------------------------------------------------
# POST /api/cellar/sync
# ---------------------------------------------------------------------------

class TestSyncCellarTracker:

    @patch("src.api.routes.cellar.os.getenv")
    def test_missing_credentials_returns_400(self, mock_getenv, client):
        mock_getenv.return_value = None

        resp = client.post("/api/cellar/sync")

        assert resp.status_code == 400
        assert "credentials" in resp.json()["detail"].lower()

    @patch("src.api.routes.cellar.os.getenv")
    def test_successful_sync(self, mock_getenv, client):
        mock_getenv.side_effect = lambda k: {"CELLAR_TRACKER_USERNAME": "user", "CELLAR_TRACKER_PASSWORD": "pass"}.get(k)

        mock_importer = MagicMock()
        mock_importer.import_all.return_value = {
            "wines_processed": 100,
            "wines_imported": 5,
            "bottles_processed": 200,
            "bottles_imported": 10,
            "producers_created": 2,
            "regions_created": 1,
            "errors": [],
        }

        with (
            patch("src.etl.cellartracker_importer.CellarTrackerImporter", return_value=mock_importer),
            patch("src.utils.utils.get_default_db_path", return_value="/tmp/test.db"),
        ):
            resp = client.post("/api/cellar/sync")

        assert resp.status_code == 200
        body = SyncResponse(**resp.json())
        assert body.success is True
        assert body.wines_imported == 5

    @patch("src.api.routes.cellar.os.getenv")
    def test_sync_failure_returns_error(self, mock_getenv, client):
        mock_getenv.side_effect = lambda k: {"CELLAR_TRACKER_USERNAME": "user", "CELLAR_TRACKER_PASSWORD": "pass"}.get(k)

        mock_importer = MagicMock()
        mock_importer.import_all.side_effect = RuntimeError("Connection failed")

        with (
            patch("src.etl.cellartracker_importer.CellarTrackerImporter", return_value=mock_importer),
            patch("src.utils.utils.get_default_db_path", return_value="/tmp/test.db"),
        ):
            resp = client.post("/api/cellar/sync")

        assert resp.status_code == 200
        body = SyncResponse(**resp.json())
        assert body.success is False
        assert "Connection failed" in body.error_message



