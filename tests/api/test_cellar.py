"""Tests for cellar API endpoints.

Uses FastAPI TestClient with patched repositories to avoid
hitting the real SQLite database.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.agents.drinking_window_service import DrinkingReadiness
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
        assert body.items[0].personal_rating >= 90  # ty:ignore[unsupported-operator]
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

    @patch(
        "src.api.routes.cellar.compute_drinking_readiness",
        return_value=DrinkingReadiness(72.0, "ready"),
    )
    @patch("src.api.routes.cellar.BottleRepository")
    def test_drink_index_computed_once_per_grouped_row(self, mock_repo_cls, mock_compute, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_inventory.return_value = [
            _make_inventory_row(wine_id=1, quantity=1, drink_index=None, drink_from_year=2020, drink_to_year=2030),
            _make_inventory_row(wine_id=1, quantity=2, drink_index=None, drink_from_year=2020, drink_to_year=2030),
        ]
        repo.get_inventory_filter_options.return_value = _make_filter_options()

        resp = client.get("/api/cellar/inventory", params={"sort_by": "drink_desc"})

        assert resp.status_code == 200
        body = InventoryResponse(**resp.json())
        assert body.items[0].drink_index == 72.0
        assert body.items[0].drink_status == "ready"
        mock_compute.assert_called_once_with(2020, 2030)

    @patch("src.api.routes.cellar.BottleRepository")
    def test_reported_window_returns_canonical_index_and_status(self, mock_repo_cls, client):
        """Inventory ignores persisted CT availability and returns window-derived readiness."""
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_inventory.return_value = [
            _make_inventory_row(
                wine_id=1,
                wine_name="Crama Bauer Cabernet Franc Clàsic 2019",
                drink_index=0.375,
                drink_from_year=2025,
                drink_to_year=2027,
                drink_window_source="cellar_tracker",
            )
        ]
        repo.get_inventory_filter_options.return_value = _make_filter_options()

        with patch("src.agents.drinking_window_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.year = 2026
            resp = client.get("/api/cellar/inventory")

        assert resp.status_code == 200
        item = InventoryResponse(**resp.json()).items[0]
        assert item.drink_index == 100.0
        assert item.drink_status == "ready"

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


class TestGetDrinkNext:
    """Tests for backend-owned Drink Next eligibility and ordering."""

    @patch("src.api.routes.cellar.BottleRepository")
    def test_community_totals_are_not_summed_across_bottle_rows(
        self, mock_repo_cls: MagicMock, client: TestClient
    ) -> None:
        """Keep wine-level totals separate from aggregated personal quantity."""
        mock_repo_cls.return_value.get_inventory.return_value = [
            _make_inventory_row(quantity=2, q_quantity=600, q_purchased=1000),
            _make_inventory_row(quantity=3, q_quantity=600, q_purchased=1000, location="Rack"),
        ]
        with patch("src.agents.drinking_window_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.year = 2026
            response = client.get("/api/cellar/drink-next")

        assert response.status_code == 200
        body = response.json()
        assert body["total_wines"] == 1
        item = body["by_type"]["Red"][0]
        assert item["quantity"] == 5
        assert item["q_quantity"] == 600
        assert item["q_purchased"] == 1000
        assert item["region_name"] == "Burgundy"
        assert item["location"] == "Cellar"

    @patch("src.api.routes.cellar.BottleRepository")
    def test_missing_community_totals_default_to_zero(self, mock_repo_cls: MagicMock, client: TestClient) -> None:
        """Return zero defaults when community fields are unavailable."""
        row = _make_inventory_row()
        del row["q_quantity"]
        del row["q_purchased"]
        mock_repo_cls.return_value.get_inventory.return_value = [row]
        with patch("src.agents.drinking_window_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.year = 2026
            response = client.get("/api/cellar/drink-next")

        assert response.status_code == 200
        item = response.json()["by_type"]["Red"][0]
        assert item["q_quantity"] == 0
        assert item["q_purchased"] == 0

    @patch("src.api.routes.cellar.BottleRepository")
    def test_returns_only_ready_phase_with_canonical_readiness(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_inventory.return_value = [
            _make_inventory_row(
                wine_id=1,
                wine_name="Ready wine",
                drink_index=0.375,
                drink_from_year=2025,
                drink_to_year=2027,
                drink_window_source="cellar_tracker",
            ),
            _make_inventory_row(
                wine_id=2,
                wine_name="Hold wine",
                drink_index=99.0,
                drink_from_year=2030,
                drink_to_year=2040,
            ),
            _make_inventory_row(
                wine_id=3,
                wine_name="Past wine",
                drink_index=99.0,
                drink_from_year=2010,
                drink_to_year=2015,
            ),
        ]

        with patch("src.agents.drinking_window_service.datetime") as mock_datetime:
            mock_datetime.now.return_value.year = 2026
            resp = client.get("/api/cellar/drink-next")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_wines"] == 1
        assert [wine["wine_name"] for wine in body["by_type"]["Red"]] == ["Ready wine"]
        assert body["by_type"]["Red"][0]["drink_index"] == 100.0
        assert body["by_type"]["Red"][0]["drink_status"] == "ready"


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
        assert "Connection failed" in body.error_message  # ty:ignore[unsupported-operator]


@pytest.mark.parametrize(
    ("sort_by", "expected"),
    [
        ("drink_desc", [2, 3, 1, 5, 6, 4, 8, 9, 7, 10, 11, 12]),
        ("drink_asc", [7, 8, 9, 4, 5, 6, 1, 2, 3, 10, 11, 12]),
    ],
)
@patch("src.api.routes.cellar.BottleRepository")
def test_inventory_drinking_order(
    mock_repo_cls: MagicMock, client: TestClient, sort_by: str, expected: list[int]
) -> None:
    """Sort every status by end year with stable ties and incomplete windows last."""
    windows = [
        (1, 2010, 2025),
        (2, 2010, 2024),
        (3, 2011, 2024),
        (4, 2020, 2030),
        (5, 2020, 2027),
        (6, 2021, 2027),
        (7, 2028, 2040),
        (8, 2027, 2035),
        (9, 2029, 2035),
        (10, None, None),
        (11, None, 2040),
        (12, 2020, None),
    ]
    rows = [
        _make_inventory_row(wine_id=wine_id, drink_from_year=start, drink_to_year=end)
        for wine_id, start, end in reversed(windows)
    ]
    # Synthetic reproduction: Ready Avincis must follow Hold Povero in Drink Later.
    next(row for row in rows if row["wine_id"] == 4).update(wine_name="Avincis Alutus Primus")
    next(row for row in rows if row["wine_id"] == 7).update(wine_name="Povero Barolo")
    rows.append({**rows[0], "quantity": 3})
    repo = mock_repo_cls.return_value
    repo.get_inventory.return_value = rows
    repo.get_inventory_filter_options.return_value = _make_filter_options()
    with patch("src.agents.drinking_window_service.datetime") as clock:
        clock.now.return_value.year = 2026
        response = client.get("/api/cellar/inventory", params={"sort_by": sort_by, "wine_type": "Red"})
    assert response.status_code == 200
    body = response.json()
    assert [item["wine_id"] for item in body["items"]] == expected
    assert body["total_wines"] == 12
    assert body["total_bottles"] == 27
    assert {item["wine_id"]: item["quantity"] for item in body["items"]} == {
        wine_id: 5 if wine_id == 12 else 2 for wine_id in range(1, 13)
    }
    assert repo.get_inventory.call_args.kwargs["wine_type"] == "Red"
