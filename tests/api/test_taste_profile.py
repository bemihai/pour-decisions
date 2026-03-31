"""Tests for taste profile API endpoints.

Uses FastAPI TestClient with patched repositories and DB connections
to avoid hitting the real SQLite database.
"""
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.schemas.taste_profile import (
    AppellationsResponse,
    ConsumedWinesResponse,
    CountriesResponse,
    ProducersResponse,
    RatingDistributionResponse,
    RatingTrendsResponse,
    RegionsResponse,
    TasteOverviewResponse,
    VarietalsResponse,
    VintagesResponse,
    WineTypesResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Create a FastAPI TestClient with the app."""
    from src.api.main import app
    return TestClient(app)


def _mock_db_rows(rows: list[dict]):
    """Build a context-manager mock for ``get_db_connection()`` that returns *rows*.

    Each dict in *rows* must support item access (``row["col"]``). The mock
    cursor returns them from ``fetchall()``.
    """
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    @contextmanager
    def _ctx(*args, **kwargs):
        yield mock_conn

    return _ctx


# ---------------------------------------------------------------------------
# GET /api/taste-profile/overview
# ---------------------------------------------------------------------------

class TestGetOverview:

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_returns_overview_metrics(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_rating_statistics.return_value = {
            "overall": {"avg_rating": 87.5, "wines_rated": 50},
            "distribution": [
                {"rating_range": "80-89", "count": 30},
                {"rating_range": "90-100", "count": 15},
            ],
        }
        repo.get_wine_type_stats.return_value = [
            {"wine_type": "Red", "wines_tasted": 30},
            {"wine_type": "White", "wines_tasted": 20},
        ]

        resp = client.get("/api/taste-profile/overview")

        assert resp.status_code == 200
        body = TasteOverviewResponse(**resp.json())
        assert body.avg_rating == 87.5
        assert body.wines_rated == 50
        assert body.favorite_type == "Red"
        assert body.highly_rated_count == 15
        assert body.highly_rated_pct == 30.0  # 15/50 * 100

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_no_wines_returns_defaults(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_rating_statistics.return_value = {
            "overall": {"avg_rating": 0, "wines_rated": 0},
            "distribution": [],
        }
        repo.get_wine_type_stats.return_value = []

        resp = client.get("/api/taste-profile/overview")

        assert resp.status_code == 200
        body = TasteOverviewResponse(**resp.json())
        assert body.wines_rated == 0
        assert body.favorite_type == "N/A"
        assert body.highly_rated_pct == 0.0


# ---------------------------------------------------------------------------
# GET /api/taste-profile/rating-distribution
# ---------------------------------------------------------------------------

class TestGetRatingDistribution:

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_returns_buckets_with_colors(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            {"personal_rating": 82},
            {"personal_rating": 85},
            {"personal_rating": 88},
            {"personal_rating": 91},
            {"personal_rating": 95},
        ])

        resp = client.get("/api/taste-profile/rating-distribution")

        assert resp.status_code == 200
        body = RatingDistributionResponse(**resp.json())
        assert body.total == 5
        assert len(body.buckets) > 0
        # Every bucket should have a non-empty color string
        for b in body.buckets:
            assert b.color
            assert b.count > 0

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_empty_ratings(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([])

        resp = client.get("/api/taste-profile/rating-distribution")

        assert resp.status_code == 200
        body = RatingDistributionResponse(**resp.json())
        assert body.total == 0
        assert body.buckets == []

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_single_bucket(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            {"personal_rating": 92},
            {"personal_rating": 93},
        ])

        resp = client.get("/api/taste-profile/rating-distribution")

        body = RatingDistributionResponse(**resp.json())
        assert body.total == 2
        assert len(body.buckets) == 1
        assert body.buckets[0].range == "90-94"


# ---------------------------------------------------------------------------
# GET /api/taste-profile/wine-types
# ---------------------------------------------------------------------------

class TestGetWineTypes:

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_returns_types(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_wine_type_stats.return_value = [
            {"wine_type": "Red", "wines_tasted": 30, "avg_rating": 88.0, "highest_rating": 96},
            {"wine_type": "White", "wines_tasted": 15, "avg_rating": 85.5, "highest_rating": 93},
        ]

        resp = client.get("/api/taste-profile/wine-types")

        assert resp.status_code == 200
        body = WineTypesResponse(**resp.json())
        assert len(body.types) == 2
        assert body.types[0].wine_type == "Red"
        assert body.types[0].avg_rating == 88.0

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_empty_types(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_wine_type_stats.return_value = []

        resp = client.get("/api/taste-profile/wine-types")

        assert resp.status_code == 200
        body = WineTypesResponse(**resp.json())
        assert body.types == []


# ---------------------------------------------------------------------------
# GET /api/taste-profile/varietals
# ---------------------------------------------------------------------------

class TestGetVarietals:

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_returns_varietals(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_varietal_preferences.return_value = [
            {"varietal": "Pinot Noir", "wines_tasted": 12, "avg_rating": 89.0, "highest_rating": 95},
        ]

        resp = client.get("/api/taste-profile/varietals")

        assert resp.status_code == 200
        body = VarietalsResponse(**resp.json())
        assert len(body.varietals) == 1
        assert body.varietals[0].varietal == "Pinot Noir"

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_custom_limit(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_varietal_preferences.return_value = []

        client.get("/api/taste-profile/varietals", params={"limit": 5})

        repo.get_varietal_preferences.assert_called_once_with(limit=5)

    def test_invalid_limit_rejected(self, client):
        resp = client.get("/api/taste-profile/varietals", params={"limit": 0})
        assert resp.status_code == 422  # FastAPI validation error

        resp = client.get("/api/taste-profile/varietals", params={"limit": 100})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/taste-profile/producers
# ---------------------------------------------------------------------------

class TestGetProducers:

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_returns_producers(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_producer_preferences.return_value = [
            {"producer_name": "Domaine Romanee", "country": "France", "wines_tasted": 8, "avg_rating": 92.0, "highest_rating": 97},
        ]

        resp = client.get("/api/taste-profile/producers")

        assert resp.status_code == 200
        body = ProducersResponse(**resp.json())
        assert len(body.producers) == 1
        assert body.producers[0].country == "France"

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_default_limit_is_5(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_producer_preferences.return_value = []

        client.get("/api/taste-profile/producers")

        repo.get_producer_preferences.assert_called_once_with(limit=5)


# ---------------------------------------------------------------------------
# GET /api/taste-profile/regions
# ---------------------------------------------------------------------------

class TestGetRegions:

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_returns_regions(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_region_preferences.return_value = [
            {"region_name": "Burgundy", "country": "France", "wines_tasted": 10, "avg_rating": 90.0, "highest_rating": 96},
        ]

        resp = client.get("/api/taste-profile/regions")

        assert resp.status_code == 200
        body = RegionsResponse(**resp.json())
        assert len(body.regions) == 1
        assert body.regions[0].region_name == "Burgundy"


# ---------------------------------------------------------------------------
# GET /api/taste-profile/countries
# ---------------------------------------------------------------------------

class TestGetCountries:

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_returns_countries(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            {"country": "France", "wines_tasted": 25, "avg_rating": 88.0, "highest_rating": 96},
            {"country": "Italy", "wines_tasted": 15, "avg_rating": 86.0, "highest_rating": 93},
        ])

        resp = client.get("/api/taste-profile/countries")

        assert resp.status_code == 200
        body = CountriesResponse(**resp.json())
        assert len(body.countries) == 2
        assert body.countries[0].country == "France"

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_empty_countries(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([])

        resp = client.get("/api/taste-profile/countries")

        assert resp.status_code == 200
        body = CountriesResponse(**resp.json())
        assert body.countries == []


# ---------------------------------------------------------------------------
# GET /api/taste-profile/vintages
# ---------------------------------------------------------------------------

class TestGetVintages:

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_returns_vintages(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            {"vintage": 2018, "wines_tasted": 5, "avg_rating": 91.0, "highest_rating": 95},
        ])

        resp = client.get("/api/taste-profile/vintages")

        assert resp.status_code == 200
        body = VintagesResponse(**resp.json())
        assert len(body.vintages) == 1
        assert body.vintages[0].vintage == 2018


# ---------------------------------------------------------------------------
# GET /api/taste-profile/appellations
# ---------------------------------------------------------------------------

class TestGetAppellations:

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_returns_appellations(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            {"appellation": "Pauillac", "country": "France", "wines_tasted": 4, "avg_rating": 93.0, "highest_rating": 97},
        ])

        resp = client.get("/api/taste-profile/appellations")

        assert resp.status_code == 200
        body = AppellationsResponse(**resp.json())
        assert len(body.appellations) == 1
        assert body.appellations[0].appellation == "Pauillac"

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_custom_limit(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([])

        resp = client.get("/api/taste-profile/appellations", params={"limit": 3})

        assert resp.status_code == 200
        body = AppellationsResponse(**resp.json())
        assert body.appellations == []


# ---------------------------------------------------------------------------
# GET /api/taste-profile/rating-trends
# ---------------------------------------------------------------------------

class TestGetRatingTrends:

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_returns_improving_trend(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_rating_timeline.return_value = [
            {"month": "2025-01", "avg_rating": 85.0, "wines_count": 3},
            {"month": "2025-02", "avg_rating": 87.0, "wines_count": 4},
            {"month": "2025-03", "avg_rating": 89.0, "wines_count": 5},
        ]

        resp = client.get("/api/taste-profile/rating-trends")

        assert resp.status_code == 200
        body = RatingTrendsResponse(**resp.json())
        assert len(body.points) == 3
        assert body.trend == "improving"

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_declining_trend(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_rating_timeline.return_value = [
            {"month": "2025-01", "avg_rating": 90.0, "wines_count": 3},
            {"month": "2025-02", "avg_rating": 85.0, "wines_count": 4},
        ]

        resp = client.get("/api/taste-profile/rating-trends")

        body = RatingTrendsResponse(**resp.json())
        assert body.trend == "declining"

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_stable_trend(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_rating_timeline.return_value = [
            {"month": "2025-01", "avg_rating": 88.0, "wines_count": 3},
            {"month": "2025-02", "avg_rating": 88.0, "wines_count": 4},
        ]

        resp = client.get("/api/taste-profile/rating-trends")

        body = RatingTrendsResponse(**resp.json())
        assert body.trend == "stable"

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_insufficient_data_returns_empty(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_rating_timeline.return_value = [
            {"month": "2025-01", "avg_rating": 88.0, "wines_count": 3},
        ]

        resp = client.get("/api/taste-profile/rating-trends")

        body = RatingTrendsResponse(**resp.json())
        assert body.points == []
        assert body.trend is None

    @patch("src.api.routes.taste_profile.StatsRepository")
    def test_caps_at_12_months(self, mock_repo_cls, client):
        repo = MagicMock()
        mock_repo_cls.return_value = repo
        repo.get_rating_timeline.return_value = [
            {"month": f"2024-{m:02d}", "avg_rating": 85.0 + m, "wines_count": m}
            for m in range(1, 16)  # 15 months
        ]

        resp = client.get("/api/taste-profile/rating-trends")

        body = RatingTrendsResponse(**resp.json())
        assert len(body.points) == 12


# ---------------------------------------------------------------------------
# GET /api/taste-profile/consumed
# ---------------------------------------------------------------------------

def _make_consumed_row(**overrides) -> dict:
    """Build a minimal consumed wine row dict with sensible defaults."""
    row = {
        "wine_id": 1,
        "bottle_id": 10,
        "wine_name": "Test Cuvee",
        "wine_type": "Red",
        "vintage": 2020,
        "varietal": "Pinot Noir",
        "producer_name": "Domaine Test",
        "country": "France",
        "region_name": "Burgundy",
        "personal_rating": 88,
        "community_rating": 4.2,
        "tasting_notes": "Lovely cherry notes.",
        "last_tasted_date": "2025-06-01",
        "consumed_date": "2025-06-01",
    }
    row.update(overrides)
    return row


class TestGetConsumed:

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_returns_consumed_with_filter_options(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, wine_type="Red"),
            _make_consumed_row(wine_id=2, wine_type="White", wine_name="Blanc", producer_name="Other"),
        ])

        resp = client.get("/api/taste-profile/consumed")

        assert resp.status_code == 200
        body = ConsumedWinesResponse(**resp.json())
        assert body.total == 2
        assert len(body.items) == 2
        assert "Red" in body.filter_options.wine_types
        assert "White" in body.filter_options.wine_types

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_filter_by_wine_type(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, wine_type="Red"),
            _make_consumed_row(wine_id=2, wine_type="White", wine_name="Blanc"),
        ])

        resp = client.get("/api/taste-profile/consumed", params={"wine_type": "Red"})

        body = ConsumedWinesResponse(**resp.json())
        assert body.total == 1
        assert body.items[0].wine_type == "Red"
        # filter_options should still show both types (from unfiltered set)
        assert "White" in body.filter_options.wine_types

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_filter_by_rating(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, personal_rating=92),
            _make_consumed_row(wine_id=2, personal_rating=75),
        ])

        resp = client.get("/api/taste-profile/consumed", params={"rating_filter": "90+"})

        body = ConsumedWinesResponse(**resp.json())
        assert body.total == 1
        assert body.items[0].personal_rating >= 90

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_filter_rated_only(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, personal_rating=88),
            _make_consumed_row(wine_id=2, personal_rating=None),
        ])

        resp = client.get("/api/taste-profile/consumed", params={"rating_filter": "rated"})

        body = ConsumedWinesResponse(**resp.json())
        assert body.total == 1
        assert body.items[0].personal_rating is not None

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_filter_unrated(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, personal_rating=88),
            _make_consumed_row(wine_id=2, personal_rating=None),
        ])

        resp = client.get("/api/taste-profile/consumed", params={"rating_filter": "unrated"})

        body = ConsumedWinesResponse(**resp.json())
        assert body.total == 1
        assert body.items[0].personal_rating is None

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_search(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, wine_name="Grand Cru"),
            _make_consumed_row(wine_id=2, wine_name="Village"),
        ])

        resp = client.get("/api/taste-profile/consumed", params={"search": "grand"})

        body = ConsumedWinesResponse(**resp.json())
        assert body.total == 1
        assert "Grand" in body.items[0].wine_name

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_sort_by_rating_desc(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, personal_rating=80),
            _make_consumed_row(wine_id=2, personal_rating=95, wine_name="Top"),
        ])

        resp = client.get("/api/taste-profile/consumed", params={"sort_by": "rating_desc"})

        body = ConsumedWinesResponse(**resp.json())
        assert body.items[0].personal_rating == 95

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_limit(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=i) for i in range(30)
        ])

        resp = client.get("/api/taste-profile/consumed", params={"limit": 5})

        body = ConsumedWinesResponse(**resp.json())
        assert len(body.items) == 5
        assert body.total == 30

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_empty_consumed(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([])

        resp = client.get("/api/taste-profile/consumed")

        assert resp.status_code == 200
        body = ConsumedWinesResponse(**resp.json())
        assert body.total == 0
        assert body.items == []

    @patch("src.api.routes.taste_profile.get_db_connection")
    def test_includes_rating_description(self, mock_conn, client):
        mock_conn.side_effect = _mock_db_rows([
            _make_consumed_row(wine_id=1, personal_rating=92),
        ])

        resp = client.get("/api/taste-profile/consumed")

        body = ConsumedWinesResponse(**resp.json())
        assert body.items[0].rating_description is not None
        assert body.items[0].rating_description != ""

