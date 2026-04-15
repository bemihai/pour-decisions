"""Tests for the FastAPI application shell and health endpoint."""
import pytest
from fastapi.testclient import TestClient


def _populate_state(app):
    """Populate all app.state attributes that lifespan normally sets."""
    app.state.local_model = None
    app.state.cloud_model = None
    app.state.model = None
    app.state.local_intelligent_agent = None
    app.state.cloud_intelligent_agent = None
    app.state.intelligent_agent = None
    app.state.local_keyword_agent = None
    app.state.cloud_keyword_agent = None
    app.state.keyword_agent = None
    app.state.retriever = None
    app.state.reranker = None


@pytest.fixture()
def client():
    """Create a FastAPI TestClient with pre-populated app state."""
    from src.api.main import app

    _populate_state(app)
    return TestClient(app)


class TestHealthCheck:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "resources" in body

    def test_health_shows_resource_status(self, client):
        resp = client.get("/health")

        resources = resp.json()["resources"]
        assert "local_model" in resources
        assert "cloud_model" in resources
        assert "local_intelligent_agent" in resources
        assert "cloud_intelligent_agent" in resources
        assert "local_keyword_agent" in resources
        assert "cloud_keyword_agent" in resources
        assert "retriever" in resources
        assert "reranker" in resources


class TestAppConfiguration:

    def test_openapi_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Pour Decisions API"
        assert schema["info"]["version"] == "1.0.0"

    def test_all_route_prefixes_registered(self, client):
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        path_keys = list(paths.keys())

        # Verify all four routers are registered
        assert any(p.startswith("/api/chat") for p in path_keys), "Chat routes missing"
        assert any(p.startswith("/api/cellar") for p in path_keys), "Cellar routes missing"
        assert any(p.startswith("/api/taste-profile") for p in path_keys), "Taste profile routes missing"
        assert any(p.startswith("/api/wines") for p in path_keys), "Wines routes missing"
        assert "/health" in path_keys, "Health endpoint missing"

    def test_cors_headers_for_localhost(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS preflight should succeed
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

