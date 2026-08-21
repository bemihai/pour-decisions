"""Tests for the FastAPI application shell and health endpoint."""
import asyncio
from types import SimpleNamespace

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
    app.state.tool_registry = None
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

        # Verify all five routers are registered
        assert any(p.startswith("/api/chat") for p in path_keys), "Chat routes missing"
        assert any(p.startswith("/api/cellar") for p in path_keys), "Cellar routes missing"
        assert any(p.startswith("/api/taste-profile") for p in path_keys), "Taste profile routes missing"
        assert any(p.startswith("/api/wines") for p in path_keys), "Wines routes missing"
        assert "/api/tools" in path_keys, "Tools route missing"
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


def test_lifespan_initializes_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifespan startup should initialize observability before loading resources."""
    from src.api import main

    cfg = SimpleNamespace(
        observability=SimpleNamespace(
            enabled=False,
            provider="none",
        ),
        model=SimpleNamespace(
            provider="ollama",
            name="gemma3:4b",
            hybrid_tool_calling=False,
        ),
    )

    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "build_tool_registry", lambda _cfg: object())

    calls: list[str] = []

    def _init_observability(config: object) -> None:
        calls.append("init")
        assert config is cfg

    monkeypatch.setattr(main, "init_observability", _init_observability)
    monkeypatch.setattr(main, "_load_local_model", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_cloud_model", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_agents", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(main, "_load_retriever", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_reranker", lambda _cfg: None)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(_run_lifespan())

    assert calls == ["init"]


def test_lifespan_local_startup_loads_ollama_when_primary_provider_is_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local startup flag should load the dedicated Ollama slot, not model.provider."""
    from src.api import main

    cfg = SimpleNamespace(
        observability=SimpleNamespace(enabled=False, provider="none"),
        api=SimpleNamespace(enable_local_model_startup=True),
        model=SimpleNamespace(
            provider="google",
            name="gemini-2.5-flash",
            fallback_provider="google",
            fallback_name="gemini-2.5-flash",
            hybrid_tool_calling=False,
            ollama=SimpleNamespace(name="gemma3:4b", base_url="http://localhost:11434"),
        ),
    )
    cloud_model = object()
    local_model = object()
    cloud_agent = object()
    local_agent = object()
    loaded_models: list[object] = []
    loaded_agents: list[tuple[object, object | None, object]] = []
    registry = object()

    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "build_tool_registry", lambda _cfg: registry)
    monkeypatch.setattr(main, "init_observability", lambda _cfg: None)
    monkeypatch.setattr(main, "is_observability_active", lambda: False)
    monkeypatch.setattr(main, "_load_cloud_model", lambda _cfg: cloud_model)
    monkeypatch.setattr(main, "_load_local_model", lambda _cfg: loaded_models.append(_cfg) or local_model)

    def _load_agents(
        llm: object | None = None,
        tool_llm: object | None = None,
        tool_registry: object | None = None,
    ) -> tuple[object, None]:
        loaded_agents.append((llm, tool_llm, tool_registry))
        return (cloud_agent if llm is cloud_model else local_agent), None

    monkeypatch.setattr(main, "_load_agents", _load_agents)
    monkeypatch.setattr(main, "_load_retriever", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_reranker", lambda _cfg: None)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(_run_lifespan())

    assert loaded_models == [cfg]
    assert loaded_agents == [
        (cloud_model, None, registry),
        (local_model, None, registry),
    ]
    assert main.app.state.tool_registry is registry
    assert main.app.state.cloud_model is cloud_model
    assert main.app.state.local_model is local_model
    assert main.app.state.cloud_intelligent_agent is cloud_agent
    assert main.app.state.local_intelligent_agent is local_agent
    assert main.app.state.model is cloud_model
    assert main.app.state.intelligent_agent is cloud_agent


def test_lifespan_local_hybrid_tool_calling_uses_cloud_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When hybrid local mode is enabled, the local agent gets the cloud model for planning."""
    from src.api import main

    cfg = SimpleNamespace(
        observability=SimpleNamespace(enabled=False, provider="none"),
        api=SimpleNamespace(enable_local_model_startup=True),
        model=SimpleNamespace(
            provider="google",
            name="gemini-2.5-flash",
            fallback_provider="google",
            fallback_name="gemini-2.5-flash",
            hybrid_tool_calling=True,
            ollama=SimpleNamespace(name="gemma3:4b", base_url="http://localhost:11434"),
        ),
    )
    cloud_model = object()
    local_model = object()
    loaded_agents: list[tuple[object, object | None, object]] = []
    registry = object()

    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "build_tool_registry", lambda _cfg: registry)
    monkeypatch.setattr(main, "init_observability", lambda _cfg: None)
    monkeypatch.setattr(main, "is_observability_active", lambda: False)
    monkeypatch.setattr(main, "_load_cloud_model", lambda _cfg: cloud_model)
    monkeypatch.setattr(main, "_load_local_model", lambda _cfg: local_model)
    def _load_agents(
        llm: object | None = None,
        tool_llm: object | None = None,
        tool_registry: object | None = None,
    ) -> tuple[object, None]:
        loaded_agents.append((llm, tool_llm, tool_registry))
        return object(), None

    monkeypatch.setattr(main, "_load_agents", _load_agents)
    monkeypatch.setattr(main, "_load_retriever", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_reranker", lambda _cfg: None)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(_run_lifespan())

    assert loaded_agents == [
        (cloud_model, None, registry),
        (local_model, cloud_model, registry),
    ]


def test_local_model_startup_flag_defaults_to_false() -> None:
    """API local startup should remain disabled when no explicit config flag is set."""
    from src.api.main import _is_local_model_startup_enabled

    cfg = SimpleNamespace()

    assert _is_local_model_startup_enabled(cfg) is False


def test_local_model_startup_flag_reads_explicit_config() -> None:
    """API local startup should follow the explicit config flag."""
    from src.api.main import _is_local_model_startup_enabled

    cfg = SimpleNamespace(api=SimpleNamespace(enable_local_model_startup=True))

    assert _is_local_model_startup_enabled(cfg) is True
