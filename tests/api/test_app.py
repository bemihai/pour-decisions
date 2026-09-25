"""Tests for the FastAPI application shell and health endpoint."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


def _populate_state(app):
    """Populate all app.state attributes that lifespan normally sets."""
    app.state.prompt_registry = None
    app.state.cloud_model = None
    app.state.intelligent_agent = None
    app.state.tool_registry = None
    app.state.tool_execution = None
    app.state.tool_execution_controller = None
    app.state.session_memory = None
    app.state.conversation_memory_manager = None
    app.state.async_rag_runtime = None
    app.state.retriever = None
    app.state.reranker = None


@pytest.fixture()
def client():
    """Create a FastAPI TestClient with pre-populated app state."""
    from src.api.main import app

    _populate_state(app)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _default_disabled_conversation_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy lifespan tests independent from durable local storage."""
    from src.agents.memory import SessionMemoryConfig
    from src.api import main

    monkeypatch.setattr(main, "load_session_memory_config", lambda _cfg: SessionMemoryConfig())
    monkeypatch.setattr(main.ConversationMemoryManager, "open", AsyncMock(return_value=None))
    runtime = MagicMock(retriever=None, reranker=None)
    runtime.close = AsyncMock()
    monkeypatch.setattr(main, "build_async_rag_runtime", AsyncMock(return_value=runtime))


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
        assert "cloud_model" in resources
        assert "intelligent_agent" in resources
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
    from src.agents.guardrails import ToolExecutionConfig

    cfg = SimpleNamespace(
        observability=SimpleNamespace(
            enabled=False,
            provider="none",
        ),
        model=SimpleNamespace(
            provider="ollama",
            name="gemma3:4b",
        ),
    )

    calls: list[str] = []
    prompt_registry = object()

    def _get_prompt_registry() -> object:
        calls.append("prompt_registry")
        return prompt_registry

    def _get_config() -> object:
        calls.append("config")
        return cfg

    monkeypatch.setattr(main, "get_prompt_registry", _get_prompt_registry)
    monkeypatch.setattr(main, "get_config", _get_config)
    monkeypatch.setattr(main, "build_tool_registry", lambda _cfg, **_kwargs: object())
    monkeypatch.setattr(main, "load_tool_execution_config", lambda _cfg: ToolExecutionConfig())

    def _init_observability(config: object) -> None:
        calls.append("init")
        assert config is cfg

    monkeypatch.setattr(main, "init_observability", _init_observability)
    monkeypatch.setattr(main, "_load_cloud_model", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_agents", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(main, "_load_retriever", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_reranker", lambda _cfg: None)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    asyncio.run(_run_lifespan())

    assert calls == ["prompt_registry", "config", "init"]
    assert main.app.state.prompt_registry is prompt_registry


def test_lifespan_owns_and_closes_enabled_conversation_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One manager should be shared with agents and closed after application shutdown."""
    from src.agents.guardrails import ToolExecutionConfig
    from src.agents.memory import SessionMemoryConfig
    from src.api import main

    cfg = SimpleNamespace(
        api=SimpleNamespace(enable_local_model_startup=False),
        model=SimpleNamespace(
            provider="ollama",
            name="gemma4:31b",
        ),
    )
    policy = SessionMemoryConfig(enabled=True, db_path="unused-test-path.db")
    manager = MagicMock()
    manager.close = AsyncMock()
    open_manager = AsyncMock(return_value=manager)
    cloud_model = object()
    load_agents = MagicMock(return_value=(object(), None))

    monkeypatch.setattr(main, "get_prompt_registry", lambda: object())
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "build_tool_registry", lambda _cfg, **_kwargs: object())
    monkeypatch.setattr(main, "load_tool_execution_config", lambda _cfg: ToolExecutionConfig())
    monkeypatch.setattr(main, "init_observability", lambda _cfg: None)
    monkeypatch.setattr(main, "is_observability_active", lambda: False)
    monkeypatch.setattr(main, "load_session_memory_config", lambda _cfg: policy)
    monkeypatch.setattr(main.ConversationMemoryManager, "open", open_manager)
    monkeypatch.setattr(main, "_load_cloud_model", lambda _cfg: cloud_model)
    monkeypatch.setattr(main, "_load_agents", load_agents)
    monkeypatch.setattr(main, "_load_retriever", lambda _cfg: None)
    monkeypatch.setattr(main, "_load_reranker", lambda _cfg: None)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            assert main.app.state.conversation_memory_manager is manager

    asyncio.run(_run_lifespan())

    open_manager.assert_awaited_once_with(policy)
    load_agents.assert_called_once_with(
        cloud_model,
        tool_registry=main.app.state.tool_registry,
        tool_execution=main.app.state.tool_execution,
        tool_execution_controller=main.app.state.tool_execution_controller,
        memory_manager=manager,
        session_memory=policy,
    )
    manager.close.assert_awaited_once_with()


def test_lifespan_builds_async_rag_before_agent_snapshot_and_closes_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG resources should precede registry snapshots and close after memory."""
    from src.agents.guardrails import ToolExecutionConfig
    from src.agents.memory import SessionMemoryConfig
    from src.api import main

    cfg = SimpleNamespace(
        api=SimpleNamespace(enable_local_model_startup=False),
        model=SimpleNamespace(provider="ollama", name="gemma4:31b"),
    )
    events: list[str] = []
    runtime = MagicMock(retriever=object(), reranker=object())

    async def _close_runtime() -> None:
        events.append("close_rag")

    async def _build_runtime(_cfg: object) -> object:
        events.append("build_rag")
        return runtime

    async def _open_memory(_policy: object) -> object:
        events.append("open_memory")
        manager = MagicMock()

        async def _close_memory() -> None:
            events.append("close_memory")

        manager.close = _close_memory
        return manager

    runtime.close = _close_runtime
    monkeypatch.setattr(main, "get_prompt_registry", lambda: object())
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "init_observability", lambda _cfg: None)
    monkeypatch.setattr(main, "is_observability_active", lambda: False)
    monkeypatch.setattr(main, "build_async_rag_runtime", _build_runtime)
    def _build_registry(_cfg: object, *, async_rag_resources: object) -> object:
        assert async_rag_resources is runtime
        events.append("build_registry")
        return object()

    monkeypatch.setattr(main, "build_tool_registry", _build_registry)
    monkeypatch.setattr(main, "load_tool_execution_config", lambda _cfg: ToolExecutionConfig())
    monkeypatch.setattr(main, "load_session_memory_config", lambda _cfg: SessionMemoryConfig())
    monkeypatch.setattr(main.ConversationMemoryManager, "open", _open_memory)
    monkeypatch.setattr(main, "_load_cloud_model", lambda _cfg: None)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            assert main.app.state.async_rag_runtime is runtime
            assert main.app.state.retriever is runtime.retriever
            assert main.app.state.reranker is runtime.reranker

    asyncio.run(_run_lifespan())

    assert events == [
        "build_rag",
        "build_registry",
        "open_memory",
        "close_memory",
        "close_rag",
    ]


def test_lifespan_closes_async_rag_after_later_startup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after RAG construction should close the completed bundle."""
    from src.api import main

    cfg = SimpleNamespace()
    runtime = MagicMock(retriever=None, reranker=None)
    runtime.close = AsyncMock()
    monkeypatch.setattr(main, "get_prompt_registry", lambda: object())
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "init_observability", lambda _cfg: None)
    monkeypatch.setattr(main, "is_observability_active", lambda: False)
    monkeypatch.setattr(main, "build_async_rag_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main, "build_tool_registry", MagicMock(side_effect=RuntimeError("registry failed")))

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="registry failed"):
        asyncio.run(_run_lifespan())

    runtime.close.assert_awaited_once_with()


def test_lifespan_closes_async_rag_when_memory_shutdown_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An independent memory close failure must not prevent RAG teardown."""
    from src.agents.guardrails import ToolExecutionConfig
    from src.agents.memory import SessionMemoryConfig
    from src.api import main

    cfg = SimpleNamespace(
        api=SimpleNamespace(enable_local_model_startup=False),
        model=SimpleNamespace(provider="ollama", name="gemma4:31b"),
    )
    runtime = MagicMock(retriever=None, reranker=None)
    runtime.close = AsyncMock()
    manager = MagicMock()
    manager.close = AsyncMock(side_effect=RuntimeError("memory close failed"))
    monkeypatch.setattr(main, "get_prompt_registry", lambda: object())
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "init_observability", lambda _cfg: None)
    monkeypatch.setattr(main, "is_observability_active", lambda: False)
    monkeypatch.setattr(main, "build_async_rag_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main, "build_tool_registry", lambda _cfg, **_kwargs: object())
    monkeypatch.setattr(main, "load_tool_execution_config", lambda _cfg: ToolExecutionConfig())
    monkeypatch.setattr(main, "load_session_memory_config", lambda _cfg: SessionMemoryConfig())
    monkeypatch.setattr(main.ConversationMemoryManager, "open", AsyncMock(return_value=manager))
    monkeypatch.setattr(main, "_load_cloud_model", lambda _cfg: None)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="memory close failed"):
        asyncio.run(_run_lifespan())

    manager.close.assert_awaited_once_with()
    runtime.close.assert_awaited_once_with()


def test_async_rag_dependency_returns_lifespan_bundle() -> None:
    """The route dependency should return the exact app-owned resource bundle."""
    from src.api.dependencies import get_async_rag_runtime

    resources = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(async_rag_runtime=resources)))

    assert get_async_rag_runtime(request) is resources


def test_lifespan_propagates_prompt_preflight_failure_before_resource_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid prompt assets should stop startup before config and model construction."""
    from src.api import main

    get_config = MagicMock()
    load_cloud_model = MagicMock()
    monkeypatch.setattr(
        main,
        "get_prompt_registry",
        MagicMock(side_effect=FileNotFoundError("missing prompt manifest")),
    )
    monkeypatch.setattr(main, "get_config", get_config)
    monkeypatch.setattr(main, "_load_cloud_model", load_cloud_model)

    async def _run_lifespan() -> None:
        async with main.lifespan(main.app):
            pass

    with pytest.raises(FileNotFoundError, match="missing prompt manifest"):
        asyncio.run(_run_lifespan())

    get_config.assert_not_called()
    load_cloud_model.assert_not_called()
