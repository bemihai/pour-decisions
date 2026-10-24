"""Real database/API tests with external generation boundaries mocked."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api.dependencies import get_description_model, get_reranker, get_retriever
from src.api.routes import grapes, wines
from src.database.db import get_db_connection, initialize_database
from src.database.repository import BottleRepository, ProducerRepository, RegionRepository, WineRepository
from src.database.repository.grape import GrapeRepository


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Build the API against a temporary cellar without application startup I/O."""
    path = tmp_path / "cellar.db"
    assert initialize_database(path)
    with get_db_connection(path) as conn:
        conn.executemany(
            "INSERT INTO wines(id, source, wine_name, wine_type, varietal) VALUES (?, 'manual', 'Wine', 'Red', ?)",
            [(1, "Syrah / Cabernet Franc"), (2, "Shiraz")],
        )
        conn.commit()
    repo = GrapeRepository(path)
    repo.sync_wine(1, "Syrah / Cabernet Franc")
    repo.sync_wine(2, "Shiraz")
    for module in (wines, grapes):
        monkeypatch.setattr(module, "GrapeRepository", lambda: repo)
        monkeypatch.setattr(module, "WineRepository", lambda: WineRepository(path))
        monkeypatch.setattr(module, "ProducerRepository", lambda: ProducerRepository(path))
    monkeypatch.setattr(wines, "BottleRepository", lambda: BottleRepository(path))
    monkeypatch.setattr(wines, "RegionRepository", lambda: RegionRepository(path))
    monkeypatch.setenv("TAVILY_API_KEY", "")
    model = MagicMock()
    model.ainvoke = AsyncMock(
        return_value=AIMessage(content="Syrah makes dark-fruited wines according to the reference.")
    )
    retriever = MagicMock()
    retriever.aretrieve = AsyncMock(return_value=[{"document": "Syrah makes dark-fruited wines.", "distance": 0.1}])
    app = FastAPI()
    app.include_router(wines.router)
    app.include_router(grapes.router)
    app.dependency_overrides[get_description_model] = lambda: model
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_reranker] = lambda: None
    return TestClient(app), repo, model, retriever


def test_read_generate_regenerate_and_shared_cache(api: tuple) -> None:
    """GET stays read-only; explicit POST updates descriptions on all linked wines."""
    client, repo, model, retriever = api
    response = client.get("/api/wines/1")
    assert response.status_code == 200
    assert [g["name"] for g in response.json()["grapes"]] == ["cabernet franc", "syrah"]
    assert set(response.json()["grapes"][0]) == {"id", "name", "description"}
    model.ainvoke.assert_not_called()
    retriever.aretrieve.assert_not_called()
    syrah = repo.get_by_wine(2)[0]
    for text in ["A supported description of Syrah.", "A regenerated description of Syrah."]:
        model.ainvoke.return_value = AIMessage(content=text)
        generated = client.post(f"/api/grapes/{syrah.id}/description", json={"use_web_search": False})
        assert generated.status_code == 200
        assert generated.json()["success"]
        for wine_id in (1, 2):
            details = client.get(f"/api/wines/{wine_id}").json()
            assert next(g for g in details["grapes"] if g["id"] == syrah.id)["description"] == text
    assert model.ainvoke.await_count == 2
    retriever.retrieve.assert_not_called()


def test_missing_grape_and_generation_failure(api: tuple) -> None:
    """Missing grapes return 404; failed regeneration gives a retryable 502."""
    client, repo, model, _ = api
    assert client.post("/api/grapes/999/description").status_code == 404
    model.ainvoke.assert_not_called()
    syrah = repo.get_by_wine(2)[0]
    repo.update_description(syrah.id, "Keep existing description")
    model.ainvoke.side_effect = RuntimeError("model unavailable")
    response = client.post(f"/api/grapes/{syrah.id}/description", json={"use_web_search": False})
    assert response.status_code == 502
    assert "retry" in response.json()["detail"]
    assert repo.get_by_id(syrah.id).description == "Keep existing description"


def test_disabled_context_options_do_not_call_external_boundaries(api: tuple) -> None:
    """The request honors both context switches without silently generating unsupported text."""
    client, repo, model, retriever = api
    grape_id = repo.get_by_wine(2)[0].id
    response = client.post(
        f"/api/grapes/{grape_id}/description", json={"use_rag_context": False, "use_web_search": False}
    )
    assert response.status_code == 502
    model.ainvoke.assert_not_called()
    retriever.aretrieve.assert_not_called()
