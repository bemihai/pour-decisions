"""Async grape generation, evidence controls, persistence, and provenance."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.description_service import DescriptionService
from src.database.db import get_db_connection, initialize_database
from src.database.repository.grape import GrapeRepository


@pytest.fixture
def setup_service(tmp_path: Path) -> tuple:
    """Use a real isolated shared grape store and mocked external boundaries."""
    path = tmp_path / "cellar.db"
    assert initialize_database(path)
    with get_db_connection(path) as conn:
        conn.executemany(
            "INSERT INTO wines(id, source, wine_name, wine_type) VALUES (?, 'manual', 'Test', 'Red')", [(1,), (2,)]
        )
        conn.commit()
    repo = GrapeRepository(path)
    repo.sync_wine(1, "Syrah")
    repo.sync_wine(2, "Shiraz")
    model = MagicMock()
    model.ainvoke = AsyncMock(
        return_value=AIMessage(content="Syrah is described in the reference as a dark-fruited grape.")
    )
    retriever = MagicMock()
    retriever.aretrieve = AsyncMock(return_value=[{"document": "Syrah produces dark-fruited wines.", "distance": 0.1}])
    reranker = MagicMock()
    reranker.arerank = AsyncMock(side_effect=lambda query, chunks, **kwargs: chunks)
    service = DescriptionService(
        model=model,
        retriever=retriever,
        reranker=reranker,
        use_web_search=False,
        config={"description_generation": {"enable_web_search": False}},
        wine_repo=MagicMock(),
        producer_repo=MagicMock(),
        grape_repo=repo,
    )
    return service, repo, model, retriever, reranker


async def test_generation_cache_regeneration_and_provenance(setup_service: tuple) -> None:
    """One successful result is visible across wines, with grape prompt provenance."""
    service, repo, model, retriever, reranker = setup_service
    grape = repo.get_by_wine(1)[0]
    result = await service.aget_grape_description(grape)
    assert result and repo.get_by_wine(2)[0].description == result
    model.ainvoke.assert_awaited_once()
    retriever.aretrieve.assert_awaited_once()
    reranker.arerank.assert_awaited_once()
    retriever.retrieve.assert_not_called()
    assert await service.aget_grape_description(repo.get_by_wine(2)[0]) == result
    assert model.ainvoke.await_count == 1
    model.ainvoke.return_value = AIMessage(content="A newly generated supported description of Syrah.")
    regenerated = await service.aget_grape_description(grape, force_regenerate=True)
    assert regenerated != result
    assert repo.get_by_wine(2)[0].description == regenerated
    provenance = service.grape_execution_provenance
    assert provenance.mode == "description_grape"
    attributes = provenance.to_trace_attributes()
    assert attributes["pour_decisions.prompt.grape_description.source_hash"] == service.grape_prompt_record.source_hash
    prompt = model.ainvoke.call_args.args[0]
    assert "Syrah produces dark-fruited wines." in prompt
    assert "Do not claim that a particular bottle" in prompt


@pytest.mark.parametrize("failure", ["llm", "empty", "rag", "unrelated", "save"])
async def test_failure_preserves_cache(setup_service: tuple, failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No failed regeneration overwrites the last successful description."""
    service, repo, model, retriever, _ = setup_service
    grape = repo.get_by_wine(1)[0]
    repo.update_description(grape.id, "Previously cached description")
    grape = repo.get_by_id(grape.id)
    if failure == "llm":
        model.ainvoke.side_effect = RuntimeError("unavailable")
    elif failure == "empty":
        model.ainvoke.return_value = AIMessage(content="")
    elif failure == "rag":
        retriever.aretrieve.side_effect = RuntimeError("unavailable")
    elif failure == "unrelated":
        retriever.aretrieve.return_value = [{"document": "Chardonnay wine", "distance": 0.1}]
    elif failure == "save":
        monkeypatch.setattr(repo, "update_description", MagicMock(return_value=False))
    assert await service.aget_grape_description(grape, force_regenerate=True) is None
    assert grape.description == "Previously cached description"
    assert repo.get_by_wine(2)[0].description == "Previously cached description"
    if failure in {"rag", "unrelated"}:
        model.ainvoke.assert_not_awaited()


async def test_web_fallback_and_disabled_controls(setup_service: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Web fallback runs only when enabled and relevant RAG evidence is absent."""
    service, repo, model, retriever, _ = setup_service
    grape = repo.get_by_wine(1)[0]
    engine = MagicMock()
    engine.search.return_value = [{"title": "Shiraz guide", "snippet": "Shiraz makes dark-fruited wine."}]
    service._web_search_engine = engine
    service.use_web_search = True
    service._web_search_available = True
    await service.aget_grape_description(grape)
    engine.search.assert_not_called()
    retriever.aretrieve.return_value = []
    assert await service.aget_grape_description(grape, force_regenerate=True)
    engine.search.assert_called_once()
    engine.search.reset_mock()
    service.use_web_search = False
    assert await service.aget_grape_description(grape, force_regenerate=True) is None
    engine.search.assert_not_called()

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    disabled = DescriptionService(
        model=model,
        use_web_search=True,
        config={"description_generation": {"enable_web_search": False}},
        wine_repo=MagicMock(),
        producer_repo=MagicMock(),
        grape_repo=repo,
    )
    assert not disabled.use_web_search
    disabled._web_search_engine = engine
    assert await disabled.aget_grape_description(grape, force_regenerate=True) is None
    engine.search.assert_not_called()


async def test_web_failure_or_unrelated_evidence_never_generates(setup_service: tuple) -> None:
    """A failed web fallback cannot produce an ungrounded replacement."""
    service, repo, model, retriever, _ = setup_service
    service.use_web_search = True
    service._web_search_available = True
    retriever.aretrieve.return_value = []
    engine = MagicMock()
    service._web_search_engine = engine
    engine.search.side_effect = RuntimeError("search unavailable")
    assert await service.aget_grape_description(repo.get_by_wine(1)[0]) is None
    engine.search.side_effect = None
    engine.search.return_value = [{"title": "Chardonnay", "snippet": "Unrelated variety"}]
    assert await service.aget_grape_description(repo.get_by_wine(1)[0]) is None
    model.ainvoke.assert_not_awaited()


async def test_grape_span_carries_its_own_provenance(setup_service: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace attributes identify the grape prompt rather than the producer prompt."""
    from src.agents import description_service as module

    service, repo, _, _, _ = setup_service
    tracer = MagicMock()
    monkeypatch.setattr(module.otel_trace, "get_tracer", lambda name: tracer)
    attributes = MagicMock()
    provenance = MagicMock()
    monkeypatch.setattr(module, "set_span_attributes", attributes)
    monkeypatch.setattr(module, "set_execution_provenance_attributes", provenance)
    grape = repo.get_by_wine(1)[0]
    assert await service.aget_grape_description(grape)
    tracer.start_as_current_span.assert_called_once_with("description_generation.grape")
    assert attributes.call_args.args[1]["entity_type"] == "grape"
    captured = provenance.call_args.args[1]
    assert captured["pour_decisions.prompt.grape_description.source_hash"] == service.grape_prompt_record.source_hash
    assert not any("producer_description" in key for key in captured)
