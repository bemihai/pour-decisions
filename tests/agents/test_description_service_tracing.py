"""Unit tests for DescriptionService observability span tagging."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

from src.agents.description_service import DescriptionService
from src.database.models import Producer, Wine


class _FakeSpan:
    """Minimal span placeholder used by tracer context manager."""


class _FakeTracer:
    """Tracer stub that returns a simple context manager with a fake span."""

    @contextmanager
    def start_as_current_span(self, _name: str):
        yield _FakeSpan()


def test_get_wine_description_sets_description_generation_span_attributes(monkeypatch) -> None:
    """Wine description generation should tag span with feature and entity metadata."""
    from src.agents import description_service as module

    captured_attributes: list[dict[str, object]] = []

    monkeypatch.setattr(module.otel_trace, "get_tracer", lambda _name: _FakeTracer())
    monkeypatch.setattr(module, "set_span_attributes", lambda _span, attrs: captured_attributes.append(attrs))

    service = object.__new__(DescriptionService)
    service._wine_prompt_template = "{wine_name} {context_section}"
    service.wine_repo = SimpleNamespace(update=lambda _wine: None)
    service._build_wine_search_query = lambda _wine: "query"
    service._build_context_section = lambda _query, _wine: "context"
    service._invoke_structured = lambda _prompt: SimpleNamespace(
        description="Generated wine description",
        drink_from_year=2024,
        drink_to_year=2030,
    )
    service._persist_llm_drinking_window = lambda _wine, _from, _to: None

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
    )

    result = service.get_wine_description(cast(Wine, wine))

    assert result == "Generated wine description"
    assert any(attrs.get("feature") == "description_generation" for attrs in captured_attributes)
    assert any(attrs.get("entity_type") == "wine" for attrs in captured_attributes)
    assert any(attrs.get("entity_id") == "7" for attrs in captured_attributes)


def test_get_producer_description_sets_description_generation_span_attributes(monkeypatch) -> None:
    """Producer description generation should tag span with feature and entity metadata."""
    from src.agents import description_service as module

    captured_attributes: list[dict[str, object]] = []

    monkeypatch.setattr(module.otel_trace, "get_tracer", lambda _name: _FakeTracer())
    monkeypatch.setattr(module, "set_span_attributes", lambda _span, attrs: captured_attributes.append(attrs))

    service = object.__new__(DescriptionService)
    service._producer_prompt_template = "{producer_name} {context_section}"
    service.producer_repo = SimpleNamespace(update=lambda _producer: None)
    service._build_producer_search_query = lambda _producer: "query"
    service._generate_with_llm = lambda _prompt: "Generated producer description"
    service.use_rag_context = False
    service.retriever = None
    service.max_context_chunks = 3

    producer = SimpleNamespace(
        id=11,
        description=None,
        name="Producer",
        country="Italy",
        region="Piedmont",
    )

    result = service.get_producer_description(cast(Producer, producer))

    assert result == "Generated producer description"
    assert any(attrs.get("feature") == "description_generation" for attrs in captured_attributes)
    assert any(attrs.get("entity_type") == "producer" for attrs in captured_attributes)
    assert any(attrs.get("entity_id") == "11" for attrs in captured_attributes)


