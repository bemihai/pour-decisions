"""Deterministic local query analysis and retrieval planning.

This module analyzes user queries to extract wine entities and build
metadata filters for improved retrieval. All processing is local.
"""
from dataclasses import dataclass, field
import re
from typing import Any

from src.chroma.metadata_extractor import (
    extract_appellations,
    extract_classifications,
    extract_grapes,
    extract_producers,
    extract_regions,
    extract_vintages,
)
from src.utils import logger

from .bm25_analyzer import analyze_bm25_text
from .query_utils import normalize_query


_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("flavour", re.compile(r"\b(?:aroma|aromas|flavou?r|flavou?rs|taste|tasting|sensory)\b")),
    ("aging", re.compile(r"\b(?:age|aging|ageing|cellar|drink|drinking|matur(?:e|ity|ation))\b")),
    ("pairing", re.compile(r"\b(?:food|dish|pair|paired|pairing|match)\b")),
    ("classification", re.compile(r"\b(?:classification|classified|requirement|regulation|docg|doc|aoc|ava)\b")),
    ("region", re.compile(r"\b(?:origin|region|regional|where|grown|grows|geography|appellation)\b")),
)
_INTENT_SEMANTIC_TERMS = {
    "flavour": "aroma flavor taste sensory profile tannin acidity body",
    "aging": "aging maturation drinking window cellar",
    "pairing": "food pairing dish match",
    "classification": "classification requirements regulations",
    "region": "origin region appellation geography",
}
_INTENT_SPARSE_TERMS = {
    "flavour": "aroma taste tannin acidity body",
    "aging": "aging maturation drinking window cellar",
    "pairing": "food pairing dish match",
    "classification": "classification requirements regulations",
    "region": "origin region appellation geography",
}


@dataclass
class QueryAnalysis:
    """
    Analysis of a user query for metadata filtering.

    Attributes:
        original_query: The original user query.
        grapes: Grape varieties detected in query.
        regions: Wine regions detected in query.
        vintages: Vintage years detected in query.
        appellations: Wine appellations detected in query.
        has_filters: Whether any filterable entities were found.
    """
    original_query: str
    grapes: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    vintages: list[str] = field(default_factory=list)
    classifications: list[str] = field(default_factory=list)
    producers: list[str] = field(default_factory=list)
    appellations: list[str] = field(default_factory=list)

    @property
    def has_filters(self) -> bool:
        """Check if any filterable entities were detected."""
        return bool(
            self.grapes
            or self.regions
            or self.vintages
            or self.classifications
            or self.producers
            or self.appellations
        )

    def to_chroma_filter(self, operator: str = "$or") -> dict[str, Any] | None:
        """
        Convert to ChromaDB where filter.

        Args:
            operator: How to combine filters ("$or" or "$and").

        Returns:
            ChromaDB where filter dict, or None if no filters.
        """
        if not self.has_filters:
            return None

        conditions = []

        # Build conditions for each entity type
        for grape in self.grapes:
            conditions.append({"grapes": {"$contains": grape}})

        for region in self.regions:
            conditions.append({"regions": {"$contains": region}})

        for vintage in self.vintages:
            conditions.append({"vintages": {"$contains": vintage}})

        for classification in self.classifications:
            conditions.append({"classifications": {"$contains": classification}})

        for producer in self.producers:
            conditions.append({"producers": {"$contains": producer}})

        for appellation in self.appellations:
            conditions.append({"appellations": {"$contains": appellation}})

        if len(conditions) == 1:
            return conditions[0]

        return {operator: conditions}

    def get_boost_terms(self) -> list[str]:
        """Get terminology that should boost relevance if found in chunks."""
        terms = []
        terms.extend(self.grapes)
        terms.extend(self.regions)
        terms.extend(self.vintages)
        terms.extend(self.classifications)
        terms.extend(self.producers)
        terms.extend(self.appellations)
        return terms


@dataclass(frozen=True)
class RetrievalQueryPlan:
    """Typed channel-specific retrieval inputs computed without an LLM."""

    original_query: str
    normalized_query: str
    semantic_query: str
    sparse_query: str
    intent: str
    grapes: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    vintages: tuple[str, ...] = ()
    classifications: tuple[str, ...] = ()
    producers: tuple[str, ...] = ()
    appellations: tuple[str, ...] = ()

    def to_analysis(self) -> QueryAnalysis:
        """Return the mutable compatibility analysis used by metadata boosting."""
        return QueryAnalysis(
            original_query=self.normalized_query,
            grapes=list(self.grapes),
            regions=list(self.regions),
            vintages=list(self.vintages),
            classifications=list(self.classifications),
            producers=list(self.producers),
            appellations=list(self.appellations),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic artifact."""
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "semantic_query": self.semantic_query,
            "sparse_query": self.sparse_query,
            "intent": self.intent,
            "entities": {
                "grapes": list(self.grapes),
                "regions": list(self.regions),
                "vintages": list(self.vintages),
                "classifications": list(self.classifications),
                "producers": list(self.producers),
                "appellations": list(self.appellations),
            },
        }


def analyze_query(query: str) -> QueryAnalysis:
    """
    Analyze a user query to extract wine entities for filtering.

    Args:
        query: User's natural language query.

    Returns:
        QueryAnalysis with detected entities.
    """
    # Extract entities using existing extractors
    grapes = sorted(extract_grapes(query), key=str.casefold)
    regions = sorted(extract_regions(query), key=str.casefold)
    vintages = sorted(extract_vintages(query))
    classifications = sorted(extract_classifications(query), key=str.casefold)
    producers = sorted(extract_producers(query), key=str.casefold)
    appellations = sorted(extract_appellations(query), key=str.casefold)

    analysis = QueryAnalysis(
        original_query=query,
        grapes=grapes,
        regions=regions,
        vintages=vintages,
        classifications=classifications,
        producers=producers,
        appellations=appellations,
    )

    if analysis.has_filters:
        logger.debug(
            f"Query analysis: grapes={grapes}, regions={regions}, "
            f"vintages={vintages}, classifications={classifications}, "
            f"producers={producers}, appellations={appellations}"
        )

    return analysis


def build_retrieval_query_plan(query: str) -> RetrievalQueryPlan:
    """Build deterministic dense and sparse query inputs from local analysis."""
    normalized_query = normalize_query(query).strip()
    analysis = analyze_query(normalized_query)
    intent = _detect_intent(normalized_query)
    entity_terms = analysis.get_boost_terms()
    semantic_query = normalized_query
    sparse_source = normalized_query
    if intent != "unknown" and entity_terms:
        semantic_query = " ".join([*entity_terms, _INTENT_SEMANTIC_TERMS[intent]])
        sparse_source = " ".join([*entity_terms, _INTENT_SPARSE_TERMS[intent]])
    sparse_query = " ".join(dict.fromkeys(analyze_bm25_text(sparse_source)))
    return RetrievalQueryPlan(
        original_query=query,
        normalized_query=normalized_query,
        semantic_query=semantic_query,
        sparse_query=sparse_query,
        intent=intent,
        grapes=tuple(analysis.grapes),
        regions=tuple(analysis.regions),
        vintages=tuple(analysis.vintages),
        classifications=tuple(analysis.classifications),
        producers=tuple(analysis.producers),
        appellations=tuple(analysis.appellations),
    )


def _detect_intent(query: str) -> str:
    """Return the first explicit supported intent or ``unknown``."""
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return "unknown"


def boost_by_metadata_match(
    docs: list[dict[str, Any]],
    analysis: QueryAnalysis,
    boost_factor: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Boost document scores based on metadata matches.

    Documents whose metadata matches query entities get a score boost.

    Args:
        docs: Retrieved documents with 'similarity' scores.
        analysis: Query analysis with detected entities.
        boost_factor: Score boost per matching entity (default: 0.1).

    Returns:
        Documents with boosted scores, re-sorted by score.
    """
    if not analysis.has_filters or not docs:
        return docs

    boosted = []
    for doc in docs:
        doc_copy = doc.copy()
        metadata = doc_copy.get('metadata', {})
        similarity = doc_copy.get('similarity', 0.5)

        # Count metadata matches
        matches = 0

        doc_grapes = metadata.get('grapes', '').lower()
        for grape in analysis.grapes:
            if grape.lower() in doc_grapes:
                matches += 1

        doc_regions = metadata.get('regions', '').lower()
        for region in analysis.regions:
            if region.lower() in doc_regions:
                matches += 1

        doc_vintages = metadata.get('vintages', '')
        for vintage in analysis.vintages:
            if vintage in doc_vintages:
                matches += 1

        doc_classifications = metadata.get("classifications", "").lower()
        for classification in analysis.classifications:
            if classification.lower() in doc_classifications:
                matches += 1

        doc_producers = metadata.get("producers", "").lower()
        for producer in analysis.producers:
            if producer.lower() in doc_producers:
                matches += 1

        doc_appellations = metadata.get('appellations', '').lower()
        for appellation in analysis.appellations:
            if appellation.lower() in doc_appellations:
                matches += 1

        # Apply boost (capped at 1.0)
        boosted_similarity = min(1.0, similarity + (matches * boost_factor))
        doc_copy['similarity'] = boosted_similarity
        doc_copy['metadata_matches'] = matches

        boosted.append(doc_copy)

    # Re-sort by boosted similarity
    boosted.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    total_matches = sum(d.get('metadata_matches', 0) for d in boosted)
    if total_matches > 0:
        logger.debug(f"Boosted {total_matches} metadata matches across {len(boosted)} docs")

    return boosted
