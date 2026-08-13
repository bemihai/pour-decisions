"""Evaluation-only search-representation helpers for M3 Phase 2."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from src.chroma.contextual_text import build_contextual_search_text


SearchRepresentation = Literal["body_only", "contextual"]
SUPPORTED_SEARCH_REPRESENTATIONS: tuple[SearchRepresentation, ...] = ("body_only", "contextual")


def build_ablation_search_text(
    body: str,
    metadata: dict[str, Any],
    representation: SearchRepresentation,
) -> str:
    """Build body-only or production-contextual search text for one record."""
    clean_body = str(body or "").strip()
    if representation == "body_only":
        return clean_body
    if representation == "contextual":
        return build_contextual_search_text(clean_body, metadata)
    raise ValueError(
        f"Unsupported search representation {representation!r}; "
        f"expected one of {SUPPORTED_SEARCH_REPRESENTATIONS}"
    )


def build_ablation_documents(
    source_documents: Sequence[dict[str, Any]],
    representation: SearchRepresentation,
) -> list[dict[str, Any]]:
    """Copy source records while changing only their indexed search text."""
    if representation not in SUPPORTED_SEARCH_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported search representation {representation!r}; "
            f"expected one of {SUPPORTED_SEARCH_REPRESENTATIONS}"
        )

    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in source_documents:
        document_id = str(source.get("id", "")).strip()
        body = str(source.get("document", "")).strip()
        if not document_id:
            raise ValueError("Ablation source record is missing an id")
        if document_id in seen_ids:
            raise ValueError(f"Duplicate ablation source id: {document_id}")
        if not body:
            raise ValueError(f"Ablation source record {document_id!r} has empty document text")

        metadata = dict(source.get("metadata", {}) or {})
        documents.append(
            {
                "id": document_id,
                "document": body,
                "metadata": metadata,
                "search_text": build_ablation_search_text(body, metadata, representation),
            }
        )
        seen_ids.add(document_id)
    return documents


def validate_aligned_ablation_documents(
    body_only: Sequence[dict[str, Any]],
    contextual: Sequence[dict[str, Any]],
) -> None:
    """Require paired corpora to differ only in ``search_text``."""
    if len(body_only) != len(contextual):
        raise ValueError(
            "Ablation variants have different record counts: "
            f"body_only={len(body_only)}, contextual={len(contextual)}"
        )

    for index, (body_record, contextual_record) in enumerate(zip(body_only, contextual)):
        for field in ("id", "document", "metadata"):
            if body_record.get(field) != contextual_record.get(field):
                raise ValueError(f"Ablation variants differ at record {index} field {field!r}")


class SearchRepresentationReranker:
    """Use one explicit search representation with an existing reranker model."""

    def __init__(self, base_reranker: Any, representation: SearchRepresentation) -> None:
        if representation not in SUPPORTED_SEARCH_REPRESENTATIONS:
            raise ValueError(
                f"Unsupported search representation {representation!r}; "
                f"expected one of {SUPPORTED_SEARCH_REPRESENTATIONS}"
            )
        self.base_reranker = base_reranker
        self.representation = representation

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank documents without thresholding."""
        return self._score(query, documents)[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        documents: list[dict[str, Any]],
        threshold: float = 0.0,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents and retain scores at or above ``threshold``."""
        results = [
            document
            for document in self._score(query, documents)
            if float(document["rerank_score"]) >= threshold
        ]
        return results if top_k is None else results[:top_k]

    def _score(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score query-document pairs with the wrapped production model."""
        if not documents:
            return []
        pairs = [
            (
                query,
                build_ablation_search_text(
                    str(document.get("document", "")),
                    dict(document.get("metadata", {}) or {}),
                    self.representation,
                ),
            )
            for document in documents
        ]
        scores = self.base_reranker.model.predict(pairs)
        scored = [
            {**document, "rerank_score": float(score)}
            for document, score in zip(documents, scores)
        ]
        return sorted(scored, key=lambda document: float(document["rerank_score"]), reverse=True)
