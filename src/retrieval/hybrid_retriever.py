"""Balanced dense and sparse candidate retrieval for shared production reranking."""

from __future__ import annotations

import time
from typing import Any

from src.utils import logger

from .keyword_search import BM25Index
from .query_analyzer import RetrievalQueryPlan, build_retrieval_query_plan
from .vector_retriever import ChromaRetriever


class HybridRetriever:
    """Build a bounded, de-duplicated union of dense and sparse candidates."""

    def __init__(
        self,
        vector_retriever: ChromaRetriever,
        bm25_index: BM25Index,
        semantic_candidate_pool: int = 25,
        bm25_candidate_pool: int = 25,
        reranker_input_limit: int = 50,
    ) -> None:
        """Configure per-channel pools and the post-union reranker bound."""
        if semantic_candidate_pool <= 0:
            raise ValueError("semantic_candidate_pool must be greater than zero")
        if bm25_candidate_pool <= 0:
            raise ValueError("bm25_candidate_pool must be greater than zero")
        if reranker_input_limit <= 0:
            raise ValueError("reranker_input_limit must be greater than zero")
        self.vector_retriever = vector_retriever
        self.bm25_index = bm25_index
        self.semantic_candidate_pool = semantic_candidate_pool
        self.bm25_candidate_pool = bm25_candidate_pool
        self.reranker_input_limit = reranker_input_limit

        logger.info(
            "Initialized HybridRetriever with pools: semantic=%d, bm25=%d, union_limit=%d",
            semantic_candidate_pool,
            bm25_candidate_pool,
            reranker_input_limit,
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        *,
        query_plan: RetrievalQueryPlan | None = None,
        use_rrf_fallback: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve complete channel pools, then return a union or fallback RRF."""
        plan = query_plan or build_retrieval_query_plan(query)

        dense_start = time.perf_counter()
        vector_results = self.vector_retriever.retrieve(
            plan.semantic_query,
            n_results=self.semantic_candidate_pool,
            **kwargs,
        )
        dense_latency_ms = (time.perf_counter() - dense_start) * 1000

        sparse_start = time.perf_counter()
        bm25_results = self.bm25_index.search(plan.sparse_query, top_k=self.bm25_candidate_pool)
        sparse_latency_ms = (time.perf_counter() - sparse_start) * 1000

        union = self._balanced_candidate_union(vector_results, bm25_results)
        diagnostics: dict[str, int | float] = {
            "dense_candidates": len(vector_results),
            "sparse_candidates": len(bm25_results),
            "unique_union_candidates": len(union),
            "reranker_candidates": min(len(union), self.reranker_input_limit),
            "dense_latency_ms": round(dense_latency_ms, 3),
            "sparse_latency_ms": round(sparse_latency_ms, 3),
        }
        for candidate in union:
            candidate["retrieval_diagnostics"] = diagnostics

        if use_rrf_fallback:
            results = self._unweighted_rrf(union)[:n_results]
            mode = "unweighted_rrf"
        else:
            results = union[: self.reranker_input_limit]
            mode = "reranker_union"

        logger.info(
            "Hybrid retrieval mode=%s dense=%d sparse=%d union=%d returned=%d",
            mode,
            len(vector_results),
            len(bm25_results),
            len(union),
            len(results),
        )
        return results

    def _balanced_candidate_union(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """De-duplicate full pools by ID while retaining ranks and channel scores."""
        candidates: dict[str, dict[str, Any]] = {}
        for rank, document in enumerate(vector_results, start=1):
            candidate = dict(document)
            candidate["dense_rank"] = rank
            candidate["sparse_rank"] = None
            candidate["dense_similarity"] = document.get("similarity")
            candidate["bm25_score"] = None
            candidate["retrieval_channels"] = ["dense"]
            candidates[str(document["id"])] = candidate

        for rank, document in enumerate(bm25_results, start=1):
            document_id = str(document["id"])
            candidate = candidates.get(document_id)
            if candidate is None:
                candidate = dict(document)
                candidate["dense_rank"] = None
                candidate["dense_similarity"] = None
                candidate["retrieval_channels"] = []
                candidates[document_id] = candidate
            candidate["sparse_rank"] = rank
            candidate["bm25_score"] = document.get("bm25_score")
            candidate["retrieval_channels"] = [*candidate["retrieval_channels"], "sparse"]

        ordered_ids: list[str] = []
        seen: set[str] = set()
        maximum_rank = max(len(vector_results), len(bm25_results))
        for index in range(maximum_rank):
            for results in (vector_results, bm25_results):
                if index >= len(results):
                    continue
                document_id = str(results[index]["id"])
                if document_id not in seen:
                    ordered_ids.append(document_id)
                    seen.add(document_id)
        return [candidates[document_id] for document_id in ordered_ids]

    @staticmethod
    def _unweighted_rrf(candidates: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
        """Rank the complete union with standard unweighted reciprocal ranks."""
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            score = 0.0
            dense_rank = candidate.get("dense_rank")
            sparse_rank = candidate.get("sparse_rank")
            if isinstance(dense_rank, int):
                score += 1 / (k + dense_rank)
            if isinstance(sparse_rank, int):
                score += 1 / (k + sparse_rank)
            scored.append({**candidate, "rrf_score": score})
        return sorted(
            scored,
            key=lambda candidate: (
                -float(candidate["rrf_score"]),
                int(candidate.get("dense_rank") or 10**9),
                int(candidate.get("sparse_rank") or 10**9),
                str(candidate["id"]),
            ),
        )
