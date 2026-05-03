"""Local retrieval metrics for the eval harness.

This module provides pure, side-effect-free implementations of ranking metrics used
in retrieval-only evaluation mode. The functions do not perform any I/O or model/API
calls and can be unit-tested deterministically.
"""

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute precision@k.

    Precision@k is the fraction of the top-k retrieved items that are relevant.
    The denominator is always ``k`` (fixed-cutoff precision), which means returning
    fewer than ``k`` results is penalized.

    Args:
        retrieved_ids: Ranked list of retrieved chunk IDs (highest rank first).
        relevant_ids: Unordered list of relevant chunk IDs.
        k: Cutoff rank. Must be positive.

    Returns:
        Precision@k in the range [0.0, 1.0]. Returns 0.0 when ``retrieved_ids``
        or ``relevant_ids`` is empty.

    Raises:
        ValueError: If ``k`` is less than or equal to zero.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Compute reciprocal rank (RR) for a single query.

    Reciprocal rank is ``1 / rank`` of the first relevant retrieved item, where
    rank is 1-based. If no relevant item is retrieved, RR is 0.0.

    Args:
        retrieved_ids: Ranked list of retrieved chunk IDs.
        relevant_ids: Unordered list of relevant chunk IDs.

    Returns:
        Reciprocal rank in the range [0.0, 1.0].
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(results: list[tuple[list[str], list[str]]]) -> float:
    """Compute mean reciprocal rank (MRR) over multiple queries.

    Args:
        results: List of ``(retrieved_ids, relevant_ids)`` tuples, one per query.

    Returns:
        Mean reciprocal rank across all tuples. Returns 0.0 for an empty list.
    """
    if not results:
        return 0.0

    rr_values = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in results]
    return sum(rr_values) / len(rr_values)


def mean_precision_at_k(results: list[tuple[list[str], list[str]]], k: int) -> float:
    """Compute mean precision@k over multiple queries.

    Args:
        results: List of ``(retrieved_ids, relevant_ids)`` tuples, one per query.
        k: Cutoff rank for precision@k. Must be positive.

    Returns:
        Mean precision@k across all tuples. Returns 0.0 for an empty list.

    Raises:
        ValueError: If ``k`` is less than or equal to zero.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    if not results:
        return 0.0

    p_at_k_values = [
        precision_at_k(retrieved_ids=retrieved, relevant_ids=relevant, k=k) for retrieved, relevant in results
    ]
    return sum(p_at_k_values) / len(p_at_k_values)

