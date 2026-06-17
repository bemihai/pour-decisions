"""Golden dataset loader and filter helpers for the eval harness."""

import json
from pathlib import Path

from src.utils import logger

from .models import GoldenSample


def load_golden_dataset(path: str | Path) -> list[GoldenSample]:
    """Load and validate the golden dataset from a JSONL file.

    Reads the file line-by-line; blank lines are skipped. Each non-blank line is
    parsed as JSON and validated against :class:`~src.eval.models.GoldenSample`.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of validated :class:`~src.eval.models.GoldenSample` objects, one per
        non-blank line in the file.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
        ValueError: If any line fails Pydantic validation.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")

    samples: list[GoldenSample] = []
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}: {exc}"
                ) from exc
            try:
                sample = GoldenSample.model_validate(data)
            except Exception as exc:
                sample_id = data.get("id", f"<line {line_number}>")
                raise ValueError(
                    f"Validation failed for sample {sample_id!r} (line {line_number}): {exc}"
                ) from exc
            samples.append(sample)

    logger.debug("Loaded %d samples from %s", len(samples), dataset_path)
    return samples


def filter_golden_samples(
    samples: list[GoldenSample],
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
    tags: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> list[GoldenSample]:
    """Return a subset of samples matching the provided filter criteria.

    All supplied filters are applied as AND conditions. Passing ``None`` (or
    omitting) a filter means "do not filter on this dimension". An empty filter
    list (e.g., ``categories=[]``) matches nothing.

    Args:
        samples: The full list of samples to filter.
        categories: If set, keep only samples whose ``category`` is in this list.
        difficulties: If set, keep only samples whose ``difficulty`` is in this list.
        tags: If set, keep only samples that have at least one of these tags.
        sample_ids: If set, keep only samples whose ``id`` is in this list.

    Returns:
        Filtered list of :class:`~src.eval.models.GoldenSample` objects. The
        original list is not modified.
    """
    result = samples
    if categories is not None:
        category_set = set(categories)
        result = [sample for sample in result if sample.category in category_set]
    if difficulties is not None:
        difficulty_set = set(difficulties)
        result = [sample for sample in result if sample.difficulty in difficulty_set]
    if tags is not None:
        tag_set = set(tags)
        result = [sample for sample in result if tag_set.intersection(sample.tags)]
    if sample_ids is not None:
        sample_id_set = set(sample_ids)
        result = [sample for sample in result if sample.id in sample_id_set]
    return result
