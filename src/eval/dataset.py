"""Golden dataset loader and filter for the eval harness.

Reads ``wine_qa_golden.jsonl`` (one JSON object per line), validates each entry
against :class:`~src.eval.models.GoldenSample`, and exposes filtering helpers so
that runners can target specific subsets (by category, difficulty, or tag).
"""

import json
from pathlib import Path

from src.utils import logger

from .models import GoldenSample


class GoldenDataset:
    """Loader and filter for the golden Q&A evaluation dataset.

    The dataset is stored as a JSONL file (one JSON object per line). Each line is
    validated against :class:`~src.eval.models.GoldenSample` on load. Validation
    errors are raised immediately with the offending sample ``id`` included in the
    message so failures are easy to locate.

    Example::

        dataset = GoldenDataset()
        samples = dataset.load("tests/eval/wine_qa_golden.jsonl")
        rag_easy = dataset.filter(samples, categories=["rag_only"], difficulties=["easy"])
    """

    def load(self, path: str | Path) -> list[GoldenSample]:
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
            ValueError: If any line fails Pydantic validation. The error message
                includes the raw line content to help locate the offending entry.

        Example::

            samples = GoldenDataset().load("tests/eval/wine_qa_golden.jsonl")
            print(f"Loaded {len(samples)} samples")
        """
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")

        samples: list[GoldenSample] = []
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue  # skip blank lines
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
                        f"Validation failed for sample {sample_id!r} "
                        f"(line {line_number}): {exc}"
                    ) from exc
                samples.append(sample)

        logger.info("Loaded %d samples from %s", len(samples), dataset_path)
        return samples

    def filter(
        self,
        samples: list[GoldenSample],
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[GoldenSample]:
        """Return a subset of samples matching the provided filter criteria.

        All supplied filters are applied as AND conditions. Passing ``None`` (or
        omitting) a filter means "do not filter on this dimension". An empty filter
        list (e.g., ``categories=[]``) matches nothing — be explicit if you want all.

        Args:
            samples: The full list of samples to filter.
            categories: If set, keep only samples whose ``category`` is in this list.
            difficulties: If set, keep only samples whose ``difficulty`` is in this list.
            tags: If set, keep only samples that have **at least one** of these tags.

        Returns:
            Filtered list of :class:`~src.eval.models.GoldenSample` objects. The
            original list is not modified.

        Example::

            easy_rag = dataset.filter(
                samples,
                categories=["rag_only"],
                difficulties=["easy"],
            )
        """
        result = samples
        if categories is not None:
            category_set = set(categories)
            result = [s for s in result if s.category in category_set]
        if difficulties is not None:
            difficulty_set = set(difficulties)
            result = [s for s in result if s.difficulty in difficulty_set]
        if tags is not None:
            tag_set = set(tags)
            result = [s for s in result if tag_set.intersection(s.tags)]
        return result

