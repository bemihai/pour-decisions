"""Integration tests for the golden dataset file itself (Phase 2).

Validates that ``tests/eval/wine_qa_golden.jsonl`` meets the structural and
distribution requirements defined in the design spec:

- All 60 entries load without validation errors.
- Category distribution matches the spec (±5 samples tolerance).
- Difficulty distribution is within the expected range.
- All ``rag_only`` samples have a non-empty ``ground_truth``.
- All ``cellar`` and ``multi_hop`` samples have expected tool calls listed.
- No duplicate IDs.
- All IDs follow the ``{category}_{NNN}`` format.

No API calls, no ChromaDB, no LLM — pure filesystem + Pydantic validation.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

from src.eval import GoldenDataset, GoldenSample

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOLDEN_JSONL = Path(__file__).parent / "wine_qa_golden.jsonl"
EXPECTED_TOTAL = 60

CATEGORY_TARGETS = {
    "rag_only": 25,
    "cellar": 15,
    "pairing": 10,
    "multi_hop": 10,
}
CATEGORY_TOLERANCE = 5  # ±5 samples per category

DIFFICULTY_TARGETS = {
    "easy": 20,
    "medium": 25,
    "hard": 15,
}
DIFFICULTY_TOLERANCE = 8  # broader tolerance since difficulty is subjective

ID_PATTERN = re.compile(r"^(rag_only|cellar|pairing|multi_hop)_\d{3}$")


@pytest.fixture(scope="module")
def golden_samples() -> list[GoldenSample]:
    """Load the actual golden dataset file once for all tests in this module."""
    if not GOLDEN_JSONL.exists():
        pytest.skip(f"Golden dataset not found at {GOLDEN_JSONL}")
    return GoldenDataset().load(GOLDEN_JSONL)


# ---------------------------------------------------------------------------
# Loading & schema tests
# ---------------------------------------------------------------------------


class TestGoldenDatasetLoading:
    """Tests that the golden JSONL file loads and validates cleanly."""

    def test_file_exists(self) -> None:
        """The golden JSONL file must exist at the expected path."""
        if not GOLDEN_JSONL.exists():
            pytest.skip(f"Golden dataset not found at {GOLDEN_JSONL}")
        assert GOLDEN_JSONL.exists()

    def test_all_entries_load_without_errors(self, golden_samples: list[GoldenSample]) -> None:
        """All entries must parse and pass Pydantic validation."""
        # If load() raised, the fixture itself would fail; this just checks count.
        assert len(golden_samples) > 0

    def test_total_sample_count(self, golden_samples: list[GoldenSample]) -> None:
        """Dataset must contain exactly the target number of samples."""
        assert len(golden_samples) == EXPECTED_TOTAL, (
            f"Expected {EXPECTED_TOTAL} samples, found {len(golden_samples)}"
        )

    def test_no_duplicate_ids(self, golden_samples: list[GoldenSample]) -> None:
        """Every sample must have a unique id."""
        ids = [s.id for s in golden_samples]
        duplicates = [id_ for id_, count in Counter(ids).items() if count > 1]
        assert not duplicates, f"Duplicate IDs found: {duplicates}"

    def test_all_ids_follow_naming_convention(self, golden_samples: list[GoldenSample]) -> None:
        """IDs must match the pattern ``{category}_{NNN}`` (three-digit zero-padded)."""
        bad_ids = [s.id for s in golden_samples if not ID_PATTERN.match(s.id)]
        assert not bad_ids, f"IDs with invalid format: {bad_ids}"

    def test_id_prefix_matches_category(self, golden_samples: list[GoldenSample]) -> None:
        """The prefix of each ID must match the sample's category field."""
        mismatched = [
            s.id for s in golden_samples
            if not s.id.startswith(s.category)
        ]
        assert not mismatched, f"ID prefix does not match category: {mismatched}"


# ---------------------------------------------------------------------------
# Category distribution tests
# ---------------------------------------------------------------------------


class TestCategoryDistribution:
    """Tests that the category distribution is within the design spec targets."""

    def test_all_four_categories_present(self, golden_samples: list[GoldenSample]) -> None:
        """All four required categories must appear in the dataset."""
        categories = {s.category for s in golden_samples}
        assert categories == {"rag_only", "cellar", "pairing", "multi_hop"}

    @pytest.mark.parametrize("category,target", CATEGORY_TARGETS.items())
    def test_category_count_within_tolerance(
        self, golden_samples: list[GoldenSample], category: str, target: int
    ) -> None:
        """Each category count must be within ±CATEGORY_TOLERANCE of the target."""
        count = sum(1 for s in golden_samples if s.category == category)
        assert abs(count - target) <= CATEGORY_TOLERANCE, (
            f"Category {category!r}: expected {target} ± {CATEGORY_TOLERANCE}, got {count}"
        )


# ---------------------------------------------------------------------------
# Difficulty distribution tests
# ---------------------------------------------------------------------------


class TestDifficultyDistribution:
    """Tests that the difficulty distribution is within spec."""

    def test_all_three_difficulties_present(self, golden_samples: list[GoldenSample]) -> None:
        """All three difficulty levels must appear in the dataset."""
        difficulties = {s.difficulty for s in golden_samples}
        assert difficulties == {"easy", "medium", "hard"}

    @pytest.mark.parametrize("difficulty,target", DIFFICULTY_TARGETS.items())
    def test_difficulty_count_within_tolerance(
        self, golden_samples: list[GoldenSample], difficulty: str, target: int
    ) -> None:
        """Each difficulty count must be within ±DIFFICULTY_TOLERANCE of the target."""
        count = sum(1 for s in golden_samples if s.difficulty == difficulty)
        assert abs(count - target) <= DIFFICULTY_TOLERANCE, (
            f"Difficulty {difficulty!r}: expected {target} ± {DIFFICULTY_TOLERANCE}, got {count}"
        )


# ---------------------------------------------------------------------------
# Content quality tests
# ---------------------------------------------------------------------------


class TestContentQuality:
    """Light-touch quality checks on ground_truth and expected_facts."""

    def test_all_samples_have_non_empty_ground_truth(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Every sample must have a non-empty ground_truth string."""
        empty = [s.id for s in golden_samples if not s.ground_truth.strip()]
        assert not empty, f"Samples with empty ground_truth: {empty}"

    def test_all_rag_only_have_substantive_ground_truth(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """RAG-only ground truths must be at least 30 characters (full sentence, not a stub)."""
        short = [
            s.id
            for s in golden_samples
            if s.category == "rag_only" and len(s.ground_truth.strip()) < 30
        ]
        assert not short, f"rag_only samples with suspiciously short ground_truth: {short}"

    def test_all_samples_have_at_least_one_expected_fact(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Every sample must declare at least one expected fact for LLM-as-judge scoring."""
        empty = [s.id for s in golden_samples if not s.expected_facts]
        assert not empty, f"Samples with no expected_facts: {empty}"

    def test_all_samples_have_at_least_one_tag(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Every sample must have at least one tag."""
        empty = [s.id for s in golden_samples if not s.tags]
        assert not empty, f"Samples with no tags: {empty}"

    def test_cellar_and_multi_hop_have_tool_calls(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Every cellar and multi_hop sample must list expected tool calls."""
        missing = [
            s.id
            for s in golden_samples
            if s.category in {"cellar", "multi_hop"} and not s.expected_tool_calls
        ]
        assert not missing, (
            f"cellar/multi_hop samples missing expected_tool_calls: {missing}"
        )

    def test_rag_only_have_no_tool_calls(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """RAG-only samples must not list tool calls (they rely on retrieval only)."""
        with_tools = [
            s.id for s in golden_samples
            if s.category == "rag_only" and s.expected_tool_calls
        ]
        assert not with_tools, f"rag_only samples unexpectedly listing tool calls: {with_tools}"

    def test_cellar_samples_have_skip_notes(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Cellar samples should mention skip behaviour in notes for empty-DB environments."""
        missing_notes = [
            s.id
            for s in golden_samples
            if s.category == "cellar" and (s.notes is None or "skip" not in s.notes.lower())
        ]
        assert not missing_notes, (
            f"cellar samples missing skip note: {missing_notes}"
        )


# ---------------------------------------------------------------------------
# Filter integration test
# ---------------------------------------------------------------------------


class TestDatasetFilter:
    """Integration tests combining load() and filter() on the real golden file."""

    def test_filter_rag_only_returns_correct_count(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Filtering for rag_only must return 25 samples (spec target)."""
        result = GoldenDataset().filter(golden_samples, categories=["rag_only"])
        assert len(result) == CATEGORY_TARGETS["rag_only"]

    def test_filter_cellar_returns_correct_count(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Filtering for cellar must return 15 samples (spec target)."""
        result = GoldenDataset().filter(golden_samples, categories=["cellar"])
        assert len(result) == CATEGORY_TARGETS["cellar"]

    def test_filter_by_tag_not_ready(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Tag 'not_ready' should return more than 0 samples."""
        result = GoldenDataset().filter(golden_samples, tags=["not_ready"])
        assert len(result) > 0

    def test_filter_by_tag_tool_required(
        self, golden_samples: list[GoldenSample]
    ) -> None:
        """Tag 'tool_required' should only appear in cellar or multi_hop categories."""
        result = GoldenDataset().filter(golden_samples, tags=["tool_required"])
        assert all(s.category in {"cellar", "multi_hop"} for s in result)

