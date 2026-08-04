"""Unit tests for src/eval/models.py and src/eval/dataset.py (Phase 1).

Tests cover:
- GoldenSample Pydantic validation (valid sample, invalid category, invalid difficulty,
  missing required field).
- load_golden_dataset() with a 3-item fixture JSONL.
- filter_golden_samples() by category, difficulty, and tag.

No API calls, no ChromaDB, no filesystem fixtures beyond tmp_path.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval import GoldenSample, filter_golden_samples, load_golden_dataset
from src.eval.models import EvalRunResult, SampleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SAMPLE_DATA: dict = {
    "id": "rag_only_001",
    "question": "What are the primary flavour characteristics of Nebbiolo?",
    "category": "rag_only",
    "difficulty": "easy",
    "expected_facts": ["high tannins", "high acidity", "tar and roses aroma"],
    "expected_tool_calls": [],
    "ground_truth": (
        "Nebbiolo is known for high tannins, high acidity, and aromas of tar, roses, and dried fruit."
    ),
    "ground_truth_chunk_ids": [],
    "tags": ["grapes", "northern_italy"],
    "notes": None,
}


def _make_sample(**overrides) -> dict:
    """Return a valid sample dict with optional field overrides."""
    return {**VALID_SAMPLE_DATA, **overrides}


def _write_jsonl(path: Path, entries: list[dict]) -> Path:
    """Write a list of dicts to a JSONL file and return the path."""
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return path


# ---------------------------------------------------------------------------
# GoldenSample validation tests
# ---------------------------------------------------------------------------


class TestGoldenSampleValidation:
    """Tests for GoldenSample field validation."""

    def test_valid_sample_loads(self) -> None:
        """A fully valid sample dict is accepted without errors."""
        sample = GoldenSample.model_validate(VALID_SAMPLE_DATA)
        assert sample.id == "rag_only_001"
        assert sample.category == "rag_only"
        assert sample.difficulty == "easy"
        assert sample.ground_truth_chunk_ids == []
        assert sample.expected_tool_calls == []
        assert sample.notes is None

    def test_optional_fields_have_defaults(self) -> None:
        """Fields with defaults are populated even when absent from the input."""
        minimal = {
            "id": "cellar_001",
            "question": "How many bottles do I have?",
            "category": "cellar",
            "difficulty": "easy",
            "expected_facts": ["a specific count"],
            "ground_truth": "The user has N bottles in their cellar.",
            "tags": ["cellar"],
        }
        sample = GoldenSample.model_validate(minimal)
        assert sample.expected_tool_calls == []
        assert sample.ground_truth_chunk_ids == []
        assert sample.notes is None

    def test_invalid_category_raises_validation_error(self) -> None:
        """An unrecognised category string raises ValidationError."""
        bad = _make_sample(category="unknown")
        with pytest.raises(ValidationError, match="category must be one of"):
            GoldenSample.model_validate(bad)

    def test_invalid_difficulty_raises_validation_error(self) -> None:
        """An unrecognised difficulty string raises ValidationError."""
        bad = _make_sample(difficulty="impossible")
        with pytest.raises(ValidationError, match="difficulty must be one of"):
            GoldenSample.model_validate(bad)

    def test_missing_required_field_raises_validation_error(self) -> None:
        """Omitting a required field (ground_truth) raises ValidationError."""
        bad = {k: v for k, v in VALID_SAMPLE_DATA.items() if k != "ground_truth"}
        with pytest.raises(ValidationError):
            GoldenSample.model_validate(bad)

    def test_missing_question_raises_validation_error(self) -> None:
        """Omitting the question field raises ValidationError."""
        bad = {k: v for k, v in VALID_SAMPLE_DATA.items() if k != "question"}
        with pytest.raises(ValidationError):
            GoldenSample.model_validate(bad)

    @pytest.mark.parametrize("category", ["rag_only", "cellar", "pairing", "multi_hop"])
    def test_all_valid_categories_accepted(self, category: str) -> None:
        """Every valid category value is accepted."""
        sample = GoldenSample.model_validate(_make_sample(category=category))
        assert sample.category == category

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_all_valid_difficulties_accepted(self, difficulty: str) -> None:
        """Every valid difficulty value is accepted."""
        sample = GoldenSample.model_validate(_make_sample(difficulty=difficulty))
        assert sample.difficulty == difficulty


# ---------------------------------------------------------------------------
# load_golden_dataset() tests
# ---------------------------------------------------------------------------


class TestGoldenDatasetLoad:
    """Tests for load_golden_dataset()."""

    def test_load_three_item_fixture(self, tmp_path: Path) -> None:
        """load() returns one GoldenSample per non-blank line."""
        entries = [
            _make_sample(id="rag_only_001"),
            _make_sample(id="cellar_001", category="cellar"),
            _make_sample(id="pairing_001", category="pairing"),
        ]
        jsonl_path = _write_jsonl(tmp_path / "fixture.jsonl", entries)

        samples = load_golden_dataset(jsonl_path)

        assert len(samples) == 3
        assert [s.id for s in samples] == ["rag_only_001", "cellar_001", "pairing_001"]

    def test_load_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines in the JSONL file are silently skipped."""
        jsonl_path = tmp_path / "blanks.jsonl"
        with jsonl_path.open("w") as fh:
            fh.write(json.dumps(_make_sample(id="rag_only_001")) + "\n")
            fh.write("\n")
            fh.write("   \n")
            fh.write(json.dumps(_make_sample(id="rag_only_002")) + "\n")

        samples = load_golden_dataset(jsonl_path)
        assert len(samples) == 2

    def test_load_raises_file_not_found(self) -> None:
        """load() raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Golden dataset not found"):
            load_golden_dataset("/nonexistent/path/to/dataset.jsonl")

    def test_load_raises_on_invalid_json_line(self, tmp_path: Path) -> None:
        """load() raises ValueError when a line is not valid JSON."""
        bad_path = tmp_path / "bad.jsonl"
        # Line 1 is a complete valid sample; line 2 is malformed JSON.
        line1 = json.dumps(VALID_SAMPLE_DATA)
        bad_path.write_text(f"{line1}\nNOT_JSON\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_golden_dataset(bad_path)

    def test_load_raises_on_schema_violation(self, tmp_path: Path) -> None:
        """load() raises ValueError with the offending sample id on schema violation."""
        bad_entry = _make_sample(id="bad_001", category="does_not_exist")
        bad_path = _write_jsonl(tmp_path / "bad_schema.jsonl", [bad_entry])

        with pytest.raises(ValueError, match="bad_001"):
            load_golden_dataset(bad_path)

    def test_load_returns_correct_types(self, tmp_path: Path) -> None:
        """Each item returned by load() is a GoldenSample instance."""
        jsonl_path = _write_jsonl(tmp_path / "types.jsonl", [VALID_SAMPLE_DATA])
        samples = load_golden_dataset(jsonl_path)

        assert len(samples) == 1
        assert isinstance(samples[0], GoldenSample)


# ---------------------------------------------------------------------------
# filter_golden_samples() tests
# ---------------------------------------------------------------------------


class TestGoldenDatasetFilter:
    """Tests for filter_golden_samples()."""

    @pytest.fixture()
    def mixed_samples(self) -> list[GoldenSample]:
        """A small mixed dataset covering all categories and difficulties."""
        raw = [
            _make_sample(id="rag_001", category="rag_only", difficulty="easy", tags=["grapes"]),
            _make_sample(id="rag_002", category="rag_only", difficulty="medium", tags=["regions"]),
            _make_sample(id="cel_001", category="cellar", difficulty="easy", tags=["cellar", "inventory"]),
            _make_sample(id="par_001", category="pairing", difficulty="hard", tags=["pairing"]),
            _make_sample(id="mh_001", category="multi_hop", difficulty="hard", tags=["cellar", "grapes"]),
        ]
        return [GoldenSample.model_validate(d) for d in raw]

    def test_filter_by_category(self, mixed_samples: list[GoldenSample]) -> None:
        """Filtering by a single category returns only matching samples."""
        result = filter_golden_samples(mixed_samples, categories=["rag_only"])
        assert len(result) == 2
        assert all(s.category == "rag_only" for s in result)

    def test_filter_by_multiple_categories(self, mixed_samples: list[GoldenSample]) -> None:
        """Multiple categories are combined as OR within that dimension."""
        result = filter_golden_samples(mixed_samples, categories=["rag_only", "cellar"])
        assert len(result) == 3

    def test_filter_by_difficulty(self, mixed_samples: list[GoldenSample]) -> None:
        """Filtering by difficulty returns only matching samples."""
        result = filter_golden_samples(mixed_samples, difficulties=["easy"])
        assert len(result) == 2
        assert all(s.difficulty == "easy" for s in result)

    def test_filter_by_category_and_difficulty(self, mixed_samples: list[GoldenSample]) -> None:
        """Category and difficulty filters are applied as AND."""
        result = filter_golden_samples(
            mixed_samples,
            categories=["rag_only"],
            difficulties=["medium"],
        )
        assert len(result) == 1
        assert result[0].id == "rag_002"

    def test_filter_by_tag(self, mixed_samples: list[GoldenSample]) -> None:
        """Filtering by tag keeps samples that have at least one matching tag."""
        result = filter_golden_samples(mixed_samples, tags=["cellar"])
        assert len(result) == 2
        assert {s.id for s in result} == {"cel_001", "mh_001"}

    def test_filter_by_sample_id(self, mixed_samples: list[GoldenSample]) -> None:
        """Filtering by sample id returns only the requested rows."""
        result = filter_golden_samples(mixed_samples, sample_ids=["rag_002", "par_001"])
        assert {s.id for s in result} == {"rag_002", "par_001"}

    def test_filter_by_sample_id_and_tag(self, mixed_samples: list[GoldenSample]) -> None:
        """Sample id filters combine with other dimensions as AND conditions."""
        result = filter_golden_samples(mixed_samples, sample_ids=["mh_001", "par_001"], tags=["cellar"])
        assert [s.id for s in result] == ["mh_001"]

    def test_filter_no_criteria_returns_all(self, mixed_samples: list[GoldenSample]) -> None:
        """Calling filter() with no criteria returns the full list unchanged."""
        result = filter_golden_samples(mixed_samples)
        assert len(result) == len(mixed_samples)

    def test_filter_does_not_mutate_input(self, mixed_samples: list[GoldenSample]) -> None:
        """filter() returns a new list and does not modify the original."""
        original_ids = [s.id for s in mixed_samples]
        filter_golden_samples(mixed_samples, categories=["cellar"])
        assert [s.id for s in mixed_samples] == original_ids

    def test_filter_empty_category_list_returns_nothing(self, mixed_samples: list[GoldenSample]) -> None:
        """An explicit empty categories list matches nothing."""
        result = filter_golden_samples(mixed_samples, categories=[])
        assert result == []

    def test_filter_nonexistent_tag_returns_empty(self, mixed_samples: list[GoldenSample]) -> None:
        """A tag that no sample has results in an empty list."""
        result = filter_golden_samples(mixed_samples, tags=["does_not_exist"])
        assert result == []


class TestEvalResultSchemaCompatibility:
    """Tests for versioned eval-result confidence fields."""

    def test_schema_v7_confidence_artifacts_round_trip(self) -> None:
        """Schema-v7 JSON should preserve confidence and a numeric zero threshold."""
        run = EvalRunResult(
            run_id="20260804T120000",
            timestamp="2026-08-04T12:00:00Z",
            mode="retrieval",
            backend="rag",
            per_sample=[
                SampleResult(
                    id="rag_only_001",
                    question="What is Barolo?",
                    retrieval_confidence=0.8,
                    low_confidence=False,
                    rerank_threshold=0.0,
                    context_chunks=[
                        {
                            "id": "chunk-1",
                            "text": "Barolo context",
                            "rerank_score": 1.3862943611198908,
                        }
                    ],
                    rag_feature_flags={"reranking": True, "rerank_thresholding": True},
                )
            ],
        )

        restored = EvalRunResult.model_validate_json(run.model_dump_json())

        assert restored.schema_version == 7
        assert restored.per_sample[0].retrieval_confidence == 0.8
        assert restored.per_sample[0].low_confidence is False
        assert restored.per_sample[0].rerank_threshold == 0.0
        assert restored.per_sample[0].context_chunks[0].rerank_score == pytest.approx(1.3862943611198908)
        assert restored.per_sample[0].rag_feature_flags["rerank_thresholding"] is True

    @pytest.mark.parametrize("schema_version", [1, 2, 3, 4, 5, 6])
    def test_legacy_schema_samples_default_confidence_fields(self, schema_version: int) -> None:
        """Versions 1–6 without confidence fields should remain readable."""
        restored = EvalRunResult.model_validate(
            {
                "schema_version": schema_version,
                "run_id": "legacy",
                "timestamp": "2026-07-28T00:00:00Z",
                "mode": "retrieval",
                "backend": "rag",
                "per_sample": [{"id": "rag_only_001", "question": "Legacy question"}],
            }
        )

        assert restored.schema_version == schema_version
        assert restored.per_sample[0].retrieval_confidence is None
        assert restored.per_sample[0].low_confidence is False
        assert restored.per_sample[0].rerank_threshold is None
