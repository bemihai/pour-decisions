"""Dataset staleness validator for the golden eval dataset.

Checks whether the cellar-dependent questions in ``wine_qa_golden.jsonl`` are still
answerable given the current state of the wine cellar database. Run this before
trusting eval results, especially after bulk cellar changes (syncs, imports, consumed
wines).

Staleness means the cellar question assumes a wine is present but it is no longer in
the DB — the system will correctly answer "not found", but the ground truth says the
wine should be there. This produces false-negative Ragas scores that look like system
regressions but are actually dataset drift.

Usage::

    python -m src.eval.scripts.dataset_validator
    python -m src.eval.scripts.dataset_validator --dataset src/eval/wine_qa_golden.jsonl
    python -m src.eval.scripts.dataset_validator --json   # machine-readable output

Exit codes:
    0 — all cellar-dependent questions are valid
    1 — one or more questions are stale or the DB is unreachable
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.database.db import get_db_connection
from src.utils import logger, get_default_db_path

from src.eval.dataset import load_golden_dataset

# ---------------------------------------------------------------------------
# Staleness checks: one function per cellar question pattern
# ---------------------------------------------------------------------------

# Maps a keyword in a question/tag to an SQL fragment that checks presence.
# Each entry is (description, SQL query returning exactly 1 row with a count).
_PRESENCE_CHECKS: list[tuple[str, list[str], str]] = [
    # (human label, trigger tags (ANY match triggers check), SQL counting matching wines)
    (
        "Nebbiolo / Barolo",
        ["nebbiolo", "barolo"],
        "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 AND varietal LIKE '%Nebbiolo%'",
    ),
    (
        "Chateauneuf-du-Pape",
        ["chateauneuf_du_pape"],
        # Appellation stored with UTF-8 multibyte â; match on the ASCII suffix instead.
        "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 AND appellation LIKE '%-du-Pape%'",
    ),
    (
        "Bandol",
        ["bandol"],
        "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 AND appellation LIKE '%Bandol%'",
    ),
    (
        "Rioja",
        ["rioja"],
        "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 AND appellation LIKE '%Rioja%'",
    ),
    (
        "Saint-Estephe",
        ["saint_estephe"],
        # Appellation stored as 'Saint-Est<è>phe' with UTF-8 accent; match on ASCII prefix.
        "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 AND appellation LIKE '%Saint-Est%'",
    ),
    (
        "Saint-Emilion Grand Cru",
        ["saint_emilion"],
        # Stored as 'Saint-<é>milion Grand Cru'; match on the ASCII suffix.
        "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 AND appellation LIKE '%milion Grand Cru%'",
    ),
    (
        "Bordeaux appellations",
        ["bordeaux"],
        (
            "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 "
            "AND (appellation LIKE '%Margaux%' OR appellation LIKE '%Saint-Est%' "
            "     OR appellation LIKE '%milion%' OR appellation LIKE '%Bordeaux%' "
            "     OR appellation LIKE '%Pauillac%' OR appellation LIKE '%Saint-Jul%')"
        ),
    ),
    (
        "Rhone Valley wines not yet ready",
        ["rhone"],
        (
            "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 "
            "AND CAST(drink_from_year AS INTEGER) > CAST(strftime('%Y', 'now') AS INTEGER) "
            "AND (appellation LIKE '%Pape%' OR appellation LIKE '%-du-Rh%' "
            "     OR wine_type = 'Red' AND varietal LIKE '%Rhone%')"
        ),
    ),
    (
        "Romanian wines not yet ready",
        ["romania"],
        (
            "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 "
            "AND CAST(drink_from_year AS INTEGER) > CAST(strftime('%Y', 'now') AS INTEGER) "
            "AND (varietal LIKE '%Fetea%' OR wine_name LIKE '%Crama%' "
            "     OR wine_name LIKE '%Prince%tirbey%' OR wine_name LIKE '%Serve%' "
            "     OR wine_name LIKE '%Avincis%' OR wine_name LIKE '%Darabont%')"
        ),
    ),
    (
        "Dealu Mare wines",
        ["dealu_mare"],
        "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 AND appellation LIKE '%Dealu%'",
    ),
    (
        "Wines not ready until 2027+",
        ["not_ready"],
        (
            "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 "
            "AND CAST(drink_from_year AS INTEGER) >= 2027"
        ),
    ),
    (
        "White wines not yet ready",
        ["white_wine", "not_ready"],
        (
            "SELECT COUNT(*) FROM wines WHERE q_quantity > 0 "
            "AND wine_type = 'White' "
            "AND CAST(drink_from_year AS INTEGER) > CAST(strftime('%Y', 'now') AS INTEGER)"
        ),
    ),
]


@dataclass
class ValidationIssue:
    """A single staleness issue found during validation.

    Attributes:
        sample_id: The ``id`` of the affected :class:`~src.eval.models.GoldenSample`.
        category: The sample's category (``cellar`` or ``multi_hop``).
        check_label: Human-readable description of the presence check that failed.
        detail: Additional context about the failure.
    """

    sample_id: str
    category: str
    check_label: str
    detail: str


@dataclass
class ValidationReport:
    """Full report from :func:`validate_dataset`.

    Attributes:
        total_samples: Total number of samples in the dataset.
        cellar_dependent: Number of samples that depend on live cellar data.
        issues: List of :class:`ValidationIssue` objects for stale questions.
        db_reachable: Whether the cellar DB was reachable during validation.
        db_path: Path to the cellar DB that was checked.
    """

    total_samples: int
    cellar_dependent: int
    issues: list[ValidationIssue] = field(default_factory=list)
    db_reachable: bool = True
    db_path: str = ""

    @property
    def is_clean(self) -> bool:
        """Return True when no staleness issues were found and DB was reachable."""
        return self.db_reachable and not self.issues

    @property
    def stale_count(self) -> int:
        """Number of stale samples found."""
        return len(self.issues)


def _tags_overlap(sample_tags: list[str], trigger_tags: list[str]) -> bool:
    """Return True only when the sample has ALL of the trigger tags.

    Using ALL-of semantics ensures that a check fires only for questions that are
    specifically about that combination. For example, the "White wines not yet ready"
    check requires both ``white_wine`` AND ``not_ready`` tags; a question about whites
    currently in their drinking window has ``white_wine`` but not ``not_ready`` and
    therefore will not trigger the not-ready presence check.

    Args:
        sample_tags: Tags from the :class:`~src.eval.models.GoldenSample`.
        trigger_tags: All tags that must be present to activate a check.

    Returns:
        True if every element of ``trigger_tags`` appears in ``sample_tags``.
    """
    sample_tag_set = set(sample_tags)
    return all(t in sample_tag_set for t in trigger_tags)


def validate_dataset(
    dataset_path: str | Path = "src/eval/wine_qa_golden.jsonl",
    db_path: str | None = None,
) -> ValidationReport:
    """Check whether cellar-dependent golden samples are still answerable.

    Loads the golden dataset, identifies samples in the ``cellar`` and
    ``multi_hop`` categories, and verifies that the wine types those samples
    query are still present (and non-zero quantity) in the live cellar DB.

    Args:
        dataset_path: Path to the golden JSONL file.
        db_path: Override the cellar DB path. Defaults to the absolute path
            returned by :func:`~src.utils.utils.get_default_db_path`.

    Returns:
        A :class:`ValidationReport` describing any staleness issues found.
    """
    samples = load_golden_dataset(dataset_path)
    cellar_samples = [s for s in samples if s.category in {"cellar", "multi_hop"}]

    report = ValidationReport(
        total_samples=len(samples),
        cellar_dependent=len(cellar_samples),
    )

    try:
        report.db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(get_default_db_path())
    except Exception:
        report.db_path = db_path or "cellar-data/wine_cellar.db"

    # Pre-run all presence checks once against the DB, build a results map.
    check_results: dict[str, bool] = {}
    try:
        with get_db_connection(report.db_path) as conn:
            for label, _trigger_tags, sql in _PRESENCE_CHECKS:
                row = conn.execute(sql).fetchone()
                count = row[0] if row else 0
                check_results[label] = count > 0
                if count == 0:
                    logger.warning("Presence check FAILED (count=0): %s", label)
    except Exception as exc:
        logger.error("Could not connect to cellar DB at %s: %s", report.db_path, exc)
        report.db_reachable = False
        return report

    # Map each cellar/multi_hop sample to the checks it depends on.
    for sample in cellar_samples:
        for label, trigger_tags, _sql in _PRESENCE_CHECKS:
            if not _tags_overlap(sample.tags, trigger_tags):
                continue
            if not check_results[label]:
                report.issues.append(
                    ValidationIssue(
                        sample_id=sample.id,
                        category=sample.category,
                        check_label=label,
                        detail=(
                            "Required wine type is no longer in the cellar "
                            f"(check: {label!r}). "
                            "Question: " + repr(sample.question[:80])
                        ),
                    )
                )

    if report.issues:
        logger.warning(
            "%d stale sample(s) detected in golden dataset. "
            "Run `python -m src.eval.scripts.dataset_validator` for details.",
            report.stale_count,
        )
    else:
        logger.info("Golden dataset validation passed — all %d cellar-dependent samples are valid.",
                    cellar_samples.__len__())

    return report


def _print_report(report: ValidationReport, use_json: bool = False) -> None:
    """Print the validation report to stdout.

    Args:
        report: The :class:`ValidationReport` to display.
        use_json: If True, output machine-readable JSON instead of human text.
    """
    if use_json:
        print(json.dumps({
            "total_samples": report.total_samples,
            "cellar_dependent": report.cellar_dependent,
            "db_reachable": report.db_reachable,
            "db_path": report.db_path,
            "stale_count": report.stale_count,
            "is_clean": report.is_clean,
            "issues": [
                {
                    "sample_id": i.sample_id,
                    "category": i.category,
                    "check_label": i.check_label,
                    "detail": i.detail,
                }
                for i in report.issues
            ],
        }, indent=2))
        return

    width = 68
    print(f"\n{'=' * width}")
    print(f"  Golden Dataset Validation Report")
    print(f"{'=' * width}")
    print(f"  DB path         : {report.db_path}")
    print(f"  Total samples   : {report.total_samples}")
    print(f"  Cellar-dependent: {report.cellar_dependent}")
    print(f"  DB reachable    : {'YES' if report.db_reachable else 'NO'}")
    print(f"  Stale questions : {report.stale_count}")
    print(f"{'=' * width}")

    if not report.db_reachable:
        print("\n  ERROR: Cellar DB is not reachable. Staleness check could not run.")
        print("  Eval results for cellar/multi_hop categories may be unreliable.\n")
        return

    if report.is_clean:
        print("\n  All cellar-dependent questions appear valid.\n")
        return

    print("\n  STALE QUESTIONS DETECTED")
    print("  These questions may produce false-negative eval scores because the")
    print("  wine they reference is no longer present in the cellar.\n")
    seen: set[str] = set()
    for issue in report.issues:
        key = issue.sample_id
        if key in seen:
            continue
        seen.add(key)
        print(f"  [{issue.category}] {issue.sample_id}")
        print(f"    Check  : {issue.check_label}")
        print(f"    Detail : {issue.detail[:100]}")
        print()
    print("  Recommended actions:")
    print("  1. Remove or update stale questions from wine_qa_golden.jsonl.")
    print("  2. Replace with questions about wines currently in the cellar.")
    print("  3. Re-run `python -m src.eval.scripts.dataset_validator` to confirm.\n")


def main() -> int:
    """Entry point for the dataset validator CLI.

    Returns:
        Exit code: 0 if clean, 1 if stale issues or DB unreachable.
    """
    parser = argparse.ArgumentParser(
        description="Validate cellar-dependent golden dataset questions against the live cellar DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="src/eval/wine_qa_golden.jsonl",
        help="Path to the golden JSONL file (default: src/eval/wine_qa_golden.jsonl)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Override the cellar DB path (default: from app_config.yml)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON instead of a human-readable report",
    )
    args = parser.parse_args()

    report = validate_dataset(dataset_path=args.dataset, db_path=args.db)
    _print_report(report, use_json=args.json)
    return 0 if report.is_clean else 1


if __name__ == "__main__":
    sys.exit(main())
