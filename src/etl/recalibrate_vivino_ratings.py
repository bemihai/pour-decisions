"""Preview or apply the approved personal Vivino rating recalibration."""

import argparse
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

from src.database import get_db_connection
from src.etl.utils import convert_vivino_personal_rating, generate_external_id
from src.etl.vivino_importer import load_full_wine_list_data
from src.utils import get_default_db_path, get_project_root, logger


@dataclass(frozen=True)
class RecalibrationReport:
    """Summary of one Vivino personal-rating recalibration run."""

    matched: int
    changed: int
    unchanged: int
    unmatched: int
    unrated: int
    invalid: int
    old_distribution: dict[str, int]
    new_distribution: dict[str, int]
    applied: bool

    def to_dict(self) -> dict[str, object]:
        """Return a serializable representation of the report."""
        return asdict(self)


def validate_database_connection(connection: sqlite3.Connection) -> None:
    """Require successful SQLite integrity and foreign-key checks.

    Args:
        connection: Open SQLite connection to validate.

    Raises:
        RuntimeError: If either database validation fails.
    """
    integrity_result = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity_result is None or integrity_result[0] != "ok":
        raise RuntimeError(f"Database integrity check failed: {integrity_result}")

    foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_errors:
        raise RuntimeError(f"Database foreign-key check failed with {len(foreign_key_errors)} violation(s)")


def verify_backup(backup_path: str | Path, database_path: str | Path) -> None:
    """Verify that a distinct, readable cellar database backup is healthy.

    Args:
        backup_path: Existing SQLite backup to verify.
        database_path: Database that will be recalibrated.

    Raises:
        ValueError: If the backup is missing or points to the live database.
        RuntimeError: If SQLite cannot validate the backup.
    """
    backup = Path(backup_path).expanduser().resolve()
    database = Path(database_path).expanduser().resolve()
    if not backup.is_file():
        raise ValueError(f"Backup does not exist or is not a file: {backup}")
    if backup == database:
        raise ValueError("Backup path must be different from the database being recalibrated")

    backup_uri = f"file:{quote(str(backup))}?mode=ro"
    try:
        with sqlite3.connect(backup_uri, uri=True) as connection:
            validate_database_connection(connection)
    except sqlite3.DatabaseError as error:
        raise RuntimeError(f"Could not verify backup {backup}: {error}") from error


def recalibrate_vivino_ratings(
    csv_path: str | Path,
    database_path: str | Path,
    *,
    apply: bool = False,
    backup_path: str | Path | None = None,
) -> RecalibrationReport:
    """Preview or atomically apply personal Vivino rating updates.

    Counts are per grouped wine except ``invalid``, which counts individual
    non-empty rating fields that are malformed or outside Vivino's 0-5 range.
    A group with both valid and invalid scans can therefore contribute to both
    ``matched`` and ``invalid``.

    Args:
        csv_path: Vivino ``full_wine_list.csv`` export.
        database_path: SQLite cellar database to inspect or update.
        apply: Whether to write the planned updates.
        backup_path: Verified SQLite backup required when ``apply`` is true.

    Returns:
        Recalibration counts and before/after score distributions.

    Raises:
        ValueError: If apply mode has no valid backup or an input is missing.
        RuntimeError: If database validation fails.
        sqlite3.DatabaseError: If a database operation fails.
    """
    csv_file = Path(csv_path)
    database_file = Path(database_path)
    if not csv_file.is_file():
        raise ValueError(f"Vivino CSV does not exist: {csv_file}")
    if not database_file.is_file():
        raise ValueError(f"Cellar database does not exist: {database_file}")
    if apply:
        if backup_path is None:
            raise ValueError("Apply mode requires --backup-path pointing to a verified cellar backup")
        verify_backup(backup_path, database_file)

    wines_data = load_full_wine_list_data(csv_file)
    matched = 0
    changed = 0
    unmatched = 0
    unrated = 0
    invalid = sum(data["invalid_ratings"] for data in wines_data.values())
    old_distribution: Counter[str] = Counter()
    new_distribution: Counter[str] = Counter()
    updates: list[tuple[int, bool, int]] = []

    with get_db_connection(database_file) as connection:
        if apply:
            connection.execute("BEGIN IMMEDIATE")

        try:
            for (winery, wine_name, vintage), data in wines_data.items():
                if not data["ratings"]:
                    if data["invalid_ratings"] == 0:
                        unrated += 1
                    continue

                raw_rating = max(data["ratings"])
                new_rating = convert_vivino_personal_rating(raw_rating)
                if new_rating is None:
                    # Ratings were parsed before grouping, so this is defensive.
                    invalid += 1
                    continue

                external_id = generate_external_id(winery, wine_name, vintage)
                row = connection.execute(
                    """
                    SELECT t.id AS tasting_id, t.personal_rating, t.do_like
                    FROM wines w
                    JOIN tastings t ON t.wine_id = w.id
                    WHERE w.source = 'vivino' AND w.external_id = ?
                    """,
                    (external_id,),
                ).fetchone()
                if row is None:
                    unmatched += 1
                    continue

                matched += 1
                old_key = "unrated" if row["personal_rating"] is None else str(row["personal_rating"])
                old_distribution[old_key] += 1
                new_distribution[str(new_rating)] += 1

                new_do_like = raw_rating >= 3.5
                old_do_like = None if row["do_like"] is None else bool(row["do_like"])
                if row["personal_rating"] == new_rating and old_do_like == new_do_like:
                    continue

                changed += 1
                updates.append((new_rating, new_do_like, row["tasting_id"]))

            if apply:
                connection.executemany(
                    """
                    UPDATE tastings
                    SET personal_rating = ?, do_like = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    updates,
                )
                validate_database_connection(connection)
                connection.commit()
        except Exception:
            if apply:
                connection.rollback()
            raise

    return RecalibrationReport(
        matched=matched,
        changed=changed,
        unchanged=matched - changed,
        unmatched=unmatched,
        unrated=unrated,
        invalid=invalid,
        old_distribution=dict(sorted(old_distribution.items())),
        new_distribution=dict(sorted(new_distribution.items())),
        applied=apply,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the recalibration command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=get_project_root() / "cellar-data/vivino/full_wine_list.csv",
        help="Vivino full_wine_list.csv export",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=get_default_db_path(),
        help="Cellar SQLite database",
    )
    parser.add_argument("--apply", action="store_true", help="Apply updates; the default is a read-only preview")
    parser.add_argument("--backup-path", type=Path, help="Verified cellar backup required with --apply")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Vivino personal-rating recalibration command."""
    args = build_parser().parse_args(argv)
    try:
        report = recalibrate_vivino_ratings(
            args.csv_path,
            args.db_path,
            apply=args.apply,
            backup_path=args.backup_path,
        )
    except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as error:
        logger.error("Vivino rating recalibration failed: %s", error)
        return 1

    mode = "APPLIED" if report.applied else "PREVIEW"
    logger.info("Vivino personal-rating recalibration %s", mode)
    logger.info(
        "matched=%d changed=%d unchanged=%d unmatched=%d unrated=%d invalid=%d",
        report.matched,
        report.changed,
        report.unchanged,
        report.unmatched,
        report.unrated,
        report.invalid,
    )
    logger.info("old_distribution=%s", report.old_distribution)
    logger.info("new_distribution=%s", report.new_distribution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
