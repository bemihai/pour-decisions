"""Reusable grape records and explicit varietal linking."""

import re
import sqlite3
from pathlib import Path

from src.database.db import get_db_connection
from src.database.models import Grape
from src.utils import get_default_db_path
from src.utils.terms import GRAPE_PATTERNS


def extract_explicit_grapes(varietal: str | None) -> list[str]:
    """Match existing terminology, preferring longer non-overlapping names.

    Args:
        varietal: Free-text varietal field; generic blends are not expanded.

    Returns:
        Unique canonical names in alphabetical order.
    """
    text = (varietal or "").lower()
    matches = []
    for term, canonical in GRAPE_PATTERNS.items():
        for match in re.finditer(rf"\b{re.escape(term)}\b", text):
            matches.append((match.start(), match.end(), canonical))
    occupied: list[tuple[int, int]] = []
    names: set[str] = set()
    for start, end, canonical in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
        if any(start < used_end and end > used_start for used_start, used_end in occupied):
            continue
        occupied.append((start, end))
        names.add(canonical)
    return sorted(names)


def link_wine_grapes(cursor: sqlite3.Cursor, wine_id: int, varietal: str | None) -> None:
    """Replace a wine's links using explicit names in the caller's transaction."""
    names = extract_explicit_grapes(varietal)
    cursor.execute("DELETE FROM wine_grapes WHERE wine_id = ?", (wine_id,))
    for name in names:
        cursor.execute("INSERT INTO grapes (name) VALUES (?) ON CONFLICT(name) DO NOTHING", (name,))
        cursor.execute(
            "INSERT INTO wine_grapes (wine_id, grape_id) SELECT ?, id FROM grapes WHERE name = ?",
            (wine_id, name),
        )


class GrapeRepository:
    """Read shared descriptions and maintain explicit wine/grape relationships."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Use an explicit test/import database or the configured cellar database."""
        self.db_path = db_path if db_path is not None else get_default_db_path()

    def get_by_id(self, grape_id: int) -> Grape | None:
        """Read a grape and its persisted description."""
        with get_db_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM grapes WHERE id = ?", (grape_id,)).fetchone()
            return Grape(**dict(row)) if row else None

    def get_by_wine(self, wine_id: int) -> list[Grape]:
        """Return every linked grape in deterministic alphabetical order."""
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                "SELECT g.* FROM grapes g JOIN wine_grapes wg ON wg.grape_id = g.id "
                "WHERE wg.wine_id = ? ORDER BY g.name COLLATE NOCASE, g.name, g.id",
                (wine_id,),
            ).fetchall()
            return [Grape(**dict(row)) for row in rows]

    def sync_wine(self, wine_id: int, varietal: str | None) -> None:
        """Synchronize links without changing cached descriptions or varietal text."""
        with get_db_connection(self.db_path) as conn:
            link_wine_grapes(conn.cursor(), wine_id, varietal)
            conn.commit()

    def update_description(self, grape_id: int, description: str) -> bool:
        """Persist a successful generation for every wine sharing the grape."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE grapes SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (description, grape_id),
            )
            conn.commit()
            return cursor.rowcount == 1
