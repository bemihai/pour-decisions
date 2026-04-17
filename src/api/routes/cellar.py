"""Cellar API endpoints.

Exposes cellar inventory, statistics, chart data, filter options,
and CellarTracker sync. Business logic lives in the repository
layer; this module handles HTTP concerns, grouping, and sorting.

Note: all route handlers are synchronous (``def``). FastAPI runs them in a
thread-pool executor so the event loop remains unblocked. Migrating to async
I/O would require async database drivers and is tracked as a future improvement.
The CellarTracker sync endpoint may take 10-60+ seconds; consider migrating it
to a background-task / polling pattern for multi-user deployments.
"""
from datetime import datetime
import os
import re

from fastapi import APIRouter, HTTPException, Query

from src.database import get_db_connection
from src.database.utils import normalize_string
from src.api.schemas.cellar import (
    CellarOverview,
    CellarStatsResponse,
    CellarValueStats,
    ChartDataResponse,
    MergeDecisionRequest,
    MergeDecisionResponse,
    MergeSuggestion,
    MergeSuggestionsResponse,
    DrinkingWindowStats,
    DrinkNextItem,
    DrinkNextResponse,
    FilterOptions,
    InventoryItem,
    InventoryResponse,
    SyncResponse,
)
from src.database.repository import BottleRepository, StatsRepository
from src.utils import logger

router = APIRouter(prefix="/api/cellar", tags=["cellar"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_dev_mode() -> bool:
    """Return true only when manual merge tooling is explicitly enabled.

    Manual merge endpoints are destructive and are disabled by default.
    Set ``ENABLE_MANUAL_MERGE=true`` (or ``1``, ``yes``, ``on``) to
    expose these endpoints in controlled development environments.
    """
    manual_merge_enabled = os.getenv("ENABLE_MANUAL_MERGE", "").strip().lower()
    return manual_merge_enabled in {"1", "true", "yes", "on"}


def _ensure_dev_mode() -> None:
    """Block manual merge endpoints outside development mode."""
    if not _is_dev_mode():
        raise HTTPException(status_code=404, detail="Not found")


def _coalesce_text(current: str | None, incoming: str | None) -> str | None:
    """Keep current non-empty text, otherwise fallback to incoming value."""
    if current is not None and str(current).strip():
        return current
    if incoming is not None and str(incoming).strip():
        return incoming
    return current


def _resolve_iso_date(current: str | None, incoming: str | None) -> str | None:
    """Return the most recent YYYY-MM-DD date from two optional values."""
    candidates = [d for d in [current, incoming] if d]
    if not candidates:
        return None
    return max(candidates)


def _collect_producer_suggestions() -> list[MergeSuggestion]:
    """Build producer duplicate suggestions using normalized name grouping."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, country
            FROM producers
            WHERE name IS NOT NULL AND TRIM(name) != ''
            ORDER BY id ASC
            """
        ).fetchall()

    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = normalize_string(row["name"])
        if key:
            groups.setdefault(key, []).append(dict(row))

    suggestions: list[MergeSuggestion] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        keep = group[0]
        for duplicate in group[1:]:
            suggestions.append(
                MergeSuggestion(
                    suggestion_type="producer",
                    keep_id=keep["id"],
                    remove_id=duplicate["id"],
                    keep_label=f"{keep['name']} (#{keep['id']})",
                    remove_label=f"{duplicate['name']} (#{duplicate['id']})",
                    reason="Normalized producer name matches",
                )
            )
    return suggestions


def _collect_region_suggestions() -> list[MergeSuggestion]:
    """Build region duplicate suggestions using normalized region identity."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, primary_name, secondary_name, country
            FROM regions
            WHERE primary_name IS NOT NULL AND TRIM(primary_name) != ''
            ORDER BY id ASC
            """
        ).fetchall()

    groups: dict[str, list[dict]] = {}
    for row in rows:
        region = dict(row)
        key = "|".join(
            [
                normalize_string(region.get("primary_name") or ""),
                normalize_string(region.get("secondary_name") or ""),
                normalize_string(region.get("country") or ""),
            ]
        )
        if key.replace("|", ""):
            groups.setdefault(key, []).append(region)

    suggestions: list[MergeSuggestion] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        keep = group[0]
        keep_label = f"{keep['primary_name']} ({keep['country']}) (#{keep['id']})"
        for duplicate in group[1:]:
            remove_label = f"{duplicate['primary_name']} ({duplicate['country']}) (#{duplicate['id']})"
            suggestions.append(
                MergeSuggestion(
                    suggestion_type="region",
                    keep_id=keep["id"],
                    remove_id=duplicate["id"],
                    keep_label=keep_label,
                    remove_label=remove_label,
                    reason="Normalized region fields match",
                )
            )
    return suggestions


def _canonical_match_text(value: str | None) -> str:
    """Normalize free-text values for duplicate matching keys."""
    normalized = normalize_string(value or "")
    if not normalized:
        return ""
    # Split glued alnum sequences (e.g., "smerenie2016" -> "smerenie 2016").
    normalized = re.sub(r"([a-z])(\d)", r"\1 \2", normalized)
    normalized = re.sub(r"(\d)([a-z])", r"\1 \2", normalized)
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return " ".join(normalized.split())


def _normalize_wine_core_name(wine_name: str | None, producer_name: str | None) -> str:
    """Return a canonical wine name with producer prefix removed when present."""
    normalized_wine = _canonical_match_text(wine_name)
    normalized_producer = _canonical_match_text(producer_name)
    if not normalized_wine:
        return ""

    if not normalized_producer:
        return normalized_wine

    wine_tokens = normalized_wine.split()
    producer_tokens = normalized_producer.split()
    if wine_tokens[: len(producer_tokens)] == producer_tokens:
        core_tokens = wine_tokens[len(producer_tokens) :]
        if core_tokens:
            return " ".join(core_tokens)

    return normalized_wine


def _strip_vintage_from_wine_core(core_name: str, vintage: int | None) -> str:
    """Remove vintage tokens from wine core to match labels that embed year in name."""
    if not core_name:
        return ""
    if not vintage:
        return core_name

    year = str(vintage)
    # Remove isolated vintage token after canonical alnum splitting.
    cleaned = re.sub(rf"\b{re.escape(year)}\b", " ", core_name)
    return " ".join(cleaned.split())


def _collect_wine_suggestions() -> list[MergeSuggestion]:
    """Build wine duplicate suggestions using normalized wine identity."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                w.id,
                w.wine_name,
                w.vintage,
                w.wine_type,
                w.producer_id,
                w.region_id,
                p.name AS producer_name,
                r.primary_name AS region_name
            FROM wines w
            LEFT JOIN producers p ON w.producer_id = p.id
            LEFT JOIN regions r ON w.region_id = r.id
            WHERE w.wine_name IS NOT NULL AND TRIM(w.wine_name) != ''
            ORDER BY w.id ASC
            """
        ).fetchall()

    groups: dict[str, list[dict]] = {}
    for row in rows:
        wine = dict(row)
        producer_name_key = _canonical_match_text(wine.get("producer_name") or "")
        producer_key = producer_name_key or f"id:{wine.get('producer_id') or ''}"
        region_name_key = _canonical_match_text(wine.get("region_name") or "")
        region_key = region_name_key or f"id:{wine.get('region_id') or ''}"
        wine_core_name = _strip_vintage_from_wine_core(
            core_name=_normalize_wine_core_name(
            wine_name=wine.get("wine_name"),
            producer_name=wine.get("producer_name"),
            ),
            vintage=wine.get("vintage"),
        )

        key = "|".join(
            [
                wine_core_name,
                str(wine.get("vintage") or ""),
                _canonical_match_text(wine.get("wine_type") or ""),
                producer_key,
                region_key,
            ]
        )
        if key.replace("|", ""):
            groups.setdefault(key, []).append(wine)

    suggestions: list[MergeSuggestion] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        keep = group[0]
        keep_label = f"{keep['wine_name']} {keep.get('vintage') or 'NV'} (#{keep['id']})"
        for duplicate in group[1:]:
            remove_label = f"{duplicate['wine_name']} {duplicate.get('vintage') or 'NV'} (#{duplicate['id']})"
            suggestions.append(
                MergeSuggestion(
                    suggestion_type="wine",
                    keep_id=keep["id"],
                    remove_id=duplicate["id"],
                    keep_label=keep_label,
                    remove_label=remove_label,
                    reason="Normalized wine core + producer/region names match",
                )
            )
    return suggestions


def _collect_possible_wine_suggestions(excluded_pairs: set[tuple[int, int]] | None = None) -> list[MergeSuggestion]:
    """Build lower-confidence wine match suggestions for manual review.

    These suggestions intentionally relax matching by ignoring wine type and
    region while still requiring the same vintage.
    """
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                w.id,
                w.wine_name,
                w.vintage,
                w.wine_type,
                w.producer_id,
                w.region_id,
                p.name AS producer_name,
                r.primary_name AS region_name
            FROM wines w
            LEFT JOIN producers p ON w.producer_id = p.id
            LEFT JOIN regions r ON w.region_id = r.id
            WHERE w.wine_name IS NOT NULL AND TRIM(w.wine_name) != ''
            ORDER BY w.id ASC
            """
        ).fetchall()

    pairs_to_skip = excluded_pairs or set()
    groups: dict[str, list[dict]] = {}
    for row in rows:
        wine = dict(row)
        producer_name_key = _canonical_match_text(wine.get("producer_name") or "")
        producer_key = producer_name_key or f"id:{wine.get('producer_id') or ''}"
        wine_core_name = _strip_vintage_from_wine_core(
            core_name=_normalize_wine_core_name(
                wine_name=wine.get("wine_name"),
                producer_name=wine.get("producer_name"),
            ),
            vintage=wine.get("vintage"),
        )
        vintage_key = str(wine.get("vintage") or "")
        key = "|".join([wine_core_name, producer_key, vintage_key])
        if key.replace("|", ""):
            groups.setdefault(key, []).append(wine)

    suggestions: list[MergeSuggestion] = []
    for group in groups.values():
        if len(group) < 2:
            continue

        keep = group[0]
        keep_label = f"{keep['wine_name']} {keep.get('vintage') or 'NV'} (#{keep['id']})"
        for duplicate in group[1:]:
            pair_key = tuple(sorted((keep["id"], duplicate["id"])))
            if pair_key in pairs_to_skip:
                continue

            wine_type_differs = _canonical_match_text(keep.get("wine_type") or "") != _canonical_match_text(
                duplicate.get("wine_type") or ""
            )
            region_differs = _canonical_match_text(keep.get("region_name") or "") != _canonical_match_text(
                duplicate.get("region_name") or ""
            )

            difference_notes: list[str] = []
            if wine_type_differs:
                difference_notes.append("wine type differs")
            if region_differs:
                difference_notes.append("region differs")

            reason = "Possible match: normalized wine core + producer name match"
            if difference_notes:
                reason = f"{reason} ({', '.join(difference_notes)})"

            remove_label = f"{duplicate['wine_name']} {duplicate.get('vintage') or 'NV'} (#{duplicate['id']})"
            suggestions.append(
                MergeSuggestion(
                    suggestion_type="wine",
                    keep_id=keep["id"],
                    remove_id=duplicate["id"],
                    keep_label=keep_label,
                    remove_label=remove_label,
                    reason=reason,
                )
            )
    return suggestions


def _merge_producers(keep_id: int, remove_id: int) -> tuple[str, dict[str, int | str | None]]:
    """Merge two producers by moving references and deleting the duplicate."""
    with get_db_connection() as conn:
        try:
            cursor = conn.cursor()
            keep = cursor.execute(
                "SELECT id, name, country, region, description FROM producers WHERE id = ?",
                (keep_id,),
            ).fetchone()
            remove = cursor.execute(
                "SELECT id, name, country, region, description FROM producers WHERE id = ?",
                (remove_id,),
            ).fetchone()

            if not keep or not remove:
                raise HTTPException(status_code=404, detail="Producer not found")

            merged_country = _coalesce_text(keep["country"], remove["country"])
            merged_region = _coalesce_text(keep["region"], remove["region"])
            merged_description = _coalesce_text(keep["description"], remove["description"])

            cursor.execute("UPDATE wines SET producer_id = ? WHERE producer_id = ?", (keep_id, remove_id))
            wines_relinked = cursor.rowcount

            cursor.execute(
                """
                UPDATE producers
                SET country = ?, region = ?, description = ?, updated_at = ?
                WHERE id = ?
                """,
                (merged_country, merged_region, merged_description, datetime.now(), keep_id),
            )
            cursor.execute("DELETE FROM producers WHERE id = ?", (remove_id,))
            deleted = cursor.rowcount
            conn.commit()

            keep_name = keep["name"]
            remove_name = remove["name"]
            summary = f"Merged producer '{remove_name}' into '{keep_name}'."
            details = {
                "wines_relinked": wines_relinked,
                "records_deleted": deleted,
            }
            return summary, details
        except Exception:
            conn.rollback()
            raise


def _merge_regions(keep_id: int, remove_id: int) -> tuple[str, dict[str, int | str | None]]:
    """Merge two regions by moving references and deleting the duplicate."""
    with get_db_connection() as conn:
        try:
            cursor = conn.cursor()
            keep = cursor.execute(
                "SELECT id, primary_name, secondary_name, country, description FROM regions WHERE id = ?",
                (keep_id,),
            ).fetchone()
            remove = cursor.execute(
                "SELECT id, primary_name, secondary_name, country, description FROM regions WHERE id = ?",
                (remove_id,),
            ).fetchone()

            if not keep or not remove:
                raise HTTPException(status_code=404, detail="Region not found")

            merged_description = _coalesce_text(keep["description"], remove["description"])

            cursor.execute("UPDATE wines SET region_id = ? WHERE region_id = ?", (keep_id, remove_id))
            wines_relinked = cursor.rowcount

            cursor.execute(
                "UPDATE regions SET description = ? WHERE id = ?",
                (merged_description, keep_id),
            )
            cursor.execute("DELETE FROM regions WHERE id = ?", (remove_id,))
            deleted = cursor.rowcount
            conn.commit()

            keep_name = keep["primary_name"]
            remove_name = remove["primary_name"]
            summary = f"Merged region '{remove_name}' into '{keep_name}'."
            details = {
                "wines_relinked": wines_relinked,
                "records_deleted": deleted,
            }
            return summary, details
        except Exception:
            conn.rollback()
            raise


def _merge_wines(keep_id: int, remove_id: int) -> tuple[str, dict[str, int | str | None]]:
    """Merge duplicate wine records and move dependent rows."""
    with get_db_connection() as conn:
        try:
            cursor = conn.cursor()
            keep = cursor.execute("SELECT * FROM wines WHERE id = ?", (keep_id,)).fetchone()
            remove = cursor.execute("SELECT * FROM wines WHERE id = ?", (remove_id,)).fetchone()

            if not keep or not remove:
                raise HTTPException(status_code=404, detail="Wine not found")

            merged_source = _coalesce_text(keep["source"], remove["source"]) or "manual"
            merged_external_id = _coalesce_text(keep["external_id"], remove["external_id"])
            merged_wine_name = _coalesce_text(keep["wine_name"], remove["wine_name"]) or "Unknown"
            merged_wine_type = _coalesce_text(keep["wine_type"], remove["wine_type"]) or "Red"
            merged_bottle_size = _coalesce_text(keep["bottle_size"], remove["bottle_size"]) or "750ml"

            merged_values = {
                "source": merged_source,
                "external_id": merged_external_id,
                "wine_name": merged_wine_name,
                "producer_id": keep["producer_id"] if keep["producer_id"] is not None else remove["producer_id"],
                "vintage": keep["vintage"] if keep["vintage"] is not None else remove["vintage"],
                "wine_type": merged_wine_type,
                "varietal": _coalesce_text(keep["varietal"], remove["varietal"]),
                "designation": _coalesce_text(keep["designation"], remove["designation"]),
                "region_id": keep["region_id"] if keep["region_id"] is not None else remove["region_id"],
                "appellation": _coalesce_text(keep["appellation"], remove["appellation"]),
                "vineyard": _coalesce_text(keep["vineyard"], remove["vineyard"]),
                "bottle_size": merged_bottle_size,
                "drink_from_year": (
                    keep["drink_from_year"] if keep["drink_from_year"] is not None else remove["drink_from_year"]
                ),
                "drink_to_year": keep["drink_to_year"] if keep["drink_to_year"] is not None else remove["drink_to_year"],
                "drink_index": keep["drink_index"] if keep["drink_index"] is not None else remove["drink_index"],
                "drink_window_source": _coalesce_text(keep["drink_window_source"], remove["drink_window_source"]),
                "description": _coalesce_text(keep["description"], remove["description"]),
                "q_purchased": int(keep["q_purchased"] or 0) + int(remove["q_purchased"] or 0),
                "q_quantity": int(keep["q_quantity"] or 0) + int(remove["q_quantity"] or 0),
                "q_consumed": int(keep["q_consumed"] or 0) + int(remove["q_consumed"] or 0),
            }

            cursor.execute(
                """
                UPDATE wines
                SET
                    source = ?,
                    external_id = ?,
                    wine_name = ?,
                    producer_id = ?,
                    vintage = ?,
                    wine_type = ?,
                    varietal = ?,
                    designation = ?,
                    region_id = ?,
                    appellation = ?,
                    vineyard = ?,
                    bottle_size = ?,
                    drink_from_year = ?,
                    drink_to_year = ?,
                    drink_index = ?,
                    drink_window_source = ?,
                    description = ?,
                    q_purchased = ?,
                    q_quantity = ?,
                    q_consumed = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    merged_values["source"],
                    merged_values["external_id"],
                    merged_values["wine_name"],
                    merged_values["producer_id"],
                    merged_values["vintage"],
                    merged_values["wine_type"],
                    merged_values["varietal"],
                    merged_values["designation"],
                    merged_values["region_id"],
                    merged_values["appellation"],
                    merged_values["vineyard"],
                    merged_values["bottle_size"],
                    merged_values["drink_from_year"],
                    merged_values["drink_to_year"],
                    merged_values["drink_index"],
                    merged_values["drink_window_source"],
                    merged_values["description"],
                    merged_values["q_purchased"],
                    merged_values["q_quantity"],
                    merged_values["q_consumed"],
                    datetime.now(),
                    keep_id,
                ),
            )

            keep_tasting = cursor.execute("SELECT * FROM tastings WHERE wine_id = ?", (keep_id,)).fetchone()
            remove_tasting = cursor.execute("SELECT * FROM tastings WHERE wine_id = ?", (remove_id,)).fetchone()
            tastings_merged = 0

            if keep_tasting and remove_tasting:
                merged_notes = _coalesce_text(keep_tasting["tasting_notes"], remove_tasting["tasting_notes"])
                merged_personal = (
                    keep_tasting["personal_rating"]
                    if keep_tasting["personal_rating"] is not None
                    else remove_tasting["personal_rating"]
                )
                merged_comm = (
                    keep_tasting["community_rating"]
                    if keep_tasting["community_rating"] is not None
                    else remove_tasting["community_rating"]
                )
                merged_like = keep_tasting["do_like"] if keep_tasting["do_like"] is not None else remove_tasting["do_like"]
                merged_like_votes = int(keep_tasting["like_votes"] or 0) + int(remove_tasting["like_votes"] or 0)
                merged_like_pct = (
                    keep_tasting["like_percentage"]
                    if keep_tasting["like_percentage"] is not None
                    else remove_tasting["like_percentage"]
                )
                merged_defective = bool(keep_tasting["is_defective"]) or bool(remove_tasting["is_defective"])
                merged_last_tasted = _resolve_iso_date(keep_tasting["last_tasted_date"], remove_tasting["last_tasted_date"])

                cursor.execute(
                    """
                    UPDATE tastings
                    SET
                        personal_rating = ?,
                        tasting_notes = ?,
                        do_like = ?,
                        community_rating = ?,
                        like_votes = ?,
                        like_percentage = ?,
                        is_defective = ?,
                        last_tasted_date = ?,
                        updated_at = ?
                    WHERE wine_id = ?
                    """,
                    (
                        merged_personal,
                        merged_notes,
                        merged_like,
                        merged_comm,
                        merged_like_votes,
                        merged_like_pct,
                        merged_defective,
                        merged_last_tasted,
                        datetime.now(),
                        keep_id,
                    ),
                )
                cursor.execute("DELETE FROM tastings WHERE wine_id = ?", (remove_id,))
                tastings_merged = 1
            elif remove_tasting and not keep_tasting:
                cursor.execute("UPDATE tastings SET wine_id = ?, updated_at = ? WHERE wine_id = ?", (keep_id, datetime.now(), remove_id))
                tastings_merged = 1

            cursor.execute("UPDATE bottles SET wine_id = ?, updated_at = ? WHERE wine_id = ?", (keep_id, datetime.now(), remove_id))
            bottles_relinked = cursor.rowcount

            cursor.execute("DELETE FROM wines WHERE id = ?", (remove_id,))
            deleted = cursor.rowcount

            conn.commit()
            summary = f"Merged wine #{remove_id} into wine #{keep_id}."
            details = {
                "bottles_relinked": bottles_relinked,
                "tastings_merged": tastings_merged,
                "records_deleted": deleted,
            }
            return summary, details
        except Exception:
            conn.rollback()
            raise

def _effective_index(w: dict) -> float | None:
    """Return the best drinking index for an inventory row on a 0-100 scale.

    Prefers the CellarTracker-sourced ``drink_index``. Falls back to a
    locally computed bell-curve value when the drinking window is known.

    Args:
        w: Inventory row dict from ``BottleRepository.get_inventory()``.

    Returns:
        Float in ``[0, 100]`` or ``None`` when no data is available.
    """
    idx = w.get("drink_index")
    src = w.get("drink_window_source")
    if idx is not None and src == "cellar_tracker":
        return max(0.0, min(100.0, float(idx) * 100))
    df, dt = w.get("drink_from_year"), w.get("drink_to_year")
    if df and dt:
        from src.agents.drinking_window_service import compute_drink_index

        return compute_drink_index(int(df), int(dt))
    return None


def _build_grouped_inventory(raw_inventory: list[dict]) -> list[dict]:
    """Group raw bottle rows by ``wine_id`` and aggregate quantities.

    Args:
        raw_inventory: Flat list from ``BottleRepository.get_inventory()``.

    Returns:
        Deduplicated list with one entry per wine, summed quantities,
        and the most recent ``created_at`` timestamp.
    """
    wine_groups: dict[int, dict] = {}
    for bottle in raw_inventory:
        wine_id = bottle.get("wine_id")
        if wine_id not in wine_groups:
            wine_groups[wine_id] = bottle.copy()
        else:
            wine_groups[wine_id]["quantity"] = (
                wine_groups[wine_id].get("quantity", 0) + bottle.get("quantity", 0)
            )
            existing_ca = wine_groups[wine_id].get("created_at")
            incoming_ca = bottle.get("created_at")
            if incoming_ca and (not existing_ca or str(incoming_ca) > str(existing_ca)):
                wine_groups[wine_id]["created_at"] = incoming_ca
    return list(wine_groups.values())


# Sort key names that the frontend can pass
_SORT_KEYS = {
    "created_at_desc": lambda w: str(w.get("created_at") or ""),
    "producer": lambda w: (w.get("producer_name") or "", w.get("vintage") or 0),
    "wine_name": lambda w: w.get("wine_name") or "",
    "vintage_desc": lambda w: w.get("vintage") or 0,
    "vintage_asc": lambda w: w.get("vintage") or 9999,
    "rating_desc": lambda w: w.get("personal_rating") or 0,
    "rating_asc": lambda w: w.get("personal_rating") or 9999,
    "drink_desc": lambda w: _effective_index(w) or 0,
    "drink_asc": lambda w: (lambda idx: idx if idx is not None else -9999)(_effective_index(w)),
}


def _apply_sort(inventory: list[dict], sort_by: str) -> list[dict]:
    """Sort inventory by the requested key.

    Args:
        inventory: Grouped inventory list.
        sort_by: Sort key name from ``_SORT_KEYS``.

    Returns:
        Sorted list (new list, does not mutate input).
    """
    reverse = sort_by.endswith("_desc")
    key_fn = _SORT_KEYS.get(sort_by, _SORT_KEYS["created_at_desc"])
    return sorted(inventory, key=key_fn, reverse=reverse)


def _row_to_item(w: dict) -> InventoryItem:
    """Convert a raw inventory dict to an ``InventoryItem`` schema.

    Args:
        w: Inventory row dict with all joined fields.

    Returns:
        Typed ``InventoryItem`` instance.
    """
    return InventoryItem(
        wine_id=w.get("wine_id", 0),
        wine_name=w.get("wine_name", ""),
        producer_name=w.get("producer_name"),
        vintage=w.get("vintage"),
        wine_type=w.get("wine_type"),
        varietal=w.get("varietal"),
        appellation=w.get("appellation"),
        vineyard=w.get("vineyard"),
        country=w.get("country"),
        region_name=w.get("region_name"),
        quantity=w.get("quantity", 0),
        personal_rating=w.get("personal_rating"),
        community_rating=w.get("community_rating"),
        do_like=bool(w["do_like"]) if w.get("do_like") is not None else None,
        drink_index=_effective_index(w),
        drink_from_year=w.get("drink_from_year"),
        drink_to_year=w.get("drink_to_year"),
        drink_window_source=w.get("drink_window_source"),
        location=w.get("location"),
        bin=w.get("bin"),
        purchase_date=str(w["purchase_date"]) if w.get("purchase_date") else None,
        purchase_price=w.get("purchase_price"),
        valuation_price=w.get("valuation_price"),
        currency=w.get("currency"),
        description=w.get("description"),
        producer_description=w.get("producer_description"),
        producer_id=w.get("producer_id"),
        bottle_note=w.get("bottle_note"),
        last_tasted_date=str(w["last_tasted_date"]) if w.get("last_tasted_date") else None,
        like_votes=w.get("like_votes"),
        like_percentage=w.get("like_percentage"),
        q_purchased=w.get("q_purchased", 0),
        q_consumed=w.get("q_consumed", 0),
        q_quantity=w.get("q_quantity", 0),
        created_at=str(w["created_at"]) if w.get("created_at") else None,
    )


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.get("/inventory", response_model=InventoryResponse)
def get_inventory(
    wine_type: str | None = Query(None, description="Filter by wine type (e.g. Red, White)"),
    country: str | None = Query(None, description="Filter by country"),
    producer: str | None = Query(None, description="Filter by producer name"),
    location: str | None = Query(None, description="Filter by storage location"),
    min_vintage: int | None = Query(None, description="Minimum vintage year (inclusive)"),
    max_vintage: int | None = Query(None, description="Maximum vintage year (inclusive)"),
    rating_filter: str | None = Query(None, description="Rating filter: rated, unrated, 90+, 80+, 70+"),
    search: str | None = Query(None, description="Free-text search (wine name, producer, varietal)"),
    sort_by: str = Query("created_at_desc", description="Sort key (see _SORT_KEYS)"),
) -> InventoryResponse:
    """Return filtered, sorted cellar inventory with filter options.

    Filtering is applied at the SQL level in the repository. Grouping by
    wine_id and sorting by computed fields (e.g. drink_index) are done in
    Python after the SQL fetch.
    """
    bottle_repo = BottleRepository()

    # Lightweight distinct queries for filter dropdowns (always over full inventory)
    raw_filter_opts = bottle_repo.get_inventory_filter_options()
    filter_options = FilterOptions(**raw_filter_opts)

    # SQL-filtered fetch
    raw = bottle_repo.get_inventory(
        location=location,
        wine_type=wine_type,
        country=country,
        producer=producer,
        min_vintage=min_vintage,
        max_vintage=max_vintage,
        rating_filter=rating_filter,
        search=search,
    )
    grouped = _build_grouped_inventory(raw)
    sorted_inv = _apply_sort(grouped, sort_by)

    items = [_row_to_item(w) for w in sorted_inv]
    total_bottles = sum(w.get("quantity", 0) for w in sorted_inv)

    return InventoryResponse(
        items=items,
        total_wines=len(items),
        total_bottles=total_bottles,
        filter_options=filter_options,
    )


@router.get("/filters", response_model=FilterOptions)
def get_filters() -> FilterOptions:
    """Return available filter values for the inventory UI.

    Uses lightweight ``SELECT DISTINCT`` queries instead of a full inventory
    scan so this endpoint is cheap to call for populating dropdowns.
    """
    bottle_repo = BottleRepository()
    return FilterOptions(**bottle_repo.get_inventory_filter_options())


@router.get("/stats", response_model=CellarStatsResponse)
def get_stats() -> CellarStatsResponse:
    """Return combined cellar overview metrics.

    Wraps ``StatsRepository.get_cellar_overview()``,
    ``get_drinking_window_stats()``, and ``get_cellar_value()``.
    """
    stats_repo = StatsRepository()

    overview_raw = stats_repo.get_cellar_overview()
    drinking_raw = stats_repo.get_drinking_window_stats()
    value_raw = stats_repo.get_cellar_value()

    return CellarStatsResponse(
        overview=CellarOverview(**overview_raw),
        drinking_stats=DrinkingWindowStats(**drinking_raw),
        value_stats=CellarValueStats(**value_raw),
    )


@router.get("/charts", response_model=ChartDataResponse)
def get_charts() -> ChartDataResponse:
    """Return pre-computed data for all cellar statistics charts.

    Wraps multiple ``StatsRepository`` methods and returns JSON-ready
    data that the frontend passes directly to a charting library.
    """
    from datetime import datetime

    stats_repo = StatsRepository()

    overview         = stats_repo.get_cellar_overview()
    varietal_dist    = stats_repo.get_varietal_distribution(limit=10)
    region_dist      = stats_repo.get_region_distribution(limit=10)
    drinking_wines   = stats_repo.get_drinking_window_wines()
    cellar_timeline  = stats_repo.get_cellar_size_over_time()
    top_rated        = stats_repo.get_top_rated_wines(limit=10)
    vintage_dist     = stats_repo.get_cellar_vintage_distribution()
    rating_dist      = stats_repo.get_cellar_rating_distribution()

    # Compute wine-age buckets from the vintage distribution (no extra DB hit).
    current_year = datetime.now().year
    age_buckets: dict[str, int] = {
        "0-5 years": 0, "6-10 years": 0, "11-15 years": 0,
        "16-20 years": 0, "20+ years": 0,
    }
    for item in vintage_dist:
        vintage = item.get("vintage")
        bottles = item.get("bottles", 0)
        if vintage:
            age = current_year - int(vintage)
            if age <= 5:
                age_buckets["0-5 years"] += bottles
            elif age <= 10:
                age_buckets["6-10 years"] += bottles
            elif age <= 15:
                age_buckets["11-15 years"] += bottles
            elif age <= 20:
                age_buckets["16-20 years"] += bottles
            else:
                age_buckets["20+ years"] += bottles
    wine_age_dist = [{"range": k, "bottles": v} for k, v in age_buckets.items() if v > 0]

    return ChartDataResponse(
        wine_type_distribution=overview.get("by_type", []),
        country_distribution=overview.get("by_country", []),
        varietal_distribution=varietal_dist,
        region_distribution=region_dist,
        drinking_window_wines=drinking_wines,
        cellar_size_over_time=cellar_timeline,
        top_rated=top_rated,
        vintage_distribution=vintage_dist,
        rating_distribution=rating_dist,
        wine_age_distribution=wine_age_dist,
    )


@router.get("/drink-next", response_model=DrinkNextResponse)
def get_drink_next(
    limit: int = Query(50, ge=1, le=200, description="Maximum wines per type")
) -> DrinkNextResponse:
    """Get wines ready to drink now, grouped by wine type.

    Returns wines with the highest drink_index (closest to peak/past peak),
    prioritizing wines that should be consumed soon. Wines are grouped by
    type (Red, White, Rosé, Sparkling, etc.) and sorted by drink_index
    descending within each group.

    Args:
        limit: Maximum number of wines to return per wine type (default 50).

    Returns:
        DrinkNextResponse with wines grouped by type.
    """
    from datetime import datetime

    bottle_repo = BottleRepository()
    raw_inventory = bottle_repo.get_inventory()
    grouped = _build_grouped_inventory(raw_inventory)

    # Filter to wines with drinking window data and currently in drinking window
    current_year = datetime.now().year
    ready_wines = []
    for wine in grouped:
        idx = _effective_index(wine)
        df = wine.get("drink_from_year")
        dt = wine.get("drink_to_year")
        
        # Include wines with drink_index >= 50 (approaching or past peak)
        # OR wines currently in their drinking window
        if idx is not None and idx >= 50:
            ready_wines.append(wine)
        elif df and dt and df <= current_year <= dt:
            ready_wines.append(wine)

    # Sort by drink_index descending (highest = drink soonest)
    ready_wines.sort(key=lambda w: _effective_index(w) or 0, reverse=True)

    # Group by wine type
    by_type: dict[str, list[DrinkNextItem]] = {}
    for wine in ready_wines:
        wine_type = wine.get("wine_type") or "Other"
        if wine_type not in by_type:
            by_type[wine_type] = []
        
        # Limit per type
        if len(by_type[wine_type]) >= limit:
            continue

        item = DrinkNextItem(
            wine_id=wine.get("wine_id", 0),
            wine_name=wine.get("wine_name", ""),
            producer_name=wine.get("producer_name"),
            vintage=wine.get("vintage"),
            wine_type=wine_type,
            varietal=wine.get("varietal"),
            region_name=wine.get("region_name"),
            country=wine.get("country"),
            quantity=wine.get("quantity", 0),
            drink_index=_effective_index(wine),
            drink_from_year=wine.get("drink_from_year"),
            drink_to_year=wine.get("drink_to_year"),
            personal_rating=wine.get("personal_rating"),
            community_rating=wine.get("community_rating"),
            location=wine.get("location"),
        )
        by_type[wine_type].append(item)

    total_wines = sum(len(wines) for wines in by_type.values())

    return DrinkNextResponse(by_type=by_type, total_wines=total_wines)


@router.get("/merge-suggestions", response_model=MergeSuggestionsResponse)
def get_merge_suggestions() -> MergeSuggestionsResponse:
    """List merge suggestions for producers, regions, and wines in development mode."""
    _ensure_dev_mode()

    producers = _collect_producer_suggestions()
    regions = _collect_region_suggestions()
    wines = _collect_wine_suggestions()
    strict_wine_pairs: set[tuple[int, int]] = set()
    for suggestion in wines:
        if isinstance(suggestion, dict):
            keep_id = int(suggestion["keep_id"])
            remove_id = int(suggestion["remove_id"])
        else:
            keep_id = suggestion.keep_id
            remove_id = suggestion.remove_id
        strict_wine_pairs.add((min(keep_id, remove_id), max(keep_id, remove_id)))
    possible_wines = _collect_possible_wine_suggestions(excluded_pairs=strict_wine_pairs)

    return MergeSuggestionsResponse(
        producers=producers,
        regions=regions,
        wines=wines,
        possible_wines=possible_wines,
        total=len(producers) + len(regions) + len(wines) + len(possible_wines),
    )


@router.post("/merge/{entity_type}/{keep_id}/{remove_id}", response_model=MergeDecisionResponse)
def execute_manual_merge(
    entity_type: str,
    keep_id: int,
    remove_id: int,
    payload: MergeDecisionRequest,
) -> MergeDecisionResponse:
    """Approve or skip a manual merge suggestion in development mode."""
    _ensure_dev_mode()

    if entity_type not in {"producer", "region", "wine"}:
        raise HTTPException(status_code=400, detail="Invalid entity_type. Expected producer, region, or wine.")
    if keep_id == remove_id:
        raise HTTPException(status_code=400, detail="keep_id and remove_id must be different.")

    if not payload.approve:
        return MergeDecisionResponse(
            approved=False,
            entity_type=entity_type,
            keep_id=keep_id,
            remove_id=remove_id,
            summary="Suggestion skipped.",
            details={"records_deleted": 0},
        )

    if entity_type == "producer":
        summary, details = _merge_producers(keep_id, remove_id)
    elif entity_type == "region":
        summary, details = _merge_regions(keep_id, remove_id)
    else:
        summary, details = _merge_wines(keep_id, remove_id)

    logger.info(
        "Manual merge completed",
        extra={
            "entity_type": entity_type,
            "keep_id": keep_id,
            "remove_id": remove_id,
            "details": details,
        },
    )
    return MergeDecisionResponse(
        approved=True,
        entity_type=entity_type,
        keep_id=keep_id,
        remove_id=remove_id,
        summary=summary,
        details=details,
    )


@router.post("/sync", response_model=SyncResponse)
def sync_cellar_tracker() -> SyncResponse:
    """Trigger a CellarTracker data sync.

    Reads credentials from server-side environment variables
    (``CELLAR_TRACKER_USERNAME``, ``CELLAR_TRACKER_PASSWORD``).

    Note: this operation can take 10-60+ seconds. For multi-user deployments
    consider moving it to a background task with a status-polling endpoint.
    """
    username = os.getenv("CELLAR_TRACKER_USERNAME")
    password = os.getenv("CELLAR_TRACKER_PASSWORD")

    if not username or not password:
        raise HTTPException(
            status_code=400,
            detail=(
                "CellarTracker credentials not configured. "
                "Set CELLAR_TRACKER_USERNAME and CELLAR_TRACKER_PASSWORD in .env."
            ),
        )

    try:
        from src.etl.cellartracker_importer import CellarTrackerImporter
        from src.utils import get_default_db_path

        db_path = get_default_db_path()
        importer = CellarTrackerImporter(username, password, str(db_path))
        stats = importer.import_all()

        return SyncResponse(
            success=True,
            wines_processed=stats.get("wines_processed", 0),
            wines_imported=stats.get("wines_imported", 0),
            bottles_processed=stats.get("bottles_processed", 0),
            bottles_imported=stats.get("bottles_imported", 0),
            producers_created=stats.get("producers_created", 0),
            regions_created=stats.get("regions_created", 0),
            errors=stats.get("errors", []),
        )

    except Exception as e:
        logger.error(f"CellarTracker sync failed: {e}", exc_info=True)
        return SyncResponse(
            success=False,
            error_message=str(e),
        )
