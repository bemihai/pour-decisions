"""
Migration: add UNIQUE(wine_id) constraint to tastings table and backfill community data.

Rationale:
    The tastings table previously had no unique constraint on wine_id, meaning multiple
    rows per wine were possible.  The new design enforces one row per wine so community
    data can be upserted for all wines — including those never personally reviewed — via
    INSERT ... ON CONFLICT(wine_id) DO UPDATE.

Steps:
    1. Deduplicate existing rows by keeping the most informative row per wine_id
       (personal_rating takes priority; community fields are merged).
    2. Recreate the tastings table with the UNIQUE(wine_id) constraint.
    3. Copy the deduplicated rows into the new table.
    4. Restore indexes.
    5. Backfill community data from the CellarTracker API (inventory CT scores and
       notes CScore/LikeVotes/LikePercent).

Usage:
    python -m src.database.migrations.add_tastings_unique_wine_id
    or call migrate(db_path) directly.
"""
import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv

from src.utils import get_default_db_path, logger

load_dotenv()


def migrate(db_path: str | None = None) -> bool:
    """
    Apply the migration to add UNIQUE(wine_id) to the tastings table.

    Args:
        db_path: Path to the SQLite database. Defaults to the project default.

    Returns:
        True if the migration succeeded or was already applied, False on error.
    """
    db_path = db_path or get_default_db_path()
    logger.info(f"Running migration: add_tastings_unique_wine_id on {db_path}")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Check if unique constraint already exists
        cursor.execute("PRAGMA index_list(tastings)")
        indexes = [dict(row) for row in cursor.fetchall()]
        for idx in indexes:
            cursor.execute(f"PRAGMA index_info({idx['name']})")
            cols = [r['name'] for r in cursor.fetchall()]
            if cols == ['wine_id'] and idx.get('unique'):
                logger.info("Schema already migrated — UNIQUE(wine_id) exists. Running backfill only.")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.close()
                backfill_community_data(db_path)
                return True

        # Step 1: deduplicate — keep one row per wine_id
        # Priority: row with personal_rating; then row with community_rating; then most recent.
        cursor.execute("""
            SELECT wine_id, COUNT(*) as cnt FROM tastings GROUP BY wine_id HAVING cnt > 1
        """)
        duplicate_wines = [row['wine_id'] for row in cursor.fetchall()]
        logger.info(f"Found {len(duplicate_wines)} wines with duplicate tasting rows — deduplicating.")

        for wine_id in duplicate_wines:
            cursor.execute("""
                SELECT * FROM tastings WHERE wine_id = ?
                ORDER BY
                    CASE WHEN personal_rating IS NOT NULL THEN 0 ELSE 1 END,
                    CASE WHEN community_rating IS NOT NULL THEN 0 ELSE 1 END,
                    updated_at DESC
            """, (wine_id,))
            rows = [dict(r) for r in cursor.fetchall()]
            keep = rows[0]
            duplicates = rows[1:]

            # Merge community fields from duplicates if missing on kept row
            for dup in duplicates:
                if keep['community_rating'] is None and dup['community_rating'] is not None:
                    keep['community_rating'] = dup['community_rating']
                if keep['like_votes'] in (None, 0) and dup.get('like_votes'):
                    keep['like_votes'] = dup['like_votes']
                if keep['like_percentage'] is None and dup['like_percentage'] is not None:
                    keep['like_percentage'] = dup['like_percentage']

            # Delete all rows for this wine and reinsert the merged keeper
            cursor.execute("DELETE FROM tastings WHERE wine_id = ?", (wine_id,))
            cursor.execute("""
                INSERT INTO tastings (
                    id, wine_id, is_defective, personal_rating, tasting_notes,
                    do_like, community_rating, like_votes, like_percentage,
                    last_tasted_date, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                keep['id'], keep['wine_id'], keep['is_defective'], keep['personal_rating'],
                keep['tasting_notes'], keep['do_like'], keep['community_rating'],
                keep['like_votes'], keep['like_percentage'], keep['last_tasted_date'],
                keep['created_at'], keep['updated_at'],
            ))

        conn.commit()
        logger.info("Deduplication complete.")

        # Step 2: recreate table with UNIQUE(wine_id)
        cursor.execute("ALTER TABLE tastings RENAME TO tastings_old")

        cursor.execute("""
            CREATE TABLE tastings (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                wine_id                 INTEGER NOT NULL REFERENCES wines(id) ON DELETE CASCADE,
                is_defective            BOOLEAN NOT NULL DEFAULT 0,
                personal_rating         INTEGER,
                tasting_notes           TEXT,
                do_like                 BOOLEAN,
                community_rating        DECIMAL(5,2),
                like_votes              INTEGER DEFAULT 0,
                like_percentage         DECIMAL(5,2),
                last_tasted_date        DATE,
                created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(wine_id),
                CHECK(personal_rating IS NULL OR (personal_rating >= 0 AND personal_rating <= 100)),
                CHECK(community_rating IS NULL OR (community_rating >= 0 AND community_rating <= 100))
            )
        """)

        # Step 3: copy rows
        cursor.execute("""
            INSERT INTO tastings (
                id, wine_id, is_defective, personal_rating, tasting_notes,
                do_like, community_rating, like_votes, like_percentage,
                last_tasted_date, created_at, updated_at
            )
            SELECT
                id, wine_id, is_defective, personal_rating, tasting_notes,
                do_like, community_rating, like_votes, like_percentage,
                last_tasted_date, created_at, updated_at
            FROM tastings_old
        """)

        cursor.execute("DROP TABLE tastings_old")

        # Step 4: restore indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tastings_wine ON tastings(wine_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tastings_personal_rating ON tastings(personal_rating)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tastings_last_tasted_date ON tastings(last_tasted_date)")

        conn.commit()
        logger.info("Migration add_tastings_unique_wine_id applied successfully.")

        # Step 5: backfill community data from cached notes.csv if available
        backfill_community_data(db_path)

        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
        return False

    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()


def backfill_community_data(db_path: str | None = None) -> int:
    """
    Backfill community rating data from the CellarTracker API.

    Fetches live inventory (CT score) and notes (CScore, LikeVotes, LikePercent)
    from the API and upserts community data into the tastings table for every wine
    that matches by external_id.  Personal fields are never overwritten.

    Inventory is processed first (broader coverage), then notes (richer data with
    like votes) -- COALESCE ensures notes data wins when both sources have values.

    Requires CELLAR_TRACKER_USERNAME and CELLAR_TRACKER_PASSWORD env vars.

    Args:
        db_path: Path to the SQLite database. Defaults to the project default.

    Returns:
        Number of tasting rows upserted.
    """
    username = os.environ.get("CELLAR_TRACKER_USERNAME", "")
    password = os.environ.get("CELLAR_TRACKER_PASSWORD", "")
    if not username or not password:
        logger.warning("CellarTracker credentials not set -- skipping community data backfill")
        return 0

    try:
        from cellartracker import cellartracker as ct_module
    except ImportError:
        logger.warning("cellartracker package not installed -- skipping community data backfill")
        return 0

    db_path = db_path or get_default_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, external_id FROM wines WHERE external_id IS NOT NULL")
    wine_map = {row["external_id"]: row["id"] for row in cur.fetchall()}

    if not wine_map:
        logger.info("No wines with external_id found -- nothing to backfill")
        conn.close()
        return 0

    def _pf(v) -> float | None:
        try:
            return float(v) if v is not None and str(v).strip() else None
        except (ValueError, TypeError):
            return None

    def _pi(v) -> int | None:
        try:
            return int(v) if v is not None and str(v).strip() else None
        except (ValueError, TypeError):
            return None

    upsert_sql = """
        INSERT INTO tastings (wine_id, community_rating, like_votes, like_percentage, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(wine_id) DO UPDATE SET
            community_rating = COALESCE(excluded.community_rating, community_rating),
            like_votes       = COALESCE(excluded.like_votes, like_votes),
            like_percentage  = COALESCE(excluded.like_percentage, like_percentage),
            updated_at       = excluded.updated_at
    """

    upserted = 0
    now = datetime.now().isoformat()

    try:
        client = ct_module.CellarTracker(username, password)

        # Phase 1: inventory -- CT column (broad coverage for in-cellar wines)
        logger.info("Backfill phase 1: fetching inventory from CellarTracker API...")
        inventory = client.get_inventory()
        inv_count = 0
        for record in inventory:
            iwine = str(record.get("iWine", "")).strip()
            wine_id = wine_map.get(iwine)
            if not wine_id:
                continue
            ct = _pf(record.get("CT"))
            if ct is None:
                continue
            cur.execute(upsert_sql, (wine_id, ct, None, None, now, now))
            inv_count += 1
        upserted += inv_count
        logger.info(f"Backfilled {inv_count} rows from inventory (CT scores)")

        # Phase 2: notes -- CScore, LikeVotes, LikePercent (richer data)
        logger.info("Backfill phase 2: fetching notes from CellarTracker API...")
        notes = client.get_notes()
        notes_count = 0
        for record in notes:
            iwine = str(record.get("iWine", "")).strip()
            if not iwine or iwine.lower() in ("true", "false", ""):
                continue
            wine_id = wine_map.get(iwine)
            if not wine_id:
                continue
            community_rating = _pf(record.get("CScore"))
            like_votes = _pi(record.get("LikeVotes"))
            like_percentage = _pf(record.get("LikePercent"))
            if community_rating is None and like_votes is None and like_percentage is None:
                continue
            cur.execute(upsert_sql, (wine_id, community_rating, like_votes, like_percentage, now, now))
            notes_count += 1
        upserted += notes_count
        logger.info(f"Backfilled {notes_count} rows from notes (CScore/likes)")

        conn.commit()
        logger.info(f"Community data backfill complete: {upserted} total upserts")

    except Exception as e:
        conn.rollback()
        logger.error(f"Community data backfill failed: {e}")
    finally:
        conn.close()

    return upserted


if __name__ == "__main__":
    success = migrate()
    print("Migration", "succeeded" if success else "FAILED")

