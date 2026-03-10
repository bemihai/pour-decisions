"""Tasting repository."""
from datetime import datetime, date

from src.database import get_db_connection, build_update_query
from src.database.models import Tasting
from src.utils import get_default_db_path, logger


class TastingRepository:
    """Repository for tasting-related database operations."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize tasting repository.

        Args:
            db_path: Optional path to database file
        """
        self.db_path = db_path or get_default_db_path()

    def get_by_id(self, tasting_id: int) -> Tasting | None:
        """
        Get tasting by ID.

        Args:
            tasting_id: Tasting ID

        Returns:
            Tasting agents or None if not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tastings WHERE id = ?", (tasting_id,))

            row = cursor.fetchone()
            if row:
                return Tasting(**dict(row))
            return None

    def get_by_wine(self, wine_id: int) -> list[Tasting]:
        """
        Get all tastings for a wine.

        Args:
            wine_id: Wine ID

        Returns:
            List of Tasting models
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tastings 
                WHERE wine_id = ?
                ORDER BY last_tasted_date DESC, created_at DESC
            """, (wine_id,))
            return [Tasting(**dict(row)) for row in cursor.fetchall()]

    def get_latest_by_wine(self, wine_id: int) -> Tasting | None:
        """
        Get the most recent tasting for a wine.

        Args:
            wine_id: Wine ID

        Returns:
            Tasting agents or None if not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tastings 
                WHERE wine_id = ?
                ORDER BY last_tasted_date DESC, created_at DESC
                LIMIT 1
            """, (wine_id,))

            row = cursor.fetchone()
            if row:
                return Tasting(**dict(row))
            return None

    def get_top_rated(
        self,
        min_rating: int = 80,
        limit: int = 10,
        wine_type: str | None = None
    ) -> list[dict]:
        """
        Get top rated wines with tasting information.

        Args:
            min_rating: Minimum personal rating (0-100 scale)
            limit: Maximum number of results
            wine_type: Optional wine type filter

        Returns:
            List of dictionaries with tasting and wine info
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT 
                    t.*,
                    w.wine_name, w.vintage, w.wine_type, w.varietal,
                    p.name as producer_name,
                    r.country, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name
                FROM tastings t
                JOIN wines w ON t.wine_id = w.id
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                WHERE t.personal_rating >= ?
            """
            params = [min_rating]

            if wine_type:
                query += " AND w.wine_type = ?"
                params.append(wine_type)

            query += " ORDER BY t.personal_rating DESC, t.community_rating DESC"
            query += " LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_all_with_wine_info(
        self,
        has_rating: bool | None = None,
        is_liked: bool | None = None,
        limit: int | None = None
    ) -> list[dict]:
        """
        Get all tastings with wine details.

        Args:
            has_rating: Filter by whether tasting has a rating
            is_liked: Filter by do_like flag
            limit: Maximum number of results

        Returns:
            List of dictionaries with combined tasting and wine info
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT 
                    t.*,
                    w.wine_name, w.vintage, w.wine_type, w.varietal,
                    p.name as producer_name,
                    r.country,
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name
                FROM tastings t
                JOIN wines w ON t.wine_id = w.id
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                WHERE 1=1
            """
            params = []

            if has_rating is not None:
                if has_rating:
                    query += " AND t.personal_rating IS NOT NULL"
                else:
                    query += " AND t.personal_rating IS NULL"

            if is_liked is not None:
                query += " AND t.do_like = ?"
                params.append(is_liked)

            query += " ORDER BY t.last_tasted_date DESC, t.created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def create(self, tasting: Tasting) -> int:
        """
        Create new tasting record.

        Args:
            tasting: Tasting agents

        Returns:
            ID of created tasting
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO tastings (
                    wine_id, is_defective, personal_rating, tasting_notes,
                    do_like, community_rating, like_votes, like_percentage,
                    last_tasted_date, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tasting.wine_id, tasting.is_defective, tasting.personal_rating,
                tasting.tasting_notes, tasting.do_like, tasting.community_rating,
                tasting.like_votes, tasting.like_percentage, tasting.last_tasted_date,
                tasting.created_at or datetime.now(), tasting.updated_at or datetime.now()
            ))

            conn.commit()
            tasting_id = cursor.lastrowid
            logger.debug(f"Created tasting for wine_id={tasting.wine_id} (ID: {tasting_id})")
            return tasting_id

    def update(self, tasting: Tasting) -> bool:
        """
        Update existing tasting record.

        Args:
            tasting: Tasting agents with updated cellar-data

        Returns:
            True if successful
        """
        if not tasting.id:
            raise ValueError("Tasting ID is required for update")

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            update_query, params = build_update_query(
                "tastings", tasting, "id"
            )
            cursor.execute(update_query, params)

            conn.commit()
            logger.debug(f"Updated tasting ID: {tasting.id}")
            return True

    def delete(self, tasting_id: int) -> bool:
        """
        Delete tasting record.

        Args:
            tasting_id: Tasting ID

        Returns:
            True if successful
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tastings WHERE id = ?", (tasting_id,))
            conn.commit()
            logger.debug(f"Deleted tasting ID: {tasting_id}")
            return True

    def upsert_community_data(
        self,
        wine_id: int,
        community_rating: float | None,
        like_votes: int | None,
        like_percentage: float | None,
    ) -> None:
        """
        Insert or update community tasting data for a wine without touching personal fields.

        Uses INSERT ... ON CONFLICT(wine_id) DO UPDATE so it is safe to call on every
        sync regardless of whether a personal review exists.  Personal fields
        (personal_rating, tasting_notes, do_like, is_defective, last_tasted_date) are
        never overwritten.

        Args:
            wine_id: Wine ID
            community_rating: Community average rating on 0-100 scale
            like_votes: Number of community like votes
            like_percentage: Fraction of likes (0.0 to 1.0)
        """
        now = datetime.now()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tastings (wine_id, community_rating, like_votes, like_percentage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(wine_id) DO UPDATE SET
                    community_rating  = COALESCE(excluded.community_rating, community_rating),
                    like_votes        = COALESCE(excluded.like_votes, like_votes),
                    like_percentage   = COALESCE(excluded.like_percentage, like_percentage),
                    updated_at        = excluded.updated_at
            """, (wine_id, community_rating, like_votes, like_percentage, now, now))
            conn.commit()
            logger.debug(f"Upserted community data for wine_id={wine_id}")

    def get_average_rating_by_wine_type(self) -> list[dict]:
        """
        Get average personal rating by wine type.

        Returns:
            List of dictionaries with wine_type and avg_rating
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    w.wine_type,
                    AVG(t.personal_rating) as avg_rating,
                    COUNT(*) as tasting_count
                FROM tastings t
                JOIN wines w ON t.wine_id = w.id
                WHERE t.personal_rating IS NOT NULL
                GROUP BY w.wine_type
                ORDER BY avg_rating DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_taste_profile_summary(self) -> dict:
        """
        Get a summary of user's taste profile.

        Returns:
            Dictionary with taste profile statistics
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_tastings,
                    AVG(personal_rating) as avg_rating,
                    MAX(personal_rating) as highest_rating,
                    MIN(personal_rating) as lowest_rating,
                    SUM(CASE WHEN do_like = 1 THEN 1 ELSE 0 END) as liked_count,
                    SUM(CASE WHEN is_defective = 1 THEN 1 ELSE 0 END) as defective_count
                FROM tastings
                WHERE personal_rating IS NOT NULL
            """)
            stats = dict(cursor.fetchone())

            # Top varietals
            cursor.execute("""
                SELECT 
                    w.varietal,
                    AVG(t.personal_rating) as avg_rating,
                    COUNT(*) as count
                FROM tastings t
                JOIN wines w ON t.wine_id = w.id
                WHERE t.personal_rating IS NOT NULL AND w.varietal IS NOT NULL
                GROUP BY w.varietal
                HAVING count >= 2
                ORDER BY avg_rating DESC
                LIMIT 5
            """)
            stats['top_varietals'] = [dict(row) for row in cursor.fetchall()]

            # Top regions
            cursor.execute("""
                SELECT 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), 'Unknown') as region,
                    AVG(t.personal_rating) as avg_rating,
                    COUNT(*) as count
                FROM tastings t
                JOIN wines w ON t.wine_id = w.id
                LEFT JOIN regions r ON w.region_id = r.id
                WHERE t.personal_rating IS NOT NULL
                GROUP BY r.id
                HAVING count >= 2
                ORDER BY avg_rating DESC
                LIMIT 5
            """)
            stats['top_regions'] = [dict(row) for row in cursor.fetchall()]

            return stats

