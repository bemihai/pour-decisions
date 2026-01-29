"""Wine repository"""
from datetime import datetime

from src.database import get_db_connection, build_update_query
from src.database.models import Wine
from src.database.utils import calculate_similarity
from src.utils import get_default_db_path, logger


class WineRepository:
    """Repository for wine-related database operations."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize wine repository.

        Args:
            db_path: Optional path to database file
        """
        self.db_path = db_path or get_default_db_path()

    def get_by_id(self, wine_id: int) -> Wine | None:
        """
        Get wine by ID.

        Args:
            wine_id: Wine ID

        Returns:
            Wine agents or None if not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    w.*, 
                    p.name as producer_name, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    r.country,
                    t.personal_rating,
                    t.community_rating,
                    t.tasting_notes,
                    t.last_tasted_date
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE w.id = ?
            """, (wine_id,))

            row = cursor.fetchone()
            if row:
                return Wine(**dict(row))
            return None

    def get_by_external_id(self, external_id: str) -> Wine | None:
        """
        Get wine by external ID.

        Args:
            external_id: External ID from source system

        Returns:
            Wine agents or None if not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    w.*, 
                    p.name as producer_name, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    r.country,
                    t.personal_rating,
                    t.community_rating,
                    t.tasting_notes,
                    t.last_tasted_date
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE w.external_id = ?
            """, (external_id,))

            row = cursor.fetchone()
            if row:
                return Wine(**dict(row))
            return None

    def get_by_name(self, wine_name: str, vintage: int | None = None) -> Wine | None:
        """
        Get wine by name using partial matching.

        Args:
            wine_name: Wine name to search for (partial match supported)
            vintage: Optional vintage to narrow down results

        Returns:
            Wine agents or None if not found. Returns first match if multiple wines found.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT 
                    w.*, 
                    p.name as producer_name, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    r.country,
                    t.personal_rating,
                    t.community_rating,
                    t.tasting_notes,
                    t.last_tasted_date
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE LOWER(w.wine_name) LIKE LOWER(?)
            """
            params = [f"%{wine_name}%"]

            if vintage is not None:
                query += " AND w.vintage = ?"
                params.append(vintage)

            query += " ORDER BY w.wine_name LIMIT 1"

            cursor.execute(query, params)
            row = cursor.fetchone()

            if row:
                return Wine(**dict(row))
            return None

    def find_duplicates(
            self, wine_name: str, producer: str, wine_type: str, vintage: int | None, confidence: float = 0.85
    ) -> list[Wine] | None:
        """
        Get duplicate wines based on wine name, producer, type, and vintage.

        Matching Algorithm:
            - Producer similarity: max 30% (weighted by string similarity)
            - Wine name similarity: max 30% (weighted by string similarity)
            - Vintage match: 40% (if both have vintage)
            - Confidence threshold: default 85%

        Returns:
            List of Wine models that are duplicates or None.
        """
        matches = []

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Get all the wines with same type
            cursor.execute("""
                SELECT w.id, w.wine_name, p.name as producer_name, w.vintage
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                WHERE w.wine_type = ?
            """, (wine_type,))
            existing_wines = cursor.fetchall()

            for existing in existing_wines:
                score = 0.0

                # Vintage match
                if vintage and existing["vintage"]:
                    if vintage == existing["vintage"]:
                        score += 30
                elif not vintage and not existing["vintage"]:
                    score += 30

                # Producer match
                if producer and existing["producer_name"]:
                    producer_similarity = calculate_similarity(producer, existing["producer_name"])
                    score += producer_similarity * 30

                # Wine name match
                if wine_name and existing["wine_name"]:
                    name_similarity = calculate_similarity(
                        wine_name,
                        existing["wine_name"]
                    )
                    score += name_similarity * 40

                if score >= 100 * confidence:
                    matches.append(
                        (existing["id"], score, existing["wine_name"], existing["producer_name"], existing["vintage"])
                    )

            # Sort matches by confidence score
            matches.sort(key=lambda x: x[1], reverse=True)

            return matches


    def get_all(
        self,
        # Optional exact match filters
        vintage: int | None = None,
        wine_type: str | None = None,
        appellation: str | None = None,
        country: str | None = None,
        min_rating: int | None = None,
        ready_to_drink: bool | None = None,
        # Search filters (partial match)
        producer_name: str | None = None,
        region_name: str | None = None,
        wine_name: str | None = None,
        varietal: str | None = None,
        # Pagination
        limit: int | None = None,
        offset: int = 0
    ) -> list[Wine]:
        """
        Get all wines with optional filters.

        Args:
            vintage: Exact vintage year filter
            wine_type: Exact wine type filter (Red, White, Rosé, Sparkling, etc.)
            appellation: Exact appellation filter
            country: Exact country filter
            min_rating: Minimum personal rating (0-100 scale)
            ready_to_drink: If True, filter wines in drinking window; if False, exclude them; if None, no filter
            producer_name: Search filter for producer name (partial match, case-insensitive)
            region_name: Search filter for region name (partial match, case-insensitive)
            wine_name: Search filter for wine name (partial match, case-insensitive)
            varietal: Search filter for varietal/grape variety (partial match, case-insensitive)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of Wine models matching the filters
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT 
                    w.*, 
                    p.name as producer_name, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    r.country,
                    t.personal_rating,
                    t.community_rating,
                    t.tasting_notes,
                    t.last_tasted_date
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE 1=1
            """
            params = []

            # Optional exact match filters
            if vintage is not None:
                query += " AND w.vintage = ?"
                params.append(vintage)

            if wine_type:
                query += " AND w.wine_type = ?"
                params.append(wine_type)

            if appellation:
                query += " AND w.appellation = ?"
                params.append(appellation)

            if country:
                query += " AND r.country = ?"
                params.append(country)

            if min_rating is not None:
                query += " AND t.personal_rating >= ?"
                params.append(min_rating)

            if ready_to_drink is not None:
                current_year = datetime.now().year
                if ready_to_drink:
                    # Wines in drinking window
                    query += " AND w.drink_from_year IS NOT NULL AND w.drink_to_year IS NOT NULL"
                    query += " AND w.drink_from_year <= ? AND w.drink_to_year >= ?"
                    params.extend([current_year, current_year])
                else:
                    # Wines not in drinking window (aging or past peak)
                    query += " AND (w.drink_from_year IS NULL OR w.drink_to_year IS NULL"
                    query += " OR w.drink_from_year > ? OR w.drink_to_year < ?)"
                    params.extend([current_year, current_year])

            # Search filters (partial match)
            if producer_name:
                query += " AND p.name LIKE ?"
                params.append(f'%{producer_name}%')

            if region_name:
                query += " AND (r.primary_name LIKE ? OR r.secondary_name LIKE ?)"
                region_param = f'%{region_name}%'
                params.extend([region_param, region_param])

            if wine_name:
                query += " AND w.wine_name LIKE ?"
                params.append(f'%{wine_name}%')

            if varietal:
                query += " AND w.varietal LIKE ?"
                params.append(f'%{varietal}%')

            query += " ORDER BY p.name, w.vintage DESC"

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor.execute(query, params)
            return [Wine(**dict(row)) for row in cursor.fetchall()]

    def create(self, wine: Wine) -> int:
        """
        Create new wine record.

        Args:
            wine: Wine agents

        Returns:
            ID of created wine
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO wines (
                    source, external_id, wine_name, producer_id, vintage,
                    wine_type, varietal, designation, region_id, appellation,
                    vineyard, bottle_size, drink_from_year, drink_to_year, drink_index,
                    q_purchased, q_quantity, q_consumed,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                wine.source, wine.external_id, wine.wine_name, wine.producer_id,
                wine.vintage, wine.wine_type, wine.varietal, wine.designation,
                wine.region_id, wine.appellation, wine.vineyard, wine.bottle_size,
                wine.drink_from_year, wine.drink_to_year, wine.drink_index,
                wine.q_purchased, wine.q_quantity, wine.q_consumed,
                wine.created_at or datetime.now(), wine.updated_at or datetime.now()
            ))

            conn.commit()
            wine_id = cursor.lastrowid
            logger.debug(f"Created wine: {wine.wine_name} (ID: {wine_id})")
            return wine_id

    def update(self, wine: Wine) -> bool:
        """
        Update existing wine record.

        Args:
            wine: Wine agents with updated cellar-data

        Returns:
            True if successful
        """
        if not wine.id:
            raise ValueError("Wine ID is required for update")

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            update_query, params = build_update_query(
                "wines", wine, "id", ["producer_name", "region_name", "country", "personal_rating", "community_rating", "tasting_notes", "last_tasted_date"]
            )
            cursor.execute(update_query, params)

            conn.commit()
            logger.debug(f"Updated wine: {wine.wine_name} (ID: {wine.id})")
            return True

    def delete(self, wine_id: int) -> bool:
        """
        Delete wine record (and cascade to bottles).

        Args:
            wine_id: Wine ID

        Returns:
            True if successful
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM wines WHERE id = ?", (wine_id,))
            conn.commit()
            logger.debug(f"Deleted wine ID: {wine_id}")
            return True

    def count(
        self,
        wine_type: str | None = None,
        country: str | None = None
    ) -> int:
        """
        Count wines with optional filters.

        Args:
            wine_type: Filter by wine type
            country: Filter by country

        Returns:
            Number of wines
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT COUNT(*) as count
                FROM wines w
                LEFT JOIN regions r ON w.region_id = r.id
                WHERE 1=1
            """
            params = []

            if wine_type:
                query += " AND w.wine_type = ?"
                params.append(wine_type)

            if country:
                query += " AND r.country = ?"
                params.append(country)

            cursor.execute(query, params)
            return cursor.fetchone()['count']

    def update_description(self, wine_id: int, description: str) -> bool:
        """
        Update description for a wine.

        Convenience method for updating only the description field.
        Useful for LLM-generated description updates.

        Args:
            wine_id: Wine ID
            description: New description text

        Returns:
            True if successful, False if wine not found

        Example:
            >>> wine_repo.update_description(123, "This Chianti Classico...")
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE wines 
                SET description = ?, updated_at = ?
                WHERE id = ?
            """, (description, datetime.now(), wine_id))

            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(f"Updated description for wine ID: {wine_id}")
                return True
            else:
                logger.warning(f"Wine ID {wine_id} not found for description update")
                return False

    def get_without_description(self, limit: int | None = None) -> list[Wine]:
        """
        Get all wines that don't have descriptions.

        Useful for batch generating descriptions for wines that need them.

        Args:
            limit: Maximum number of wines to return (None for all)

        Returns:
            List of Wine models without descriptions

        Example:
            >>> wines_to_describe = wine_repo.get_without_description(limit=50)
            >>> for wine in wines_to_describe:
            >>>     description = service.get_wine_description(wine)
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT 
                    w.*, 
                    p.name as producer_name, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    r.country,
                    t.personal_rating,
                    t.community_rating,
                    t.tasting_notes,
                    t.last_tasted_date
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE w.description IS NULL
                ORDER BY w.wine_name
            """

            if limit:
                query += " LIMIT ?"
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)

            return [Wine(**dict(row)) for row in cursor.fetchall()]

    def count_with_description(self) -> dict[str, int]:
        """
        Count wines with and without descriptions.

        Returns:
            Dict with counts: {"with_description": N, "without_description": M, "total": T}

        Example:
            >>> stats = wine_repo.count_with_description()
            >>> print(f"{stats['with_description']} of {stats['total']} wines have descriptions")
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN description IS NOT NULL THEN 1 ELSE 0 END) as with_desc,
                    SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END) as without_desc
                FROM wines
            """)

            row = cursor.fetchone()
            return {
                "with_description": row['with_desc'],
                "without_description": row['without_desc'],
                "total": row['total']
            }

