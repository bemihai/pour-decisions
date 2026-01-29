"""Producer repository."""
from datetime import datetime

from src.database import get_db_connection, build_update_query
from src.database.models import Producer
from src.utils import get_default_db_path, logger


class ProducerRepository:
    """Repository for producer-related database operations."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize producer repository.

        Args:
            db_path: Optional path to database file
        """
        self.db_path = db_path or get_default_db_path()

    def get_by_id(self, producer_id: int) -> Producer | None:
        """Get producer by ID."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM producers WHERE id = ?", (producer_id,))

            row = cursor.fetchone()
            if row:
                return Producer(**dict(row))
            return None

    def get_by_name(self, name: str) -> Producer | None:
        """Get producer by name (case-insensitive)."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM producers WHERE LOWER(name) = LOWER(?)",
                (name,)
            )

            row = cursor.fetchone()
            if row:
                return Producer(**dict(row))
            return None

    def get_or_create(self, name: str, country: str | None = None, region: str | None = None, description: str | None = None) -> int:
        """
        Get an existing producer by name or create a new one if the name does not exist.

        Args:
            name: Producer name
            country: Optional country
            region: Optional region
            description: Optional description/notes

        Returns:
            Producer ID
        """
        existing = self.get_by_name(name)
        if existing:
            return existing.id

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO producers (name, country, region, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, country, region, description, datetime.now(), datetime.now()))

            conn.commit()
            producer_id = cursor.lastrowid
            logger.debug(f"Created producer: {name} (ID: {producer_id})")
            return producer_id

    def update(self, producer: Producer) -> bool:
        """
        Update existing producer record.

        Args:
            producer: Producer agents with updated cellar-data

        Returns:
            True if successful
        """
        if not producer.id:
            raise ValueError("Producer ID is required for update")

        update_query, params = build_update_query("producers", producer)
        if not update_query:
            logger.info(f"No fields to update for producer ID: {producer.id}")
            return False

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(update_query, params)
            conn.commit()
            logger.debug(f"Updated producer ID: {producer.id}")
            return True

    def get_all(self) -> list[Producer]:
        """Get all producers."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM producers ORDER BY name")
            return [Producer(**dict(row)) for row in cursor.fetchall()]

    def update_description(self, producer_id: int, description: str) -> bool:
        """
        Update description for a producer.

        Convenience method for updating only the description field.
        Useful for LLM-generated description updates.

        Args:
            producer_id: Producer ID
            description: New description text

        Returns:
            True if successful, False if producer not found

        Example:
            >>> producer_repo.update_description(42, "Antinori is one of Italy's oldest...")
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE producers 
                SET description = ?, updated_at = ?
                WHERE id = ?
            """, (description, datetime.now(), producer_id))

            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(f"Updated description for producer ID: {producer_id}")
                return True
            else:
                logger.warning(f"Producer ID {producer_id} not found for description update")
                return False

    def get_without_description(self, limit: int | None = None) -> list[Producer]:
        """
        Get all producers that don't have descriptions.

        Useful for batch generating descriptions for producers that need them.

        Args:
            limit: Maximum number of producers to return (None for all)

        Returns:
            List of Producer models without descriptions

        Example:
            >>> producers_to_describe = producer_repo.get_without_description(limit=20)
            >>> for producer in producers_to_describe:
            >>>     description = service.get_producer_description(producer)
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT * FROM producers
                WHERE description IS NULL
                ORDER BY name
            """

            if limit:
                query += " LIMIT ?"
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)

            return [Producer(**dict(row)) for row in cursor.fetchall()]

    def count_with_description(self) -> dict[str, int]:
        """
        Count producers with and without descriptions.

        Returns:
            Dict with counts: {"with_description": N, "without_description": M, "total": T}

        Example:
            >>> stats = producer_repo.count_with_description()
            >>> print(f"{stats['with_description']} of {stats['total']} producers have descriptions")
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN description IS NOT NULL THEN 1 ELSE 0 END) as with_desc,
                    SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END) as without_desc
                FROM producers
            """)

            row = cursor.fetchone()
            return {
                "with_description": row['with_desc'],
                "without_description": row['without_desc'],
                "total": row['total']
            }

