"""Bottle repository"""
from datetime import datetime, date

from src.database import get_db_connection, build_update_query
from src.database.models import Bottle
from src.utils import get_default_db_path, logger


class BottleRepository:
    """Repository for bottle-related database operations."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize bottle repository.

        Args:
            db_path: Optional path to database file
        """
        self.db_path = db_path or get_default_db_path()

    def get_by_id(self, bottle_id: int) -> Bottle | None:
        """
        Get bottle by ID.

        Args:
            bottle_id: Bottle ID

        Returns:
            Bottle agents or None if not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM bottles WHERE id = ?", (bottle_id,))

            row = cursor.fetchone()
            if row:
                return Bottle(**dict(row))
            return None

    def get_by_wine_and_external_id(self, wine_id: int, external_bottle_id: str) -> Bottle | None:
        """
        Get bottle by external bottle ID.

        Args:
            wine_id: Wine ID
            external_bottle_id: External bottle ID from source system

        Returns:
            Bottle agents or None if not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM bottles WHERE wine_id = ? AND external_bottle_id = ?",
                (wine_id, external_bottle_id,)
            )

            row = cursor.fetchone()
            if row:
                return Bottle(**dict(row))
            return None

    def get_by_wine(self, wine_id: int, status: str | None = None) -> list[Bottle]:
        """
        Get all bottles for a wine.

        Args:
            wine_id: Wine ID
            status: Optional status filter (in_cellar, consumed, etc.)

        Returns:
            List of Bottle models
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM bottles WHERE wine_id = ?"
            params = [wine_id]

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY location, bin"

            cursor.execute(query, params)
            return [Bottle(**dict(row)) for row in cursor.fetchall()]

    def get_owned_quantity(self, wine_id: int) -> int:
        """
        Get total quantity of owned bottles for a wine (in cellar).

        Args:
            wine_id: Wine ID

        Returns:
            Total number of bottles owned (in_cellar status)
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT SUM(quantity) as total FROM bottles WHERE wine_id = ? AND status = 'in_cellar'",
                (wine_id,)
            )
            row = cursor.fetchone()
            return row['total'] if row and row['total'] else 0

    def get_inventory(self, location : str | None = None, wine_type: str | None = None) -> list[dict]:
        """
        Get current inventory with wine details.

        Args:
            location: Filter by location
            wine_type: Filter by wine type

        Returns:
            List of dictionaries with combined bottle and wine info
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT 
                    b.*,
                    w.wine_name, w.wine_type, w.vintage, w.varietal, w.drink_index,
                    w.drink_from_year, w.drink_to_year, w.drink_window_source, w.description,
                    w.q_purchased, w.q_consumed, w.q_quantity,
                    w.producer_id,
                    p.name as producer_name, p.description as producer_description,
                    r.country, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    t.personal_rating, t.community_rating, t.like_votes, t.like_percentage, t.last_tasted_date
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'in_cellar'
            """
            params = []

            if location:
                query += " AND b.location = ?"
                params.append(location)

            if wine_type:
                query += " AND w.wine_type = ?"
                params.append(wine_type)

            query += " ORDER BY p.name, w.vintage DESC, b.location"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def create(self, bottle: Bottle) -> int:
        """
        Create new bottle record.

        Args:
            bottle: Bottle agents

        Returns:
            ID of created bottle
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO bottles (
                    wine_id, source, external_bottle_id, quantity, status,
                    location, bin, purchase_date, purchase_price, valuation_price, currency,
                    store_name, consumed_date, bottle_note,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bottle.wine_id, bottle.source, bottle.external_bottle_id,
                bottle.quantity, bottle.status, bottle.location, bottle.bin,
                bottle.purchase_date, bottle.purchase_price, bottle.valuation_price, bottle.currency,
                bottle.store_name, bottle.consumed_date, bottle.bottle_note,
                bottle.created_at or datetime.now(), bottle.updated_at or datetime.now()
            ))

            conn.commit()
            bottle_id = cursor.lastrowid
            logger.debug(f"Created bottle for wine_id={bottle.wine_id} (ID: {bottle_id})")
            return bottle_id

    def update(self, bottle: Bottle) -> bool:
        """
        Update existing bottle record.

        Args:
            bottle: Bottle agents with updated cellar-data

        Returns:
            True if successful
        """
        if not bottle.id:
            raise ValueError("Bottle ID is required for update")

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            update_query, params = build_update_query(
                "bottles", bottle, "id", exclude_fields=["wine_name", "producer_name", "region_name", "vintage"]
            )
            cursor.execute(update_query, params)

            conn.commit()
            logger.debug(f"Updated bottle ID: {bottle.id}")
            return True

    def mark_consumed(self, bottle_id: int, consumed_date: date | None = None) -> bool:
        """
        Mark bottle as consumed.

        Args:
            bottle_id: Bottle ID
            consumed_date: Date consumed (defaults to today)

        Returns:
            True if successful
        """
        consumed_date = consumed_date or date.today()

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE bottles
                SET status = 'consumed', consumed_date = ?, updated_at = ?
                WHERE id = ?
            """, (consumed_date, datetime.now(), bottle_id))

            conn.commit()
            logger.debug(f"Marked bottle {bottle_id} as consumed on {consumed_date}")
            return True

    def get_total_bottles(self, status: str = "in_cellar") -> int:
        """
        Get total number of bottles by status.

        Args:
            status: Bottle status

        Returns:
            Total bottle count
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(quantity) as total
                FROM bottles
                WHERE status = ?
            """, (status,))

            result = cursor.fetchone()
            return result['total'] or 0
