"""Stats repository"""
from datetime import datetime

from src.database import get_db_connection
from src.utils import get_default_db_path, logger


class StatsRepository:
    """Repository for wine cellar statistics and aggregations."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize statistics repository.

        Args:
            db_path: Optional path to database file
        """
        self.db_path = db_path or get_default_db_path()

    def get_cellar_overview(self) -> dict:
        """
        Get overall cellar statistics.

        Returns:
            Dictionary with statistics
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Total bottles
            cursor.execute("""
                SELECT SUM(quantity) as total
                FROM bottles WHERE status = 'in_cellar'
            """)
            total_bottles = cursor.fetchone()['total'] or 0

            # Unique wines
            cursor.execute("""
                SELECT COUNT(DISTINCT wine_id) as count
                FROM bottles WHERE status = 'in_cellar'
            """)
            unique_wines = cursor.fetchone()['count'] or 0

            # By type
            cursor.execute("""
                SELECT 
                    w.wine_type,
                    COUNT(DISTINCT w.id) as unique_wines,
                    SUM(b.quantity) as bottles
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'in_cellar'
                GROUP BY w.wine_type
                ORDER BY bottles DESC
            """)
            by_type = [dict(row) for row in cursor.fetchall()]

            # By country
            cursor.execute("""
                SELECT 
                    r.country,
                    COUNT(DISTINCT w.id) as unique_wines,
                    SUM(b.quantity) as bottles
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                LEFT JOIN regions r ON w.region_id = r.id
                WHERE b.status = 'in_cellar'
                GROUP BY r.country
                ORDER BY bottles DESC
                LIMIT 10
            """)
            by_country = [dict(row) for row in cursor.fetchall()]

            return {
                'total_bottles': total_bottles,
                'unique_wines': unique_wines,
                'by_type': by_type,
                'by_country': by_country
            }

    def get_top_rated_wines(self, limit: int = 10) -> list[dict]:
        """Get the highest rated wines in collection."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    w.id, w.wine_name, w.vintage, w.wine_type,
                    p.name as producer,
                    r.country,
                    t.personal_rating,
                    t.community_rating,
                    COUNT(b.id) as bottles_owned
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                LEFT JOIN bottles b ON w.id = b.wine_id AND b.status = 'in_cellar'
                WHERE t.personal_rating IS NOT NULL
                GROUP BY w.id
                ORDER BY t.personal_rating DESC, t.community_rating DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_drinking_window_wines(self) -> dict[str, list[dict]]:
        """Get wines organized by drinking window status."""
        current_year = datetime.now().year

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Wines ready to drink now
            cursor.execute("""
                SELECT 
                    w.id, w.wine_name, w.vintage, w.wine_type,
                    p.name as producer,
                    w.drink_from_year, w.drink_to_year,
                    COUNT(b.id) as bottles
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'in_cellar'
                  AND w.drink_from_year <= ?
                  AND (w.drink_to_year >= ? OR w.drink_to_year IS NULL)
                GROUP BY w.id
                ORDER BY w.drink_to_year
            """, (current_year, current_year))
            ready_now = [dict(row) for row in cursor.fetchall()]

            # Wines to drink soon (window closing)
            cursor.execute("""
                SELECT 
                    w.id, w.wine_name, w.vintage, w.wine_type,
                    p.name as producer,
                    w.drink_from_year, w.drink_to_year,
                    COUNT(b.id) as bottles
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'in_cellar'
                  AND w.drink_to_year BETWEEN ? AND ?
                GROUP BY w.id
                ORDER BY w.drink_to_year
            """, (current_year, current_year + 2))
            drink_soon = [dict(row) for row in cursor.fetchall()]

            # Wines for aging
            cursor.execute("""
                SELECT 
                    w.id, w.wine_name, w.vintage, w.wine_type,
                    p.name as producer,
                    w.drink_from_year, w.drink_to_year,
                    COUNT(b.id) as bottles
                FROM wines w
                LEFT JOIN producers p ON w.producer_id = p.id
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'in_cellar'
                  AND w.drink_from_year > ?
                GROUP BY w.id
                ORDER BY w.drink_from_year
            """, (current_year,))
            for_aging = [dict(row) for row in cursor.fetchall()]

            return {
                'ready_now': ready_now,
                'drink_soon': drink_soon,
                'for_aging': for_aging
            }

    def get_consumed_with_ratings(self, wine_type: str | None = None, limit: int | None = None) -> list[dict]:
        """
        Get consumed bottles with wine details and ratings.

        Args:
            wine_type: Filter by wine type
            limit: Maximum number of results

        Returns:
            List of dictionaries with combined bottle and wine info, sorted by rating (descending)
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            query = """
                SELECT 
                    b.*,
                    w.wine_name, w.wine_type, w.vintage,
                    p.name as producer_name,
                    r.country, 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    t.personal_rating,
                    t.tasting_notes
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND t.personal_rating IS NOT NULL
            """
            params = []

            if wine_type:
                query += " AND w.wine_type = ?"
                params.append(wine_type)

            query += " ORDER BY t.personal_rating DESC, b.consumed_date DESC"
            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_cellar_value(self) -> dict:
        """
        Calculate total cellar value based on purchase prices.

        Returns:
            Dictionary with value statistics by currency
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Get total value by currency
            cursor.execute("""
                SELECT 
                    currency,
                    SUM(quantity * purchase_price) as total_value,
                    COUNT(DISTINCT wine_id) as wines_with_price
                FROM bottles
                WHERE status = 'in_cellar' AND purchase_price IS NOT NULL
                GROUP BY currency
                ORDER BY total_value DESC
            """)
            by_currency = [dict(row) for row in cursor.fetchall()]

            # Get bottles without price info
            cursor.execute("""
                SELECT SUM(quantity) as count
                FROM bottles
                WHERE status = 'in_cellar' AND purchase_price IS NULL
            """)
            bottles_without_price = cursor.fetchone()['count'] or 0

            return {
                'by_currency': by_currency,
                'bottles_without_price': bottles_without_price
            }

    def get_drinking_window_stats(self) -> dict:
        """
        Get statistics about bottles by drinking window status.

        Returns:
            Dictionary with counts for ready, hold, and unknown categories
        """
        current_year = datetime.now().year

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Bottles ready to drink
            cursor.execute("""
                SELECT SUM(b.quantity) as count
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                WHERE b.status = 'in_cellar'
                  AND w.drink_from_year IS NOT NULL
                  AND w.drink_from_year <= ?
                  AND (w.drink_to_year >= ? OR w.drink_to_year IS NULL)
            """, (current_year, current_year))
            ready_to_drink = cursor.fetchone()['count'] or 0

            # Bottles to hold (not yet in drinking window)
            cursor.execute("""
                SELECT SUM(b.quantity) as count
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                WHERE b.status = 'in_cellar'
                  AND w.drink_from_year IS NOT NULL
                  AND w.drink_from_year > ?
            """, (current_year,))
            to_hold = cursor.fetchone()['count'] or 0

            # Bottles with unknown drinking window
            cursor.execute("""
                SELECT SUM(b.quantity) as count
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                WHERE b.status = 'in_cellar'
                  AND w.drink_from_year IS NULL
            """)
            unknown = cursor.fetchone()['count'] or 0

            return {
                'ready_to_drink': ready_to_drink,
                'to_hold': to_hold,
                'unknown': unknown
            }

    def get_rating_statistics(self) -> dict:
        """
        Get comprehensive rating statistics for consumed wines.

        Returns:
            Dictionary with rating metrics (avg, min, max, count, distribution)
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Overall rating stats
            cursor.execute("""
                SELECT 
                    AVG(t.personal_rating) as avg_rating,
                    MIN(t.personal_rating) as min_rating,
                    MAX(t.personal_rating) as max_rating,
                    COUNT(DISTINCT b.id) as wines_rated
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND t.personal_rating IS NOT NULL
            """)
            overall = dict(cursor.fetchone())

            # Rating distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN t.personal_rating < 50 THEN '0-49'
                        WHEN t.personal_rating < 70 THEN '50-69'
                        WHEN t.personal_rating < 80 THEN '70-79'
                        WHEN t.personal_rating < 90 THEN '80-89'
                        ELSE '90-100'
                    END as rating_range,
                    COUNT(*) as count
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND t.personal_rating IS NOT NULL
                GROUP BY rating_range
                ORDER BY rating_range
            """)
            distribution = [dict(row) for row in cursor.fetchall()]

            return {
                'overall': overall,
                'distribution': distribution
            }

    def get_wine_type_stats(self) -> list[dict]:
        """
        Get statistics by wine type for consumed wines.

        Returns:
            List of dicts with type, count, avg_rating, highest_rated, most_recent
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    w.wine_type,
                    COUNT(DISTINCT b.id) as wines_tasted,
                    AVG(t.personal_rating) as avg_rating,
                    MAX(t.personal_rating) as highest_rating,
                    MAX(b.consumed_date) as most_recent_date
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed'
                GROUP BY w.wine_type
                ORDER BY wines_tasted DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def get_varietal_preferences(self, limit: int = 10) -> list[dict]:
        """
        Get top varietal preferences based on consumed wines.

        Args:
            limit: Maximum number of varietals to return

        Returns:
            List of dicts with varietal, count, avg_rating
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    w.varietal,
                    COUNT(DISTINCT b.id) as wines_tasted,
                    AVG(t.personal_rating) as avg_rating,
                    MAX(t.personal_rating) as highest_rating
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND w.varietal IS NOT NULL
                GROUP BY w.varietal
                HAVING COUNT(DISTINCT b.id) >= 1
                ORDER BY wines_tasted DESC, avg_rating DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_producer_preferences(self, limit: int = 10) -> list[dict]:
        """
        Get top producer preferences based on consumed wines.

        Args:
            limit: Maximum number of producers to return

        Returns:
            List of dicts with producer info and stats
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    p.name as producer_name,
                    p.country,
                    COUNT(DISTINCT b.id) as wines_tasted,
                    AVG(t.personal_rating) as avg_rating,
                    MAX(t.personal_rating) as highest_rating
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN producers p ON w.producer_id = p.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND p.name IS NOT NULL
                GROUP BY p.id
                HAVING COUNT(DISTINCT b.id) >= 1
                ORDER BY wines_tasted DESC, avg_rating DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_region_preferences(self, limit: int = 10) -> list[dict]:
        """
        Get top region preferences based on consumed wines.

        Args:
            limit: Maximum number of regions to return

        Returns:
            List of dicts with region info and stats
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), 'Unknown') as region_name,
                    r.country,
                    COUNT(DISTINCT b.id) as wines_tasted,
                    AVG(t.personal_rating) as avg_rating,
                    MAX(t.personal_rating) as highest_rating
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND r.primary_name IS NOT NULL
                GROUP BY r.id
                HAVING COUNT(DISTINCT b.id) >= 1
                ORDER BY wines_tasted DESC, avg_rating DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_rating_timeline(self) -> list[dict]:
        """
        Get rating trends over time (by month).

        Returns:
            List of dicts with month, avg_rating, count
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    strftime('%Y-%m', b.consumed_date) as month,
                    AVG(t.personal_rating) as avg_rating,
                    COUNT(DISTINCT b.id) as wines_count
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' 
                  AND b.consumed_date IS NOT NULL
                  AND t.personal_rating IS NOT NULL
                GROUP BY month
                ORDER BY month
            """)

            return [dict(row) for row in cursor.fetchall()]

    def get_tasting_streak_days(self) -> int:
        """
        Calculate the number of consecutive months with tastings.

        Returns:
            Number of consecutive months with wine tastings
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Get months with tastings, ordered by date descending
            cursor.execute("""
                SELECT DISTINCT strftime('%Y-%m', b.consumed_date) as month
                FROM bottles b
                WHERE b.status = 'consumed' AND b.consumed_date IS NOT NULL
                ORDER BY month DESC
            """)

            months = [row['month'] for row in cursor.fetchall()]

            if not months:
                return 0

            # Count consecutive months from most recent
            from datetime import datetime
            streak = 1
            current = datetime.strptime(months[0], '%Y-%m')

            for i in range(1, len(months)):
                prev = datetime.strptime(months[i], '%Y-%m')
                # Check if months are consecutive
                month_diff = (current.year - prev.year) * 12 + (current.month - prev.month)
                if month_diff == 1:
                    streak += 1
                    current = prev
                else:
                    break

            return streak


    def get_varietal_distribution(self, limit: int = 5) -> list[dict]:
        """
        Get distribution of wines by main grape/varietal.

        Args:
            limit: Maximum number of varietals to return

        Returns:
            List of dictionaries with varietal and bottle counts
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    w.varietal,
                    COUNT(DISTINCT w.id) as unique_wines,
                    SUM(b.quantity) as bottles
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'in_cellar' 
                  AND w.varietal IS NOT NULL 
                  AND w.varietal != ''
                GROUP BY w.varietal
                ORDER BY bottles DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_region_distribution(self, limit: int = 5) -> list[dict]:
        """
        Get distribution of wines by region.

        Args:
            limit: Maximum number of regions to return

        Returns:
            List of dictionaries with region and bottle counts
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region,
                    r.country,
                    COUNT(DISTINCT w.id) as unique_wines,
                    SUM(b.quantity) as bottles
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                LEFT JOIN regions r ON w.region_id = r.id
                WHERE b.status = 'in_cellar' 
                  AND r.primary_name IS NOT NULL 
                  AND r.primary_name != ''
                GROUP BY r.id, r.country
                ORDER BY bottles DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_cellar_size_over_time(self) -> list[dict]:
        """
        Get cellar size progression over time for CellarTracker bottles only.
        Tracks cumulative bottle count by month based on purchase dates.

        Returns:
            List of dictionaries with month and cumulative bottle count
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Get monthly cellar size progression for CellarTracker bottles only
            cursor.execute("""
                WITH monthly_purchases AS (
                    SELECT 
                        DATE(purchase_date, 'start of month') as month,
                        SUM(quantity) as bottles_added
                    FROM bottles b
                    JOIN wines w ON b.wine_id = w.id
                    WHERE b.source = 'cellar_tracker' 
                      AND b.purchase_date IS NOT NULL
                    GROUP BY DATE(purchase_date, 'start of month')
                ),
                monthly_consumption AS (
                    SELECT 
                        DATE(consumed_date, 'start of month') as month,
                        SUM(quantity) as bottles_consumed
                    FROM bottles b
                    JOIN wines w ON b.wine_id = w.id
                    WHERE b.source = 'cellar_tracker' 
                      AND b.consumed_date IS NOT NULL
                      AND b.status = 'consumed'
                    GROUP BY DATE(consumed_date, 'start of month')
                ),
                all_months AS (
                    SELECT month FROM monthly_purchases
                    UNION 
                    SELECT month FROM monthly_consumption
                ),
                monthly_net_change AS (
                    SELECT 
                        am.month,
                        COALESCE(mp.bottles_added, 0) - COALESCE(mc.bottles_consumed, 0) as net_change
                    FROM all_months am
                    LEFT JOIN monthly_purchases mp ON am.month = mp.month
                    LEFT JOIN monthly_consumption mc ON am.month = mc.month
                )
                SELECT 
                    month,
                    net_change,
                    SUM(net_change) OVER (ORDER BY month) as cumulative_bottles
                FROM monthly_net_change
                ORDER BY month
            """)

            results = [dict(row) for row in cursor.fetchall()]

            # Format month for better display
            for result in results:
                if result['month']:
                    # Convert YYYY-MM-DD to YYYY-MM format for display
                    result['month_display'] = result['month'][:7]

            return results

    def get_rating_distribution(self) -> dict:
        """Return rating distribution bucketed into 5-point intervals for consumed wines.

        Buckets: ``0-49`` (catch-all low), ``50-54`` … ``90-94`` (5-point), ``95-100``
        (catch-all high). Empty buckets are omitted.

        Returns:
            Dict with ``buckets`` (list of ``{"range": str, "count": int}``) and
            ``total`` (int) keys.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.personal_rating
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND t.personal_rating IS NOT NULL
            """)
            ratings = [row["personal_rating"] for row in cursor.fetchall()]

        if not ratings:
            return {"buckets": [], "total": 0}

        # Single-pass bucketing to avoid repeatedly scanning the ratings list.
        bucket_counts: dict[str, int] = {}
        for rating in ratings:
            if rating < 50:
                label = "0-49"
            elif rating >= 95:
                label = "95-100"
            else:
                start = (int(rating) // 5) * 5
                label = f"{start}-{start + 4}"
            bucket_counts[label] = bucket_counts.get(label, 0) + 1

        bucket_order = ["0-49"] + [f"{i}-{i + 4}" for i in range(50, 95, 5)] + ["95-100"]
        buckets = [
            {"range": label, "count": bucket_counts[label]}
            for label in bucket_order
            if label in bucket_counts
        ]
        return {"buckets": buckets, "total": len(ratings)}

    def get_country_stats(self, limit: int = 5) -> list[dict]:
        """Return consumed-wine statistics grouped by country.

        Args:
            limit: Maximum number of countries to return.

        Returns:
            List of dicts with ``country``, ``wines_tasted``, ``avg_rating``,
            and ``highest_rating`` keys, ordered by ``wines_tasted`` descending.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    r.country,
                    COUNT(DISTINCT b.id) as wines_tasted,
                    AVG(t.personal_rating) as avg_rating,
                    MAX(t.personal_rating) as highest_rating
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND r.country IS NOT NULL
                GROUP BY r.country
                HAVING COUNT(DISTINCT b.id) >= 1
                ORDER BY wines_tasted DESC, avg_rating DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_vintage_stats(self, limit: int = 5) -> list[dict]:
        """Return consumed-wine statistics grouped by vintage year.

        Only vintages with at least 2 wines tasted are included.

        Args:
            limit: Maximum number of vintages to return.

        Returns:
            List of dicts with ``vintage``, ``wines_tasted``, ``avg_rating``,
            and ``highest_rating`` keys, ordered by ``avg_rating`` descending.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    w.vintage,
                    COUNT(DISTINCT b.id) as wines_tasted,
                    AVG(t.personal_rating) as avg_rating,
                    MAX(t.personal_rating) as highest_rating
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND w.vintage IS NOT NULL
                GROUP BY w.vintage
                HAVING COUNT(DISTINCT b.id) >= 2
                ORDER BY avg_rating DESC, wines_tasted DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_appellation_stats(self, limit: int = 5) -> list[dict]:
        """Return consumed-wine statistics grouped by appellation.

        Args:
            limit: Maximum number of appellations to return.

        Returns:
            List of dicts with ``appellation``, ``country``, ``wines_tasted``,
            ``avg_rating``, and ``highest_rating`` keys.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    w.appellation,
                    r.country,
                    COUNT(DISTINCT b.id) as wines_tasted,
                    AVG(t.personal_rating) as avg_rating,
                    MAX(t.personal_rating) as highest_rating
                FROM bottles b
                JOIN wines w ON b.wine_id = w.id
                LEFT JOIN regions r ON w.region_id = r.id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'consumed' AND w.appellation IS NOT NULL
                GROUP BY w.appellation
                HAVING COUNT(DISTINCT b.id) >= 1
                ORDER BY wines_tasted DESC, avg_rating DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_cellar_vintage_distribution(self) -> list[dict]:
        """Return bottle counts grouped by vintage year for in-cellar wines.

        Returns:
            List of dicts with ``vintage`` (int) and ``bottles`` (int) keys,
            ordered by vintage year ascending.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    w.vintage,
                    SUM(b.quantity) AS bottles
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'in_cellar' AND w.vintage IS NOT NULL
                GROUP BY w.vintage
                ORDER BY w.vintage ASC
                """,
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_cellar_rating_distribution(self) -> list[dict]:
        """Return personal-rating tier counts for rated wines in the cellar.

        Only wines with ``status = 'in_cellar'`` and a personal rating are
        included.  Tiers match the Streamlit show_cellar_statistics() display.

        Returns:
            List of dicts with ``tier`` (str) and ``wines`` (int) keys for
            non-empty tiers only, ordered from highest to lowest tier.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT w.id, t.personal_rating
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                LEFT JOIN tastings t ON w.id = t.wine_id
                WHERE b.status = 'in_cellar' AND t.personal_rating IS NOT NULL
                """,
            )
            ratings = [row["personal_rating"] for row in cursor.fetchall()]

        if not ratings:
            return []

        tiers = [
            ("Exceptional (98-100)", lambda r: r >= 98),
            ("Outstanding (94-97)", lambda r: 94 <= r < 98),
            ("Excellent (90-93)",   lambda r: 90 <= r < 94),
            ("Very Good (86-89)",   lambda r: 86 <= r < 90),
            ("Good (80-85)",        lambda r: 80 <= r < 86),
            ("Average (70-79)",     lambda r: 70 <= r < 80),
        ]
        return [
            {"tier": label, "wines": sum(1 for r in ratings if pred(r))}
            for label, pred in tiers
            if any(pred(r) for r in ratings)
        ]

    def get_consumed_filter_options(self) -> dict:
        """Return distinct filter values derived from all consumed wines.

        Runs lightweight ``SELECT DISTINCT`` queries instead of loading
        every consumed row.

        Returns:
            Dict with ``wine_types``, ``countries``, ``producers``,
            ``min_vintage``, and ``max_vintage`` keys.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DISTINCT w.wine_type
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'consumed' AND w.wine_type IS NOT NULL
                ORDER BY w.wine_type
            """)
            wine_types = [row[0] for row in cursor.fetchall()]

            cursor.execute("""
                SELECT DISTINCT r.country
                FROM regions r
                JOIN wines w ON w.region_id = r.id
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'consumed' AND r.country IS NOT NULL
                ORDER BY r.country
            """)
            countries = [row[0] for row in cursor.fetchall()]

            cursor.execute("""
                SELECT DISTINCT p.name
                FROM producers p
                JOIN wines w ON w.producer_id = p.id
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'consumed' AND p.name IS NOT NULL
                ORDER BY p.name
            """)
            producers = [row[0] for row in cursor.fetchall()]

            cursor.execute("""
                SELECT MIN(w.vintage) as min_v, MAX(w.vintage) as max_v
                FROM wines w
                JOIN bottles b ON w.id = b.wine_id
                WHERE b.status = 'consumed' AND w.vintage IS NOT NULL
            """)
            row = cursor.fetchone()
            min_vintage = row["min_v"] or 2000
            max_vintage = row["max_v"] or 2025

            return {
                "wine_types": wine_types,
                "countries": countries,
                "producers": producers,
                "min_vintage": min_vintage,
                "max_vintage": max_vintage,
            }

    _CONSUMED_SQL_SORT: dict[str, str] = {
        "consumed_date_desc": "b.consumed_date DESC",
        "consumed_date_asc": "b.consumed_date ASC",
        "rating_desc": "t.personal_rating DESC",
        "rating_asc": "t.personal_rating ASC",
        "producer": "p.name ASC, w.vintage ASC",
        "wine_name": "w.wine_name ASC",
    }

    def get_consumed_wines(
        self,
        wine_type: str | None = None,
        country: str | None = None,
        producer: str | None = None,
        min_vintage: int | None = None,
        max_vintage: int | None = None,
        rating_filter: str | None = None,
        search: str | None = None,
        sort_by: str = "consumed_date_desc",
        limit: int = 20,
    ) -> dict:
        """Return consumed wines filtered and sorted at the SQL level.

        Args:
            wine_type: Filter by exact wine type string.
            country: Filter by country of origin.
            producer: Filter by exact producer name.
            min_vintage: Minimum vintage year (inclusive).
            max_vintage: Maximum vintage year (inclusive).
            rating_filter: One of ``"rated"``, ``"unrated"``, ``"90+"``,
                ``"80+"``, ``"70+"`` or ``None`` for all.
            search: Free-text search across wine name, producer, and varietal.
            sort_by: Sort key from ``_CONSUMED_SQL_SORT``.
            limit: Maximum number of rows to return.

        Returns:
            Dict with ``items`` (list of row dicts) and ``total`` (untruncated
            count after filtering) keys.
        """
        base_query = """
            FROM bottles b
            JOIN wines w ON b.wine_id = w.id
            LEFT JOIN producers p ON w.producer_id = p.id
            LEFT JOIN regions r ON w.region_id = r.id
            LEFT JOIN tastings t ON w.id = t.wine_id
            WHERE b.status = 'consumed'
        """
        params: list = []

        if wine_type:
            base_query += " AND w.wine_type = ?"
            params.append(wine_type)
        if country:
            base_query += " AND r.country = ?"
            params.append(country)
        if producer:
            base_query += " AND p.name = ?"
            params.append(producer)
        if min_vintage is not None:
            base_query += " AND w.vintage >= ?"
            params.append(min_vintage)
        if max_vintage is not None:
            base_query += " AND w.vintage <= ?"
            params.append(max_vintage)
        if rating_filter:
            rf = rating_filter.lower()
            if rf == "rated":
                base_query += " AND t.personal_rating IS NOT NULL"
            elif rf == "unrated":
                base_query += " AND t.personal_rating IS NULL"
            elif rf.endswith("+"):
                try:
                    threshold = int(rf.rstrip("+"))
                    base_query += " AND t.personal_rating >= ?"
                    params.append(threshold)
                except ValueError:
                    logger.warning(f"Invalid rating_filter value: {rating_filter}")
        if search:
            base_query += (
                " AND (LOWER(w.wine_name) LIKE '%' || LOWER(?) || '%'"
                " OR LOWER(p.name) LIKE '%' || LOWER(?) || '%'"
                " OR LOWER(w.varietal) LIKE '%' || LOWER(?) || '%')"
            )
            params.extend([search, search, search])

        order_clause = self._CONSUMED_SQL_SORT.get(sort_by, self._CONSUMED_SQL_SORT["consumed_date_desc"])

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(f"SELECT COUNT(*) as total {base_query}", params)
            total = cursor.fetchone()["total"] or 0

            select_clause = """
                SELECT
                    w.id as wine_id,
                    b.id as bottle_id,
                    w.wine_name, w.wine_type, w.vintage, w.varietal,
                    p.name as producer_name,
                    r.country,
                    COALESCE(r.primary_name || COALESCE(' - ' || r.secondary_name, ''), '') as region_name,
                    t.personal_rating, t.community_rating, t.tasting_notes, t.last_tasted_date,
                    b.consumed_date
            """
            cursor.execute(
                f"{select_clause} {base_query} ORDER BY {order_clause} LIMIT ?",
                params + [limit],
            )
            items = [dict(row) for row in cursor.fetchall()]

        return {"items": items, "total": total}

