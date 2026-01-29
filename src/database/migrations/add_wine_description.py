"""
Migration script to add description column to wines table.

This migration adds the description field to the wines table to store
LLM-generated wine descriptions for enhanced UI display.
"""

import sqlite3
from pathlib import Path

ALLOWED_TABLES: set[str] = {"wines", "producers"}


def check_column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    """
    Check if a column exists in a table.

    Args:
        cursor: Database cursor
        table_name: Name of the table to check. Must be in ALLOWED_TABLES.
        column_name: Name of the column to check

    Returns:
        True if column exists, False otherwise

    Raises:
        ValueError: If the table_name is not in the allowed whitelist.
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Invalid table name '{table_name}' for schema inspection.")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def add_wine_description_column(conn: sqlite3.Connection):
    """Add description column to wines table."""
    cursor = conn.cursor()

    # Check if column already exists
    if check_column_exists(cursor, "wines", "description"):
        print("  - Column 'description' already exists in wines table, skipping")
        return

    # Add description column
    cursor.execute("""
        ALTER TABLE wines 
        ADD COLUMN description TEXT
    """)

    conn.commit()
    print("✓ Added 'description' column to wines table")


def run_migration(db_path: str):
    """
    Run the migration to add description column to wines table.

    Args:
        db_path: Path to the SQLite database file
    """
    print(f"Running migration on database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        add_wine_description_column(conn)
        print("✓ Migration completed successfully")
    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from src.utils import get_default_db_path

    # Use the same method as the rest of the application
    db_path = get_default_db_path()
    run_migration(str(db_path))
