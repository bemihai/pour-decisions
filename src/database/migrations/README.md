# Database Migrations

> **Project version**: 0.8.3 — last verified 2026-08-30.
> Current migrations reflect schema up to v0.7.0. New migrations will be added as Milestone 14
> (knowledge graph) and other milestones introduce schema changes.

This folder contains database migration scripts for the Pour Decisions wine cellar database.

## Running Migrations

Each migration script can be run independently:

```bash
# From project root
python src/database/migrations/<migration_name>.py
```

The migration scripts will:
- Use the default database path from `get_default_db_path()`
- Check if changes already exist before applying them
- Provide clear output about what was done

## Available Migrations

### add_drink_window_source.py
**Purpose**: Add `drink_window_source` column to `wines` table to track the provenance of each
wine's drinking window estimate.

**When to run**:
- After upgrading to a version that includes drinking window estimation
- On existing databases that don't have the `drink_window_source` column

**What it does**:
- Adds `drink_window_source TEXT` column to `wines` table
- Backfills existing CellarTracker rows: sets `drink_window_source = 'cellar_tracker'`
  where `source = 'cellar_tracker' AND drink_from_year IS NOT NULL`

**Usage**:
```bash
python src/database/migrations/add_drink_window_source.py
```

### add_wine_description.py
**Purpose**: Add `description` column to `wines` table for LLM-generated wine descriptions.

**When to run**: 
- After upgrading to a version that includes LLM description generation
- On existing databases that don't have the `description` column

**What it does**:
- Adds `description TEXT` column to `wines` table
- Safely checks if column already exists before adding

**Usage**:
```bash
python src/database/migrations/add_wine_description.py
```

### create_food_pairing_rules.py
**Purpose**: Create `food_pairing_rules` table and populate with initial pairing data.

**When to run**: 
- When setting up food pairing functionality
- On databases created before this feature was added

**What it does**:
- Creates `food_pairing_rules` table with indexes
- Inserts ~25 common food pairing rules
- Skips rules that already exist

**Usage**:
```bash
python src/database/migrations/create_food_pairing_rules.py
```

## Creating New Migrations

When creating a new migration script:

1. **Follow the naming convention**: `<action>_<table>_<feature>.py`
2. **Include a docstring** explaining what the migration does
3. **Check before modifying**: Always check if changes already exist
4. **Use transactions**: Wrap changes in try/except with rollback on error
5. **Provide feedback**: Print clear messages about what's happening
6. **Make it idempotent**: Running the same migration twice should be safe

### Migration Template

```python
"""
Migration script to <brief description>.

This migration <detailed explanation of what it does>.
"""

import sqlite3
from pathlib import Path


def check_exists(cursor: sqlite3.Cursor) -> bool:
    """Check if migration was already applied."""
    # Check logic here
    pass


def apply_migration(conn: sqlite3.Connection):
    """Apply the migration changes."""
    cursor = conn.cursor()
    
    if check_exists(cursor):
        print("  - Migration already applied, skipping")
        return
    
    # Apply changes here
    
    conn.commit()
    print("✓ Migration applied successfully")


def run_migration(db_path: str):
    """Run the migration."""
    print(f"Running migration on database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        apply_migration(conn)
        print("✓ Migration completed successfully")
    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    from src.utils import get_default_db_path
    
    db_path = get_default_db_path()
    run_migration(str(db_path))
```

## Rollback

Currently, migrations do not have automatic rollback scripts. If you need to undo a migration:

1. Restore from backup (recommended)
2. Manually reverse the changes using SQL

Always backup your database before running migrations in production.
