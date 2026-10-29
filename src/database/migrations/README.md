# Database Migrations

> **Project version**: 0.14.0 — last updated 2026-10-28.
> Current migrations include raw CellarTracker availability storage, declared preferences, the
> Shopping List and saved-conversation schemas, reusable grape records, tasting uniqueness, and
> explicit producer- and wine-description cleanup.

This folder contains explicit migration scripts for Pour Decisions SQLite databases.

## Running Migrations

Each migration script can be run independently:

```bash
# From project root
python src/database/migrations/<migration_name>.py
```

The migration scripts will:
- Use the default database path from `get_default_db_path()` unless a destructive migration requires explicit paths
- Check if changes already exist before applying them
- Provide clear output about what was done

## Available Migrations

`fix_bottles_check_constraint.py` is currently an empty placeholder and has no runnable migration.

### add_tastings_unique_wine_id.py

**Purpose**: Deduplicate tastings, enforce one tasting row per wine with `UNIQUE(wine_id)`, and then
optionally backfill CellarTracker community data.

```bash
PYTHONPATH=. python3 -m src.database.migrations.add_tastings_unique_wine_id
```

Stop writers and create a verified cellar backup first. The schema rewrite preserves the most
informative row for each wine and merges available community fields. If CellarTracker credentials
are present, the command also makes live CellarTracker requests for the backfill; without them it
skips that external step. This older migration returns a failure status rather than raising for
some errors, so inspect its log and validate database integrity before restarting writers.

### create_grapes.py

**Purpose**: Add reusable grapes and wine/grape links, backfilling explicitly named varieties
from existing `wines.varietal` values. Generic blends remain unlinked. The persistence matcher
prefers longer names over overlapping aliases; retrieval and preference matching are unchanged.

Stop cellar writers and back up the database with `make cellar-backup`, then run:

```bash
PYTHONPATH=. python -m src.database.migrations.create_grapes
```

Run this explicit migration before serving wine details or importing into an existing database.
Fresh databases include the same schema. The migration checks schema parity, integrity and foreign
keys in one transaction and rolls back on failure. Re-running preserves cached descriptions and
reconciles links without duplicates. It never calls an LLM or web search and never runs at startup.

### clear_wine_descriptions.py

**Purpose**: Clear all stored wine descriptions so they can be regenerated individually under the
wine-specific evidence policy. Producer and region descriptions and all drinking-window fields are
preserved.

**Before running**:

1. Stop API, import, and sync writers.
2. Run `make cellar-backup`.
3. Select the backup that exactly matches the current wine records. The migration verifies its
   integrity, foreign keys, and complete wine snapshot before changing data.

**Usage**:

```bash
PYTHONPATH=. python3 -m src.database.migrations.clear_wine_descriptions \
  --database cellar-data/wine_cellar.db \
  --backup backups/wine_cellar/<verified-backup>.db \
  --confirm-writers-stopped
```

The migration takes an exclusive transaction, clears only non-null `wines.description` values,
validates the result, and reports the affected count. Repeating it clears zero rows. It does not
call web search or an LLM, regenerate descriptions, or run at startup.

### clear_producer_descriptions.py

**Purpose**: Clear all stored producer descriptions so they can be regenerated individually under
the producer-specific evidence policy.

**Before running**:

1. Stop API, import, and sync writers.
2. Run `make cellar-backup`.
3. Select the backup that exactly matches the current producer records. The migration verifies its
   integrity, foreign keys, and producer identity/description snapshot before changing data.

**Usage**:

```bash
PYTHONPATH=. python3 -m src.database.migrations.clear_producer_descriptions \
  --database cellar-data/wine_cellar.db \
  --backup backups/wine_cellar/<verified-backup>.db \
  --confirm-writers-stopped
```

The migration takes an exclusive transaction, clears only non-null `producers.description` values,
runs integrity and foreign-key checks before commit, and reports the affected count. Repeating it
clears zero rows. It does not call web search or an LLM, regenerate descriptions, or run at startup.

Restore the verified backup with `make cellar-restore BACKUP_FILE=<verified-backup>` if the previous
producer descriptions are needed again.

### add_cellar_tracker_available.py

**Purpose**: Add nullable raw CellarTracker `Available` storage without changing the canonical
drinking-readiness calculation.

**Usage**:

```bash
PYTHONPATH=. python3 -m src.database.migrations.add_cellar_tracker_available
```

The migration copies recoverable legacy `drink_index` values for CellarTracker wines whose window
source is CellarTracker or unset. It runs in one explicit transaction, validates database integrity
and foreign keys before commit, and is safe to run repeatedly. API startup never runs it.

### create_declared_preferences.py

**Purpose**: Create the strict single-profile `declared_preferences` table and indexes.

**Before running on a real cellar**:

1. Stop API, import, and sync writers.
2. Record row counts for `producers`, `regions`, `wines`, `bottles`, `tastings`, and `sync_log`.
3. Run `make cellar-backup`.
4. Confirm the selected backup is non-empty and run `PRAGMA quick_check` against that backup.

**Usage**:

```bash
PYTHONPATH=. python3 -m src.database.migrations.create_declared_preferences
```

The migration opens one explicit transaction, creates or verifies the schema, and then runs
`PRAGMA integrity_check` and `PRAGMA foreign_key_check` before commit. It is safe to run twice. An
incompatible partial table fails and rolls back without repair. API startup never runs it.

**Failure recovery**:

- Leave all writers stopped after any migration or validation failure.
- Before the first preference write, restore the verified backup with
  `make cellar-restore BACKUP_FILE=backups/wine_cellar/<verified-backup>.db`.
- Run `PRAGMA integrity_check` and `PRAGMA foreign_key_check` on the restored database and compare
  the six recorded core-table counts before restarting writers.
- After any preference has been written, do not restore the pre-migration backup or drop the table.
  Keep preference list/create/update/delete/reset management available while preference consumption
  is repaired or disabled, so user-managed records are not stranded or lost.

### create_chat_sessions.py

**Purpose**: Add versioned application-owned saved-conversation sessions, public turns, operation
records, and indexes to the conversation-memory database. LangGraph retains ownership of its saver
tables. The migration never reads or rewrites saver tables and never runs during API startup.

**Before running on a real conversation database**:

1. Resolve `session_memory.db_path` from `app_config.yml` and confirm that exact file is the intended
   target.
2. Stop the API and every process using the conversation database. Do not copy only the SQLite main
   file while writers are active because committed WAL content may be omitted.
3. Create a timestamped backup with SQLite's `.backup` command or online backup API, then run
   `PRAGMA quick_check` and `PRAGMA foreign_key_check` against the backup.
4. Record the backup path and application revision together. The pre-upgrade database must be paired
   with pre-upgrade code for rollback.

**Usage**:

```bash
CONVERSATION_DB=cellar-data/conversation_memory.db
CONVERSATION_BACKUP=backups/conversations/conversation_memory_pre_chat_sessions_YYYYMMDD_HHMMSS.db
mkdir -p backups/conversations
sqlite3 "$CONVERSATION_DB" ".backup '$CONVERSATION_BACKUP'"
sqlite3 "$CONVERSATION_BACKUP" "PRAGMA quick_check; PRAGMA foreign_key_check;"
PYTHONPATH=. uv run python -m src.database.migrations.create_chat_sessions --db-path "$CONVERSATION_DB"
sqlite3 "$CONVERSATION_DB" \
  "PRAGMA integrity_check; PRAGMA foreign_key_check; SELECT id, version FROM chat_schema_version;"
```

For a new installation without a conversation database, create the configured empty SQLite target
first with `sqlite3 "$CONVERSATION_DB" "PRAGMA quick_check;"`, then run the same migration command.
The migration requires an existing explicit target, uses one immediate transaction, preserves legacy
thread and saver content, rejects incompatible partial or newer schemas, and is safe to run again as
a verification step.

**Recovery and compatibility**:

- On migration or validation failure, keep writers stopped. The migration rolls back its transaction;
  investigate before choosing restore or forward repair.
- To restore, stop writers, pair the older application revision with the verified pre-upgrade backup,
  and use SQLite `.restore` or the online backup API. Re-run integrity and foreign-key checks before
  starting that pair.
- Never start an older application against an upgraded conversation database. Older code can expire
  or delete thread state without honoring saved-session retention.
- Restoring the pre-upgrade backup discards every saved conversation and other conversation-memory
  write made after the backup. Once saved conversations exist, prefer a forward fix unless that loss
  is explicitly accepted.
- Saved sessions and their operation results remain until explicit deletion. Disk growth does not
  authorize checkpoint compaction or expiry; those require a separate reviewed decision.

Disposable restore rehearsal, or real rollback after explicitly accepting post-backup data loss:

```bash
RESTORED_CONVERSATION_DB=/tmp/conversation_memory_restored.db
sqlite3 "$RESTORED_CONVERSATION_DB" ".restore '$CONVERSATION_BACKUP'"
sqlite3 "$RESTORED_CONVERSATION_DB" "PRAGMA integrity_check; PRAGMA foreign_key_check;"
```

For real rollback, validate the disposable restored copy first. Keep writers stopped, replace the
configured database only through the operator's established recoverable file procedure, and start
only the application revision recorded with that backup.

### create_shopping_list.py

**Purpose**: Add the strict `shopping_list_items` table and indexes to an existing cellar database.
Fresh databases already contain the same schema. The migration is additive, explicit, idempotent,
and never runs during API startup.

**Before running on a real cellar**:

1. Resolve the configured cellar database path and confirm it is the intended target. The module
   uses `get_default_db_path()`; a command-scoped Make variable does not retarget it.
2. Stop API, import, and sync writers.
3. Record row counts for `producers`, `regions`, `wines`, `bottles`, `tastings`, `sync_log`, and
   `declared_preferences`.
4. Run `make cellar-backup` against that same database path.
5. Confirm the backup is non-empty and run `PRAGMA quick_check` against it.

**Usage**:

```bash
PYTHONPATH=. python3 -m src.database.migrations.create_shopping_list
```

The migration requires an existing database, uses one immediate transaction, rejects incompatible
partial schemas, and runs both `PRAGMA integrity_check` and `PRAGMA foreign_key_check` before commit.
Re-running it verifies the existing schema without changing Shopping List rows.

**Recovery and compatibility**:

- On migration or validation failure, leave writers stopped and investigate; the transaction rolls
  back without attempting a repair.
- Before the first Shopping List write, the verified full-database backup can restore the exact
  pre-migration state. After restoring, rerun integrity, foreign-key, and recorded row-count checks.
- After any Shopping List write, restoring the pre-migration backup loses that intent and all other
  post-backup cellar changes. Prefer reverting application code while retaining the additive table;
  older code can ignore it.
- Never drop only `shopping_list_items` as a routine rollback. Catalogue identity and list history
  recover together only through a consistent full-database restore.

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
