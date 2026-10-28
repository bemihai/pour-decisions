"""Explicit, versioned initialization of application-owned conversation storage."""

import argparse
import sqlite3
from pathlib import Path

from src.utils import logger

SCHEMA_VERSION = 1
SCHEMA = {
    "chat_schema_version": """CREATE TABLE chat_schema_version (
        id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL CHECK (version = 1))""",
    "chat_sessions": """CREATE TABLE chat_sessions (
        id TEXT PRIMARY KEY NOT NULL,
        agent_mode TEXT CHECK (agent_mode IN ('intelligent', 'rag_only')),
        title TEXT CHECK (length(title) BETWEEN 1 AND 120),
        version INTEGER CHECK (version > 0),
        state TEXT NOT NULL CHECK (state IN ('active', 'deleting', 'deleted')),
        created_at TEXT, updated_at TEXT, creation_fingerprint TEXT, deleted_at TEXT,
        CHECK ((state = 'deleted' AND agent_mode IS NULL AND title IS NULL AND version IS NULL
            AND created_at IS NULL AND updated_at IS NULL AND creation_fingerprint IS NULL AND deleted_at IS NOT NULL)
            OR (state != 'deleted' AND agent_mode IS NOT NULL AND title IS NOT NULL AND version IS NOT NULL
            AND created_at IS NOT NULL AND updated_at IS NOT NULL AND creation_fingerprint IS NOT NULL
            AND deleted_at IS NULL)))""",
    "chat_turns": """CREATE TABLE chat_turns (
        session_id TEXT NOT NULL REFERENCES chat_sessions(id),
        sequence INTEGER NOT NULL CHECK (sequence > 0),
        message TEXT NOT NULL CHECK (length(trim(message)) > 0 AND length(message) <= 32000),
        display_message TEXT NOT NULL CHECK (length(trim(display_message)) > 0 AND length(display_message) <= 32000),
        answer TEXT CHECK (length(answer) BETWEEN 1 AND 128000),
        sources TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(sources) AND json_type(sources) = 'array'
            AND json_array_length(sources) <= 100),
        web_sources TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(web_sources) AND json_type(web_sources) = 'array'
            AND json_array_length(web_sources) <= 100),
        status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id, sequence), CHECK (status != 'completed' OR answer IS NOT NULL))""",
    "chat_operations": """CREATE TABLE chat_operations (
        id TEXT PRIMARY KEY NOT NULL, session_id TEXT NOT NULL REFERENCES chat_sessions(id),
        turn_sequence INTEGER NOT NULL,
        action TEXT NOT NULL CHECK (action IN ('append', 'replace_last')),
        fingerprint TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
        response TEXT CHECK (response IS NULL OR json_valid(response)), error_code TEXT,
        conversation_version INTEGER NOT NULL CHECK (conversation_version > 0),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        FOREIGN KEY (session_id, turn_sequence) REFERENCES chat_turns(session_id, sequence),
        CHECK ((status = 'completed' AND response IS NOT NULL) OR (status != 'completed' AND response IS NULL)))""",
    "chat_sessions_activity": """CREATE INDEX chat_sessions_activity ON chat_sessions(updated_at DESC, id DESC)""",
    "chat_operations_running": """CREATE UNIQUE INDEX chat_operations_running ON chat_operations(session_id)
        WHERE status = 'running'""",
}


def normalized_sql(sql: str) -> str:
    """Normalize whitespace for strict application schema identity checks."""
    return " ".join(sql.split()).lower()


def validate_schema(objects: dict[str, str], version: int | None) -> None:
    """Reject missing, partial or incompatible application schema definitions."""
    if version != SCHEMA_VERSION or any(
        normalized_sql(objects.get(name, "")) != normalized_sql(sql) for name, sql in SCHEMA.items()
    ):
        raise ValueError("Conversation schema is missing or incompatible; run the explicit migration.")


def run_migration(db_path: str | Path) -> None:
    """Initialize a new explicit target or migrate a legacy database atomically.

    Existing saver and thread tables are left unchanged. Partial application schemas
    and unsupported versions fail without modification. API startup never calls this.
    """
    path = Path(db_path)
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        objects = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))
        present = set(SCHEMA).intersection(objects)
        if not present:
            for sql in SCHEMA.values():
                connection.execute(sql)
            connection.execute("INSERT INTO chat_schema_version VALUES (1, ?)", (SCHEMA_VERSION,))
        objects = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))
        rows = connection.execute("SELECT id, version FROM chat_schema_version").fetchall()
        validate_schema(objects, rows[0][1] if len(rows) == 1 and rows[0][0] == 1 else None)
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Conversation migration failed foreign-key validation.")
        connection.commit()
        logger.info("Conversation schema initialized and verified")
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    """Require an explicit database path rather than selecting a live default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, type=Path)
    run_migration(parser.parse_args().db_path)


if __name__ == "__main__":
    main()
