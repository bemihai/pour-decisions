"""Transactional repository for manually managed purchase intentions."""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

from src.database import get_db_connection
from src.database.models import (
    ShoppingListCatalogueOption,
    ShoppingListItem,
    ShoppingListPriority,
    ShoppingListStatus,
)
from src.utils import get_default_db_path

WRITE_BUSY_TIMEOUT_MS = 1_000
_MUTABLE_FIELDS = {"name", "vintage", "quantity", "priority", "notes", "retailer_url", "status"}


class ShoppingListRepositoryError(Exception):
    """Base class for bounded Shopping List repository failures."""


class ShoppingListValidationError(ShoppingListRepositoryError):
    """Raised when input violates the Shopping List contract."""


class ShoppingListNotFoundError(ShoppingListRepositoryError):
    """Raised when an item or selected catalogue wine is missing."""


class ShoppingListDuplicateError(ShoppingListRepositoryError):
    """Raised when a Wanted identity already exists."""

    def __init__(self, existing_item_id: int) -> None:
        super().__init__("A Wanted Shopping List item already uses this identity.")
        self.existing_item_id = existing_item_id


class ShoppingListVersionConflictError(ShoppingListRepositoryError):
    """Raised when an optimistic version does not match current state."""


class ShoppingListStoreBusyError(ShoppingListRepositoryError):
    """Raised after the bounded SQLite write-lock wait is exhausted."""


class ShoppingListStoreUnavailableError(ShoppingListRepositoryError):
    """Raised when the explicit Shopping List schema is unavailable."""


def normalize_shopping_list_name(value: str) -> tuple[str, str]:
    """Return NFC display text and case-folded identity text."""
    if not isinstance(value, str):
        raise ShoppingListValidationError("Name must be text.")
    if any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value):
        raise ShoppingListValidationError("Name contains unsupported control characters.")
    display_name = unicodedata.normalize("NFC", " ".join(value.strip().split()))
    if not 1 <= len(display_name) <= 300:
        raise ShoppingListValidationError("Name must contain 1 to 300 characters.")
    return display_name, display_name.casefold()


def _validate_vintage(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1000 <= value <= 9999:
        raise ShoppingListValidationError("Vintage must be between 1000 and 9999.")
    return value


def _validate_quantity(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 999:
        raise ShoppingListValidationError("Quantity must be between 1 and 999.")
    return value


def _validate_priority(value: object) -> ShoppingListPriority:
    try:
        return ShoppingListPriority(value)
    except (TypeError, ValueError) as error:
        raise ShoppingListValidationError("Priority must be normal or high.") from error


def _validate_status(value: object) -> ShoppingListStatus:
    try:
        return ShoppingListStatus(value)
    except (TypeError, ValueError) as error:
        raise ShoppingListValidationError("Status must be wanted or purchased.") from error


def _clean_notes(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ShoppingListValidationError("Notes must be text.")
    cleaned = value.strip()
    if len(cleaned) > 2000:
        raise ShoppingListValidationError("Notes cannot exceed 2000 characters.")
    return cleaned or None


def _clean_retailer_url(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ShoppingListValidationError("Retailer URL must be text.")
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > 2048:
        raise ShoppingListValidationError("Retailer URL cannot exceed 2048 characters.")
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ShoppingListValidationError("Retailer URL must use HTTP(S) and include a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ShoppingListValidationError("Retailer URL cannot contain embedded credentials.")
    return cleaned


def _literal_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _translate_operational_error(error: sqlite3.OperationalError) -> None:
    message = str(error).casefold()
    if "locked" in message or "busy" in message:
        raise ShoppingListStoreBusyError("Shopping List storage is busy.") from error
    if "no such table" in message or "no such column" in message:
        raise ShoppingListStoreUnavailableError("Shopping List schema is unavailable.") from error
    raise error


class ShoppingListRepository:
    """Raw-SQL Shopping List reads and transactional lifecycle writes."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path if db_path is not None else get_default_db_path()

    @staticmethod
    def _begin_write(connection: sqlite3.Connection) -> None:
        connection.execute(f"PRAGMA busy_timeout = {WRITE_BUSY_TIMEOUT_MS}")
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _to_item(row: sqlite3.Row) -> ShoppingListItem:
        values = dict(row)
        values["link_available"] = values.pop("linked_wine_exists") == 1
        values.pop("normalized_name", None)
        return ShoppingListItem(**values)

    @staticmethod
    def _select_item(cursor: sqlite3.Cursor, item_id: int) -> sqlite3.Row | None:
        return cursor.execute(
            """
            SELECT sli.*, CASE WHEN w.id IS NULL THEN 0 ELSE 1 END AS linked_wine_exists
            FROM shopping_list_items sli
            LEFT JOIN wines w ON w.id = sli.wine_id
            WHERE sli.id = ?
            """,
            (item_id,),
        ).fetchone()

    @staticmethod
    def _existing_identity_id(
        cursor: sqlite3.Cursor,
        *,
        catalogue_identity_id: int | None = None,
        normalized_name: str | None = None,
        vintage: int | None = None,
    ) -> int:
        if catalogue_identity_id is not None:
            row = cursor.execute(
                "SELECT id FROM shopping_list_items WHERE status = 'wanted' AND identity_kind = 'catalogue' "
                "AND catalogue_identity_id = ?",
                (catalogue_identity_id,),
            ).fetchone()
        else:
            row = cursor.execute(
                "SELECT id FROM shopping_list_items WHERE status = 'wanted' AND identity_kind = 'manual' "
                "AND normalized_name = ? AND COALESCE(vintage, -1) = COALESCE(?, -1)",
                (normalized_name, vintage),
            ).fetchone()
        if row is None:
            raise ShoppingListValidationError("Shopping List identity violates the storage contract.")
        return int(row[0])

    def get_by_id(self, item_id: int) -> ShoppingListItem | None:
        """Return one Shopping List item or ``None``."""
        try:
            with get_db_connection(self.db_path) as connection:
                row = self._select_item(connection.cursor(), item_id)
                return self._to_item(row) if row else None
        except sqlite3.OperationalError as error:
            _translate_operational_error(error)

    def list_items(
        self,
        status: ShoppingListStatus | str = ShoppingListStatus.WANTED,
        *,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ShoppingListItem], int]:
        """List one lifecycle view with stable order and a filtered total."""
        resolved_status = _validate_status(status)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ShoppingListValidationError("Limit must be between 1 and 100.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ShoppingListValidationError("Offset cannot be negative.")
        conditions = ["sli.status = ?"]
        parameters: list[object] = [resolved_status.value]
        if search is not None:
            if not isinstance(search, str):
                raise ShoppingListValidationError("Search must be text.")
            cleaned_search = " ".join(search.strip().split())
            if cleaned_search:
                pattern = f"%{_literal_like(cleaned_search)}%"
                conditions.append("(sli.name LIKE ? ESCAPE '\\' OR COALESCE(sli.notes, '') LIKE ? ESCAPE '\\')")
                parameters.extend([pattern, pattern])
        where_clause = " AND ".join(conditions)
        try:
            with get_db_connection(self.db_path) as connection:
                cursor = connection.cursor()
                total = cursor.execute(
                    f"SELECT COUNT(*) FROM shopping_list_items sli WHERE {where_clause}", parameters
                ).fetchone()[0]
                rows = cursor.execute(
                    f"""
                    SELECT sli.*, CASE WHEN w.id IS NULL THEN 0 ELSE 1 END AS linked_wine_exists
                    FROM shopping_list_items sli LEFT JOIN wines w ON w.id = sli.wine_id
                    WHERE {where_clause}
                    ORDER BY CASE sli.priority WHEN 'high' THEN 0 ELSE 1 END,
                             sli.created_at DESC, sli.id DESC
                    LIMIT ? OFFSET ?
                    """,
                    [*parameters, limit, offset],
                ).fetchall()
                return [self._to_item(row) for row in rows], int(total)
        except sqlite3.OperationalError as error:
            _translate_operational_error(error)

    def list_catalogue_options(
        self, search: str, *, limit: int = 20, offset: int = 0
    ) -> tuple[list[ShoppingListCatalogueOption], bool]:
        """Search a bounded local catalogue page without generated descriptions."""
        display_search, _ = normalize_shopping_list_name(search)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ShoppingListValidationError("Catalogue limit must be between 1 and 20.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ShoppingListValidationError("Offset cannot be negative.")
        pattern = f"%{_literal_like(display_search)}%"
        try:
            with get_db_connection(self.db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT w.id, w.wine_name AS name, w.vintage, p.name AS producer, w.bottle_size
                    FROM wines w LEFT JOIN producers p ON p.id = w.producer_id
                    WHERE w.wine_name LIKE ? ESCAPE '\\' OR COALESCE(p.name, '') LIKE ? ESCAPE '\\'
                    ORDER BY w.wine_name COLLATE NOCASE, w.vintage DESC, w.id
                    LIMIT ? OFFSET ?
                    """,
                    (pattern, pattern, limit + 1, offset),
                ).fetchall()
                return [ShoppingListCatalogueOption(**dict(row)) for row in rows[:limit]], len(rows) > limit
        except sqlite3.OperationalError as error:
            _translate_operational_error(error)

    def create_catalogue(
        self,
        wine_id: int,
        *,
        quantity: int = 1,
        priority: ShoppingListPriority | str = ShoppingListPriority.NORMAL,
        notes: str | None = None,
        retailer_url: str | None = None,
    ) -> ShoppingListItem:
        """Create a Wanted item from an authoritative catalogue snapshot."""
        if isinstance(wine_id, bool) or not isinstance(wine_id, int) or wine_id <= 0:
            raise ShoppingListValidationError("Catalogue wine ID must be positive.")
        values = (
            _validate_quantity(quantity),
            _validate_priority(priority),
            _clean_notes(notes),
            _clean_retailer_url(retailer_url),
        )
        with get_db_connection(self.db_path) as connection:
            try:
                self._begin_write(connection)
                cursor = connection.cursor()
                wine = cursor.execute("SELECT id, wine_name, vintage FROM wines WHERE id = ?", (wine_id,)).fetchone()
                if wine is None:
                    raise ShoppingListNotFoundError("Selected catalogue wine was not found.")
                display_name, _ = normalize_shopping_list_name(wine["wine_name"])
                vintage = _validate_vintage(wine["vintage"])
                cursor.execute(
                    "INSERT INTO shopping_list_items (wine_id, identity_kind, catalogue_identity_id, name, vintage, "
                    "quantity, priority, notes, retailer_url) VALUES (?, 'catalogue', ?, ?, ?, ?, ?, ?, ?)",
                    (wine_id, wine_id, display_name, vintage, values[0], values[1].value, values[2], values[3]),
                )
                row = self._select_item(cursor, int(cursor.lastrowid))
                connection.commit()
                return self._to_item(row)
            except sqlite3.IntegrityError as error:
                existing_id = self._existing_identity_id(connection.cursor(), catalogue_identity_id=wine_id)
                connection.rollback()
                raise ShoppingListDuplicateError(existing_id) from error
            except sqlite3.OperationalError as error:
                connection.rollback()
                _translate_operational_error(error)
            except Exception:
                connection.rollback()
                raise

    def create_manual(
        self,
        name: str,
        *,
        vintage: int | None = None,
        quantity: int = 1,
        priority: ShoppingListPriority | str = ShoppingListPriority.NORMAL,
        notes: str | None = None,
        retailer_url: str | None = None,
    ) -> ShoppingListItem:
        """Create a Wanted item from an explicit manual identity."""
        display_name, normalized_name = normalize_shopping_list_name(name)
        resolved_vintage = _validate_vintage(vintage)
        values = (
            _validate_quantity(quantity),
            _validate_priority(priority),
            _clean_notes(notes),
            _clean_retailer_url(retailer_url),
        )
        with get_db_connection(self.db_path) as connection:
            try:
                self._begin_write(connection)
                cursor = connection.cursor()
                cursor.execute(
                    "INSERT INTO shopping_list_items (identity_kind, name, normalized_name, vintage, quantity, "
                    "priority, notes, retailer_url) VALUES ('manual', ?, ?, ?, ?, ?, ?, ?)",
                    (display_name, normalized_name, resolved_vintage, values[0], values[1].value, values[2], values[3]),
                )
                row = self._select_item(cursor, int(cursor.lastrowid))
                connection.commit()
                return self._to_item(row)
            except sqlite3.IntegrityError as error:
                existing_id = self._existing_identity_id(
                    connection.cursor(), normalized_name=normalized_name, vintage=resolved_vintage
                )
                connection.rollback()
                raise ShoppingListDuplicateError(existing_id) from error
            except sqlite3.OperationalError as error:
                connection.rollback()
                _translate_operational_error(error)
            except Exception:
                connection.rollback()
                raise

    def update(self, item_id: int, expected_version: int, changes: dict[str, object]) -> ShoppingListItem:
        """Apply one validated optimistic update, including purchase or restore."""
        if not changes:
            raise ShoppingListValidationError("At least one mutable field is required.")
        if set(changes) - _MUTABLE_FIELDS:
            raise ShoppingListValidationError("Update contains immutable or unknown fields.")
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version <= 0:
            raise ShoppingListValidationError("Expected version must be positive.")
        with get_db_connection(self.db_path) as connection:
            normalized_name: str | None = None
            vintage: int | None = None
            catalogue_identity_id: int | None = None
            try:
                self._begin_write(connection)
                cursor = connection.cursor()
                current = cursor.execute("SELECT * FROM shopping_list_items WHERE id = ?", (item_id,)).fetchone()
                if current is None:
                    raise ShoppingListNotFoundError("Shopping List item was not found.")
                if current["version"] != expected_version:
                    raise ShoppingListVersionConflictError("Shopping List item version changed.")
                if current["identity_kind"] == "catalogue" and ({"name", "vintage"} & set(changes)):
                    raise ShoppingListValidationError("Catalogue identity is immutable; delete and recreate the item.")
                assignments: list[str] = []
                parameters: list[object] = []
                normalized_name, vintage = current["normalized_name"], current["vintage"]
                catalogue_identity_id = current["catalogue_identity_id"]
                if "name" in changes:
                    display_name, normalized_name = normalize_shopping_list_name(changes["name"])
                    assignments.extend(["name = ?", "normalized_name = ?"])
                    parameters.extend([display_name, normalized_name])
                if "vintage" in changes:
                    vintage = _validate_vintage(changes["vintage"])
                    assignments.append("vintage = ?")
                    parameters.append(vintage)
                validators = {
                    "quantity": _validate_quantity,
                    "priority": lambda value: _validate_priority(value).value,
                    "notes": _clean_notes,
                    "retailer_url": _clean_retailer_url,
                }
                for field, validator in validators.items():
                    if field in changes:
                        assignments.append(f"{field} = ?")
                        parameters.append(validator(changes[field]))
                if "status" in changes:
                    status = _validate_status(changes["status"])
                    purchased_at = (
                        None
                        if status == ShoppingListStatus.WANTED
                        else current["purchased_at"] or _current_timestamp(cursor)
                    )
                    assignments.extend(["status = ?", "purchased_at = ?"])
                    parameters.extend([status.value, purchased_at])
                assignments.extend(["version = version + 1", "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"])
                parameters.extend([item_id, expected_version])
                cursor.execute(
                    f"UPDATE shopping_list_items SET {', '.join(assignments)} WHERE id = ? AND version = ?", parameters
                )
                if cursor.rowcount != 1:
                    raise ShoppingListVersionConflictError("Shopping List item version changed.")
                row = self._select_item(cursor, item_id)
                connection.commit()
                return self._to_item(row)
            except sqlite3.IntegrityError as error:
                existing_id = self._existing_identity_id(
                    connection.cursor(),
                    catalogue_identity_id=catalogue_identity_id,
                    normalized_name=normalized_name,
                    vintage=vintage,
                )
                connection.rollback()
                raise ShoppingListDuplicateError(existing_id) from error
            except sqlite3.OperationalError as error:
                connection.rollback()
                _translate_operational_error(error)
            except Exception:
                connection.rollback()
                raise

    def delete(self, item_id: int, expected_version: int) -> tuple[int, int]:
        """Delete one item using optimistic concurrency."""
        with get_db_connection(self.db_path) as connection:
            try:
                self._begin_write(connection)
                current = connection.execute(
                    "SELECT version FROM shopping_list_items WHERE id = ?", (item_id,)
                ).fetchone()
                if current is None:
                    raise ShoppingListNotFoundError("Shopping List item was not found.")
                if current["version"] != expected_version:
                    raise ShoppingListVersionConflictError("Shopping List item version changed.")
                cursor = connection.execute(
                    "DELETE FROM shopping_list_items WHERE id = ? AND version = ?", (item_id, expected_version)
                )
                if cursor.rowcount != 1:
                    raise ShoppingListVersionConflictError("Shopping List item version changed.")
                connection.commit()
                return item_id, expected_version
            except sqlite3.OperationalError as error:
                connection.rollback()
                _translate_operational_error(error)
            except Exception:
                connection.rollback()
                raise


def _current_timestamp(cursor: sqlite3.Cursor) -> str:
    return str(cursor.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0])
