"""Repository and normalization for explicit wine preferences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from typing import TypeVar
import unicodedata

from src.database import get_db_connection
from src.database.models import (
    DeclaredPreference,
    PreferenceCurrency,
    PreferenceStance,
    PreferenceSubjectKind,
    WineStyle,
)
from src.utils import GRAPE_SYNONYMS, REGION_VARIATIONS, get_default_db_path

MAX_DECLARED_PREFERENCES = 100
WRITE_BUSY_TIMEOUT_MS = 1_000
EnumType = TypeVar("EnumType", bound=Enum)

_STYLE_DISPLAY_VALUES: dict[WineStyle, str] = {
    WineStyle.RED: "Red",
    WineStyle.WHITE: "White",
    WineStyle.ROSE: "Rosé",
    WineStyle.SPARKLING: "Sparkling",
    WineStyle.DESSERT: "Dessert",
    WineStyle.FORTIFIED: "Fortified",
}


class PreferenceRepositoryError(Exception):
    """Base class for bounded declared-preference repository failures."""


class PreferenceValueUnresolvedError(PreferenceRepositoryError):
    """Raised when a domain value cannot be resolved deterministically."""


class PreferenceCombinationError(PreferenceRepositoryError):
    """Raised for an invalid subject/field combination."""


class PreferenceDuplicateError(PreferenceRepositoryError):
    """Raised when a preference identity already exists."""


class PreferenceLimitReachedError(PreferenceRepositoryError):
    """Raised when the single profile has reached its record limit."""


class PreferenceNotFoundError(PreferenceRepositoryError):
    """Raised when a requested preference ID does not exist."""


class PreferenceVersionConflictError(PreferenceRepositoryError):
    """Raised when an optimistic version does not match current state."""


class PreferenceCountChangedError(PreferenceRepositoryError):
    """Raised when reset confirmation uses a stale record count."""


class PreferenceStoreBusyError(PreferenceRepositoryError):
    """Raised after the bounded SQLite write-lock wait is exhausted."""


@dataclass(frozen=True)
class _NormalizedPreference:
    """Validated values ready for one repository insert."""

    subject_kind: PreferenceSubjectKind
    stance: PreferenceStance | None
    normalized_value: str
    display_value: str | None
    price_minor_units: int | None
    currency: PreferenceCurrency | None


def normalize_preference_text(value: str) -> tuple[str, str]:
    """Normalize a user-entered domain label into display and identity forms.

    Args:
        value: User-entered grape, region, or producer label.

    Returns:
        Tuple of NFC display text and NFKC/case-folded identity text.

    Raises:
        PreferenceCombinationError: If the value is empty, too long, or contains
            C0/C1 control characters.
    """
    if not isinstance(value, str):
        raise PreferenceCombinationError("Preference value must be text.")
    if any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value):
        raise PreferenceCombinationError("Preference value contains unsupported control characters.")

    collapsed = " ".join(value.strip().split())
    display_value = unicodedata.normalize("NFC", collapsed)
    if not display_value or len(display_value) > 120:
        raise PreferenceCombinationError("Preference value must contain 1 to 120 characters.")

    normalized_value = unicodedata.normalize("NFKC", display_value).casefold()
    if len(normalized_value) > 120:
        raise PreferenceCombinationError("Normalized preference value exceeds 120 characters.")
    return display_value, normalized_value


def _enum_value(enum_type: type[EnumType], value: object, message: str) -> EnumType:
    """Convert a raw value to a string enum with a bounded failure."""
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise PreferenceCombinationError(message) from error


def _terminology_lookup(values: dict[str, list[str]]) -> dict[str, str]:
    """Build an exact normalized terminology lookup."""
    lookup: dict[str, str] = {}
    for canonical, aliases in values.items():
        canonical_display, canonical_identity = normalize_preference_text(canonical)
        lookup[canonical_identity] = canonical_display.title()
        for alias in aliases:
            _, alias_identity = normalize_preference_text(alias)
            lookup[alias_identity] = canonical_display.title()
    return lookup


_GRAPE_LOOKUP = _terminology_lookup(GRAPE_SYNONYMS)
_REGION_LOOKUP = _terminology_lookup(REGION_VARIATIONS)


def _resolve_grape(cursor: sqlite3.Cursor, value: str) -> tuple[str, str]:
    """Resolve a grape through cellar values or the terminology dictionary."""
    _, requested_identity = normalize_preference_text(value)
    cursor.execute("SELECT DISTINCT varietal FROM wines WHERE TRIM(COALESCE(varietal, '')) <> ''")
    for row in cursor.fetchall():
        display_value, identity = normalize_preference_text(row[0])
        if identity == requested_identity:
            return display_value, identity

    display_value = _GRAPE_LOOKUP.get(requested_identity)
    if display_value is None:
        raise PreferenceValueUnresolvedError("Grape value could not be resolved.")
    _, identity = normalize_preference_text(display_value)
    return display_value, identity


def _resolve_region(cursor: sqlite3.Cursor, value: str) -> tuple[str, str]:
    """Resolve a region through cellar values or the terminology dictionary."""
    _, requested_identity = normalize_preference_text(value)
    cursor.execute("SELECT primary_name, secondary_name FROM regions")
    for row in cursor.fetchall():
        labels = [row[0]]
        if row[1]:
            labels.extend([row[1], f"{row[0]} - {row[1]}"])
        for label in labels:
            display_value, identity = normalize_preference_text(label)
            if identity == requested_identity:
                return display_value, identity

    display_value = _REGION_LOOKUP.get(requested_identity)
    if display_value is None:
        raise PreferenceValueUnresolvedError("Region value could not be resolved.")
    _, identity = normalize_preference_text(display_value)
    return display_value, identity


def _resolve_producer(cursor: sqlite3.Cursor, value: str) -> tuple[str, str]:
    """Resolve a producer against an existing producer row."""
    _, requested_identity = normalize_preference_text(value)
    cursor.execute("SELECT name FROM producers")
    for row in cursor.fetchall():
        display_value, identity = normalize_preference_text(row[0])
        if identity == requested_identity:
            return display_value, identity
    raise PreferenceValueUnresolvedError("Producer value could not be resolved.")


def _normalize_create(
    cursor: sqlite3.Cursor,
    subject_kind: PreferenceSubjectKind | str,
    *,
    stance: PreferenceStance | str | None = None,
    value: str | None = None,
    price_minor_units: int | None = None,
    currency: PreferenceCurrency | str | None = None,
) -> _NormalizedPreference:
    """Validate and resolve one create request inside its write transaction."""
    subject = _enum_value(PreferenceSubjectKind, subject_kind, "Unsupported preference subject.")

    if subject == PreferenceSubjectKind.PRICE_CEILING:
        if stance is not None or value is not None:
            raise PreferenceCombinationError("Price ceilings do not accept stance or value fields.")
        if isinstance(price_minor_units, bool) or not isinstance(price_minor_units, int):
            raise PreferenceCombinationError("Price ceiling must use integer minor units.")
        if not 1 <= price_minor_units <= 99_999_999:
            raise PreferenceCombinationError("Price ceiling is outside the supported range.")
        resolved_currency = _enum_value(PreferenceCurrency, currency, "Unsupported preference currency.")
        return _NormalizedPreference(
            subject_kind=subject,
            stance=None,
            normalized_value="price_ceiling",
            display_value=None,
            price_minor_units=price_minor_units,
            currency=resolved_currency,
        )

    if price_minor_units is not None or currency is not None:
        raise PreferenceCombinationError("Non-price preferences do not accept price fields.")
    resolved_stance = _enum_value(PreferenceStance, stance, "Unsupported preference stance.")
    if value is None:
        raise PreferenceCombinationError("Preference value is required.")

    if subject == PreferenceSubjectKind.GRAPE:
        display_value, normalized_value = _resolve_grape(cursor, value)
    elif subject == PreferenceSubjectKind.REGION:
        display_value, normalized_value = _resolve_region(cursor, value)
    elif subject == PreferenceSubjectKind.PRODUCER:
        display_value, normalized_value = _resolve_producer(cursor, value)
    else:
        style = _enum_value(WineStyle, value, "Unsupported wine style.")
        display_value = _STYLE_DISPLAY_VALUES[style]
        normalized_value = style.value

    return _NormalizedPreference(
        subject_kind=subject,
        stance=resolved_stance,
        normalized_value=normalized_value,
        display_value=display_value,
        price_minor_units=None,
        currency=None,
    )


def _raise_if_busy(error: sqlite3.OperationalError) -> None:
    """Translate SQLite lock exhaustion and re-raise other operational errors."""
    message = str(error).casefold()
    if "locked" in message or "busy" in message:
        raise PreferenceStoreBusyError("Preference store is busy.") from error
    raise error


class DeclaredPreferenceRepository:
    """Read and mutate the single local declared-preference profile."""

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the repository with an explicit or configured database."""
        self.db_path = db_path if db_path is not None else get_default_db_path()

    @staticmethod
    def _to_model(row: sqlite3.Row) -> DeclaredPreference:
        """Convert one SQLite row into the persisted model."""
        return DeclaredPreference(**dict(row))

    @staticmethod
    def _begin_write(conn: sqlite3.Connection) -> None:
        """Start one serialized write after applying the bounded lock wait."""
        conn.execute(f"PRAGMA busy_timeout = {WRITE_BUSY_TIMEOUT_MS}")
        conn.execute("BEGIN IMMEDIATE")

    def list_all(self) -> list[DeclaredPreference]:
        """List current preferences in stable subject/value order."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM declared_preferences
                ORDER BY subject_kind, normalized_value, id
                """
            )
            return [self._to_model(row) for row in cursor.fetchall()]

    def get_by_id(self, preference_id: int) -> DeclaredPreference | None:
        """Return one current preference or ``None``."""
        with get_db_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM declared_preferences WHERE id = ?", (preference_id,)).fetchone()
            return self._to_model(row) if row else None

    def list_options(self) -> dict[str, list[str]]:
        """Return sorted domain options accepted by preference creation."""
        grapes: dict[str, str] = {}
        regions: dict[str, str] = {}
        producers: dict[str, str] = {}

        for display_value in _GRAPE_LOOKUP.values():
            _, identity = normalize_preference_text(display_value)
            grapes[identity] = display_value
        for display_value in _REGION_LOOKUP.values():
            _, identity = normalize_preference_text(display_value)
            regions[identity] = display_value

        with get_db_connection(self.db_path) as conn:
            for row in conn.execute("SELECT DISTINCT varietal FROM wines WHERE TRIM(COALESCE(varietal, '')) <> ''"):
                display_value, identity = normalize_preference_text(row[0])
                grapes[identity] = display_value
            for row in conn.execute("SELECT primary_name, secondary_name FROM regions"):
                labels = [row[0]]
                if row[1]:
                    labels.extend([row[1], f"{row[0]} - {row[1]}"])
                for label in labels:
                    display_value, identity = normalize_preference_text(label)
                    regions[identity] = display_value
            for row in conn.execute("SELECT name FROM producers"):
                display_value, identity = normalize_preference_text(row[0])
                producers[identity] = display_value

        return {
            "grapes": sorted(grapes.values(), key=str.casefold),
            "regions": sorted(regions.values(), key=str.casefold),
            "producers": sorted(producers.values(), key=str.casefold),
        }

    def create(
        self,
        subject_kind: PreferenceSubjectKind | str,
        *,
        stance: PreferenceStance | str | None = None,
        value: str | None = None,
        price_minor_units: int | None = None,
        currency: PreferenceCurrency | str | None = None,
    ) -> DeclaredPreference:
        """Validate, resolve, and atomically create one preference."""
        with get_db_connection(self.db_path) as conn:
            try:
                self._begin_write(conn)
                cursor = conn.cursor()
                normalized = _normalize_create(
                    cursor,
                    subject_kind,
                    stance=stance,
                    value=value,
                    price_minor_units=price_minor_units,
                    currency=currency,
                )
                count = cursor.execute("SELECT COUNT(*) FROM declared_preferences").fetchone()[0]
                if count >= MAX_DECLARED_PREFERENCES:
                    raise PreferenceLimitReachedError("Declared preference limit reached.")

                cursor.execute(
                    """
                    INSERT INTO declared_preferences (
                        subject_kind, stance, normalized_value, display_value,
                        price_minor_units, currency
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized.subject_kind.value,
                        normalized.stance.value if normalized.stance else None,
                        normalized.normalized_value,
                        normalized.display_value,
                        normalized.price_minor_units,
                        normalized.currency.value if normalized.currency else None,
                    ),
                )
                row = cursor.execute(
                    "SELECT * FROM declared_preferences WHERE id = ?", (cursor.lastrowid,)
                ).fetchone()
                conn.commit()
                return self._to_model(row)
            except sqlite3.IntegrityError as error:
                conn.rollback()
                if "UNIQUE constraint failed" in str(error):
                    raise PreferenceDuplicateError("Declared preference already exists.") from error
                raise PreferenceCombinationError("Preference violates the storage contract.") from error
            except sqlite3.OperationalError as error:
                conn.rollback()
                _raise_if_busy(error)
            except Exception:
                conn.rollback()
                raise

    def update_mutable(
        self,
        preference_id: int,
        expected_version: int,
        *,
        stance: PreferenceStance | str | None = None,
        price_minor_units: int | None = None,
        currency: PreferenceCurrency | str | None = None,
    ) -> DeclaredPreference:
        """Update only stance or price fields using optimistic concurrency."""
        with get_db_connection(self.db_path) as conn:
            try:
                self._begin_write(conn)
                cursor = conn.cursor()
                current = cursor.execute(
                    "SELECT * FROM declared_preferences WHERE id = ?", (preference_id,)
                ).fetchone()
                if current is None:
                    raise PreferenceNotFoundError("Declared preference was not found.")
                if current["version"] != expected_version:
                    raise PreferenceVersionConflictError("Declared preference version changed.")

                if current["subject_kind"] == PreferenceSubjectKind.PRICE_CEILING.value:
                    if stance is not None:
                        raise PreferenceCombinationError("Price ceilings cannot update stance.")
                    if isinstance(price_minor_units, bool) or not isinstance(price_minor_units, int):
                        raise PreferenceCombinationError("Price ceiling must use integer minor units.")
                    if not 1 <= price_minor_units <= 99_999_999:
                        raise PreferenceCombinationError("Price ceiling is outside the supported range.")
                    resolved_currency = _enum_value(
                        PreferenceCurrency, currency, "Unsupported preference currency."
                    )
                    cursor.execute(
                        """
                        UPDATE declared_preferences
                        SET price_minor_units = ?, currency = ?, version = version + 1,
                            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE id = ? AND version = ?
                        """,
                        (price_minor_units, resolved_currency.value, preference_id, expected_version),
                    )
                else:
                    if price_minor_units is not None or currency is not None:
                        raise PreferenceCombinationError("Non-price preferences cannot update price fields.")
                    resolved_stance = _enum_value(PreferenceStance, stance, "Unsupported preference stance.")
                    cursor.execute(
                        """
                        UPDATE declared_preferences
                        SET stance = ?, version = version + 1,
                            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                        WHERE id = ? AND version = ?
                        """,
                        (resolved_stance.value, preference_id, expected_version),
                    )

                if cursor.rowcount != 1:
                    raise PreferenceVersionConflictError("Declared preference version changed.")
                row = cursor.execute(
                    "SELECT * FROM declared_preferences WHERE id = ?", (preference_id,)
                ).fetchone()
                conn.commit()
                return self._to_model(row)
            except sqlite3.OperationalError as error:
                conn.rollback()
                _raise_if_busy(error)
            except Exception:
                conn.rollback()
                raise

    def delete(self, preference_id: int, expected_version: int) -> tuple[int, int]:
        """Delete one preference using optimistic concurrency."""
        with get_db_connection(self.db_path) as conn:
            try:
                self._begin_write(conn)
                cursor = conn.cursor()
                current = cursor.execute(
                    "SELECT version FROM declared_preferences WHERE id = ?", (preference_id,)
                ).fetchone()
                if current is None:
                    raise PreferenceNotFoundError("Declared preference was not found.")
                if current["version"] != expected_version:
                    raise PreferenceVersionConflictError("Declared preference version changed.")
                cursor.execute(
                    "DELETE FROM declared_preferences WHERE id = ? AND version = ?",
                    (preference_id, expected_version),
                )
                if cursor.rowcount != 1:
                    raise PreferenceVersionConflictError("Declared preference version changed.")
                conn.commit()
                return preference_id, expected_version
            except sqlite3.OperationalError as error:
                conn.rollback()
                _raise_if_busy(error)
            except Exception:
                conn.rollback()
                raise

    def reset(self, expected_count: int | None = None) -> int:
        """Atomically delete every preference, optionally checking current count."""
        with get_db_connection(self.db_path) as conn:
            try:
                self._begin_write(conn)
                cursor = conn.cursor()
                current_count = cursor.execute("SELECT COUNT(*) FROM declared_preferences").fetchone()[0]
                if expected_count is not None and current_count != expected_count:
                    raise PreferenceCountChangedError("Declared preference count changed.")
                cursor.execute("DELETE FROM declared_preferences")
                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count
            except sqlite3.OperationalError as error:
                conn.rollback()
                _raise_if_busy(error)
            except Exception:
                conn.rollback()
                raise
