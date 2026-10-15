"""Focused repository and agent-tool tests for consumed tasting history."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.agents.provenance import build_tool_contract_provenance
from src.agents.tools import taste_profile_tools
from src.agents.tools.registry import ToolSelectionSnapshot
from src.database import get_db_connection, initialize_database
from src.database.repository import StatsRepository


@pytest.fixture()
def tasting_history_db(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create representative rated and unrated consumed history."""
    db_path = temp_dir / "tasting-history.db"
    assert initialize_database(db_path)

    with get_db_connection(db_path) as conn:
        deau_mare = conn.execute(
            "INSERT INTO regions (primary_name, secondary_name, country) VALUES (?, ?, ?)",
            ("Muntenia", "Dealu Mare", "Romania"),
        ).lastrowid
        burgundy = conn.execute(
            "INSERT INTO regions (primary_name, country) VALUES (?, ?)",
            ("Burgundy", "France"),
        ).lastrowid
        prince_matei = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Domeniile Prince Matei", "Romania"),
        ).lastrowid
        test_producer = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Test Producer", "Romania"),
        ).lastrowid
        french_producer = conn.execute(
            "INSERT INTO producers (name, country) VALUES (?, ?)",
            ("Domaine Test", "France"),
        ).lastrowid

        wines = (
            (1, "Rezerva Villa Zorilor", prince_matei, 2016, "Red", "Merlot", deau_mare),
            (2, "Unrated Dealu Mare Merlot", test_producer, 2019, "Red", "Merlot", deau_mare),
            (3, "Recent Burgundy", french_producer, 2020, "Red", "Pinot Noir", burgundy),
            (4, "Older Burgundy", french_producer, 2018, "Red", "Pinot Noir", burgundy),
        )
        for wine in wines:
            conn.execute(
                """
                INSERT INTO wines (
                    id, source, wine_name, producer_id, vintage, wine_type, varietal, region_id
                ) VALUES (?, 'manual', ?, ?, ?, ?, ?, ?)
                """,
                wine,
            )

        bottles = (
            (1, "2026-05-24"),
            (2, "2026-05-24"),
            (3, "2026-06-10"),
            (4, "2026-01-15"),
        )
        for wine_id, consumed_date in bottles:
            conn.execute(
                """
                INSERT INTO bottles (wine_id, source, status, consumed_date)
                VALUES (?, 'manual', 'consumed', ?)
                """,
                (wine_id, consumed_date),
            )

        conn.execute(
            """
            INSERT INTO tastings (
                wine_id, personal_rating, community_rating, tasting_notes,
                do_like, is_defective, last_tasted_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (1, 89, 91.2, "Black cherry, plum, cedar, and a long savory finish.", 1, 0, "2026-05-24"),
        )
        conn.execute(
            """
            INSERT INTO tastings (wine_id, personal_rating, last_tasted_date)
            VALUES (?, ?, ?)
            """,
            (3, 95, "2026-06-10"),
        )
        conn.execute(
            """
            INSERT INTO tastings (wine_id, personal_rating, last_tasted_date)
            VALUES (?, ?, ?)
            """,
            (4, 84, "2026-01-15"),
        )
        conn.commit()

    monkeypatch.setattr(taste_profile_tools, "get_default_db_path", lambda: db_path)
    return db_path


def test_repository_combines_region_varietal_and_inclusive_date_filters(
    tasting_history_db: Path,
) -> None:
    """The UI repository path should support the agent's combined history filters."""
    result = StatsRepository(tasting_history_db).get_consumed_wines(
        region="Dealu Mare",
        varietal="merlot",
        consumed_from="2026-05-24",
        consumed_to="2026-05-24",
        limit=100,
    )

    assert result["total"] == 2
    assert [item["wine_id"] for item in result["items"]] == [2, 1]
    assert result["items"][0]["personal_rating"] is None
    assert result["items"][1]["personal_rating"] == 89


def test_tasting_history_returns_reported_prince_matei_record(tasting_history_db: Path) -> None:
    """The reported below-90 tasting should be discoverable with exact stored data."""
    result = taste_profile_tools.get_tasting_history.invoke({"search": "Prince Matei"})

    assert result["total"] == 1
    assert result["items"] == [
        {
            "wine_id": 1,
            "bottle_id": 1,
            "name": "Rezerva Villa Zorilor",
            "producer": "Domeniile Prince Matei",
            "vintage": 2016,
            "wine_type": "Red",
            "varietal": "Merlot",
            "country": "Romania",
            "region": "Muntenia - Dealu Mare",
            "personal_rating": 89,
            "community_rating": 91.2,
            "tasting_notes": "Black cherry, plum, cedar, and a long savory finish.",
            "do_like": True,
            "is_defective": False,
            "tasting_date": "2026-05-24",
            "consumed_date": "2026-05-24",
        }
    ]


def test_tasting_history_keeps_unrated_rows_and_stable_newest_first_order(
    tasting_history_db: Path,
) -> None:
    """History should not silently impose a rating threshold or unstable tie order."""
    result = taste_profile_tools.get_tasting_history.invoke({})

    assert [item["wine_id"] for item in result["items"]] == [3, 2, 1, 4]
    unrated = result["items"][1]
    assert unrated["personal_rating"] is None
    assert unrated["community_rating"] is None
    assert unrated["tasting_notes"] is None
    assert unrated["do_like"] is None
    assert unrated["is_defective"] is None
    assert unrated["tasting_date"] is None


def test_tasting_history_empty_match_and_argument_validation(tasting_history_db: Path) -> None:
    """Empty results stay explicit and ISO date or bounded-range errors fail early."""
    assert taste_profile_tools.get_tasting_history.invoke({"search": "not present"}) == {
        "items": [],
        "total": 0,
    }

    with pytest.raises(ValidationError):
        taste_profile_tools.get_tasting_history.invoke({"consumed_from": "24-05-2026"})
    with pytest.raises(ValueError, match="consumed_from"):
        taste_profile_tools.get_tasting_history.invoke({"consumed_from": "2026-05-25", "consumed_to": "2026-05-24"})
    with pytest.raises(ValueError, match="limit"):
        taste_profile_tools.get_tasting_history.invoke({"limit": 101})


def test_tasting_history_contract_is_registered_and_hashed() -> None:
    """Catalogue metadata and provenance should cover the new local tool contract."""
    definition = next(
        item for item in taste_profile_tools.TOOL_DEFINITIONS if item.metadata.name == "get_tasting_history"
    )
    provenance = build_tool_contract_provenance(ToolSelectionSnapshot(definitions=(definition,), readiness=()))
    schema = definition.tool.tool_call_schema.model_json_schema()

    assert provenance.selected_names == ("get_tasting_history",)
    assert provenance.contract_hash.startswith("sha256:")
    assert {"region", "varietal", "consumed_from", "consumed_to", "limit"} <= set(schema["properties"])
    assert schema["properties"]["limit"]["default"] == 20
    assert schema["properties"]["limit"]["maximum"] == 100
    assert definition.metadata.cost_class.value == "free"
    assert definition.metadata.latency_class.value == "fast"
    assert definition.metadata.idempotent is True


def test_tasting_history_uses_default_limit_and_local_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool should call only the shared local repository with the default limit."""
    repository = MagicMock()
    repository.get_consumed_wines.return_value = {"items": [], "total": 0}
    repository_factory = MagicMock(return_value=repository)
    monkeypatch.setattr(taste_profile_tools, "StatsRepository", repository_factory)
    monkeypatch.setattr(taste_profile_tools, "get_default_db_path", lambda: "local.db")

    assert taste_profile_tools.get_tasting_history.invoke({}) == {"items": [], "total": 0}
    repository_factory.assert_called_once_with("local.db")
    repository.get_consumed_wines.assert_called_once_with(
        search=None,
        producer=None,
        region=None,
        varietal=None,
        wine_type=None,
        min_vintage=None,
        max_vintage=None,
        consumed_from=None,
        consumed_to=None,
        sort_by="consumed_date_desc",
        limit=20,
    )
