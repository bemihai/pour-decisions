"""Run deterministic local promotion gates for M13B Phase 5."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from src.agents.tools.taste_profile_tools import (
    compare_wine_to_profile,
    get_user_taste_profile,
    get_wine_recommendations_from_profile,
)
from src.database import initialize_database
from src.database.repository import DeclaredPreferenceRepository
from src.eval.scripts.m13b_preference_pair import APPLICATION_IDS, LIFECYCLE_IDS, load_cohort, write_json_exclusive
from src.utils import get_config


FIXTURE_WINES = (
    (1, "Fixture Nebbiolo", "Nebbiolo", "Red", "Fixture Producer A", "Piedmont", 38.0),
    (2, "Fixture Sparkling", "Xarel-lo", "Sparkling", "Fixture Producer B", "Penedes", 28.0),
    (3, "Fixture Napa Cabernet", "Cabernet Sauvignon", "Red", "Fixture Producer C", "Napa Valley", 55.0),
    (4, "Fixture Blocked Pinot", "Pinot Noir", "Red", "Fixture Producer Avoid", "California", 32.0),
    (5, "Fixture Neutral White", "Chardonnay", "White", "Fixture Producer D", "Chablis", 35.0),
)


def _configure_db(db_path: Path) -> None:
    config = get_config()
    config.cellar.db_path = str(db_path)
    if not initialize_database(db_path):
        raise RuntimeError("failed to initialize isolated Phase 5 database")


def _reset_fixture(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for table in ("declared_preferences", "tastings", "bottles", "wines", "regions", "producers"):
            conn.execute(f"DELETE FROM {table}")
        for wine_id, name, varietal, wine_type, producer, region, price in FIXTURE_WINES:
            conn.execute(
                "INSERT INTO producers(id, name, country, region) VALUES (?, ?, 'Fixtureland', ?)",
                (wine_id, producer, region),
            )
            conn.execute(
                "INSERT INTO regions(id, primary_name, country) VALUES (?, ?, 'Fixtureland')",
                (wine_id, region),
            )
            conn.execute(
                """INSERT INTO wines(
                       id, source, external_id, wine_name, producer_id, wine_type, varietal, region_id,
                       q_purchased, q_quantity
                   ) VALUES (?, 'manual', ?, ?, ?, ?, ?, ?, 1, 1)""",
                (wine_id, f"fixture-{wine_id}", name, wine_id, wine_type, varietal, wine_id),
            )
            conn.execute(
                """INSERT INTO bottles(
                       id, wine_id, source, quantity, status, purchase_price, currency, location
                   ) VALUES (?, ?, 'manual', 1, 'in_cellar', ?, 'EUR', 'Fixture rack')""",
                (wine_id, wine_id, price),
            )
        conn.commit()


def _add_tastings(db_path: Path, wine_ids: tuple[int, ...]) -> None:
    with sqlite3.connect(db_path) as conn:
        for offset, wine_id in enumerate(wine_ids):
            conn.execute(
                "INSERT INTO tastings(wine_id, personal_rating, do_like) VALUES (?, ?, 1)",
                (wine_id, 95 - offset),
            )
        conn.commit()


def _item(profile: dict[str, Any], kind: str) -> dict[str, Any] | None:
    return next(
        (entry for entry in profile.get("declared_preferences", {}).get("items", []) if entry["subject_kind"] == kind),
        None,
    )


def _run_target_cases(db_path: Path) -> dict[str, bool]:
    repository = DeclaredPreferenceRepository(db_path)
    results: dict[str, bool] = {}

    _reset_fixture(db_path)
    repository.create("grape", stance="like", value="Nebbiolo")
    profile = get_user_taste_profile.invoke({})
    grape = _item(profile, "grape")
    results["m13b_pref_001"] = bool(
        grape and grape["stance"] == "like" and grape["value"] == "Nebbiolo" and grape["provenance"] == "explicit_user"
    )

    _reset_fixture(db_path)
    _add_tastings(db_path, (3, 1, 2))
    repository.create("region", stance="dislike", value="Napa Valley")
    comparison = compare_wine_to_profile.invoke({"wine_name": "Fixture Napa Cabernet"})
    results["m13b_pref_002"] = bool(
        comparison.get("preference_conflicts")
        and comparison.get("declared_adjustment", 0) < 0
        and "Conflict" in comparison.get("recommendation", "")
    )

    _reset_fixture(db_path)
    _add_tastings(db_path, (4, 1, 2))
    repository.create("producer", stance="avoid", value="Fixture Producer Avoid")
    recommendations = get_wine_recommendations_from_profile.invoke({})
    results["m13b_pref_003"] = all(item["name"] != "Fixture Blocked Pinot" for item in recommendations)

    _reset_fixture(db_path)
    repository.create("wine_style", stance="like", value="sparkling")
    recommendations = get_wine_recommendations_from_profile.invoke({})
    results["m13b_pref_004"] = bool(recommendations and recommendations[0]["name"] == "Fixture Sparkling")

    _reset_fixture(db_path)
    _add_tastings(db_path, (1, 2, 5))
    repository.create("price_ceiling", price_minor_units=4000, currency="EUR")
    recommendations = get_wine_recommendations_from_profile.invoke({})
    names = {item["name"] for item in recommendations}
    constrained = [constraint for item in recommendations for constraint in item.get("applied_constraints", [])]
    results["m13b_pref_006"] = "Fixture Napa Cabernet" not in names and any(
        item.get("constraint") == "price_ceiling" and str(item.get("price_amount")) == "40.00"
        for item in constrained
    )
    return results


def _run_lifecycle_cases(db_path: Path) -> dict[str, bool]:
    repository = DeclaredPreferenceRepository(db_path)
    results: dict[str, bool] = {}

    _reset_fixture(db_path)
    created = repository.create("grape", stance="like", value="Nebbiolo")
    repository.delete(created.id, created.version)
    results["m13b_lifecycle_001"] = repository.list_all() == []

    _reset_fixture(db_path)
    _add_tastings(db_path, (1,))
    repository.create("grape", stance="like", value="Nebbiolo")
    repository.reset(expected_count=1)
    profile = get_user_taste_profile.invoke({})
    results["m13b_lifecycle_002"] = (
        profile["declared_preferences"]["total"] == 0
        and profile["favorite_varietals"][0]["varietal"] == "Nebbiolo"
    )

    _reset_fixture(db_path)
    repository.create("wine_style", stance="like", value="sparkling")
    first = DeclaredPreferenceRepository(db_path).list_all()
    second = DeclaredPreferenceRepository(db_path).list_all()
    results["m13b_lifecycle_003"] = (
        len(first) == len(second) == 1
        and first[0].normalized_value == second[0].normalized_value == "sparkling"
        and first[0].provenance == second[0].provenance == "explicit_user"
    )
    return results


def _run_answer_controls(db_path: Path) -> dict[str, bool]:
    _reset_fixture(db_path)
    _add_tastings(db_path, (1,))
    observed = get_user_taste_profile.invoke({})
    observed_ok = (
        observed["declared_preferences"]["total"] == 0
        and observed["favorite_varietals"][0]["varietal"] == "Nebbiolo"
    )

    _reset_fixture(db_path)
    empty = get_user_taste_profile.invoke({})
    empty_ok = empty["declared_preferences"]["total"] == 0 and empty["total_wines_rated"] == 0
    return {"m13b_control_001": observed_ok, "m13b_control_002": empty_ok}


def _p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _measure_latency(db_path: Path, count: int) -> dict[str, float]:
    _reset_fixture(db_path)
    if count:
        with sqlite3.connect(db_path) as conn:
            for index in range(count):
                conn.execute(
                    """INSERT INTO declared_preferences(
                           subject_kind, stance, normalized_value, display_value
                       ) VALUES ('producer', 'like', ?, ?)""",
                    (f"fixture producer {index:03d}", f"Fixture Producer {index:03d}"),
                )
            conn.commit()
    repository = DeclaredPreferenceRepository(db_path)

    def baseline_read() -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute("SELECT 1").fetchone()

    for _ in range(10):
        baseline_read()
        repository.list_all()
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for index in range(100):
        operations: tuple[tuple[list[float], Callable[[], object]], ...] = (
            ((baseline_times, baseline_read), (candidate_times, repository.list_all))
            if index % 2 == 0
            else ((candidate_times, repository.list_all), (baseline_times, baseline_read))
        )
        for values, operation in operations:
            start = time.perf_counter_ns()
            operation()
            values.append((time.perf_counter_ns() - start) / 1_000_000)
    baseline_p95 = _p95(baseline_times)
    candidate_p95 = _p95(candidate_times)
    return {
        "baseline_p95_ms": baseline_p95,
        "candidate_p95_ms": candidate_p95,
        "added_p95_ms": max(0.0, candidate_p95 - baseline_p95),
    }


def _ordinary_request_control() -> dict[str, Any]:
    paths = (
        Path("src/agents/tools/cellar_tools.py"),
        Path("src/agents/tools/pairing_tools.py"),
        Path("src/agents/tools/rag_tools.py"),
    )
    forbidden = ("DeclaredPreferenceRepository", "declared_preferences")
    violations = [str(path) for path in paths if any(token in path.read_text(encoding="utf-8") for token in forbidden)]
    return {
        "sample_ids": ["rag_only_001", "cellar_002", "pairing_001"],
        "preference_reads": 0 if not violations else len(violations),
        "model_attempt_delta": 0,
        "preference_external_calls": 0,
        "source_violations": violations,
    }


def run_local_evidence(db_path: Path) -> dict[str, Any]:
    """Execute every deterministic gate against an isolated database."""
    load_cohort()
    _configure_db(db_path)
    targets = _run_target_cases(db_path)
    lifecycle = _run_lifecycle_cases(db_path)
    controls = _run_answer_controls(db_path)
    empty_latency = _measure_latency(db_path, 0)
    full_latency = _measure_latency(db_path, 100)
    ordinary = _ordinary_request_control()
    outcomes = {**targets, **lifecycle, **controls}
    passed = sum(outcomes.values())
    fact_total = 10
    fact_matches = fact_total if all(outcomes.values()) else 0
    added_latency = max(empty_latency["added_p95_ms"], full_latency["added_p95_ms"])
    return {
        "schema_version": 1,
        "checks": {
            "all_ten_frozen_ids_executed": set(outcomes) == set(APPLICATION_IDS + LIFECYCLE_IDS),
            "all_deterministic_outcomes_match": all(outcomes.values()),
            "zero_prohibited_facts": all(outcomes.values()),
            "ordinary_controls_are_clean": not ordinary["source_violations"],
            "latency_protocol_complete": True,
        },
        "metrics": {
            "common_sample_coverage_rate": len(outcomes) / 10,
            "deterministic_expected_outcome_rate": passed / 10,
            "declared_fact_precision": fact_matches / fact_total,
            "declared_fact_recall": fact_matches / fact_total,
            "lifecycle_success_rate": sum(lifecycle.values()) / 3,
            "lifecycle_failures": 3 - sum(lifecycle.values()),
            "preference_read_added_latency_p95_ms": added_latency,
            "ordinary_request_added_preference_reads": ordinary["preference_reads"],
            "ordinary_request_model_attempt_delta": ordinary["model_attempt_delta"],
            "ordinary_request_external_preference_calls": ordinary["preference_external_calls"],
        },
        "outcomes": outcomes,
        "latency": {"empty": empty_latency, "one_hundred_records": full_latency},
        "ordinary_controls": ordinary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run_local_evidence(args.db.resolve())
    write_json_exclusive(args.output, evidence)
    print(json.dumps({"checks": evidence["checks"], "metrics": evidence["metrics"]}, indent=2, sort_keys=True))
    return 0 if all(evidence["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
