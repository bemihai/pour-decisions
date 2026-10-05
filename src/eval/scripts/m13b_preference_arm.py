"""Capture one fail-closed M13B Phase 5 cloud evaluation arm."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.agents.llm import load_base_model, validate_cloud_model_config
from src.database import initialize_database
from src.eval.models import GoldenSample
from src.eval.runner import EvalRunner
from src.eval.utils import resolve_execution_model_config, run_agent_sample_sync
from src.utils import get_config


BASELINE_COMMIT = "d53ac92a602eb236e7395d972a201fd09a07cac1"
APPLICATION_IDS = (
    "m13b_pref_001",
    "m13b_pref_002",
    "m13b_pref_003",
    "m13b_pref_004",
    "m13b_pref_006",
    "m13b_control_001",
    "m13b_control_002",
)
TARGET_IDS = APPLICATION_IDS[:5]
REPETITIONS = 2
MAX_CALLS_PER_EXECUTION = 5
MAX_ARM_ATTEMPTS = len(APPLICATION_IDS) * REPETITIONS * MAX_CALLS_PER_EXECUTION
EXPECTED_MODEL = "gemma4:31b"
EXPECTED_BASE_URL = "https://ollama.com"

FIXTURE_WINES = (
    (1, "Fixture Nebbiolo", "Nebbiolo", "Red", "Fixture Producer A", "Piedmont", 38.0),
    (2, "Fixture Sparkling", "Xarel-lo", "Sparkling", "Fixture Producer B", "Penedes", 28.0),
    (3, "Fixture Napa Cabernet", "Cabernet Sauvignon", "Red", "Fixture Producer C", "Napa Valley", 55.0),
    (4, "Fixture Blocked Pinot", "Pinot Noir", "Red", "Fixture Producer Avoid", "California", 32.0),
    (5, "Fixture Neutral White", "Chardonnay", "White", "Fixture Producer D", "Chablis", 35.0),
)


class UsageCallback(BaseCallbackHandler):
    """Reserve provider attempts before calls and collect reported usage."""

    raise_error = True
    run_inline = True

    def __init__(self) -> None:
        self.attempts = 0
        self.completions = 0
        self.errors = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.token_reports = 0
        self._lock = Lock()

    def _reserve(self) -> None:
        with self._lock:
            if self.attempts >= MAX_ARM_ATTEMPTS:
                raise RuntimeError(f"provider attempt ceiling reached ({MAX_ARM_ATTEMPTS})")
            self.attempts += 1

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        self._reserve()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._reserve()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = _extract_usage(response)
        with self._lock:
            self.completions += 1
            if usage is not None:
                self.token_reports += 1
                self.input_tokens += usage["input_tokens"]
                self.output_tokens += usage["output_tokens"]

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "attempts": self.attempts,
                "completions": self.completions,
                "errors": self.errors,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "token_reports": self.token_reports,
            }


def _extract_usage(response: LLMResult) -> dict[str, int] | None:
    input_tokens = 0
    output_tokens = 0
    found = False
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message is not None else None
            if isinstance(usage, dict):
                input_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                output_tokens += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
                found = True
    if found:
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}
    raw = (response.llm_output or {}).get("token_usage") or (response.llm_output or {}).get("usage")
    if not isinstance(raw, dict):
        return None
    return {
        "input_tokens": int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0),
    }


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], Any, str, str, dict[str, Any]]:
    if args.output.exists():
        raise FileExistsError(args.output)
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    samples = [sample for sample in cohort.get("samples", []) if sample.get("sample_id") in APPLICATION_IDS]
    if (
        cohort.get("status") != "frozen_and_approved"
        or len(samples) != len(APPLICATION_IDS)
        or {sample["sample_id"] for sample in samples} != set(APPLICATION_IDS)
    ):
        raise ValueError("arm requires the exact approved seven application IDs")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != args.source_commit:
        raise ValueError("source worktree does not match --source-commit")
    if args.arm == "baseline" and commit != BASELINE_COMMIT:
        raise ValueError("baseline source is not the frozen released commit")

    config = get_config()
    provider, model_name, kwargs = resolve_execution_model_config(config)
    base_url = str(kwargs.get("base_url", ""))
    timeout = float(kwargs.get("timeout", 0))
    if provider != "ollama" or model_name != EXPECTED_MODEL or base_url != EXPECTED_BASE_URL:
        raise ValueError("Phase 5 model fingerprint differs from the approved contract")
    call_budget = config.agents.guardrails.call_budget
    if not bool(call_budget.enabled) or int(call_budget.max_llm_calls_per_query) != MAX_CALLS_PER_EXECUTION:
        raise ValueError("Phase 5 requires the enabled five-call application budget")
    validate_cloud_model_config(provider, model_name, base_url, timeout)
    return cohort, config, provider, model_name, kwargs


def _configure_isolated_paths(config: Any, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=False)
    db_path = (work_dir / "cellar.db").resolve()
    config.cellar.db_path = str(db_path)
    if hasattr(config, "session_memory"):
        config.session_memory.db_path = str((work_dir / "memory.db").resolve())
    if hasattr(config, "web_search") and hasattr(config.web_search, "cache"):
        config.web_search.cache.db_path = str((work_dir / "web-cache.db").resolve())
    if hasattr(config, "web_search") and hasattr(config.web_search, "auto_fallback"):
        config.web_search.auto_fallback = False
    if not initialize_database(db_path):
        raise RuntimeError("failed to initialize isolated arm database")
    return db_path


def _reset_fixture(db_path: Path, sample_id: str, candidate: bool) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        has_preferences = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='declared_preferences'"
        ).fetchone() is not None
        tables = ["tastings", "bottles", "wines", "regions", "producers"]
        if has_preferences:
            tables.insert(0, "declared_preferences")
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        for wine_id, name, varietal, wine_type, producer, region, price in FIXTURE_WINES:
            conn.execute(
                "INSERT INTO producers(id, name, country, region) VALUES (?, ?, 'Fixtureland', ?)",
                (wine_id, producer, region),
            )
            conn.execute(
                "INSERT INTO regions(id, primary_name, country) VALUES (?, ?, 'Fixtureland')", (wine_id, region)
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
        tasting_ids = {
            "m13b_pref_002": (3, 1, 2),
            "m13b_pref_003": (4, 1, 2),
            "m13b_pref_006": (1, 2, 5),
            "m13b_control_001": (1,),
        }.get(sample_id, ())
        for offset, wine_id in enumerate(tasting_ids):
            conn.execute(
                "INSERT INTO tastings(wine_id, personal_rating, do_like) VALUES (?, ?, 1)",
                (wine_id, 95 - offset),
            )
        preference = {
            "m13b_pref_001": ("grape", "like", "nebbiolo", "Nebbiolo", None, None),
            "m13b_pref_002": ("region", "dislike", "napa valley", "Napa Valley", None, None),
            "m13b_pref_003": (
                "producer", "avoid", "fixture producer avoid", "Fixture Producer Avoid", None, None
            ),
            "m13b_pref_004": ("wine_style", "like", "sparkling", "Sparkling", None, None),
            "m13b_pref_006": ("price_ceiling", None, "price_ceiling", None, 4000, "EUR"),
        }.get(sample_id)
        if candidate and preference is not None:
            if not has_preferences:
                raise RuntimeError("candidate source lacks declared-preference schema")
            conn.execute(
                """INSERT INTO declared_preferences(
                       subject_kind, stance, normalized_value, display_value, price_minor_units, currency
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                preference,
            )
        conn.commit()


def _parse_content(content: str) -> Any:
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(content)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return content


def _tool_values(tool_outputs: list[dict[str, Any]], tool_name: str) -> list[Any]:
    return [
        _parse_content(str(output.get("content", "")))
        for output in tool_outputs
        if output.get("tool_name") == tool_name
    ]


def _control_signature(tool_outputs: list[dict[str, Any]]) -> Any:
    profiles = _tool_values(tool_outputs, "get_user_taste_profile")
    if not profiles or not isinstance(profiles[-1], dict):
        return None
    profile = dict(profiles[-1])
    profile.pop("declared_preferences", None)
    profile.pop("preference_conflicts", None)
    return profile


def _assess(sample_id: str, answer: str, tool_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    folded = answer.casefold()
    profiles = _tool_values(tool_outputs, "get_user_taste_profile")
    comparisons = _tool_values(tool_outputs, "compare_wine_to_profile")
    recommendations = _tool_values(tool_outputs, "get_wine_recommendations_from_profile")
    recommendation_items = next((value for value in reversed(recommendations) if isinstance(value, list)), [])
    names = {str(item.get("name", "")) for item in recommendation_items if isinstance(item, dict)}
    visible = False
    conflict_disclosed = False
    hard_avoid = 0
    prohibited = 0
    if sample_id == "m13b_pref_001":
        visible = "nebbiolo" in folded and any(word in folded for word in ("explicit", "declared", "preference", "like"))
        prohibited = int("barbera" in folded and "dislike" in folded)
    elif sample_id == "m13b_pref_002":
        conflict_disclosed = "napa" in folded and any(word in folded for word in ("conflict", "dislike", "declared"))
        visible = conflict_disclosed
        prohibited = int("avoid napa" in folded)
    elif sample_id == "m13b_pref_003":
        hard_avoid = int("Fixture Blocked Pinot" in names)
        visible = hard_avoid == 0 and "fixture producer avoid" in folded and any(
            word in folded for word in ("avoid", "excluded", "not recommend", "never")
        )
    elif sample_id == "m13b_pref_004":
        visible = bool(recommendation_items and recommendation_items[0].get("name") == "Fixture Sparkling") and (
            "fixture sparkling" in folded or "sparkling" in folded
        )
        prohibited = int("avoid still" in folded)
    elif sample_id == "m13b_pref_006":
        prohibited = int("Fixture Napa Cabernet" in names)
        visible = prohibited == 0 and any(word in folded for word in ("€40", "40 eur", "budget", "ceiling", "price"))
    if sample_id in TARGET_IDS and not (profiles or comparisons or recommendations):
        visible = False
    return {
        "visible_effect": visible,
        "hard_avoid_violations": hard_avoid,
        "prohibited_or_fabricated_facts": prohibited,
        "conflict_disclosed": conflict_disclosed,
        "control_regression": False,
        "control_signature": _control_signature(tool_outputs) if sample_id.startswith("m13b_control_") else None,
    }


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def _sample(record: dict[str, Any]) -> GoldenSample:
    return GoldenSample(
        id=record["sample_id"],
        question=record["question"],
        category="multi_hop",
        difficulty="medium",
        expected_facts=list(record.get("required_evidence", [])),
        expected_tool_calls=[record["expected_tool"]],
        ground_truth=record["expected_visible_effect"],
        tags=["m13b", record["cohort_role"]],
    )


def capture_arm(args: argparse.Namespace) -> dict[str, Any]:
    cohort, config, provider, model_name, kwargs = _preflight(args)
    db_path = _configure_isolated_paths(config, args.work_dir.resolve())
    _reset_fixture(db_path, "m13b_control_002", args.arm == "candidate")
    callback = UsageCallback()
    model = load_base_model(provider, model_name, **kwargs)
    model.callbacks = [callback]
    runner = EvalRunner(backend="agent", config=config, model=model)
    asyncio.run(runner._prepare_backend_resources())
    if runner._agent is None:
        raise RuntimeError("agent preflight did not construct an agent")
    execution_snapshot = runner.config_snapshot.get("execution", {})
    samples = {record["sample_id"]: record for record in cohort["samples"] if record["sample_id"] in APPLICATION_IDS}
    executions: list[dict[str, Any]] = []
    for sample_id in APPLICATION_IDS:
        for repetition in range(1, REPETITIONS + 1):
            _reset_fixture(db_path, sample_id, args.arm == "candidate")
            before = callback.snapshot()
            started = time.perf_counter()
            try:
                result = run_agent_sample_sync(runner._agent, _sample(samples[sample_id]))
                elapsed = (time.perf_counter() - started) * 1000
                after = callback.snapshot()
                outputs = [output.model_dump(mode="json") for output in result.tool_outputs]
                usage = result.token_usage
                status = "passed" if result.answer.strip() and usage is not None else "failed"
                error = None if status == "passed" else "missing answer or provider token usage"
                execution = {
                    "sample_id": sample_id,
                    "repetition": repetition,
                    "status": status,
                    "error": error,
                    "answer": result.answer,
                    "tool_calls": result.tool_calls,
                    "tool_call_records": [record.model_dump(mode="json") for record in result.tool_call_records],
                    "tool_outputs": outputs,
                    "latency_ms": elapsed,
                    "model_attempts": after["attempts"] - before["attempts"],
                    "input_tokens": usage["input_tokens"] if usage else None,
                    "output_tokens": usage["output_tokens"] if usage else None,
                    "failures": result.guardrail_events,
                    "external_calls": [name for name in result.tool_calls if "web" in name],
                    "preference_external_calls": [name for name in result.tool_calls if "web" in name],
                    **_assess(sample_id, result.answer, outputs),
                }
            except Exception as error:
                after = callback.snapshot()
                execution = {
                    "sample_id": sample_id,
                    "repetition": repetition,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "answer": "",
                    "tool_calls": [],
                    "tool_call_records": [],
                    "tool_outputs": [],
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "model_attempts": after["attempts"] - before["attempts"],
                    "input_tokens": None,
                    "output_tokens": None,
                    "failures": [],
                    "external_calls": [],
                    "preference_external_calls": [],
                    **_assess(sample_id, "", []),
                }
            executions.append(execution)
    usage = callback.snapshot()
    configuration = {key: value for key, value in runner.config_snapshot.items() if key != "execution"}
    return {
        "schema_version": 1,
        "arm": args.arm,
        "commit": args.source_commit,
        "model_id": model_name,
        "configuration_fingerprint": _canonical_hash(configuration),
        "prompt_fingerprint": execution_snapshot.get("prompt_bundle_hash"),
        "tool_snapshot": execution_snapshot.get("tools"),
        "provider_attempts": usage["attempts"],
        "provider_completions": usage["completions"],
        "provider_errors": usage["errors"],
        "provider_token_reports": usage["token_reports"],
        "executions": executions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arm = capture_arm(args)
    _write_exclusive(args.output, arm)
    passed = sum(execution["status"] == "passed" for execution in arm["executions"])
    print(json.dumps({"arm": args.arm, "passed": passed, "provider_attempts": arm["provider_attempts"]}, indent=2))
    return 0 if passed == len(APPLICATION_IDS) * REPETITIONS else 1


if __name__ == "__main__":
    raise SystemExit(main())
