"""Run the authorized M12 paired generation and faithfulness gate.

The command is intentionally limited to the six frozen Gate 0 IDs and the one
cloud judge metric required for rollout: faithfulness. A shared callback counts
every execution and judge invocation and raises before the approved provider
attempt ceiling can be exceeded.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from omegaconf import DictConfig

from src.eval.dataset import load_golden_dataset
from src.eval.models import GoldenSample, SampleResult
from src.eval.ragas_scorer import RagasScorer
from src.eval.runner import EvalRunner
from src.eval.scripts.corrective_retrieval_gate import DEFAULT_COHORT_PATH, DEFAULT_DATASET_PATH, load_cohort
from src.eval.scripts.corrective_retrieval_pair import (
    FROZEN_SAMPLE_IDS,
    arm_config,
    retriever_cache_stats,
    validate_frozen_cohort,
)
from src.eval.utils import get_git_metadata, resolve_eval_model_config, resolve_execution_model_config
from src.agents.llm import load_base_model
from src.utils import get_config, logger
from src.utils.env import load_env


APPROVED_PROVIDER_ATTEMPT_CEILING = 108
GENERATION_EXECUTIONS = 12
FAITHFULNESS_JUDGE_CALLS_PER_SAMPLE_BOUND = 3
DEFAULT_OUTPUT_PATH = Path("eval-results/m12_phase4_corrective_generation_pair.json")


@dataclass
class ProviderAttemptBudget:
    """Thread-safe shared ceiling across generation and judge invocations."""

    maximum: int = APPROVED_PROVIDER_ATTEMPT_CEILING
    attempts: int = 0
    attempts_by_role: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def reserve(self, role: str) -> None:
        """Reserve one provider attempt or fail before issuing it."""
        with self._lock:
            if self.attempts >= self.maximum:
                raise RuntimeError(f"provider attempt ceiling reached ({self.maximum})")
            self.attempts += 1
            self.attempts_by_role[role] = self.attempts_by_role.get(role, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        """Return bounded counters for artifact reporting."""
        with self._lock:
            return {
                "attempts": self.attempts,
                "attempts_by_role": dict(self.attempts_by_role),
                "maximum": self.maximum,
                "remaining": self.maximum - self.attempts,
            }


class ProviderUsageCallback(BaseCallbackHandler):
    """Count model attempts, completions, failures, and reported token usage."""

    raise_error = True
    run_inline = True

    def __init__(self, role: str, budget: ProviderAttemptBudget) -> None:
        """Bind the callback to one provider role and the shared attempt budget."""
        self.role = role
        self.budget = budget
        self.completions = 0
        self.errors = 0
        self.token_usage_reports = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self._lock = Lock()

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        """Reserve one attempt immediately before a chat-model invocation."""
        self.budget.reserve(self.role)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Reserve one attempt for non-chat model adapters."""
        self.budget.reserve(self.role)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record completion and any provider-reported token usage."""
        usage = _extract_usage(response)
        with self._lock:
            self.completions += 1
            if usage is not None:
                self.token_usage_reports += 1
                self.input_tokens += usage["input_tokens"]
                self.output_tokens += usage["output_tokens"]
                self.total_tokens += usage["total_tokens"]

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Count failed provider invocations."""
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, int]:
        """Return cumulative counters without model content."""
        with self._lock:
            return {
                "completions": self.completions,
                "errors": self.errors,
                "token_usage_reports": self.token_usage_reports,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            }


def _extract_usage(response: LLMResult) -> dict[str, int] | None:
    """Extract token usage from LangChain message or provider metadata."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    found = False
    for generation_list in response.generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message is not None else None
            if not isinstance(usage, dict):
                continue
            input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
            total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["total_tokens"] += total_tokens
            found = True
    if found:
        return totals

    llm_output = response.llm_output or {}
    raw_usage = llm_output.get("token_usage") or llm_output.get("usage")
    if not isinstance(raw_usage, dict):
        return None
    input_tokens = int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(raw_usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    """Subtract cumulative callback snapshots."""
    return {key: int(after[key]) - int(before[key]) for key in after}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def prepare_arm_config(config: DictConfig, *, correction_enabled: bool) -> DictConfig:
    """Create one arm config with only faithfulness judging and no web fallback."""
    copied = arm_config(config, correction_enabled=correction_enabled)
    copied.eval.ragas.metrics = ["faithfulness"]
    copied.eval.ragas.max_retries = 1
    copied.eval.ragas.max_workers = 1
    copied.eval.max_concurrency = 1
    return copied


def validate_cloud_contract(config: DictConfig) -> dict[str, Any]:
    """Fail before external calls when the approved model or attempt contract differs."""
    execution_provider, execution_model, execution_kwargs = resolve_execution_model_config(config)
    judge_provider, judge_model, judge_kwargs = resolve_eval_model_config(config)
    if (execution_provider, execution_model) != ("ollama", "gemma4:cloud"):
        raise ValueError("M12 cloud execution requires ollama/gemma4:cloud")
    if (judge_provider, judge_model) != ("ollama", "gemma4:31b-cloud"):
        raise ValueError("M12 cloud judging requires ollama/gemma4:31b-cloud")

    judge_retry_multiplier = 2
    maximum_judge_attempts = (
        len(FROZEN_SAMPLE_IDS) * 2 * FAITHFULNESS_JUDGE_CALLS_PER_SAMPLE_BOUND * judge_retry_multiplier
    )
    estimated_maximum = GENERATION_EXECUTIONS + maximum_judge_attempts
    if estimated_maximum > APPROVED_PROVIDER_ATTEMPT_CEILING:
        raise ValueError("Configured experiment can exceed the approved provider attempt ceiling")
    return {
        "execution_provider": execution_provider,
        "execution_model": execution_model,
        "execution_kwargs": execution_kwargs,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_kwargs": judge_kwargs,
        "generation_executions": GENERATION_EXECUTIONS,
        "maximum_judge_attempts": maximum_judge_attempts,
        "estimated_retry_inclusive_maximum": estimated_maximum,
        "approved_ceiling": APPROVED_PROVIDER_ATTEMPT_CEILING,
    }


def build_counted_model(
    provider: str,
    model_name: str,
    kwargs: dict[str, Any],
    callback: ProviderUsageCallback,
) -> Any:
    """Build an approved model with the shared bounded accounting callback."""
    return load_base_model(provider, model_name, callbacks=[callback], **deepcopy(kwargs))


def _source_attribution_valid(result: SampleResult) -> bool:
    context_ids = {chunk.id for chunk in result.context_chunks}
    source_ids = {source.chunk_id for source in result.rag_sources}
    metadata_valid = all(
        chunk.metadata.get("source") or chunk.metadata.get("filename") for chunk in result.context_chunks
    )
    return metadata_valid and context_ids.issubset(source_ids)


def run_arm(
    name: str,
    config: DictConfig,
    samples: list[GoldenSample],
    execution_model: Any,
    judge_model: Any,
    execution_callback: ProviderUsageCallback,
    judge_callback: ProviderUsageCallback,
) -> dict[str, Any]:
    """Run and judge one arm while capturing role-specific counter deltas."""
    execution_before = execution_callback.snapshot()
    runner = EvalRunner(
        backend="rag",
        config=config,
        generation_enabled=True,
        model=execution_model,
    )
    results = asyncio.run(runner.run(samples=samples, max_concurrency=1))
    execution_usage = _counter_delta(execution_callback.snapshot(), execution_before)

    judge_before = judge_callback.snapshot()
    scorer = RagasScorer(llm=judge_model, config=config)
    scorer.score(results)
    judge_usage = _counter_delta(judge_callback.snapshot(), judge_before)
    cache = retriever_cache_stats(runner._retriever) if runner._retriever is not None else {"available": False}

    logger.info(
        "M12 cloud arm=%s samples=%d generation_attempts=%d judge_attempts=%d",
        name,
        len(results),
        execution_usage["completions"] + execution_usage["errors"],
        judge_usage["completions"] + judge_usage["errors"],
    )
    return {
        "name": name,
        "execution_usage": execution_usage,
        "judge_usage": judge_usage,
        "cache": cache,
        "results": [result.model_dump(mode="json") for result in results],
    }


def evaluate_cloud_gate(disabled: dict[str, Any], enabled: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Compare paired generation and faithfulness results on common scored IDs."""
    disabled_results = {result["id"]: result for result in disabled["results"]}
    enabled_results = {result["id"]: result for result in enabled["results"]}
    all_ids = sorted(FROZEN_SAMPLE_IDS)
    disabled_success = {sample_id for sample_id in all_ids if disabled_results[sample_id]["status"] == "passed"}
    enabled_success = {sample_id for sample_id in all_ids if enabled_results[sample_id]["status"] == "passed"}
    disabled_blank = {sample_id for sample_id in disabled_success if not disabled_results[sample_id]["answer"].strip()}
    enabled_blank = {sample_id for sample_id in enabled_success if not enabled_results[sample_id]["answer"].strip()}
    disabled_scored = {
        sample_id for sample_id in disabled_success if "faithfulness" in disabled_results[sample_id]["scores"]
    }
    enabled_scored = {
        sample_id for sample_id in enabled_success if "faithfulness" in enabled_results[sample_id]["scores"]
    }
    common_scored = sorted(disabled_scored & enabled_scored)
    disabled_mean = (
        sum(float(disabled_results[sample_id]["scores"]["faithfulness"]) for sample_id in common_scored)
        / len(common_scored)
        if common_scored
        else None
    )
    enabled_mean = (
        sum(float(enabled_results[sample_id]["scores"]["faithfulness"]) for sample_id in common_scored)
        / len(common_scored)
        if common_scored
        else None
    )
    faithfulness_delta = (
        enabled_mean - disabled_mean if disabled_mean is not None and enabled_mean is not None else None
    )
    source_regressions = [
        sample_id
        for sample_id in sorted(disabled_success & enabled_success)
        if _source_attribution_valid(SampleResult.model_validate(disabled_results[sample_id]))
        and not _source_attribution_valid(SampleResult.model_validate(enabled_results[sample_id]))
    ]
    enabled_correction = [enabled_results[sample_id]["correction"] for sample_id in enabled_success]
    web_calls = sum(
        int(result["rag_feature_flags"].get("web_fallback", False))
        for arm in (disabled_results, enabled_results)
        for result in arm.values()
        if result["status"] == "passed"
    )
    judge_failures = {
        "disabled": {
            sample_id: disabled_results[sample_id]["metric_errors"].get("faithfulness")
            for sample_id in all_ids
            if disabled_results[sample_id]["metric_errors"].get("faithfulness")
        },
        "enabled": {
            sample_id: enabled_results[sample_id]["metric_errors"].get("faithfulness")
            for sample_id in all_ids
            if enabled_results[sample_id]["metric_errors"].get("faithfulness")
        },
    }
    checks = {
        "all_execution_ids_succeeded": disabled_success == set(all_ids) and enabled_success == set(all_ids),
        "no_new_blank_answers": not (enabled_blank - disabled_blank),
        "zero_generation_provider_errors": (
            disabled["execution_usage"]["errors"] == 0 and enabled["execution_usage"]["errors"] == 0
        ),
        "all_ids_scored_for_faithfulness": len(common_scored) == len(all_ids),
        "faithfulness_regression_at_most_two_points": faithfulness_delta is not None and faithfulness_delta >= -0.02,
        "no_source_attribution_regressions": not source_regressions,
        "ineligible_queries_have_zero_attempts": all(
            correction["attempt_count"] == 0 for correction in enabled_correction if not correction["eligible"]
        ),
        "eligible_queries_have_at_most_one_attempt": all(
            correction["attempt_count"] <= 1 for correction in enabled_correction if correction["eligible"]
        ),
        "zero_correction_model_attempts": all(correction["model_attempts"] == 0 for correction in enabled_correction),
        "zero_web_calls": web_calls == 0,
        "no_judge_failures": not judge_failures["disabled"] and not judge_failures["enabled"],
        "provider_attempts_within_ceiling": budget["attempts"] <= budget["maximum"],
    }
    return {
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "coverage": {
            "expected_ids_per_arm": len(all_ids),
            "disabled_success_ids": len(disabled_success),
            "enabled_success_ids": len(enabled_success),
            "disabled_faithfulness_ids": len(disabled_scored),
            "enabled_faithfulness_ids": len(enabled_scored),
            "common_faithfulness_ids": len(common_scored),
            "common_sample_ids": common_scored,
        },
        "faithfulness": {
            "disabled_common_mean": disabled_mean,
            "enabled_common_mean": enabled_mean,
            "enabled_minus_disabled": faithfulness_delta,
            "per_sample": {
                sample_id: {
                    "disabled": disabled_results[sample_id]["scores"]["faithfulness"],
                    "enabled": enabled_results[sample_id]["scores"]["faithfulness"],
                }
                for sample_id in common_scored
            },
        },
        "blank_answer_ids": {"disabled": sorted(disabled_blank), "enabled": sorted(enabled_blank)},
        "execution_error_ids": {
            "disabled": sorted(set(all_ids) - disabled_success),
            "enabled": sorted(set(all_ids) - enabled_success),
        },
        "source_attribution_regressions": source_regressions,
        "judge_failures": judge_failures,
        "token_coverage": {
            "disabled_generation": disabled["execution_usage"],
            "enabled_generation": enabled["execution_usage"],
            "disabled_judge": disabled["judge_usage"],
            "enabled_judge": enabled["judge_usage"],
        },
        "provider_attempts": budget,
        "web_calls": web_calls,
    }


def main() -> int:
    """Run the bounded cloud pair and persist its full local evidence artifact."""
    args = parse_args()
    disabled_checkpoint = args.output.with_suffix(".disabled.json")
    enabled_checkpoint = args.output.with_suffix(".enabled.json")
    existing_outputs = [path for path in (args.output, disabled_checkpoint, enabled_checkpoint) if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite existing cloud evidence: {existing_outputs}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_env()
    config = get_config()
    git_metadata = get_git_metadata()
    if git_metadata["is_dirty"] is not False:
        raise ValueError("Commit the cloud evaluation implementation before issuing provider calls")
    cohort = load_cohort(args.cohort)
    validate_frozen_cohort(cohort)
    contract = validate_cloud_contract(config)
    dataset = {sample.id: sample for sample in load_golden_dataset(args.dataset)}
    samples = [dataset[entry["sample_id"]] for entry in cohort["targets"] + cohort["controls"]]

    budget = ProviderAttemptBudget()
    execution_callback = ProviderUsageCallback("generation", budget)
    judge_callback = ProviderUsageCallback("judge", budget)
    disabled_execution_model = build_counted_model(
        contract["execution_provider"],
        contract["execution_model"],
        contract["execution_kwargs"],
        execution_callback,
    )
    enabled_execution_model = build_counted_model(
        contract["execution_provider"],
        contract["execution_model"],
        contract["execution_kwargs"],
        execution_callback,
    )
    disabled_judge_model = build_counted_model(
        contract["judge_provider"],
        contract["judge_model"],
        contract["judge_kwargs"],
        judge_callback,
    )
    enabled_judge_model = build_counted_model(
        contract["judge_provider"],
        contract["judge_model"],
        contract["judge_kwargs"],
        judge_callback,
    )

    disabled = run_arm(
        "disabled",
        prepare_arm_config(config, correction_enabled=False),
        samples,
        disabled_execution_model,
        disabled_judge_model,
        execution_callback,
        judge_callback,
    )
    disabled_checkpoint.write_text(json.dumps(disabled, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    enabled = run_arm(
        "enabled",
        prepare_arm_config(config, correction_enabled=True),
        samples,
        enabled_execution_model,
        enabled_judge_model,
        execution_callback,
        judge_callback,
    )
    enabled_checkpoint.write_text(json.dumps(enabled, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    budget_snapshot = budget.snapshot()
    gate = evaluate_cloud_gate(disabled, enabled, budget_snapshot)
    artifact = {
        "schema_version": 1,
        "evaluation_scope": "paired_generation_and_faithfulness",
        "git": git_metadata,
        "dataset": {"path": str(args.dataset), "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest()},
        "cohort": {"path": str(args.cohort), "sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest()},
        "contract": contract,
        "gate": gate,
        "arms": {"disabled": disabled, "enabled": enabled},
    }
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info(
        "M12 cloud gate decision=%s provider_attempts=%d evidence=%s",
        gate["decision"],
        budget_snapshot["attempts"],
        args.output,
    )
    return 0 if gate["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
