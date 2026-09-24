"""Run bounded direct Ollama Cloud capability checks for M13A Gate 0.

This probe never changes production configuration or writes generated descriptions.
It records metadata and validation outcomes, not model text or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ollama import AsyncClient, Client

from src.agents.description_service import WineAnalysis
from src.eval.models import SampleResult
from src.eval.ragas_scorer import RagasScorer
from src.utils import get_config


CONTRACT_PATH = Path("src/eval/m13a_gate0_contract.json")
CALL_CAP_AMENDMENT_PATH = Path("src/eval/m13a_gate0_call_cap_amendment.json")
_WINE_CASES = (
    ("Cantine Povero Barolo Priore", "Cantine Povero", "2019", "red", "Nebbiolo", "Piedmont", "Italy", "Barolo"),
    ("Chianti Classico", "A Tuscan producer", "2020", "red", "Sangiovese", "Tuscany", "Italy", "Chianti Classico"),
    (
        "Blanc de Blancs Champagne", "A Champagne producer", "Non-vintage", "sparkling",
        "Chardonnay", "Champagne", "France", "Champagne",
    ),
)


def _validate_contract(contract: dict[str, Any]) -> None:
    """Reject drift before any paid or free-tier inference request."""
    if contract["status"] != "frozen_before_live_results":
        raise ValueError("Gate 0 contract is not frozen")
    if contract["connection"]["base_url"] != "https://ollama.com":
        raise ValueError("Gate 0 endpoint is not direct Ollama Cloud")
    expected = contract["provenance"]
    files = {
        **expected["tool_source_sha256"],
        **{f"src/agents/prompts/{name}": digest for name, digest in expected["prompt_files_sha256"].items()},
        contract["release_cohort"]["dataset_path"]: contract["release_cohort"]["dataset_sha256"],
        contract["release_cohort"]["manifest_path"]: contract["release_cohort"]["manifest_sha256"],
        "app_config.yml": expected["app_config_sha256"],
    }
    for name, digest in files.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen Gate 0 input changed: {name}")


@dataclass
class CallBudget:
    """Count actual Ollama SDK chat attempts across sync and async clients."""

    limit: int
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    token_reports: int = 0

    def reserve(self) -> None:
        """Fail before a request exceeds the approved call cap."""
        if self.attempts >= self.limit:
            raise RuntimeError("Gate 0 model-call cap exhausted")
        self.attempts += 1

    def observe(self, response: Any) -> None:
        """Record token counts only when the provider returned them."""
        input_count = getattr(response, "prompt_eval_count", None)
        output_count = getattr(response, "eval_count", None)
        if isinstance(input_count, int) and isinstance(output_count, int):
            self.input_tokens += input_count
            self.output_tokens += output_count
            self.token_reports += 1


def _model(contract: dict[str, Any], *, judge: bool = False) -> ChatOllama:
    """Construct the approved candidate using the existing LangChain client."""
    connection = contract["connection"]
    sampling = contract["judge_sampling"] if judge else contract["application_sampling"]
    timeout = connection["judge_timeout_seconds"] if judge else connection["app_timeout_seconds"]
    kwargs = {key: value for key, value in sampling.items() if value is not None}
    return ChatOllama(
        model=connection["judge_model"] if judge else connection["application_model"],
        base_url=connection["base_url"],
        client_kwargs={"timeout": timeout},
        **kwargs,
    )


def _record(name: str, operation: Callable[[], Any], checks: list[dict[str, Any]]) -> Any | None:
    """Run one sync check and retain bounded evidence even after failure."""
    started = time.monotonic()
    try:
        result = operation()
    except Exception as exc:
        checks.append({
            "name": name, "passed": False, "error_type": type(exc).__name__,
            "latency_ms": round((time.monotonic() - started) * 1000),
        })
        return None
    checks.append({"name": name, "passed": True, "latency_ms": round((time.monotonic() - started) * 1000)})
    return result


async def _record_async(name: str, operation: Callable[[], Any], checks: list[dict[str, Any]]) -> Any | None:
    """Run one async check and retain bounded evidence even after failure."""
    started = time.monotonic()
    try:
        result = await operation()
    except Exception as exc:
        checks.append({
            "name": name, "passed": False, "error_type": type(exc).__name__,
            "latency_ms": round((time.monotonic() - started) * 1000),
        })
        return None
    checks.append({"name": name, "passed": True, "latency_ms": round((time.monotonic() - started) * 1000)})
    return result


def _tool_schema(name: str, description: str) -> dict[str, Any]:
    """Build one local, inert tool definition for a model capability check."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"wine": {"type": "string"}}, "required": ["wine"]},
        },
    }


def _wine_prompt(case: tuple[str, ...]) -> str:
    """Render the registered wine-description asset without changing its text."""
    template = Path("src/agents/prompts/wine_description_prompt.md").read_text()
    return template.format(
        wine_name=case[0], producer_name=case[1], vintage=case[2], wine_type=case[3],
        varietal=case[4], region=case[5], country=case[6], appellation=case[7], context_section="",
    )


def _judge_sample(contract: dict[str, Any]) -> SampleResult:
    """Use one frozen golden case as a short, deterministic judge fixture."""
    sample_id = contract["capability_probes"]["judge_scoring"]["sample_id"]
    for line in Path(contract["release_cohort"]["dataset_path"]).read_text().splitlines():
        sample = json.loads(line)
        if sample["id"] == sample_id:
            reference = sample["ground_truth"]
            return SampleResult(
                id=sample_id, question=sample["question"], answer=reference,
                ground_truth=reference, expected_facts=sample["expected_facts"],
                contexts=[reference], status="passed",
            )
    raise ValueError("Frozen judge sample is absent from the dataset")


def _run_judge(contract: dict[str, Any], checks: list[dict[str, Any]], *, cpu_embeddings: bool = False) -> None:
    """Check the existing Ragas judge path and all four configured metrics."""
    def score_judge() -> SampleResult:
        embedder = None
        if cpu_embeddings:
            model_name = str(get_config().chroma.settings.embedder)
            embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
        # Exercise the scorer's configured-client lifecycle: its second metric
        # batch needs a fresh async Ollama client after Ragas closes the first loop.
        with patch("src.eval.ragas_scorer.load_eval_model", lambda cfg: _model(contract, judge=True)):
            scorer = RagasScorer(embedder=embedder)
            scorer.evaluator_provider = "ollama"
            scorer.evaluator_model = contract["connection"]["judge_model"]
            return scorer.score([_judge_sample(contract)])[0]

    scored = _record("judge_scoring", score_judge, checks)
    if scored is not None:
        expected = set(contract["capability_probes"]["judge_scoring"]["metrics"])
        checks[-1]["passed"] = expected <= set(scored.scores)
        checks[-1]["score_coverage"] = len(expected & set(scored.scores))
        checks[-1]["metric_error_types"] = {
            metric: ":".join(reason.split(":")[:2])
            for metric, reason in scored.metric_errors.items()
        }


def run_probe(
    contract: dict[str, Any], *, judge_only: bool = False, prior_attempts: int = 0,
    approved_call_cap: int | None = None,
) -> dict[str, Any]:
    """Execute the frozen probes with an SDK-level hard request cap."""
    budget = CallBudget(limit=approved_call_cap or contract["connection"]["max_probe_calls"], attempts=prior_attempts)
    checks: list[dict[str, Any]] = []
    original_sync = Client.chat
    original_async = AsyncClient.chat

    def counted_sync(client: Client, *args: Any, **kwargs: Any) -> Any:
        budget.reserve()
        response = original_sync(client, *args, **kwargs)
        budget.observe(response)
        return response

    async def counted_async(client: AsyncClient, *args: Any, **kwargs: Any) -> Any:
        budget.reserve()
        response = await original_async(client, *args, **kwargs)
        budget.observe(response)
        return response

    with patch.object(Client, "chat", counted_sync), patch.object(AsyncClient, "chat", counted_async):
        if judge_only:
            _run_judge(contract, checks, cpu_embeddings=True)
            return _result(contract, checks, budget, prior_attempts=prior_attempts, judge_only=True)

        model = _model(contract)
        sync_reply = _record(
            "sync_chat", lambda: model.invoke("In one short sentence, name Nebbiolo as a wine grape."), checks
        )
        if sync_reply is None:
            return _result(contract, checks, budget)
        checks[-1]["passed"] = bool(getattr(sync_reply, "content", "").strip())

        async_reply = asyncio.run(_record_async(
            "async_chat", lambda: model.ainvoke("In one short sentence, name Sangiovese as a wine grape."), checks
        ))
        if async_reply is None:
            return _result(contract, checks, budget)
        checks[-1]["passed"] = bool(getattr(async_reply, "content", "").strip())

        wine_tool = _tool_schema("lookup_wine_category", "Look up the category of a named wine.")
        region_tool = _tool_schema("lookup_wine_region", "Look up the region of a named wine.")
        single = _record(
            "single_tool", lambda: model.bind_tools([wine_tool]).invoke("Call lookup_wine_category for Barolo."), checks
        )
        if single is not None:
            checks[-1]["passed"] = "lookup_wine_category" in {call.get("name") for call in single.tool_calls}
        multi = _record(
            "multi_tool",
            lambda: model.bind_tools([wine_tool, region_tool]).invoke(
                "Call BOTH lookup_wine_category and lookup_wine_region for Barolo in this reply."
            ),
            checks,
        )
        if multi is not None:
            expected = {"lookup_wine_category", "lookup_wine_region"}
            checks[-1]["passed"] = expected <= {call.get("name") for call in multi.tool_calls}

        structured = _record(
            "wine_analysis_binding",
            lambda: model.with_structured_output(WineAnalysis, method="function_calling"),
            checks,
        )
        if structured is None:
            return _result(contract, checks, budget)
        for index, case in enumerate(_WINE_CASES, start=1):
            result = _record(f"wine_analysis_{index}", lambda case=case: structured.invoke(_wine_prompt(case)), checks)
            if result is not None:
                checks[-1]["passed"] = isinstance(result, WineAnalysis) and bool(result.description.strip())

        producer_template = Path("src/agents/prompts/producer_description_prompt.md").read_text()
        producer_prompt = producer_template.format(
            producer_name="Cantine Povero", country="Italy", region="Piedmont", context_section=""
        )
        producer = _record("producer_description", lambda: model.invoke(producer_prompt), checks)
        if producer is not None:
            checks[-1]["passed"] = bool(getattr(producer, "content", "").strip())

        _run_judge(contract, checks)

    return _result(contract, checks, budget)


def _result(
    contract: dict[str, Any], checks: list[dict[str, Any]], budget: CallBudget,
    *, prior_attempts: int = 0, judge_only: bool = False,
) -> dict[str, Any]:
    """Build a secret-free, bounded evidence record."""
    return {
        "schema_version": 1,
        "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        "model": contract["connection"]["application_model"],
        "base_url": contract["connection"]["base_url"],
        "checks": checks,
        "run_mode": "judge_followup" if judge_only else "full_probe",
        "all_passed": len(checks) == (1 if judge_only else 10) and all(check["passed"] for check in checks),
        "sdk_chat_attempts": budget.attempts,
        "sdk_chat_attempts_this_run": budget.attempts - prior_attempts,
        "max_sdk_chat_attempts": budget.limit,
        "token_reports": budget.token_reports,
        "input_tokens_when_reported": budget.input_tokens,
        "output_tokens_when_reported": budget.output_tokens,
        "actual_billed_cost_usd": None,
    }


def main() -> None:
    """Load the frozen contract, run bounded calls, and write evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Required to authorize live Cloud inference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-only", action="store_true", help="Recheck the judge within the remaining call budget")
    parser.add_argument("--prior-evidence", type=Path, help="Required with --judge-only to preserve cumulative attempts")
    parser.add_argument("--approved-cap-amendment", action="store_true", help="Use the separately approved 32-call cap")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for live inference")
    load_dotenv()
    if not os.environ.get("OLLAMA_API_KEY"):
        parser.error("OLLAMA_API_KEY is not configured")
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract["connection"]["additional_paid_spend_usd"] != 0:
        parser.error("Only included free-tier credits are authorized")
    _validate_contract(contract)
    prior_attempts = 0
    approved_call_cap = contract["connection"]["max_probe_calls"]
    if args.judge_only:
        if args.prior_evidence is None:
            parser.error("--judge-only requires --prior-evidence")
        prior = json.loads(args.prior_evidence.read_text())
        if prior["contract_sha256"] != hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest():
            parser.error("Prior evidence used a different frozen contract")
        prior_attempts = int(prior["sdk_chat_attempts"])
        if args.approved_cap_amendment:
            amendment = json.loads(CALL_CAP_AMENDMENT_PATH.read_text())
            if amendment["contract_sha256"] != hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest():
                parser.error("Call-cap amendment does not match the frozen contract")
            if amendment["additional_paid_spend_usd"] != 0:
                parser.error("Only included free-tier credits are authorized")
            if prior_attempts != amendment["prior_sdk_chat_attempts"]:
                parser.error("Prior call count does not match the approved amendment")
            approved_call_cap = amendment["cumulative_max_sdk_chat_attempts"]
        if prior_attempts >= approved_call_cap:
            parser.error("No approved model-call attempts remain")
    elif args.approved_cap_amendment:
        parser.error("The cap amendment applies only to the judge follow-up")
    result = run_probe(
        contract, judge_only=args.judge_only, prior_attempts=prior_attempts,
        approved_call_cap=approved_call_cap,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "all_passed": result["all_passed"],
        "sdk_chat_attempts": result["sdk_chat_attempts"],
        "checks": [{"name": check["name"], "passed": check["passed"]} for check in result["checks"]],
    }))


if __name__ == "__main__":
    main()
