"""Phoenix experiment integration for eval harness results.

Pushes each eval run to a running Phoenix server as a named experiment so that
results are browsable and comparable in the Phoenix UI at http://localhost:6006.

The integration uses the Phoenix REST API directly — no arize-phoenix client SDK
is required. The only dependency is ``httpx``, which is already installed as part
of the project's dev dependencies.

Architecture
------------
Five REST calls are made per eval run:

1. ``POST /v1/datasets/upload`` — upload the golden dataset as a Phoenix dataset
   (one version per run, ``action=create``).
2. ``GET /v1/datasets/{id}/examples`` — recover per-example Phoenix IDs so that
   experiment runs can be linked back to dataset examples.
3. ``POST /v1/datasets/{id}/experiments`` — create a named experiment.
4. ``POST /v1/experiments/{id}/runs`` — one run per evaluated sample.
5. ``POST /v1/experiment_evaluations`` — one evaluation record per (run, metric).

All errors are non-fatal: a failed push logs a warning and returns ``None``
rather than aborting the eval run.

Usage::

    from src.eval.phoenix_reporter import PhoenixReporter
    reporter = PhoenixReporter()
    url = reporter.push(result, samples)
    if url:
        logger.info("Phoenix experiment: %s", url)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from src.eval.models import EvalRunResult, GoldenSample
from src.utils import get_config, logger

_DATASET_NAME = "eval_golden_dataset"
_DATASET_DESCRIPTION = "Pour Decisions golden Q&A eval dataset (auto-uploaded by eval harness)"


def _extract_base_url(endpoint: str) -> str:
    """Strip the path from a Phoenix OTLP endpoint to obtain the base URL.

    Args:
        endpoint: Full OTLP endpoint, e.g. ``http://localhost:6006/v1/traces``.

    Returns:
        Base URL with no trailing slash, e.g. ``http://localhost:6006``.
    """
    parsed = urlsplit(endpoint.strip())
    return f"{parsed.scheme}://{parsed.netloc}"


class PhoenixReporter:
    """Push eval run results to a Phoenix server as named experiments.

    Each call to :meth:`push` uploads the golden samples as a versioned dataset,
    creates a named experiment, and populates it with per-sample runs and per-metric
    evaluation scores.

    Attributes:
        base_url: Base URL of the Phoenix server (no trailing slash).
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the reporter.

        Args:
            base_url: Override for the Phoenix base URL. If not provided,
                reads from ``observability.phoenix.endpoint`` in ``app_config.yml``
                and strips the path to recover the server origin.
        """
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            cfg = get_config()
            phoenix_cfg = getattr(cfg.observability, "phoenix", None)
            endpoint = str(getattr(phoenix_cfg, "endpoint", "http://localhost:6006/v1/traces"))
            self.base_url = _extract_base_url(endpoint)

    def push(self, result: EvalRunResult, samples: list[GoldenSample]) -> str | None:
        """Push one eval run to Phoenix as a named experiment.

        Uploads the golden dataset as a new version, creates an experiment tied to
        that version, then populates runs and metric evaluations.

        Args:
            result: Aggregated eval run result from :class:`~src.eval.reporter.EvalReporter`.
            samples: Original golden samples used in the run (for dataset upload).

        Returns:
            URL string for the Phoenix experiment page on success, or ``None``
            if Phoenix is unreachable or any step fails.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("PhoenixReporter: httpx not installed; skipping Phoenix push")
            return None

        try:
            return self._push(httpx_module=httpx, result=result, samples=samples)
        except Exception as exc:
            logger.warning("PhoenixReporter: push failed (%s); continuing without Phoenix", exc)
            return None

    def _push(
        self,
        httpx_module: Any,
        result: EvalRunResult,
        samples: list[GoldenSample],
    ) -> str | None:
        """Internal push implementation.

        Args:
            httpx_module: The imported httpx module (injected for testability).
            result: Aggregated eval run result.
            samples: Golden samples for dataset upload.

        Returns:
            Phoenix experiment URL or None.
        """
        client = httpx_module.Client(base_url=self.base_url, timeout=30.0)

        with client:
            dataset_id, version_id = self._upload_dataset(client=client, samples=samples)
            example_id_map = self._list_example_ids(
                client=client,
                dataset_id=dataset_id,
                version_id=version_id,
                samples=samples,
            )
            experiment_id = self._create_experiment(
                client=client,
                dataset_id=dataset_id,
                version_id=version_id,
                result=result,
            )
            run_id_map = self._push_runs(
                client=client,
                experiment_id=experiment_id,
                result=result,
                example_id_map=example_id_map,
            )
            self._push_evaluations(
                client=client,
                result=result,
                run_id_map=run_id_map,
            )

        url = f"{self.base_url}/experiments/{experiment_id}"
        logger.info("PhoenixReporter: experiment published at %s", url)
        return url

    def _upload_dataset(
        self,
        client: Any,
        samples: list[GoldenSample],
    ) -> tuple[str, str]:
        """Upload the golden dataset as a new version and return (dataset_id, version_id).

        Uses ``action=create`` so each eval run gets a versioned snapshot of the
        dataset, keeping the history of what questions were active at each eval.

        Args:
            client: Active httpx client.
            samples: Golden samples to upload.

        Returns:
            Tuple of ``(dataset_id, version_id)`` strings.

        Raises:
            httpx.HTTPStatusError: On non-2xx response.
        """
        payload: dict[str, Any] = {
            "action": "create",
            "name": _DATASET_NAME,
            "description": _DATASET_DESCRIPTION,
            "inputs": [
                {"id": sample.id, "question": sample.question, "category": sample.category}
                for sample in samples
            ],
            "outputs": [
                {"ground_truth": sample.ground_truth}
                for sample in samples
            ],
            "metadata": [
                {
                    "difficulty": sample.difficulty,
                    "tags": ",".join(sample.tags),
                    "expected_tool_calls": ",".join(sample.expected_tool_calls),
                }
                for sample in samples
            ],
        }
        response = client.post("/v1/datasets/upload", json=payload)
        response.raise_for_status()
        data = response.json()["data"]
        dataset_id: str = data["dataset_id"]
        version_id: str = data["version_id"]
        logger.info(
            "PhoenixReporter: uploaded dataset '%s' dataset_id=%s version_id=%s",
            _DATASET_NAME,
            dataset_id,
            version_id,
        )
        return dataset_id, version_id

    def _list_example_ids(
        self,
        client: Any,
        dataset_id: str,
        version_id: str,
        samples: list[GoldenSample],
    ) -> dict[str, str]:
        """Build a mapping from golden sample ID to Phoenix dataset example ID.

        Args:
            client: Active httpx client.
            dataset_id: Phoenix dataset ID.
            version_id: Dataset version to list.
            samples: Golden samples (used to validate the mapping).

        Returns:
            Dict mapping ``sample.id`` to Phoenix ``example.id``.
        """
        response = client.get(
            f"/v1/datasets/{dataset_id}/examples",
            params={"version_id": version_id},
        )
        response.raise_for_status()
        examples = response.json()["data"]["examples"]

        example_id_map: dict[str, str] = {}
        for example in examples:
            sample_id = example.get("input", {}).get("id")
            if sample_id:
                example_id_map[sample_id] = example["id"]

        logger.info(
            "PhoenixReporter: resolved %d/%d example IDs",
            len(example_id_map),
            len(samples),
        )
        return example_id_map

    def _create_experiment(
        self,
        client: Any,
        dataset_id: str,
        version_id: str,
        result: EvalRunResult,
    ) -> str:
        """Create a named Phoenix experiment and return its ID.

        Args:
            client: Active httpx client.
            dataset_id: Phoenix dataset ID.
            version_id: Dataset version to run the experiment over.
            result: Eval run result (used for name, description, and metadata).

        Returns:
            Phoenix experiment ID string.
        """
        experiment_name = f"eval_{result.mode}_{result.backend}_{result.run_id}"
        description = (
            f"Eval harness run — mode: {result.mode}, backend: {result.backend}, "
            f"git: {result.git_sha}, samples: {result.summary.get('evaluated', 0)}"
        )
        payload: dict[str, Any] = {
            "name": experiment_name,
            "description": description,
            "version_id": version_id,
            "metadata": {
                "run_id": result.run_id,
                "git_sha": result.git_sha,
                "mode": result.mode,
                "backend": result.backend,
                **result.config_snapshot,
            },
        }
        response = client.post(f"/v1/datasets/{dataset_id}/experiments", json=payload)
        response.raise_for_status()
        experiment_id: str = response.json()["data"]["id"]
        logger.info(
            "PhoenixReporter: created experiment '%s' id=%s",
            experiment_name,
            experiment_id,
        )
        return experiment_id

    def _push_runs(
        self,
        client: Any,
        experiment_id: str,
        result: EvalRunResult,
        example_id_map: dict[str, str],
    ) -> dict[str, str]:
        """Push one experiment run per evaluated sample.

        Samples without a matching Phoenix example ID (e.g., cellar samples that
        were not uploaded because they had no ground truth) are skipped.

        Args:
            client: Active httpx client.
            experiment_id: Phoenix experiment ID.
            result: Eval run result.
            example_id_map: Map from sample ID to Phoenix example ID.

        Returns:
            Dict mapping sample ID to Phoenix experiment run ID.
        """
        run_id_map: dict[str, str] = {}
        run_ts = datetime.now(UTC)

        for sample_result in result.per_sample:
            example_id = example_id_map.get(sample_result.id)
            if not example_id:
                continue

            start_time = run_ts.isoformat()
            end_time = (run_ts + timedelta(milliseconds=sample_result.latency_ms)).isoformat()

            output: dict[str, Any] = {
                "answer": sample_result.answer,
                "latency_ms": sample_result.latency_ms,
                "contexts_count": len(sample_result.contexts),
            }
            if sample_result.tool_calls_made:
                output["tool_calls_made"] = ",".join(sample_result.tool_calls_made)

            payload: dict[str, Any] = {
                "dataset_example_id": example_id,
                "output": output,
                "repetition_number": 1,
                "start_time": start_time,
                "end_time": end_time,
            }
            if sample_result.error:
                payload["error"] = sample_result.error

            response = client.post(f"/v1/experiments/{experiment_id}/runs", json=payload)
            response.raise_for_status()
            phoenix_run_id: str = response.json()["data"]["id"]
            run_id_map[sample_result.id] = phoenix_run_id

        logger.info(
            "PhoenixReporter: pushed %d experiment runs", len(run_id_map)
        )
        return run_id_map

    def _push_evaluations(
        self,
        client: Any,
        result: EvalRunResult,
        run_id_map: dict[str, str],
    ) -> None:
        """Push one evaluation record per (sample, metric) pair.

        Only samples with a Phoenix run ID and non-empty scores are processed.
        The ``annotator_kind`` is set to ``"CODE"`` for local retrieval metrics
        (MRR, precision@k) and ``"LLM"`` for Ragas-based metrics.

        Args:
            client: Active httpx client.
            result: Eval run result.
            run_id_map: Map from sample ID to Phoenix experiment run ID.
        """
        llm_metrics = frozenset(
            {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        )
        eval_ts = datetime.now(UTC)
        eval_end = (eval_ts + timedelta(seconds=1)).isoformat()
        eval_start = eval_ts.isoformat()

        count = 0
        for sample_result in result.per_sample:
            phoenix_run_id = run_id_map.get(sample_result.id)
            if not phoenix_run_id or not sample_result.scores:
                continue

            for metric_name, score in sample_result.scores.items():
                annotator_kind = "LLM" if metric_name in llm_metrics else "CODE"
                payload: dict[str, Any] = {
                    "experiment_run_id": phoenix_run_id,
                    "name": metric_name,
                    "annotator_kind": annotator_kind,
                    "start_time": eval_start,
                    "end_time": eval_end,
                    "result": {
                        "score": float(score),
                        "label": _score_to_label(metric_name, score),
                    },
                }
                response = client.post("/v1/experiment_evaluations", json=payload)
                response.raise_for_status()
                count += 1

        logger.info("PhoenixReporter: pushed %d evaluation records", count)


def _score_to_label(metric_name: str, score: float) -> str:
    """Convert a numeric metric score to a human-readable quality label.

    Uses general thresholds applicable to all supported metrics. The label is
    stored as a secondary annotation in Phoenix alongside the numeric score.

    Args:
        metric_name: Name of the metric (unused in threshold logic, reserved for
            future per-metric calibration).
        score: Numeric score in the range [0.0, 1.0].

    Returns:
        One of ``"excellent"``, ``"good"``,  ``"fair"``, or ``"poor"``.
    """
    if score >= 0.85:
        return "excellent"
    if score >= 0.70:
        return "good"
    if score >= 0.50:
        return "fair"
    return "poor"

