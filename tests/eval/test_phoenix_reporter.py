"""Unit tests for PhoenixReporter (Phase 7).

Tests use a fake httpx module and a mock Client so no real network calls are made.
All tests run without a running Phoenix server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.eval.models import EvalRunResult, GoldenSample, SampleResult
from src.eval.phoenix_reporter import PhoenixReporter, _extract_base_url, _score_to_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(sid: str = "rag_only_001", category: str = "rag_only") -> GoldenSample:
    return GoldenSample(
        id=sid,
        question="What is Burgundy?",
        category=category,
        difficulty="easy",
        expected_facts=["A French wine region"],
        ground_truth="Burgundy is a prestigious French wine region.",
        tags=["rag"],
    )


def _make_result(sample_id: str = "rag_only_001") -> EvalRunResult:
    sr = SampleResult(
        id=sample_id,
        question="What is Burgundy?",
        answer="Burgundy is a French wine region.",
        contexts=["Burgundy produces fine Pinot Noir."],
        latency_ms=120.0,
        scores={"faithfulness": 0.9, "mrr": 0.75},
    )
    return EvalRunResult(
        run_id="20260504T120000",
        timestamp="2026-05-04T12:00:00",
        mode="full",
        backend="rag",
        git_sha="abc1234",
        config_snapshot={"model": "gemini-2.0-flash"},
        aggregate_metrics={"faithfulness": 0.9, "mrr": 0.75},
        per_sample=[sr],
        summary={"evaluated": 1, "errors": 0},
    )


def _make_http_client(
    dataset_id: str = "ds-1",
    version_id: str = "ver-1",
    experiment_id: str = "exp-1",
    example_phoenix_id: str = "ex-1",
    run_id: str = "run-1",
    sample_input_id: str = "rag_only_001",
) -> MagicMock:
    """Build a mock httpx.Client that returns sensible responses for all five REST calls."""
    client = MagicMock()

    # 1. POST /v1/datasets/upload
    upload_resp = MagicMock()
    upload_resp.raise_for_status = MagicMock()
    upload_resp.json.return_value = {
        "data": {"dataset_id": dataset_id, "version_id": version_id}
    }

    # 2. GET /v1/datasets/{id}/examples
    list_resp = MagicMock()
    list_resp.raise_for_status = MagicMock()
    list_resp.json.return_value = {
        "data": {
            "examples": [{"id": example_phoenix_id, "input": {"id": sample_input_id}}]
        }
    }

    # 3. POST /v1/datasets/{id}/experiments
    create_exp_resp = MagicMock()
    create_exp_resp.raise_for_status = MagicMock()
    create_exp_resp.json.return_value = {"data": {"id": experiment_id}}

    # 4. POST /v1/experiments/{id}/runs
    push_run_resp = MagicMock()
    push_run_resp.raise_for_status = MagicMock()
    push_run_resp.json.return_value = {"data": {"id": run_id}}

    # 5. POST /v1/experiment_evaluations
    push_eval_resp = MagicMock()
    push_eval_resp.raise_for_status = MagicMock()
    push_eval_resp.json.return_value = {"data": {"id": "eval-1"}}

    client.post.side_effect = [upload_resp, create_exp_resp, push_run_resp, push_eval_resp, push_eval_resp]
    client.get.side_effect = [list_resp]

    # Support context manager protocol
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)

    return client


def _make_httpx_module(client: MagicMock) -> MagicMock:
    """Return a fake httpx module whose Client() returns the given mock client."""
    httpx_module = MagicMock()
    httpx_module.Client.return_value = client
    return httpx_module


# ---------------------------------------------------------------------------
# _extract_base_url
# ---------------------------------------------------------------------------

class TestExtractBaseUrl:
    def test_strips_path(self) -> None:
        assert _extract_base_url("http://localhost:6006/v1/traces") == "http://localhost:6006"

    def test_no_path_unchanged(self) -> None:
        assert _extract_base_url("http://localhost:6006") == "http://localhost:6006"

    def test_https_scheme_preserved(self) -> None:
        assert _extract_base_url("https://phoenix.example.com/v1/traces") == "https://phoenix.example.com"

    def test_port_preserved(self) -> None:
        assert _extract_base_url("http://192.168.1.10:9090/v1/traces") == "http://192.168.1.10:9090"


# ---------------------------------------------------------------------------
# _score_to_label
# ---------------------------------------------------------------------------

class TestScoreToLabel:
    def test_excellent_at_boundary(self) -> None:
        assert _score_to_label("faithfulness", 0.85) == "excellent"

    def test_excellent_above_boundary(self) -> None:
        assert _score_to_label("faithfulness", 1.0) == "excellent"

    def test_good_at_boundary(self) -> None:
        assert _score_to_label("mrr", 0.70) == "good"

    def test_good_below_excellent(self) -> None:
        assert _score_to_label("mrr", 0.80) == "good"

    def test_fair_at_boundary(self) -> None:
        assert _score_to_label("context_precision", 0.50) == "fair"

    def test_fair_above_poor(self) -> None:
        assert _score_to_label("context_precision", 0.65) == "fair"

    def test_poor_below_fair(self) -> None:
        assert _score_to_label("answer_relevancy", 0.49) == "poor"

    def test_poor_at_zero(self) -> None:
        assert _score_to_label("answer_relevancy", 0.0) == "poor"


# ---------------------------------------------------------------------------
# PhoenixReporter.__init__
# ---------------------------------------------------------------------------

class TestPhoenixReporterInit:
    def test_explicit_base_url_stored(self) -> None:
        reporter = PhoenixReporter(base_url="http://localhost:6006")
        assert reporter.base_url == "http://localhost:6006"

    def test_trailing_slash_stripped(self) -> None:
        reporter = PhoenixReporter(base_url="http://localhost:6006/")
        assert reporter.base_url == "http://localhost:6006"


# ---------------------------------------------------------------------------
# PhoenixReporter.push — happy path
# ---------------------------------------------------------------------------

class TestPhoenixReporterPushHappyPath:
    def test_returns_experiment_url(self) -> None:
        samples = [_make_sample()]
        result = _make_result()
        client = _make_http_client(experiment_id="exp-42")
        httpx_module = _make_httpx_module(client)

        reporter = PhoenixReporter(base_url="http://localhost:6006")
        url = reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        assert url == "http://localhost:6006/experiments/exp-42"

    def test_dataset_upload_called_once(self) -> None:
        samples = [_make_sample()]
        result = _make_result()
        client = _make_http_client()
        httpx_module = _make_httpx_module(client)

        reporter = PhoenixReporter(base_url="http://localhost:6006")
        reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        upload_call = client.post.call_args_list[0]
        assert upload_call[0][0] == "/v1/datasets/upload"
        payload = upload_call[1]["json"]
        assert payload["name"] == "eval_golden_dataset"
        assert len(payload["inputs"]) == 1
        assert payload["inputs"][0]["id"] == "rag_only_001"

    def test_example_ids_fetched_with_version(self) -> None:
        samples = [_make_sample()]
        result = _make_result()
        client = _make_http_client(dataset_id="ds-99", version_id="ver-99")
        httpx_module = _make_httpx_module(client)

        reporter = PhoenixReporter(base_url="http://localhost:6006")
        reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        get_call = client.get.call_args_list[0]
        assert "ds-99" in get_call[0][0]
        assert get_call[1]["params"]["version_id"] == "ver-99"

    def test_experiment_run_pushed_per_sample(self) -> None:
        samples = [_make_sample()]
        result = _make_result()
        client = _make_http_client(experiment_id="exp-1")
        httpx_module = _make_httpx_module(client)

        reporter = PhoenixReporter(base_url="http://localhost:6006")
        reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        run_call = client.post.call_args_list[2]
        assert "/v1/experiments/exp-1/runs" in run_call[0][0]

    def test_evaluation_records_pushed_per_metric(self) -> None:
        samples = [_make_sample()]
        # Two metrics: faithfulness, mrr
        result = _make_result()
        client = _make_http_client()
        httpx_module = _make_httpx_module(client)

        reporter = PhoenixReporter(base_url="http://localhost:6006")
        reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        # 5 post calls: upload + create_experiment + push_run + 2 evaluations
        assert client.post.call_count == 5

    def test_annotator_kind_llm_for_ragas_metrics(self) -> None:
        samples = [_make_sample()]
        result = _make_result()
        client = _make_http_client()
        httpx_module = _make_httpx_module(client)

        reporter = PhoenixReporter(base_url="http://localhost:6006")
        reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        # Fourth post call is the first evaluation (faithfulness -> LLM)
        eval_call_payload = client.post.call_args_list[3][1]["json"]
        assert eval_call_payload["name"] == "faithfulness"
        assert eval_call_payload["annotator_kind"] == "LLM"

    def test_annotator_kind_code_for_retrieval_metrics(self) -> None:
        samples = [_make_sample()]
        result = _make_result()
        client = _make_http_client()
        httpx_module = _make_httpx_module(client)

        reporter = PhoenixReporter(base_url="http://localhost:6006")
        reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        # Fifth post call is the second evaluation (mrr -> CODE)
        eval_call_payload = client.post.call_args_list[4][1]["json"]
        assert eval_call_payload["name"] == "mrr"
        assert eval_call_payload["annotator_kind"] == "CODE"


# ---------------------------------------------------------------------------
# PhoenixReporter.push — fail-open behaviour
# ---------------------------------------------------------------------------

class TestPhoenixReporterFailOpen:
    def test_returns_none_when_httpx_missing(self) -> None:
        """push() must return None (not raise) when httpx is not importable."""
        samples = [_make_sample()]
        result = _make_result()
        reporter = PhoenixReporter(base_url="http://localhost:6006")

        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("httpx not found")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            url = reporter.push(result=result, samples=samples)

        assert url is None

    def test_returns_none_on_http_error(self) -> None:
        """push() must return None when the server returns a non-2xx status."""
        import httpx as real_httpx

        samples = [_make_sample()]
        result = _make_result()
        reporter = PhoenixReporter(base_url="http://localhost:6006")

        bad_response = MagicMock()
        bad_response.raise_for_status.side_effect = real_httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        bad_client = MagicMock()
        bad_client.post.side_effect = [bad_response]
        bad_client.__enter__ = MagicMock(return_value=bad_client)
        bad_client.__exit__ = MagicMock(return_value=False)

        fake_httpx = MagicMock()
        fake_httpx.Client.return_value = bad_client

        url = reporter._push.__wrapped__(reporter, httpx_module=fake_httpx, result=result, samples=samples) \
            if hasattr(reporter._push, "__wrapped__") else None

        # Call the public method which wraps with try/except
        with patch.object(reporter, "_push", side_effect=real_httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )):
            url = reporter.push(result=result, samples=samples)

        assert url is None

    def test_returns_none_on_connection_error(self) -> None:
        """push() must return None when Phoenix server is unreachable."""
        import httpx as real_httpx

        samples = [_make_sample()]
        result = _make_result()
        reporter = PhoenixReporter(base_url="http://localhost:9999")

        with patch.object(reporter, "_push", side_effect=Exception("connection refused")):
            url = reporter.push(result=result, samples=samples)

        assert url is None

    def test_sample_without_example_id_is_skipped(self) -> None:
        """Samples missing a Phoenix example ID must be silently skipped."""
        samples = [_make_sample(sid="rag_only_001")]
        result = _make_result(sample_id="rag_only_001")
        # Return empty example list so no ID mapping exists
        client = MagicMock()
        upload_resp = MagicMock()
        upload_resp.raise_for_status = MagicMock()
        upload_resp.json.return_value = {"data": {"dataset_id": "ds-1", "version_id": "ver-1"}}

        list_resp = MagicMock()
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = {"data": {"examples": []}}  # no examples

        create_exp_resp = MagicMock()
        create_exp_resp.raise_for_status = MagicMock()
        create_exp_resp.json.return_value = {"data": {"id": "exp-1"}}

        client.post.side_effect = [upload_resp, create_exp_resp]
        client.get.side_effect = [list_resp]
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)

        httpx_module = _make_httpx_module(client)
        reporter = PhoenixReporter(base_url="http://localhost:6006")
        url = reporter._push(httpx_module=httpx_module, result=result, samples=samples)

        # Experiment URL still returned; no run/eval posts made
        assert url == "http://localhost:6006/experiments/exp-1"
        assert client.post.call_count == 2  # upload + create_experiment only

