"""Unit tests for eval CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
import types

import pytest

from src.eval.__main__ import _build_run_metadata, _validate_cli_filters
from src.eval.models import GoldenSample
from src.eval.preflight import (
    preflight_eval_local_only_guardrail,
    preflight_full_mode,
    preflight_model_backend,
    preflight_rag_backend,
    run_preflight,
)


@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    """Build a minimal parser for CLI validation tests."""
    return argparse.ArgumentParser(prog="eval")


@pytest.fixture()
def dataset() -> list[GoldenSample]:
    """Build a minimal golden dataset for filter validation tests."""
    return [
        GoldenSample(
            id="rag_only_001",
            question="What is Barolo?",
            category="rag_only",
            difficulty="easy",
            expected_facts=["A Piedmont wine"],
            ground_truth="Barolo is a red DOCG wine from Piedmont.",
            ground_truth_chunk_ids=["chunk-1"],
            tags=["barolo", "italy"],
        ),
        GoldenSample(
            id="cellar_001",
            question="How many Barolo bottles do I have?",
            category="cellar",
            difficulty="medium",
            expected_facts=["A bottle count"],
            ground_truth="The answer should report the Barolo bottle count.",
            ground_truth_chunk_ids=[],
            tags=["cellar", "inventory"],
        ),
    ]


@pytest.fixture()
def preflight_config() -> object:
    """Build a minimal config object for eval preflight tests."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            provider="google",
            name="gemini-2.5-flash",
            ollama=types.SimpleNamespace(base_url="http://localhost:11434"),
        ),
        chroma=types.SimpleNamespace(
            client=types.SimpleNamespace(host="localhost", port=8100),
            collections=[types.SimpleNamespace(name="wine_books")],
        ),
        eval=types.SimpleNamespace(
            execution_provider="ollama",
            execution_model="llama3.2:3b",
            ollama=types.SimpleNamespace(base_url="http://localhost:11434"),
            ragas=types.SimpleNamespace(
                evaluator_provider="ollama",
                evaluator_model="gemma2:2b",
            )
        ),
    )


def test_validate_cli_filters_accepts_valid_values(
    parser: argparse.ArgumentParser,
    dataset: list[GoldenSample],
) -> None:
    """Valid categories, difficulties, and tags should pass validation."""
    _validate_cli_filters(
        parser=parser,
        dataset=dataset,
        categories=["rag_only", "cellar"],
        difficulties=["easy", "medium"],
        tags=["barolo", "inventory"],
        sample_ids=["rag_only_001", "cellar_001"],
        validate_tag_filters=True,
    )


def test_validate_cli_filters_rejects_invalid_category(
    parser: argparse.ArgumentParser,
    dataset: list[GoldenSample],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown categories should fail fast with a helpful message."""
    with pytest.raises(SystemExit):
        _validate_cli_filters(
            parser=parser,
            dataset=dataset,
            categories=["ragonly"],
            difficulties=None,
            tags=None,
            sample_ids=None,
            validate_tag_filters=True,
        )

    captured = capsys.readouterr()
    assert "Invalid categories: ragonly" in captured.err
    assert "rag_only" in captured.err


def test_validate_cli_filters_rejects_invalid_difficulty(
    parser: argparse.ArgumentParser,
    dataset: list[GoldenSample],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown difficulties should fail fast with a helpful message."""
    with pytest.raises(SystemExit):
        _validate_cli_filters(
            parser=parser,
            dataset=dataset,
            categories=None,
            difficulties=["medum"],
            tags=None,
            sample_ids=None,
            validate_tag_filters=True,
        )

    captured = capsys.readouterr()
    assert "Invalid difficulties: medum" in captured.err
    assert "medium" in captured.err


def test_validate_cli_filters_rejects_invalid_tags_when_strict(
    parser: argparse.ArgumentParser,
    dataset: list[GoldenSample],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown tags should fail fast when strict tag validation is enabled."""
    with pytest.raises(SystemExit):
        _validate_cli_filters(
            parser=parser,
            dataset=dataset,
            categories=None,
            difficulties=None,
            tags=["does_not_exist"],
            sample_ids=None,
            validate_tag_filters=True,
        )

    captured = capsys.readouterr()
    assert "Invalid tags: does_not_exist" in captured.err
    assert "barolo" in captured.err
    assert "inventory" in captured.err


def test_validate_cli_filters_allows_invalid_tags_when_strict_mode_disabled(
    parser: argparse.ArgumentParser,
    dataset: list[GoldenSample],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown tags should only warn when strict tag validation is disabled."""
    _validate_cli_filters(
        parser=parser,
        dataset=dataset,
        categories=None,
        difficulties=None,
        tags=["does_not_exist"],
        sample_ids=None,
        validate_tag_filters=False,
    )

    assert "Unknown tag filters requested: does_not_exist" in caplog.text


def test_validate_cli_filters_rejects_invalid_sample_id(
    parser: argparse.ArgumentParser,
    dataset: list[GoldenSample],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown sample ids should fail fast with a helpful message."""
    with pytest.raises(SystemExit):
        _validate_cli_filters(
            parser=parser,
            dataset=dataset,
            categories=None,
            difficulties=None,
            tags=None,
            sample_ids=["does_not_exist"],
            validate_tag_filters=True,
        )

    captured = capsys.readouterr()
    assert "Invalid sample ids: does_not_exist" in captured.err
    assert "rag_only_001" in captured.err


def test_build_run_metadata_captures_dataset_identity_and_filters(
    tmp_path: Path,
    dataset: list[GoldenSample],
) -> None:
    """Run metadata should include dataset path, hash, counts, filters, and CLI settings."""
    dataset_path = tmp_path / "golden.jsonl"
    dataset_path.write_text('{"id":"rag_only_001"}\n', encoding="utf-8")

    args = argparse.Namespace(
        mode="retrieval",
        backend="rag",
        max_concurrency=2,
        push_to_phoenix=False,
        phoenix_url=None,
    )
    config = types.SimpleNamespace(eval=types.SimpleNamespace(sample_timeout_seconds=30))

    metadata = _build_run_metadata(
        dataset_path=dataset_path,
        dataset=dataset,
        selected_samples=dataset[:1],
        categories=["rag_only"],
        difficulties=["easy"],
        tags=["barolo"],
        sample_ids=["rag_only_001"],
        args=args,
        config=config,
        git_metadata={"sha": "abc123", "branch": "main", "is_dirty": True},
    )

    assert metadata["dataset"]["path"] == str(dataset_path.resolve())
    assert metadata["dataset"]["content_hash"]
    assert metadata["dataset"]["total_sample_count"] == 2
    assert metadata["dataset"]["selected_sample_count"] == 1
    assert metadata["filters"]["categories"] == ["rag_only"]
    assert metadata["filters"]["difficulties"] == ["easy"]
    assert metadata["filters"]["tags"] == ["barolo"]
    assert metadata["filters"]["sample_ids"] == ["rag_only_001"]
    assert metadata["execution"]["mode"] == "retrieval"
    assert metadata["execution"]["backend"] == "rag"
    assert metadata["execution"]["max_concurrency"] == 2
    assert metadata["execution"]["sample_timeout_seconds"] == 30.0
    assert metadata["git"]["sha"] == "abc123"
    assert metadata["git"]["branch"] == "main"
    assert metadata["git"]["is_dirty"] is True


def test_preflight_model_backend_accepts_reachable_ollama(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    mocker,
) -> None:
    """Ollama-backed preflight should pass when the endpoint is reachable."""
    mocker.patch("src.eval.preflight._ollama_available", return_value=True)

    preflight_model_backend(parser, preflight_config)


def test_preflight_eval_local_only_guardrail_rejects_cloud_execution_model(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Eval guardrail should reject cloud-backed execution models."""
    preflight_config.eval.execution_provider = "google"
    preflight_config.eval.execution_model = "gemini-2.5-flash"

    with pytest.raises(SystemExit):
        preflight_eval_local_only_guardrail(parser, preflight_config)

    captured = capsys.readouterr()
    assert "Eval execution must use Ollama only" in captured.err


def test_preflight_eval_local_only_guardrail_rejects_cloud_evaluator_model(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Eval guardrail should reject cloud-backed evaluator models."""
    preflight_config.eval.ragas.evaluator_provider = "google"
    preflight_config.eval.ragas.evaluator_model = "gemini-2.5-flash"

    with pytest.raises(SystemExit):
        preflight_eval_local_only_guardrail(parser, preflight_config)

    captured = capsys.readouterr()
    assert "Eval judge scoring must use Ollama only" in captured.err


def test_preflight_model_backend_rejects_unreachable_ollama(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ollama-backed preflight should fail fast when the endpoint is unreachable."""
    mocker.patch("src.eval.preflight._ollama_available", return_value=False)

    with pytest.raises(SystemExit):
        preflight_model_backend(parser, preflight_config)

    captured = capsys.readouterr()
    assert "Ollama is unreachable" in captured.err


def test_preflight_rag_backend_rejects_missing_collection(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RAG preflight should fail when the configured Chroma collection is unavailable."""
    client = mocker.Mock()
    client.get_collection.side_effect = RuntimeError("missing collection")
    mocker.patch("src.eval.preflight.initialize_chroma_client", return_value=client)

    with pytest.raises(SystemExit):
        preflight_rag_backend(parser, preflight_config)

    captured = capsys.readouterr()
    assert "Chroma preflight failed" in captured.err
    assert "wine_books" in captured.err


def test_preflight_full_mode_requires_ragas(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Full mode should fail fast when ragas is not installed."""
    mocker.patch("src.eval.preflight.importlib.import_module", side_effect=ImportError("missing ragas"))

    with pytest.raises(SystemExit):
        preflight_full_mode(parser, preflight_config)

    captured = capsys.readouterr()
    assert "Full eval requires `ragas`" in captured.err


def test_run_preflight_runs_backend_and_mode_checks(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    mocker,
) -> None:
    """Full-mode RAG preflight should invoke model, backend, and scorer checks."""
    model_check = mocker.patch("src.eval.preflight.preflight_model_backend")
    rag_check = mocker.patch("src.eval.preflight.preflight_rag_backend")
    full_check = mocker.patch("src.eval.preflight.preflight_full_mode")

    run_preflight(parser=parser, config=preflight_config, mode="full", backend="rag")

    model_check.assert_called_once_with(parser, preflight_config)
    rag_check.assert_called_once_with(parser, preflight_config)
    full_check.assert_called_once_with(parser, preflight_config)


def test_run_preflight_skips_model_check_for_retrieval_rag(
    parser: argparse.ArgumentParser,
    preflight_config: object,
    mocker,
) -> None:
    """Retrieval-only RAG preflight should not require the execution LLM backend."""
    model_check = mocker.patch("src.eval.preflight.preflight_model_backend")
    rag_check = mocker.patch("src.eval.preflight.preflight_rag_backend")
    full_check = mocker.patch("src.eval.preflight.preflight_full_mode")

    run_preflight(parser=parser, config=preflight_config, mode="retrieval", backend="rag")

    model_check.assert_not_called()
    rag_check.assert_called_once_with(parser, preflight_config)
    full_check.assert_not_called()
