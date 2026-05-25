"""Evaluation harness for the Pour Decisions RAG pipeline and agent.

This package provides tools for running automated evaluations against a curated golden
dataset of wine Q&A pairs. It supports two evaluation modes:

- ``retrieval``: Computes MRR and precision@k locally with zero LLM calls.
- ``full``: Adds Ragas-based metrics (faithfulness, answer relevancy, context precision,
  context recall) using the evaluator provider/model configured in ``app_config.yml``
  (defaults to ``model.provider``/``model.name`` when eval overrides are empty).

Usage::

    from src.eval import GoldenDataset, GoldenSample, EvalRunResult, EvalRunner
    from src.eval import validate_dataset

    dataset = GoldenDataset()
    samples = dataset.load("src/eval/wine_qa_golden.jsonl")
    filtered = dataset.filter(samples, categories=["rag_only"], difficulties=["easy"])

    report = validate_dataset()
    if not report.is_clean:
        logger.warning("Detected %d stale questions", report.stale_count)
"""

from .dataset import GoldenDataset
from .dataset_validator import ValidationReport, validate_dataset
from .models import EvalRunResult, GoldenSample, SampleResult
from .runner import EvalRunner
from .ragas_scorer import RagasScorer
from .reporter import EvalReporter
from .phoenix_reporter import PhoenixReporter

__all__ = [
    "GoldenDataset",
    "GoldenSample",
    "SampleResult",
    "EvalRunResult",
    "validate_dataset",
    "ValidationReport",
    "EvalRunner",
    "RagasScorer",
    "EvalReporter",
    "PhoenixReporter",
]
