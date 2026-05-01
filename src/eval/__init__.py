"""Evaluation harness for the Pour Decisions RAG pipeline and agent.

This package provides tools for running automated evaluations against a curated golden
dataset of wine Q&A pairs. It supports two evaluation modes:

- ``retrieval``: Computes MRR and precision@k locally with zero LLM calls.
- ``full``: Adds Ragas-based metrics (faithfulness, answer relevancy, context precision,
  context recall) using Gemini Flash as the evaluator LLM.

Usage::

    from src.eval import GoldenDataset, GoldenSample, EvalRunResult

    dataset = GoldenDataset()
    samples = dataset.load("tests/eval/wine_qa_golden.jsonl")
    filtered = dataset.filter(samples, categories=["rag_only"], difficulties=["easy"])
"""

from .dataset import GoldenDataset
from .models import EvalRunResult, GoldenSample, SampleResult

__all__ = [
    "GoldenDataset",
    "GoldenSample",
    "SampleResult",
    "EvalRunResult",
]

