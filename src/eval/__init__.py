from typing import TYPE_CHECKING, Any

from .dataset import filter_golden_samples, load_golden_dataset
from .models import (
    AgentToolOutput,
    EvalRunResult,
    GoldenSample,
    MetricCoverage,
    MetricOutcome,
    MetricSupportCounts,
    SampleResult,
)
from .reporter import EvalReporter
from .runner import EvalRunner

if TYPE_CHECKING:
    from .phoenix_reporter import PhoenixReporter
    from .ragas_scorer import RagasScorer
    from .scripts.dataset_validator import ValidationReport

__all__ = [
    "load_golden_dataset",
    "filter_golden_samples",
    "GoldenSample",
    "AgentToolOutput",
    "MetricOutcome",
    "MetricSupportCounts",
    "MetricCoverage",
    "SampleResult",
    "EvalRunResult",
    "validate_dataset",
    "ValidationReport",
    "EvalRunner",
    "RagasScorer",
    "EvalReporter",
    "PhoenixReporter",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional integrations and standalone script exports."""
    if name == "RagasScorer":
        from .ragas_scorer import RagasScorer

        return RagasScorer
    if name == "PhoenixReporter":
        from .phoenix_reporter import PhoenixReporter

        return PhoenixReporter
    if name in {"validate_dataset", "ValidationReport"}:
        from .scripts.dataset_validator import ValidationReport, validate_dataset

        return {
            "validate_dataset": validate_dataset,
            "ValidationReport": ValidationReport,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
