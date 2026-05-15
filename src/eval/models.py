"""Pydantic data models for the eval harness.

Defines the schema for golden dataset entries, per-sample evaluation results, and
the aggregated run result written to disk after each eval run.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Valid values for constrained string fields.
CATEGORIES = frozenset({"rag_only", "cellar", "pairing", "multi_hop"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})


class GoldenSample(BaseModel):
    """A single entry in the golden Q&A dataset.

    Each sample represents one question that is sent to the system under test along
    with the metadata needed to score the system's response.

    Attributes:
        id: Unique identifier in the format ``{category}_{NNN}``.
        question: The exact question string passed to the system.
        category: Primary evaluation category. One of ``rag_only``, ``cellar``,
            ``pairing``, or ``multi_hop``.
        difficulty: Perceived difficulty level. One of ``easy``, ``medium``, ``hard``.
        expected_facts: Human-readable facts the answer must contain. Used as input
            to LLM-as-judge prompts.
        expected_tool_calls: Tool names expected to be invoked when using the agent
            backend. Optional — only meaningful for ``cellar`` and ``multi_hop``
            categories.
        ground_truth: Reference answer used for ``context_recall`` and LLM-as-judge
            scoring. Should be a complete, factual sentence.
        ground_truth_chunk_ids: ChromaDB chunk IDs known to contain the information
            needed to answer the question. Used for MRR and precision@k. Leave empty
            when the answer does not come from the RAG index (e.g., cellar queries).
        tags: Multi-labels for result slicing (e.g., ``cellar``, ``tool_required``).
        notes: Optional human notes about special conditions or skip criteria.
    """

    id: str = Field(..., description="Unique identifier, format {category}_{NNN}")
    question: str = Field(..., description="The exact question passed to the system")
    category: str = Field(..., description="Primary category: rag_only, cellar, pairing, multi_hop")
    difficulty: str = Field(..., description="easy, medium, or hard")
    expected_facts: list[str] = Field(
        ..., description="Human-readable facts the answer must contain"
    )
    expected_tool_calls: list[str] = Field(
        default_factory=list,
        description="Tool names expected to be invoked (agent backend only)",
    )
    ground_truth: str = Field(
        ..., description="Reference answer for context_recall and LLM-as-judge scoring"
    )
    ground_truth_chunk_ids: list[str] = Field(
        default_factory=list,
        description="ChromaDB chunk IDs known to contain the answer (for retrieval metrics)",
    )
    tags: list[str] = Field(..., description="Multi-labels for result slicing")
    notes: str | None = Field(None, description="Human notes about special conditions or skip criteria")

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        """Validate that category is one of the allowed values.

        Args:
            value: The category string to validate.

        Returns:
            The validated category string.

        Raises:
            ValueError: If the category is not in the allowed set.
        """
        if value not in CATEGORIES:
            raise ValueError(f"category must be one of {sorted(CATEGORIES)}, got {value!r}")
        return value

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, value: str) -> str:
        """Validate that difficulty is one of the allowed values.

        Args:
            value: The difficulty string to validate.

        Returns:
            The validated difficulty string.

        Raises:
            ValueError: If the difficulty is not in the allowed set.
        """
        if value not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}, got {value!r}")
        return value


class SampleResult(BaseModel):
    """The system's output and computed scores for a single golden sample.

    Attributes:
        id: Matches the ``id`` from the corresponding :class:`GoldenSample`.
        question: The question that was asked.
        answer: The answer produced by the system under test.
        ground_truth: Optional reference answer associated with the sample.
        contexts: List of text chunks retrieved by the RAG pipeline.
        retrieved_chunk_ids: IDs of the retrieved chunks (used for retrieval metrics).
        tool_calls_made: Names of tools invoked during the run (agent backend only).
        latency_ms: Wall-clock time for the pipeline call in milliseconds.
        error: Error message if the run failed; ``None`` on success.
        scores: Metric name → score mapping, populated by the scorer components.
    """

    id: str = Field(..., description="Matches the GoldenSample id")
    question: str = Field(..., description="The question that was asked")
    answer: str = Field(default="", description="The answer produced by the system")
    ground_truth: str | None = Field(default=None, description="Optional reference answer")
    contexts: list[str] = Field(default_factory=list, description="Retrieved text chunks")
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="IDs of retrieved chunks"
    )
    tool_calls_made: list[str] = Field(
        default_factory=list, description="Tool names invoked (agent backend only)"
    )
    latency_ms: float = Field(default=0.0, description="Wall-clock time in milliseconds")
    error: str | None = Field(None, description="Error message if the run failed")
    scores: dict[str, float] = Field(default_factory=dict, description="Metric name to score")


class EvalRunResult(BaseModel):
    """Aggregated result of a complete eval run written to disk as JSON.

    Attributes:
        run_id: ISO-format timestamp string used as a unique run identifier.
        timestamp: Full ISO 8601 timestamp of when the run started.
        mode: Eval mode — ``retrieval`` (no Ragas) or ``full`` (with Ragas).
        backend: System under test — ``rag`` (pipeline only) or ``agent`` (full agent).
        git_sha: Short git commit hash at the time of the run.
        config_snapshot: Snapshot of relevant ``app_config.yml`` settings.
        aggregate_metrics: Mean score per metric across all evaluated samples.
        metrics_by_category: Per-category breakdown of aggregate scores.
        per_sample: Individual :class:`SampleResult` for each sample.
        summary: High-level run statistics (counts, total latency, LLM calls).
    """

    run_id: str = Field(..., description="ISO-format timestamp used as run identifier")
    timestamp: str = Field(..., description="Full ISO 8601 timestamp of run start")
    mode: Literal["retrieval", "full"] = Field(..., description="retrieval or full")
    backend: Literal["rag", "agent"] = Field(..., description="rag or agent")
    git_sha: str = Field(default="unknown", description="Short git commit hash")
    config_snapshot: dict = Field(default_factory=dict, description="Relevant config settings")
    aggregate_metrics: dict[str, float] = Field(
        default_factory=dict, description="Mean score per metric"
    )
    metrics_by_category: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Per-category metric breakdown"
    )
    per_sample: list[SampleResult] = Field(
        default_factory=list, description="Individual sample results"
    )
    summary: dict = Field(default_factory=dict, description="Run statistics")

