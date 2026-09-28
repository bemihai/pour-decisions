"""Deterministic corrective-retrieval contracts and selection primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from threading import Lock
from typing import Any, Literal, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf

from .query_analyzer import RetrievalQueryPlan


MAX_CORRECTION_ATTEMPTS_PER_REQUEST = 1
DEFAULT_CORRECTION_TIMEOUT_SECONDS = 3.0

CorrectionTriggerReason = Literal["aging_classifications", "dated_classification"]
CorrectionStatus = Literal[
    "disabled",
    "ineligible",
    "budget_exhausted",
    "completed",
    "timed_out",
    "failed",
]
CorrectionSelectedResult = Literal["first_pass", "corrected"]
CorrectionSelectionReason = Literal[
    "correction_disabled",
    "trigger_not_matched",
    "empty_first_pass",
    "budget_exhausted",
    "corrected_novel_chunks",
    "corrected_empty",
    "no_novel_chunks",
    "correction_failed",
    "correction_timed_out",
]
CorrectionFailureReason = Literal["retrieval_error", "rerank_error", "timeout"]


@dataclass(frozen=True)
class CorrectionConfig:
    """Validated settings for the bounded deterministic correction path."""

    enabled: bool = False
    timeout_seconds: float = DEFAULT_CORRECTION_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CorrectionQuery:
    """One bounded alternate query derived without an LLM."""

    template_id: str
    query: str
    sha256: str
    trigger_reason: CorrectionTriggerReason


@dataclass(frozen=True)
class CorrectionSelection:
    """Deterministic choice between valid first-pass and corrected documents."""

    documents: list[dict[str, Any]]
    selected_result: CorrectionSelectedResult
    selection_reason: CorrectionSelectionReason
    novel_corrected_chunk_count: int


@dataclass(frozen=True)
class RAGCorrectionDiagnostic:
    """Bounded correction state owned by one production RAG result."""

    enabled: bool = False
    eligible: bool = False
    trigger_reason: CorrectionTriggerReason | None = None
    attempt_reserved: bool = False
    attempt_count: int = 0
    mode: Literal["deterministic"] | None = None
    alternate_query_id: str | None = None
    alternate_query_sha256: str | None = None
    status: CorrectionStatus = "disabled"
    selected_result: CorrectionSelectedResult = "first_pass"
    selection_reason: CorrectionSelectionReason = "correction_disabled"
    first_pass_chunk_count: int = 0
    corrected_chunk_count: int = 0
    novel_corrected_chunk_count: int = 0
    added_latency_ms: float = 0.0
    model_attempts: int = 0
    failure_reason: CorrectionFailureReason | None = None

    def __post_init__(self) -> None:
        """Reject states that violate the approved bounded contract."""
        if self.attempt_count not in {0, 1}:
            raise ValueError("correction attempt_count must be 0 or 1")
        if self.model_attempts != 0:
            raise ValueError("deterministic correction model_attempts must be 0")
        counts = (
            self.first_pass_chunk_count,
            self.corrected_chunk_count,
            self.novel_corrected_chunk_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("correction chunk counts must be non-negative")
        if self.added_latency_ms < 0:
            raise ValueError("correction added_latency_ms must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic."""
        return asdict(self)


@dataclass
class CorrectionAttemptBudget:
    """Thread-safe request-local reservation with a hard one-attempt limit."""

    _attempt_count: int = 0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @property
    def attempt_count(self) -> int:
        """Return the number of successful reservations."""
        with self._lock:
            return self._attempt_count

    def reserve(self) -> bool:
        """Reserve the request's only correction attempt if still available."""
        with self._lock:
            if self._attempt_count >= MAX_CORRECTION_ATTEMPTS_PER_REQUEST:
                return False
            self._attempt_count += 1
            return True


def load_correction_config(config: DictConfig) -> CorrectionConfig:
    """Load and validate the approved disabled-by-default correction settings."""
    enabled = OmegaConf.select(config, "chroma.retrieval.correction.enabled", default=False)
    timeout_seconds = OmegaConf.select(
        config,
        "chroma.retrieval.correction.timeout_seconds",
        default=DEFAULT_CORRECTION_TIMEOUT_SECONDS,
    )
    if type(enabled) is not bool:
        raise ValueError("chroma.retrieval.correction.enabled must be a boolean")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("chroma.retrieval.correction.timeout_seconds must be numeric")
    if float(timeout_seconds) <= 0:
        raise ValueError("chroma.retrieval.correction.timeout_seconds must be greater than zero")
    return CorrectionConfig(enabled=enabled, timeout_seconds=float(timeout_seconds))


def correction_trigger_reason(query_plan: RetrievalQueryPlan) -> CorrectionTriggerReason | None:
    """Return the approved trigger reason for one deterministic query plan."""
    original = query_plan.original_query
    plural_classification = re.search(r"\bclassifications\b", original, re.IGNORECASE) is not None
    dated_classification = (
        re.search(r"\b\d{4}\b", original) is not None
        and re.search(r"\bclassification\b", original, re.IGNORECASE) is not None
    )
    if query_plan.intent == "aging" and plural_classification:
        return "aging_classifications"
    if query_plan.intent == "classification" and dated_classification:
        return "dated_classification"
    return None


def build_correction_query(query_plan: RetrievalQueryPlan) -> CorrectionQuery | None:
    """Build the approved alternate query for an eligible deterministic plan."""
    trigger_reason = correction_trigger_reason(query_plan)
    if trigger_reason is None:
        return None

    original = query_plan.original_query
    numeric_anchors = re.findall(r"\b\d{4}\b", original)
    if trigger_reason == "aging_classifications":
        anchors = list(query_plan.regions) + numeric_anchors
        terms = ["aging", "classification", "categories", "minimum", "requirements", "oak", "bottle"]
        template_id = "aging_classifications_v1"
    else:
        stop_words = {"how", "does", "the", "rank", "classification"}
        anchors = [
            "château" if token.casefold() == "châteaux" else token
            for token in re.findall(r"[\wÀ-ÿ]+", original, re.UNICODE)
            if token.casefold() not in stop_words
        ]
        terms = ["classification", "ranking", "tiers", "growths"]
        template_id = "dated_classification_v1"

    query = " ".join(dict.fromkeys(anchors + terms))
    return CorrectionQuery(
        template_id=template_id,
        query=query,
        sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        trigger_reason=trigger_reason,
    )


def select_correction_result(
    first_pass: Sequence[Mapping[str, Any]],
    corrected: Sequence[Mapping[str, Any]],
    *,
    correction_succeeded: bool,
) -> CorrectionSelection:
    """Apply the approved novelty rule without comparing cross-query scores."""
    first_documents = [dict(document) for document in first_pass]
    corrected_documents = [dict(document) for document in corrected]
    if not correction_succeeded:
        return CorrectionSelection(first_documents, "first_pass", "correction_failed", 0)
    if not corrected_documents:
        return CorrectionSelection(first_documents, "first_pass", "corrected_empty", 0)

    first_ids = {_document_id(document) for document in first_documents}
    novel_count = sum(
        1 for document in corrected_documents if _document_id(document) not in first_ids
    )
    if novel_count == 0:
        return CorrectionSelection(first_documents, "first_pass", "no_novel_chunks", 0)
    return CorrectionSelection(
        corrected_documents,
        "corrected",
        "corrected_novel_chunks",
        novel_count,
    )


def correction_trace_attributes(diagnostic: RAGCorrectionDiagnostic) -> dict[str, Any]:
    """Return bounded low-cardinality attributes safe for tracing."""
    attributes: dict[str, Any] = {
        "correction_enabled": diagnostic.enabled,
        "correction_eligible": diagnostic.eligible,
        "correction_attempt_reserved": diagnostic.attempt_reserved,
        "correction_attempt_count": diagnostic.attempt_count,
        "correction_status": diagnostic.status,
        "correction_selected_result": diagnostic.selected_result,
        "correction_selection_reason": diagnostic.selection_reason,
        "correction_first_pass_chunk_count": diagnostic.first_pass_chunk_count,
        "correction_corrected_chunk_count": diagnostic.corrected_chunk_count,
        "correction_novel_chunk_count": diagnostic.novel_corrected_chunk_count,
        "correction_added_latency_ms": diagnostic.added_latency_ms,
        "correction_model_attempts": diagnostic.model_attempts,
    }
    optional_values = {
        "correction_trigger_reason": diagnostic.trigger_reason,
        "correction_mode": diagnostic.mode,
        "correction_alternate_query_id": diagnostic.alternate_query_id,
        "correction_failure_reason": diagnostic.failure_reason,
    }
    attributes.update({key: value for key, value in optional_values.items() if value is not None})
    return attributes


def _document_id(document: Mapping[str, Any]) -> str:
    """Return a stable chunk identifier for novelty comparison."""
    return str(document.get("id", ""))
