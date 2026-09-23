"""Typed state and result contracts for the adaptive workflow."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from adaptive_rag.domain.models import (
    ConversationTurn,
    EvidenceAssessment,
    EvidenceDocument,
    QueryRoute,
    QueryRoutingDecision,
)

type TerminationReason = Literal[
    "evidence_sufficient",
    "max_retrieval_attempts",
    "general_chat",
]


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    """Private evaluation trace; never included in the HTTP response or access logs."""

    query: str
    retrieval_query: str
    context: str
    answer: str
    evidence: tuple[EvidenceDocument, ...]
    attempt: int


class AdaptiveRAGState(TypedDict):
    """Complete state passed between LangGraph nodes."""

    original_query: str
    snapshots: tuple[GenerationSnapshot, ...]
    resolved_query: str
    conversation_history: tuple[ConversationTurn, ...]
    routing_decision: QueryRoutingDecision | None
    current_query: str
    query_history: tuple[str, ...]
    evidence: tuple[EvidenceDocument, ...]
    context: str
    answer: str
    assessment: EvidenceAssessment | None
    retrieval_attempts: int
    termination_reason: TerminationReason | None
    on_answer_start: Callable[[], None] | None
    on_answer_delta: Callable[[str], None] | None


class AdaptiveRAGUpdate(TypedDict, total=False):
    """Partial state update returned by one graph node."""

    routing_decision: QueryRoutingDecision | None
    snapshots: tuple[GenerationSnapshot, ...]
    resolved_query: str
    current_query: str
    query_history: tuple[str, ...]
    evidence: tuple[EvidenceDocument, ...]
    context: str
    answer: str
    assessment: EvidenceAssessment | None
    retrieval_attempts: int
    termination_reason: TerminationReason | None


@dataclass(frozen=True, slots=True)
class AdaptiveRAGResult:
    """Stable public result projected from final internal graph state."""

    original_query: str
    response_mode: QueryRoute
    routing_reason: str
    final_query: str
    query_history: tuple[str, ...]
    evidence: tuple[EvidenceDocument, ...]
    answer: str
    evidence_sufficient: bool | None
    verification_reason: str | None
    retrieval_attempts: int
    termination_reason: TerminationReason
    snapshots: tuple[GenerationSnapshot, ...] = ()
