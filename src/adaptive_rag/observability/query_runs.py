"""Storage-neutral online query-run records."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID, uuid4

from adaptive_rag.graph.state import AdaptiveRAGResult, TerminationReason

type QueryRunStatus = Literal["succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class QueryEvidenceRecord:
    doc_id: str
    title: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class QueryRunRecord:
    """One immutable online execution record ready for persistence."""

    id: UUID
    request_id: str
    trace_id: str
    status: QueryRunStatus
    original_query: str
    final_query: str | None
    query_history: tuple[str, ...]
    answer: str | None
    evidence_sufficient: bool | None
    verification_reason: str | None
    retrieval_attempts: int
    termination_reason: TerminationReason | None
    evidence: tuple[QueryEvidenceRecord, ...]
    latency_ms: float
    error_type: str | None
    created_at: datetime

    @classmethod
    def succeeded(
        cls,
        *,
        request_id: str,
        trace_id: str,
        result: AdaptiveRAGResult,
        latency_ms: float,
    ) -> "QueryRunRecord":
        return cls(
            id=uuid4(),
            request_id=request_id,
            trace_id=trace_id,
            status="succeeded",
            original_query=result.original_query,
            final_query=result.final_query,
            query_history=result.query_history,
            answer=result.answer,
            evidence_sufficient=result.evidence_sufficient,
            verification_reason=result.verification_reason,
            retrieval_attempts=result.retrieval_attempts,
            termination_reason=result.termination_reason,
            evidence=tuple(
                QueryEvidenceRecord(
                    doc_id=document.doc_id,
                    title=document.title,
                    score=document.score,
                    rank=document.rank,
                )
                for document in result.evidence
            ),
            latency_ms=latency_ms,
            error_type=None,
            created_at=datetime.now(UTC),
        )

    @classmethod
    def failed(
        cls,
        *,
        request_id: str,
        trace_id: str,
        original_query: str,
        latency_ms: float,
        error_type: str,
    ) -> "QueryRunRecord":
        return cls(
            id=uuid4(),
            request_id=request_id,
            trace_id=trace_id,
            status="failed",
            original_query=original_query,
            final_query=None,
            query_history=(original_query,),
            answer=None,
            evidence_sufficient=None,
            verification_reason=None,
            retrieval_attempts=0,
            termination_reason=None,
            evidence=(),
            latency_ms=latency_ms,
            error_type=error_type,
            created_at=datetime.now(UTC),
        )


class QueryRunRecorder(Protocol):
    """Non-critical persistence boundary used by the API layer."""

    def save(self, record: QueryRunRecord) -> None: ...
