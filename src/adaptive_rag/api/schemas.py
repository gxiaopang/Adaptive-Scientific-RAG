"""Validated public request and response models."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from adaptive_rag.domain.models import QueryRoute
from adaptive_rag.graph.state import TerminationReason

QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]


class ApiModel(BaseModel):
    """Base contract that rejects silently ignored client fields."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    """Liveness response; it deliberately does not claim dependency readiness."""

    status: Literal["ok"] = "ok"
    service: str
    version: str


class ReadinessResponse(ApiModel):
    """Dependency-readiness response used by deployment health checks."""

    status: Literal["ready"] = "ready"
    service: str
    version: str


class QueryRequest(ApiModel):
    """One natural-language question submitted to the adaptive workflow."""

    query: QueryText
    conversation_id: UUID | None = None


class EvidenceResponse(ApiModel):
    """Compact source metadata returned with an answer."""

    doc_id: str
    title: str
    excerpt: str = Field(max_length=500)
    score: float
    rank: int = Field(ge=1)


class QueryResponse(ApiModel):
    """Stable HTTP projection of the internal workflow result."""

    conversation_id: UUID
    original_query: str
    response_mode: QueryRoute
    routing_reason: str
    final_query: str
    query_history: tuple[str, ...]
    answer: str
    evidence_sufficient: bool | None
    verification_reason: str | None
    retrieval_attempts: int = Field(ge=0)
    termination_reason: TerminationReason
    evidence: tuple[EvidenceResponse, ...]
