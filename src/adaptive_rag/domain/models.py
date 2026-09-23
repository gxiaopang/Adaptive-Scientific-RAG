"""Small immutable values shared across retrieval and storage implementations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

type QueryRoute = Literal["scientific_evidence", "general_chat"]


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """One ranked retrieval result using the original corpus document ID."""

    doc_id: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class DenseVectorRecord:
    """One canonical SciFact document plus its dense vector."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    text: str
    source: str
    corpus_version: str
    content_hash: str
    metadata: dict[str, object]
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    """A storage-neutral dense search result."""

    doc_id: str
    score: float


@dataclass(frozen=True, slots=True)
class EmbeddingIndexSpec:
    """Compatibility identity and expected contents of one dense index."""

    name: str
    provider: str
    model: str
    dimension: int
    distance_metric: Literal["cosine"]
    source_checksum: str
    doc_ids_checksum: str
    expected_records: int
    configuration_version: str


@dataclass(frozen=True, slots=True)
class EmbeddingIndexSnapshot:
    """Safe metadata snapshot used by readiness and evaluation reports."""

    name: str
    status: Literal["building", "ready", "failed"]
    provider: str
    model: str
    dimension: int
    distance_metric: Literal["cosine"]
    source_checksum: str
    doc_ids_checksum: str
    expected_records: int
    actual_records: int
    configuration_version: str
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    """One retrieved document projected into the adaptive workflow."""

    doc_id: str
    title: str
    text: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Structured evidence-sufficiency decision used for graph routing."""

    sufficient: bool
    reason: str


@dataclass(frozen=True, slots=True)
class QueryRoutingDecision:
    """Structured entry decision plus a history-resolved standalone request."""

    route: QueryRoute
    reason: str
    standalone_query: str


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One completed user/assistant exchange available as bounded context."""

    sequence_no: int
    user_message: str
    assistant_message: str
    response_mode: QueryRoute
