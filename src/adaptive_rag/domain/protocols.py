"""Interfaces implemented by concrete retrieval, embedding, and storage adapters."""

from collections.abc import Callable, Sequence
from typing import Protocol
from uuid import UUID

from adaptive_rag.domain.models import (
    ConversationTurn,
    DenseSearchHit,
    DenseVectorRecord,
    EmbeddingIndexSnapshot,
    EmbeddingIndexSpec,
    EvidenceAssessment,
    EvidenceDocument,
    QueryRoute,
    QueryRoutingDecision,
    ScoredDocument,
)


class ProviderInvocationError(RuntimeError):
    """Raised when an external provider cannot satisfy a protocol call."""


class Retriever(Protocol):
    """Synchronous retrieval boundary shared by offline evaluation strategies."""

    @property
    def strategy(self) -> str: ...

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]: ...


class PairwiseReranker(Protocol):
    """Query-document scoring boundary used after first-stage retrieval."""

    @property
    def model_name(self) -> str: ...

    @property
    def provider(self) -> str: ...

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class AnswerGenerator(Protocol):
    """Provider-neutral answer generation boundary."""

    def generate(
        self,
        query: str,
        context: str,
        on_delta: Callable[[str], None] | None = None,
    ) -> str: ...


class QueryRouter(Protocol):
    """Provider-neutral boundary deciding whether a query needs scientific evidence."""

    def classify(
        self,
        query: str,
        history: Sequence[ConversationTurn],
    ) -> QueryRoutingDecision: ...


class GeneralChatGenerator(Protocol):
    """Provider-neutral answer boundary for queries that do not need retrieval."""

    def chat(
        self,
        query: str,
        history: Sequence[ConversationTurn],
        on_delta: Callable[[str], None] | None = None,
    ) -> str: ...


class ConversationRepository(Protocol):
    """Storage boundary for ordered, completed conversation exchanges."""

    def load(self, conversation_id: UUID, limit: int) -> tuple[ConversationTurn, ...] | None: ...

    def append_exchange(
        self,
        *,
        conversation_id: UUID,
        request_id: str,
        user_message: str,
        assistant_message: str,
        response_mode: QueryRoute,
        create_conversation: bool,
    ) -> UUID: ...


class EvidenceVerifier(Protocol):
    """Provider-neutral answerability decision boundary."""

    def verify(
        self,
        query: str,
        answer: str,
        context: str,
        evidence: Sequence[EvidenceDocument],
    ) -> EvidenceAssessment: ...


class QueryRewriter(Protocol):
    """Provider-neutral query rewrite boundary for bounded re-retrieval."""

    def rewrite(
        self,
        *,
        original_query: str,
        current_query: str,
        context: str,
        assessment: EvidenceAssessment,
        retrieval_attempt: int,
    ) -> str: ...


class EmbeddingProvider(Protocol):
    """Query/document encoder boundary used by dense retrieval."""

    @property
    def model_name(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class DenseVectorStore(Protocol):
    """PostgreSQL vector-store operations used by indexing and retrieval."""

    @property
    def index_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def prepare(self, spec: EmbeddingIndexSpec) -> bool: ...

    def existing_content_hashes(self) -> dict[str, str]: ...

    def upsert(self, records: Sequence[DenseVectorRecord]) -> None: ...

    def delete_missing(self, chunk_ids: Sequence[str]) -> None: ...

    def finalize(self, spec: EmbeddingIndexSpec) -> EmbeddingIndexSnapshot: ...

    def snapshot(self) -> EmbeddingIndexSnapshot: ...

    def count(self) -> int: ...

    def search(self, vector: Sequence[float], top_k: int) -> list[DenseSearchHit]: ...
