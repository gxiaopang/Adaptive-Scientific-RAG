"""SQLAlchemy schema for valuable online and offline execution records."""

from datetime import datetime
from typing import Final
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_JSON_DOCUMENT: Final = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Metadata boundary for the application's PostgreSQL schema."""


class ConversationRow(Base):
    """One durable short-term conversation boundary."""

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ConversationTurnRow(Base):
    """One completed exchange, ordered within its conversation."""

    __tablename__ = "conversation_turns"
    __table_args__ = (
        CheckConstraint("sequence_no >= 1", name="ck_conversation_turns_sequence"),
        CheckConstraint(
            "response_mode IN ('scientific_evidence', 'general_chat')",
            name="ck_conversation_turns_response_mode",
        ),
        UniqueConstraint(
            "conversation_id",
            "sequence_no",
            name="uq_conversation_turns_sequence",
        ),
        UniqueConstraint("request_id", name="uq_conversation_turns_request_id"),
        Index("ix_conversation_turns_recent", "conversation_id", "sequence_no"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_message: Mapped[str] = mapped_column(Text, nullable=False)
    response_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryRunRow(Base):
    """One online adaptive RAG request and its correlation metadata."""

    __tablename__ = "query_runs"
    __table_args__ = (
        CheckConstraint("status IN ('succeeded', 'failed')", name="ck_query_runs_status"),
        CheckConstraint("retrieval_attempts >= 0", name="ck_query_runs_attempts"),
        CheckConstraint("latency_ms >= 0", name="ck_query_runs_latency"),
        Index("ix_query_runs_trace_created", "trace_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    final_query: Mapped[str | None] = mapped_column(Text)
    query_history: Mapped[list[str]] = mapped_column(_JSON_DOCUMENT, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    evidence_sufficient: Mapped[bool | None] = mapped_column(Boolean)
    verification_reason: Mapped[str | None] = mapped_column(Text)
    retrieval_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    termination_reason: Mapped[str | None] = mapped_column(String(32))
    evidence: Mapped[list[dict[str, object]]] = mapped_column(_JSON_DOCUMENT, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EvaluationRunRow(Base):
    """One offline retrieval experiment with reproducible configuration and results."""

    __tablename__ = "evaluation_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_evaluation_runs_status",
        ),
        Index("ix_evaluation_runs_strategy_created", "strategy", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_split: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    retrieval_config: Mapped[dict[str, object]] = mapped_column(_JSON_DOCUMENT, nullable=False)
    metrics: Mapped[dict[str, object] | None] = mapped_column(_JSON_DOCUMENT)
    latency: Mapped[dict[str, object] | None] = mapped_column(_JSON_DOCUMENT)
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EmbeddingIndexRow(Base):
    """Database-owned compatibility and lifecycle metadata for one dense index."""

    __tablename__ = "embedding_indexes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('building', 'ready', 'failed')",
            name="ck_embedding_indexes_status",
        ),
        CheckConstraint("dimension = 1024", name="ck_embedding_indexes_dimension"),
        CheckConstraint(
            "distance_metric = 'cosine'",
            name="ck_embedding_indexes_distance_metric",
        ),
        CheckConstraint(
            "expected_records >= 0 AND actual_records >= 0",
            name="ck_embedding_indexes_record_counts",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(16), nullable=False)
    source_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    doc_ids_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_records: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_records: Mapped[int] = mapped_column(Integer, nullable=False)
    configuration_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DocumentEmbeddingRow(Base):
    """One canonical document chunk and its 1024-dimensional online embedding."""

    __tablename__ = "document_embeddings"
    __table_args__ = (
        CheckConstraint("chunk_index >= 0", name="ck_document_embeddings_chunk_index"),
        UniqueConstraint(
            "index_id",
            "doc_id",
            "chunk_index",
            name="uq_document_embeddings_doc_chunk",
        ),
        Index("ix_document_embeddings_index_doc", "index_id", "doc_id"),
        Index(
            "ix_document_embeddings_embedding_cosine_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    index_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("embedding_indexes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    corpus_version: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    document_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata", _JSON_DOCUMENT, nullable=False
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
