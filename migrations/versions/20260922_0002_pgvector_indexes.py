"""Add pgvector-backed dense embedding indexes.

Revision ID: 20260922_0002
Revises: 20260831_0001
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260922_0002"
down_revision: str | None = "20260831_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "embedding_indexes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("distance_metric", sa.String(length=16), nullable=False),
        sa.Column("source_checksum", sa.String(length=64), nullable=False),
        sa.Column("doc_ids_checksum", sa.String(length=64), nullable=False),
        sa.Column("expected_records", sa.Integer(), nullable=False),
        sa.Column("actual_records", sa.Integer(), nullable=False),
        sa.Column("configuration_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "distance_metric = 'cosine'", name="ck_embedding_indexes_distance_metric"
        ),
        sa.CheckConstraint("dimension = 1024", name="ck_embedding_indexes_dimension"),
        sa.CheckConstraint(
            "expected_records >= 0 AND actual_records >= 0",
            name="ck_embedding_indexes_record_counts",
        ),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'failed')",
            name="ck_embedding_indexes_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "document_embeddings",
        sa.Column("index_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.String(length=256), nullable=False),
        sa.Column("doc_id", sa.String(length=128), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("corpus_version", sa.String(length=64), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=1024), nullable=False),
        sa.CheckConstraint(
            "chunk_index >= 0", name="ck_document_embeddings_chunk_index"
        ),
        sa.ForeignKeyConstraint(
            ["index_id"], ["embedding_indexes.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("index_id", "chunk_id"),
        sa.UniqueConstraint(
            "index_id", "doc_id", "chunk_index", name="uq_document_embeddings_doc_chunk"
        ),
    )
    op.create_index(
        "ix_document_embeddings_index_doc",
        "document_embeddings",
        ["index_id", "doc_id"],
    )
    op.create_index(
        "ix_document_embeddings_embedding_cosine_hnsw",
        "document_embeddings",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_embeddings_embedding_cosine_hnsw",
        table_name="document_embeddings",
        postgresql_using="hnsw",
    )
    op.drop_index("ix_document_embeddings_index_doc", table_name="document_embeddings")
    op.drop_table("document_embeddings")
    op.drop_table("embedding_indexes")
