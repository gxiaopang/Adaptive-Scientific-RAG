"""Create query, evaluation, and conversation tables.

Revision ID: 20260831_0001
Revises: None
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260831_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "conversation_turns",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("assistant_message", sa.Text(), nullable=False),
        sa.Column("response_mode", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("sequence_no >= 1", name="ck_conversation_turns_sequence"),
        sa.CheckConstraint(
            "response_mode IN ('scientific_evidence', 'general_chat')",
            name="ck_conversation_turns_response_mode",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id", name="uq_conversation_turns_request_id"),
        sa.UniqueConstraint(
            "conversation_id",
            "sequence_no",
            name="uq_conversation_turns_sequence",
        ),
    )
    op.create_index(
        "ix_conversation_turns_recent",
        "conversation_turns",
        ["conversation_id", "sequence_no"],
    )
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("dataset_split", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("retrieval_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("latency", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("artifact_path", sa.String(length=512), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_evaluation_runs_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evaluation_runs_strategy_created",
        "evaluation_runs",
        ["strategy", "created_at"],
    )
    op.create_table(
        "query_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("final_query", sa.Text(), nullable=True),
        sa.Column("query_history", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("evidence_sufficient", sa.Boolean(), nullable=True),
        sa.Column("verification_reason", sa.Text(), nullable=True),
        sa.Column("retrieval_attempts", sa.Integer(), nullable=False),
        sa.Column("termination_reason", sa.String(length=32), nullable=True),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("latency_ms >= 0", name="ck_query_runs_latency"),
        sa.CheckConstraint("retrieval_attempts >= 0", name="ck_query_runs_attempts"),
        sa.CheckConstraint("status IN ('succeeded', 'failed')", name="ck_query_runs_status"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_runs_request_id", "query_runs", ["request_id"])
    op.create_index(
        "ix_query_runs_trace_created",
        "query_runs",
        ["trace_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_query_runs_trace_created", table_name="query_runs")
    op.drop_index("ix_query_runs_request_id", table_name="query_runs")
    op.drop_table("query_runs")
    op.drop_index("ix_evaluation_runs_strategy_created", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
    op.drop_index("ix_conversation_turns_recent", table_name="conversation_turns")
    op.drop_table("conversation_turns")
    op.drop_table("conversations")
