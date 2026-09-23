"""PostgreSQL migration and repository integration tests."""

import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from adaptive_rag.domain.models import DenseVectorRecord, EmbeddingIndexSpec
from adaptive_rag.storage.postgres.models import ConversationRow
from adaptive_rag.storage.postgres.repository import SQLAlchemyConversationRepository
from adaptive_rag.storage.postgres.vector_store import PgVectorDenseStore, PgVectorStoreError


@pytest.mark.integration
def test_alembic_round_trip_and_conversation_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = os.getenv("ASR_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("ASR_TEST_POSTGRES_DSN is not configured")
    monkeypatch.setenv("ASR_POSTGRES_DSN", dsn)
    alembic_config = Config("alembic.ini")

    command.upgrade(alembic_config, "head")
    engine = create_engine(dsn)
    expected_tables = {
        "alembic_version",
        "conversations",
        "conversation_turns",
        "evaluation_runs",
        "query_runs",
        "embedding_indexes",
        "document_embeddings",
    }
    assert set(inspect(engine).get_table_names()) == expected_tables
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        ).scalar_one()
    index_names = {
        item["name"] for item in inspect(engine).get_indexes("document_embeddings")
    }
    assert "ix_document_embeddings_embedding_cosine_hnsw" in index_names
    command.check(alembic_config)

    command.downgrade(alembic_config, "base")
    application_tables = expected_tables - {"alembic_version"}
    assert application_tables.isdisjoint(inspect(engine).get_table_names())
    command.upgrade(alembic_config, "head")

    sessions = sessionmaker(engine, class_=Session, expire_on_commit=False)
    repository = SQLAlchemyConversationRepository(sessions)
    conversation_id = uuid4()
    persisted_id = repository.append_exchange(
        conversation_id=conversation_id,
        request_id=f"integration-{uuid4().hex}",
        user_message="What does the evidence show?",
        assistant_message="The evidence supports the claim.",
        response_mode="scientific_evidence",
        create_conversation=True,
    )

    history = repository.load(conversation_id, 8)

    assert persisted_id == conversation_id
    assert history is not None
    assert [turn.user_message for turn in history] == ["What does the evidence show?"]

    dense_store = PgVectorDenseStore(
        sessions,
        index_name="integration_dense",
        dimension=1024,
    )
    spec = EmbeddingIndexSpec(
        name="integration_dense",
        provider="integration-provider",
        model="integration-model",
        dimension=1024,
        distance_metric="cosine",
        source_checksum="a" * 64,
        doc_ids_checksum="b" * 64,
        expected_records=2,
        configuration_version="1",
    )
    assert dense_store.prepare(spec) is False
    records = [
        DenseVectorRecord(
            chunk_id="doc-1:0",
            doc_id="doc-1",
            chunk_index=0,
            title="First",
            text="First text",
            source="integration",
            corpus_version="test",
            content_hash="c" * 64,
            metadata={},
            embedding=(1.0, *([0.0] * 1023)),
        ),
        DenseVectorRecord(
            chunk_id="doc-2:0",
            doc_id="doc-2",
            chunk_index=0,
            title="Second",
            text="Second text",
            source="integration",
            corpus_version="test",
            content_hash="d" * 64,
            metadata={},
            embedding=(0.0, 1.0, *([0.0] * 1022)),
        ),
    ]
    dense_store.upsert(records)
    dense_store.upsert(records)
    assert dense_store.count() == 2
    assert dense_store.finalize(spec).status == "ready"
    assert [hit.doc_id for hit in dense_store.search(records[0].embedding, 2)] == [
        "doc-1",
        "doc-2",
    ]
    assert dense_store.prepare(spec) is True
    incompatible = EmbeddingIndexSpec(
        name=spec.name,
        provider=spec.provider,
        model="different-model",
        dimension=spec.dimension,
        distance_metric=spec.distance_metric,
        source_checksum=spec.source_checksum,
        doc_ids_checksum=spec.doc_ids_checksum,
        expected_records=spec.expected_records,
        configuration_version=spec.configuration_version,
    )
    with pytest.raises(PgVectorStoreError, match="incompatible"):
        dense_store.prepare(incompatible)

    with sessions.begin() as session:
        conversation = session.get(ConversationRow, conversation_id)
        assert conversation is not None
        session.delete(conversation)
        index = session.execute(
            text("DELETE FROM embedding_indexes WHERE name = 'integration_dense'")
        )
        assert index.rowcount == 1
    engine.dispose()
