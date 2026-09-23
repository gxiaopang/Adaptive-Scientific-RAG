from dataclasses import replace
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateTable

from adaptive_rag.observability.query_runs import QueryRunRecord
from adaptive_rag.storage.postgres.models import (
    Base,
    ConversationRow,
    ConversationTurnRow,
    EvaluationRunRow,
    QueryRunRow,
)
from adaptive_rag.storage.postgres.repository import (
    ConversationPersistenceError,
    QueryRunPersistenceError,
    SQLAlchemyConversationRepository,
    SQLAlchemyQueryRunRepository,
)


def test_postgresql_schema_uses_uuid_and_jsonb_for_structured_data() -> None:
    query_ddl = str(
        CreateTable(QueryRunRow.__table__).compile(dialect=postgresql.dialect())
    )
    evaluation_ddl = str(
        CreateTable(EvaluationRunRow.__table__).compile(dialect=postgresql.dialect())
    )
    conversation_ddl = str(
        CreateTable(ConversationTurnRow.__table__).compile(dialect=postgresql.dialect())
    )

    assert "UUID" in query_ddl
    assert query_ddl.count("JSONB") == 2
    assert evaluation_ddl.count("JSONB") == 3
    assert "ck_query_runs_status" in query_ddl
    assert "ck_evaluation_runs_status" in evaluation_ddl
    assert "UUID" in conversation_ddl
    assert "uq_conversation_turns_sequence" in conversation_ddl
    assert "uq_conversation_turns_request_id" in conversation_ddl
    assert "FOREIGN KEY(conversation_id) REFERENCES conversations" in conversation_ddl


def test_repository_commits_query_run_with_real_session_transaction() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    repository = SQLAlchemyQueryRunRepository(session_factory)
    record = QueryRunRecord.failed(
        request_id="request-1",
        trace_id="trace-1",
        original_query="scientific query",
        latency_ms=4.5,
        error_type="WorkflowError",
    )

    repository.save(record)

    with session_factory() as session:
        stored = session.scalar(select(QueryRunRow))

    assert stored is not None
    assert stored.id == record.id
    assert stored.request_id == "request-1"
    assert stored.status == "failed"
    assert stored.query_history == ["scientific query"]
    assert stored.evidence == []
    assert stored.latency_ms == 4.5


def test_repository_rolls_back_constraint_failure() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    repository = SQLAlchemyQueryRunRepository(session_factory)
    record = QueryRunRecord.failed(
        request_id="request-1",
        trace_id="trace-1",
        original_query="scientific query",
        latency_ms=1.0,
        error_type="WorkflowError",
    )

    with pytest.raises(QueryRunPersistenceError, match=str(record.id)):
        repository.save(replace(record, latency_ms=-1.0))

    with session_factory() as session:
        assert session.scalar(select(QueryRunRow)) is None


def test_conversation_repository_appends_loads_and_limits_ordered_turns() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    repository = SQLAlchemyConversationRepository(session_factory)
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    repository.append_exchange(
        conversation_id=conversation_id,
        request_id="request-1",
        user_message="first question",
        assistant_message="first answer",
        response_mode="scientific_evidence",
        create_conversation=True,
    )
    repository.append_exchange(
        conversation_id=conversation_id,
        request_id="request-2",
        user_message="second question",
        assistant_message="second answer",
        response_mode="general_chat",
        create_conversation=False,
    )

    history = repository.load(conversation_id, 1)

    assert history is not None
    assert len(history) == 1
    assert history[0].sequence_no == 2
    assert history[0].user_message == "second question"
    assert history[0].response_mode == "general_chat"


def test_conversation_repository_is_idempotent_by_request_id() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    repository = SQLAlchemyConversationRepository(session_factory)
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")
    values = {
        "conversation_id": conversation_id,
        "request_id": "request-1",
        "user_message": "question",
        "assistant_message": "answer",
        "response_mode": "general_chat",
        "create_conversation": True,
    }

    first_conversation_id = repository.append_exchange(**values)
    retry_values = {
        **values,
        "conversation_id": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    }
    retried_conversation_id = repository.append_exchange(**retry_values)

    with session_factory() as session:
        assert len(session.scalars(select(ConversationRow)).all()) == 1
        assert len(session.scalars(select(ConversationTurnRow)).all()) == 1
    assert first_conversation_id == conversation_id
    assert retried_conversation_id == conversation_id


def test_conversation_repository_rejects_missing_existing_conversation() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    repository = SQLAlchemyConversationRepository(session_factory)

    with pytest.raises(ConversationPersistenceError, match="disappeared"):
        repository.append_exchange(
            conversation_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            request_id="request-1",
            user_message="question",
            assistant_message="answer",
            response_mode="general_chat",
            create_conversation=False,
        )
