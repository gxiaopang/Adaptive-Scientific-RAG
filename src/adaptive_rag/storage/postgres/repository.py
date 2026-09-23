"""SQLAlchemy implementations of online persistence boundaries."""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from adaptive_rag.domain.models import ConversationTurn, QueryRoute
from adaptive_rag.observability.query_runs import QueryRunRecord
from adaptive_rag.storage.postgres.models import (
    ConversationRow,
    ConversationTurnRow,
    QueryRunRow,
)


class QueryRunPersistenceError(RuntimeError):
    """Raised when a query-run transaction cannot be committed."""


class ConversationPersistenceError(RuntimeError):
    """Raised when a conversation transaction cannot be completed."""


class SQLAlchemyQueryRunRepository:
    """Persist each query run in its own short transaction."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, record: QueryRunRecord) -> None:
        row = QueryRunRow(
            id=record.id,
            request_id=record.request_id,
            trace_id=record.trace_id,
            status=record.status,
            original_query=record.original_query,
            final_query=record.final_query,
            query_history=list(record.query_history),
            answer=record.answer,
            evidence_sufficient=record.evidence_sufficient,
            verification_reason=record.verification_reason,
            retrieval_attempts=record.retrieval_attempts,
            termination_reason=record.termination_reason,
            evidence=[
                {
                    "doc_id": item.doc_id,
                    "title": item.title,
                    "score": item.score,
                    "rank": item.rank,
                }
                for item in record.evidence
            ],
            latency_ms=record.latency_ms,
            error_type=record.error_type,
            created_at=record.created_at,
        )
        try:
            with self._session_factory.begin() as session:
                session.add(row)
        except SQLAlchemyError as error:
            msg = f"unable to persist query run {record.id}"
            raise QueryRunPersistenceError(msg) from error


class SQLAlchemyConversationRepository:
    """Load recent turns and append exchanges in serialized transactions."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def load(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> tuple[ConversationTurn, ...] | None:
        if limit < 1:
            raise ValueError("limit must be positive")
        try:
            with self._session_factory() as session:
                exists = session.get(ConversationRow, conversation_id)
                if exists is None:
                    return None
                rows = session.scalars(
                    select(ConversationTurnRow)
                    .where(ConversationTurnRow.conversation_id == conversation_id)
                    .order_by(ConversationTurnRow.sequence_no.desc())
                    .limit(limit)
                ).all()
        except SQLAlchemyError as error:
            msg = f"unable to load conversation {conversation_id}"
            raise ConversationPersistenceError(msg) from error

        return tuple(
            ConversationTurn(
                sequence_no=row.sequence_no,
                user_message=row.user_message,
                assistant_message=row.assistant_message,
                response_mode=cast(QueryRoute, row.response_mode),
            )
            for row in reversed(rows)
        )

    def append_exchange(
        self,
        *,
        conversation_id: UUID,
        request_id: str,
        user_message: str,
        assistant_message: str,
        response_mode: QueryRoute,
        create_conversation: bool,
    ) -> UUID:
        now = datetime.now(UTC)
        try:
            with self._session_factory.begin() as session:
                existing_conversation_id = self._request_conversation_id(
                    session, request_id
                )
                if existing_conversation_id is not None:
                    return existing_conversation_id
                conversation = session.get(
                    ConversationRow,
                    conversation_id,
                    with_for_update=True,
                )
                if conversation is None:
                    if not create_conversation:
                        raise ConversationPersistenceError(
                            f"conversation disappeared: {conversation_id}"
                        )
                    conversation = ConversationRow(
                        id=conversation_id,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(conversation)
                    session.flush()

                latest_sequence = session.scalar(
                    select(func.max(ConversationTurnRow.sequence_no)).where(
                        ConversationTurnRow.conversation_id == conversation_id
                    )
                )
                conversation.updated_at = now
                session.add(
                    ConversationTurnRow(
                        id=uuid4(),
                        conversation_id=conversation_id,
                        request_id=request_id,
                        sequence_no=(latest_sequence or 0) + 1,
                        user_message=user_message,
                        assistant_message=assistant_message,
                        response_mode=response_mode,
                        created_at=now,
                    )
                )
            return conversation_id
        except IntegrityError as error:
            existing_conversation_id = self._request_conversation_id_in_new_session(
                request_id
            )
            if existing_conversation_id is not None:
                return existing_conversation_id
            msg = f"unable to append conversation {conversation_id}"
            raise ConversationPersistenceError(msg) from error
        except SQLAlchemyError as error:
            msg = f"unable to append conversation {conversation_id}"
            raise ConversationPersistenceError(msg) from error

    @staticmethod
    def _request_conversation_id(session: Session, request_id: str) -> UUID | None:
        return session.scalar(
            select(ConversationTurnRow.conversation_id).where(
                ConversationTurnRow.request_id == request_id
            )
        )

    def _request_conversation_id_in_new_session(self, request_id: str) -> UUID | None:
        try:
            with self._session_factory() as session:
                return self._request_conversation_id(session, request_id)
        except SQLAlchemyError:
            return None
