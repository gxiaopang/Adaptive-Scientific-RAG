"""Small dependency-injection boundaries used by the HTTP layer."""

from collections.abc import Callable, Sequence
from typing import Protocol
from uuid import UUID

from fastapi import HTTPException, status

from adaptive_rag.domain.models import ConversationTurn, QueryRoute
from adaptive_rag.graph.state import AdaptiveRAGResult
from adaptive_rag.memory.service import ConversationSession


class WorkflowRunner(Protocol):
    """Minimal workflow capability required by the query endpoint."""

    def run(
        self,
        query: str,
        history: Sequence[ConversationTurn] = (),
        *,
        on_answer_start: Callable[[], None] | None = None,
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> AdaptiveRAGResult: ...


class ConversationMemory(Protocol):
    """Conversation capability required by the query endpoint."""

    def start(self, conversation_id: UUID | None) -> ConversationSession: ...

    def remember(
        self,
        session: ConversationSession,
        *,
        request_id: str,
        user_message: str,
        assistant_message: str,
        response_mode: QueryRoute,
    ) -> UUID: ...


class ReadinessProbe(Protocol):
    """Runtime dependency probe used by the readiness endpoint."""

    def check(self) -> None: ...


def create_workflow_dependency(
    workflow: WorkflowRunner | None,
) -> Callable[[], WorkflowRunner]:
    """Bind one process-wide workflow instance to a FastAPI dependency."""

    def get_workflow() -> WorkflowRunner:
        if workflow is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="adaptive RAG workflow is not configured",
            )
        return workflow

    return get_workflow


def create_memory_dependency(
    memory: ConversationMemory | None,
) -> Callable[[], ConversationMemory]:
    """Bind one process-wide memory service to a FastAPI dependency."""

    def get_memory() -> ConversationMemory:
        if memory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="conversation memory is not configured",
            )
        return memory

    return get_memory


def create_readiness_dependency(
    readiness: ReadinessProbe | None,
) -> Callable[[], ReadinessProbe]:
    """Bind the runtime probe while keeping the import-safe app unavailable."""

    def get_readiness() -> ReadinessProbe:
        if readiness is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="application dependencies are not configured",
            )
        return readiness

    return get_readiness
