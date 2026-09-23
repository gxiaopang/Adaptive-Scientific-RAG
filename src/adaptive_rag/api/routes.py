"""HTTP routes and query-run observability integration."""

import json
import logging
from collections.abc import Callable, Iterator
from contextvars import copy_context
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from adaptive_rag import __version__
from adaptive_rag.api.dependencies import (
    ConversationMemory,
    ReadinessProbe,
    WorkflowRunner,
    create_memory_dependency,
    create_readiness_dependency,
    create_workflow_dependency,
)
from adaptive_rag.api.schemas import (
    EvidenceResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from adaptive_rag.core.config import Settings
from adaptive_rag.core.context import get_request_context
from adaptive_rag.graph.state import AdaptiveRAGResult
from adaptive_rag.graph.workflow import AdaptiveRAGWorkflowError
from adaptive_rag.memory import ConversationMemoryError, ConversationNotFoundError
from adaptive_rag.memory.service import ConversationSession
from adaptive_rag.observability.query_runs import QueryRunRecord, QueryRunRecorder

logger = logging.getLogger(__name__)


def create_api_router(
    *,
    settings: Settings,
    workflow: WorkflowRunner | None,
    conversation_memory: ConversationMemory | None,
    readiness_probe: ReadinessProbe | None,
    query_run_recorder: QueryRunRecorder | None,
) -> APIRouter:
    """Build routes with an explicitly injected workflow dependency."""

    router = APIRouter()
    get_workflow = create_workflow_dependency(workflow)
    get_memory = create_memory_dependency(conversation_memory)
    get_readiness = create_readiness_dependency(readiness_probe)

    @router.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Check process liveness",
    )
    def health() -> HealthResponse:
        return HealthResponse(service=settings.app_name, version=__version__)

    @router.get(
        "/ready",
        response_model=ReadinessResponse,
        tags=["system"],
        summary="Check required runtime dependencies",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "One or more runtime dependencies are unavailable"
            }
        },
    )
    def ready(
        probe: Annotated[ReadinessProbe, Depends(get_readiness)],
    ) -> ReadinessResponse:
        try:
            probe.check()
        except Exception as error:
            logger.exception("application readiness check failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="application dependencies are not ready",
            ) from error
        return ReadinessResponse(service=settings.app_name, version=__version__)

    @router.post(
        "/api/v1/query",
        response_model=QueryResponse,
        tags=["rag"],
        summary="Run the adaptive scientific RAG workflow",
        responses={
            status.HTTP_502_BAD_GATEWAY: {
                "description": "A workflow component violated its runtime contract"
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "The workflow, memory, or database is unavailable"
            },
            status.HTTP_404_NOT_FOUND: {"description": "Conversation not found"},
        },
    )
    def query(
        request: QueryRequest,
        runner: Annotated[WorkflowRunner, Depends(get_workflow)],
        memory: Annotated[ConversationMemory, Depends(get_memory)],
    ) -> QueryResponse:
        started_at = perf_counter()
        request_id, trace_id = _correlation_ids()
        conversation = _start_conversation(memory, request.conversation_id)
        return _execute_query(
            request=request,
            runner=runner,
            memory=memory,
            conversation=conversation,
            query_run_recorder=query_run_recorder,
            request_id=request_id,
            trace_id=trace_id,
            started_at=started_at,
        )

    @router.post(
        "/api/v1/query/stream",
        response_class=StreamingResponse,
        tags=["rag"],
        summary="Stream an adaptive scientific RAG workflow",
        responses={
            status.HTTP_404_NOT_FOUND: {"description": "Conversation not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid query"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "description": "The workflow, memory, or database is unavailable"
            },
        },
    )
    def query_stream(
        request: QueryRequest,
        runner: Annotated[WorkflowRunner, Depends(get_workflow)],
        memory: Annotated[ConversationMemory, Depends(get_memory)],
    ) -> StreamingResponse:
        started_at = perf_counter()
        request_id, trace_id = _correlation_ids()
        conversation = _start_conversation(memory, request.conversation_id)
        events: Queue[str | None] = Queue()
        context = copy_context()

        def on_answer_start() -> None:
            events.put(_encode_stream_event({"type": "answer_start"}))

        def on_answer_delta(delta: str) -> None:
            events.put(_encode_stream_event({"type": "answer_delta", "text": delta}))

        def run_workflow() -> None:
            try:
                response = _execute_query(
                    request=request,
                    runner=runner,
                    memory=memory,
                    conversation=conversation,
                    query_run_recorder=query_run_recorder,
                    request_id=request_id,
                    trace_id=trace_id,
                    started_at=started_at,
                    on_answer_start=on_answer_start,
                    on_answer_delta=on_answer_delta,
                )
                events.put(
                    _encode_stream_event(
                        {"type": "result", "response": response.model_dump(mode="json")}
                    )
                )
            except HTTPException as error:
                events.put(
                    _encode_stream_event(
                        {
                            "type": "error",
                            "status": error.status_code,
                            "detail": error.detail,
                        }
                    )
                )
            except Exception:
                logger.exception("streaming query failed unexpectedly")
                events.put(
                    _encode_stream_event(
                        {
                            "type": "error",
                            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "detail": "streaming query interrupted",
                        }
                    )
                )
            finally:
                events.put(None)

        Thread(
            target=lambda: context.run(run_workflow),
            name=f"query-stream-{request_id[:8]}",
            daemon=True,
        ).start()

        def iterate_events() -> Iterator[str]:
            while (event := events.get()) is not None:
                yield event

        return StreamingResponse(
            iterate_events(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return router


def _start_conversation(
    memory: ConversationMemory,
    conversation_id: UUID | None,
) -> ConversationSession:
    try:
        return memory.start(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="conversation not found",
        ) from error
    except ConversationMemoryError as error:
        logger.exception("conversation memory load failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="conversation memory unavailable",
        ) from error


def _execute_query(
    *,
    request: QueryRequest,
    runner: WorkflowRunner,
    memory: ConversationMemory,
    conversation: ConversationSession,
    query_run_recorder: QueryRunRecorder | None,
    request_id: str,
    trace_id: str,
    started_at: float,
    on_answer_start: Callable[[], None] | None = None,
    on_answer_delta: Callable[[str], None] | None = None,
) -> QueryResponse:
    try:
        if on_answer_start is None and on_answer_delta is None:
            result = runner.run(request.query, conversation.history)
        else:
            result = runner.run(
                request.query,
                conversation.history,
                on_answer_start=on_answer_start,
                on_answer_delta=on_answer_delta,
            )
    except AdaptiveRAGWorkflowError as error:
        latency_ms = (perf_counter() - started_at) * 1_000
        _persist_query_run(
            query_run_recorder,
            QueryRunRecord.failed(
                request_id=request_id,
                trace_id=trace_id,
                original_query=request.query,
                latency_ms=latency_ms,
                error_type=type(error).__name__,
            ),
        )
        logger.exception("adaptive RAG workflow failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="adaptive RAG workflow failed",
        ) from error

    latency_ms = (perf_counter() - started_at) * 1_000
    try:
        persisted_conversation_id = memory.remember(
            conversation,
            request_id=request_id,
            user_message=request.query,
            assistant_message=result.answer,
            response_mode=result.response_mode,
        )
    except ConversationMemoryError as error:
        _persist_query_run(
            query_run_recorder,
            QueryRunRecord.failed(
                request_id=request_id,
                trace_id=trace_id,
                original_query=request.query,
                latency_ms=latency_ms,
                error_type=type(error).__name__,
            ),
        )
        logger.exception("conversation memory persistence failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="conversation memory unavailable",
        ) from error
    _persist_query_run(
        query_run_recorder,
        QueryRunRecord.succeeded(
            request_id=request_id,
            trace_id=trace_id,
            result=result,
            latency_ms=latency_ms,
        ),
    )
    return _query_response(result, persisted_conversation_id)


def _query_response(result: AdaptiveRAGResult, conversation_id: UUID) -> QueryResponse:
    return QueryResponse(
        conversation_id=conversation_id,
        original_query=result.original_query,
        response_mode=result.response_mode,
        routing_reason=result.routing_reason,
        final_query=result.final_query,
        query_history=result.query_history,
        answer=result.answer,
        evidence_sufficient=result.evidence_sufficient,
        verification_reason=result.verification_reason,
        retrieval_attempts=result.retrieval_attempts,
        termination_reason=result.termination_reason,
        evidence=tuple(
            EvidenceResponse(
                doc_id=document.doc_id,
                title=document.title,
                excerpt=document.text[:500],
                score=document.score,
                rank=document.rank,
            )
            for document in result.evidence
        ),
    )


def _encode_stream_event(event: dict[str, object]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"


def _correlation_ids() -> tuple[str, str]:
    context = get_request_context()
    request_id = context.request_id or uuid4().hex
    trace_id = context.trace_id or request_id
    return request_id, trace_id


def _persist_query_run(
    recorder: QueryRunRecorder | None,
    record: QueryRunRecord,
) -> None:
    if recorder is None:
        return
    try:
        recorder.save(record)
    except Exception:
        logger.exception(
            "query run persistence failed",
            extra={
                "request_id": record.request_id,
                "trace_id": record.trace_id,
                "query_run_id": str(record.id),
            },
        )
