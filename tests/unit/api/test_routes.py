import json
from collections.abc import Callable, Sequence
from uuid import UUID

from fastapi.testclient import TestClient

from adaptive_rag.core.config import Settings
from adaptive_rag.core.context import get_request_context
from adaptive_rag.domain.models import ConversationTurn, EvidenceDocument, QueryRoute
from adaptive_rag.graph.state import AdaptiveRAGResult
from adaptive_rag.graph.workflow import AdaptiveRAGWorkflowError
from adaptive_rag.main import create_app
from adaptive_rag.memory import (
    ConversationMemoryError,
    ConversationNotFoundError,
    ConversationSession,
)
from adaptive_rag.observability.query_runs import QueryRunRecord


class FakeWorkflowRunner:
    def __init__(self, result: AdaptiveRAGResult) -> None:
        self.result = result
        self.calls: list[str] = []
        self.histories: list[tuple[ConversationTurn, ...]] = []

    def run(
        self,
        query: str,
        history: Sequence[ConversationTurn] = (),
        *,
        on_answer_start: Callable[[], None] | None = None,
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> AdaptiveRAGResult:
        self.calls.append(query)
        self.histories.append(tuple(history))
        if on_answer_start is not None:
            on_answer_start()
        if on_answer_delta is not None:
            midpoint = len(self.result.answer) // 2
            on_answer_delta(self.result.answer[:midpoint])
            on_answer_delta(self.result.answer[midpoint:])
        return self.result


class FailingWorkflowRunner:
    def run(
        self,
        query: str,
        history: Sequence[ConversationTurn] = (),
        *,
        on_answer_start: Callable[[], None] | None = None,
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> AdaptiveRAGResult:
        raise AdaptiveRAGWorkflowError(f"failed for {query}")


class FakeConversationMemory:
    conversation_id = UUID("12345678-1234-5678-1234-567812345678")

    def __init__(self, history: tuple[ConversationTurn, ...] = ()) -> None:
        self.history = history
        self.started_with: list[UUID | None] = []
        self.remembered: list[dict[str, object]] = []

    def start(self, conversation_id: UUID | None) -> ConversationSession:
        self.started_with.append(conversation_id)
        if conversation_id not in {None, self.conversation_id}:
            raise ConversationNotFoundError("missing")
        return ConversationSession(
            conversation_id=self.conversation_id,
            history=self.history,
            is_new=conversation_id is None,
        )

    def remember(
        self,
        session: ConversationSession,
        *,
        request_id: str,
        user_message: str,
        assistant_message: str,
        response_mode: QueryRoute,
    ) -> UUID:
        self.remembered.append(
            {
                "session": session,
                "request_id": request_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "response_mode": response_mode,
            }
        )
        return session.conversation_id


class FailingConversationMemory(FakeConversationMemory):
    def remember(
        self,
        session: ConversationSession,
        *,
        request_id: str,
        user_message: str,
        assistant_message: str,
        response_mode: QueryRoute,
    ) -> UUID:
        raise ConversationMemoryError("database unavailable")


class FakeReadinessProbe:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.checks = 0

    def check(self) -> None:
        self.checks += 1
        if self.failure is not None:
            raise self.failure


class FakeQueryRunRecorder:
    def __init__(self) -> None:
        self.records: list[QueryRunRecord] = []

    def save(self, record: QueryRunRecord) -> None:
        self.records.append(record)


class FailingQueryRunRecorder:
    def save(self, record: QueryRunRecord) -> None:
        raise RuntimeError(f"database unavailable for {record.id}")


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        app_name="Test Scientific RAG",
        app_env="test",
        log_level="CRITICAL",
    )


def _result() -> AdaptiveRAGResult:
    return AdaptiveRAGResult(
        original_query="What supports the claim?",
        response_mode="scientific_evidence",
        routing_reason="The question asks for scientific support.",
        final_query="scientific evidence supporting the claim",
        query_history=(
            "What supports the claim?",
            "scientific evidence supporting the claim",
        ),
        evidence=(
            EvidenceDocument(
                doc_id="123",
                title="Evidence title",
                text="The full document body is internal context.",
                score=0.91,
                rank=1,
            ),
        ),
        answer="The retrieved study supports the claim.",
        evidence_sufficient=True,
        verification_reason="The study directly addresses the claim.",
        retrieval_attempts=2,
        termination_reason="evidence_sufficient",
    )


def _general_chat_result() -> AdaptiveRAGResult:
    return AdaptiveRAGResult(
        original_query="Hello",
        response_mode="general_chat",
        routing_reason="This is a casual greeting.",
        final_query="Hello",
        query_history=("Hello",),
        evidence=(),
        answer="Hello! How can I help?",
        evidence_sufficient=None,
        verification_reason=None,
        retrieval_attempts=0,
        termination_reason="general_chat",
    )


def test_health_reports_process_liveness_without_workflow() -> None:
    with TestClient(create_app(_settings())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Test Scientific RAG",
        "version": "0.1.0",
    }
    assert len(response.headers["x-request-id"]) == 32
    assert response.headers["x-trace-id"] == response.headers["x-request-id"]
    assert get_request_context().request_id is None


def test_readiness_checks_runtime_dependencies() -> None:
    readiness = FakeReadinessProbe()
    with TestClient(create_app(_settings(), readiness_probe=readiness)) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "Test Scientific RAG",
        "version": "0.1.0",
    }
    assert readiness.checks == 1


def test_readiness_returns_503_when_dependencies_fail() -> None:
    readiness = FakeReadinessProbe(RuntimeError("database detail"))
    with TestClient(create_app(_settings(), readiness_probe=readiness)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "application dependencies are not ready"}


def test_readiness_returns_503_for_import_safe_app() -> None:
    with TestClient(create_app(_settings())) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "application dependencies are not configured"}


def test_query_normalizes_input_and_projects_workflow_result() -> None:
    runner = FakeWorkflowRunner(_result())
    recorder = FakeQueryRunRecorder()
    memory = FakeConversationMemory()
    with TestClient(
        create_app(
            _settings(),
            workflow=runner,
            conversation_memory=memory,
            query_run_recorder=recorder,
        )
    ) as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "  What supports the claim?  "},
            headers={
                "x-request-id": "client-request-123",
                "x-trace-id": "client-trace-456",
            },
        )

    assert response.status_code == 200
    assert runner.calls == ["What supports the claim?"]
    assert runner.histories == [()]
    assert response.json() == {
        "conversation_id": str(memory.conversation_id),
        "original_query": "What supports the claim?",
        "response_mode": "scientific_evidence",
        "routing_reason": "The question asks for scientific support.",
        "final_query": "scientific evidence supporting the claim",
        "query_history": [
            "What supports the claim?",
            "scientific evidence supporting the claim",
        ],
        "answer": "The retrieved study supports the claim.",
        "evidence_sufficient": True,
        "verification_reason": "The study directly addresses the claim.",
        "retrieval_attempts": 2,
        "termination_reason": "evidence_sufficient",
        "evidence": [
            {
                "doc_id": "123",
                "title": "Evidence title",
                "excerpt": "The full document body is internal context.",
                "score": 0.91,
                "rank": 1,
            }
        ],
    }
    assert "text" not in response.json()["evidence"][0]
    assert len(response.json()["evidence"][0]["excerpt"]) <= 500
    assert response.headers["x-request-id"] == "client-request-123"
    assert response.headers["x-trace-id"] == "client-trace-456"
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.request_id == "client-request-123"
    assert record.trace_id == "client-trace-456"
    assert record.status == "succeeded"
    assert record.final_query == "scientific evidence supporting the claim"
    assert record.retrieval_attempts == 2
    assert record.latency_ms >= 0.0
    assert [evidence.doc_id for evidence in record.evidence] == ["123"]
    assert memory.remembered[0]["request_id"] == "client-request-123"
    assert (
        memory.remembered[0]["assistant_message"]
        == "The retrieved study supports the claim."
    )


def test_general_chat_response_has_no_fake_retrieval_metadata() -> None:
    runner = FakeWorkflowRunner(_general_chat_result())
    recorder = FakeQueryRunRecorder()
    memory = FakeConversationMemory()
    with TestClient(
        create_app(
            _settings(),
            workflow=runner,
            conversation_memory=memory,
            query_run_recorder=recorder,
        )
    ) as client:
        response = client.post("/api/v1/query", json={"query": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(memory.conversation_id),
        "original_query": "Hello",
        "response_mode": "general_chat",
        "routing_reason": "This is a casual greeting.",
        "final_query": "Hello",
        "query_history": ["Hello"],
        "answer": "Hello! How can I help?",
        "evidence_sufficient": None,
        "verification_reason": None,
        "retrieval_attempts": 0,
        "termination_reason": "general_chat",
        "evidence": [],
    }
    assert recorder.records[0].retrieval_attempts == 0
    assert recorder.records[0].termination_reason == "general_chat"
    assert recorder.records[0].evidence == ()


def test_query_stream_emits_answer_deltas_before_final_result() -> None:
    runner = FakeWorkflowRunner(_general_chat_result())
    memory = FakeConversationMemory()
    with TestClient(
        create_app(
            _settings(),
            workflow=runner,
            conversation_memory=memory,
        )
    ) as client:
        with client.stream(
            "POST",
            "/api/v1/query/stream",
            json={"query": "Hello"},
        ) as response:
            events = [json.loads(line) for line in response.iter_lines()]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    assert response.headers["x-accel-buffering"] == "no"
    assert [event["type"] for event in events] == [
        "answer_start",
        "answer_delta",
        "answer_delta",
        "result",
    ]
    assert "".join(event["text"] for event in events[1:3]) == _general_chat_result().answer
    assert events[-1]["response"]["answer"] == _general_chat_result().answer
    assert memory.remembered[0]["assistant_message"] == _general_chat_result().answer


def test_query_stream_maps_workflow_failure_to_error_event() -> None:
    with TestClient(
        create_app(
            _settings(),
            workflow=FailingWorkflowRunner(),
            conversation_memory=FakeConversationMemory(),
        )
    ) as client:
        response = client.post("/api/v1/query/stream", json={"query": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "type": "error",
        "status": 502,
        "detail": "adaptive RAG workflow failed",
    }


def test_existing_conversation_history_is_passed_to_workflow() -> None:
    history = (
        ConversationTurn(1, "Tell me about aspirin", "Prior answer", "scientific_evidence"),
    )
    memory = FakeConversationMemory(history)
    runner = FakeWorkflowRunner(_result())
    with TestClient(
        create_app(_settings(), workflow=runner, conversation_memory=memory)
    ) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "query": "What are its adverse effects?",
                "conversation_id": str(memory.conversation_id),
            },
        )

    assert response.status_code == 200
    assert memory.started_with == [memory.conversation_id]
    assert runner.histories == [history]
    remembered_session = memory.remembered[0]["session"]
    assert isinstance(remembered_session, ConversationSession)
    assert remembered_session.is_new is False


def test_query_returns_404_for_unknown_conversation() -> None:
    runner = FakeWorkflowRunner(_result())
    with TestClient(
        create_app(
            _settings(),
            workflow=runner,
            conversation_memory=FakeConversationMemory(),
        )
    ) as client:
        response = client.post(
            "/api/v1/query",
            json={
                "query": "follow-up",
                "conversation_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            },
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation not found"}
    assert runner.calls == []


def test_query_returns_503_when_conversation_memory_is_not_configured() -> None:
    with TestClient(create_app(_settings(), workflow=FakeWorkflowRunner(_result()))) as client:
        response = client.post("/api/v1/query", json={"query": "valid question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "conversation memory is not configured"}


def test_query_returns_503_when_completed_exchange_cannot_be_saved() -> None:
    recorder = FakeQueryRunRecorder()
    with TestClient(
        create_app(
            _settings(),
            workflow=FakeWorkflowRunner(_result()),
            conversation_memory=FailingConversationMemory(),
            query_run_recorder=recorder,
        )
    ) as client:
        response = client.post("/api/v1/query", json={"query": "valid question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "conversation memory unavailable"}
    assert recorder.records[0].status == "failed"
    assert recorder.records[0].error_type == "ConversationMemoryError"


def test_query_returns_503_when_workflow_is_not_configured() -> None:
    with TestClient(create_app(_settings())) as client:
        response = client.post("/api/v1/query", json={"query": "valid question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "adaptive RAG workflow is not configured"}


def test_query_maps_known_workflow_failure_to_502() -> None:
    recorder = FakeQueryRunRecorder()
    with TestClient(
        create_app(
            _settings(),
            workflow=FailingWorkflowRunner(),
            conversation_memory=FakeConversationMemory(),
            query_run_recorder=recorder,
        )
    ) as client:
        response = client.post("/api/v1/query", json={"query": "valid question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "adaptive RAG workflow failed"}
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.status == "failed"
    assert record.original_query == "valid question"
    assert record.error_type == "AdaptiveRAGWorkflowError"
    assert record.answer is None


def test_query_response_survives_non_critical_persistence_failure() -> None:
    runner = FakeWorkflowRunner(_result())
    with TestClient(
        create_app(
            _settings(),
            workflow=runner,
            conversation_memory=FakeConversationMemory(),
            query_run_recorder=FailingQueryRunRecorder(),
        )
    ) as client:
        response = client.post("/api/v1/query", json={"query": "valid question"})

    assert response.status_code == 200
    assert response.json()["answer"] == "The retrieved study supports the claim."


def test_invalid_correlation_headers_are_replaced() -> None:
    with TestClient(create_app(_settings())) as client:
        response = client.get(
            "/health",
            headers={"x-request-id": "contains spaces", "x-trace-id": ""},
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] != "contains spaces"
    assert len(response.headers["x-request-id"]) == 32
    assert response.headers["x-trace-id"] == response.headers["x-request-id"]


def test_query_rejects_blank_input_before_calling_workflow() -> None:
    runner = FakeWorkflowRunner(_result())
    with TestClient(
        create_app(
            _settings(), workflow=runner, conversation_memory=FakeConversationMemory()
        )
    ) as client:
        response = client.post("/api/v1/query", json={"query": "   "})

    assert response.status_code == 422
    assert runner.calls == []


def test_query_rejects_unknown_request_fields() -> None:
    runner = FakeWorkflowRunner(_result())
    with TestClient(
        create_app(
            _settings(), workflow=runner, conversation_memory=FakeConversationMemory()
        )
    ) as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "valid question", "model": "unapproved-provider"},
        )

    assert response.status_code == 422
    assert runner.calls == []


def test_openapi_contains_only_phase_seven_business_contracts() -> None:
    with TestClient(create_app(_settings())) as client:
        openapi = client.get("/openapi.json").json()

    assert set(openapi["paths"]) == {
        "/health",
        "/ready",
        "/api/v1/query",
        "/api/v1/query/stream",
    }
    assert openapi["paths"]["/api/v1/query"]["post"]["responses"].keys() >= {
        "200",
        "422",
        "502",
        "503",
    }
