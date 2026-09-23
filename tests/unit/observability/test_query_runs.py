from adaptive_rag.domain.models import EvidenceDocument
from adaptive_rag.graph.state import AdaptiveRAGResult
from adaptive_rag.observability.query_runs import QueryRunRecord


def test_success_record_projects_stable_workflow_fields() -> None:
    result = AdaptiveRAGResult(
        original_query="original",
        response_mode="scientific_evidence",
        routing_reason="needs evidence",
        final_query="rewritten",
        query_history=("original", "rewritten"),
        evidence=(EvidenceDocument("d1", "Title", "private body", 0.8, 1),),
        answer="answer",
        evidence_sufficient=True,
        verification_reason="supported",
        retrieval_attempts=2,
        termination_reason="evidence_sufficient",
    )

    record = QueryRunRecord.succeeded(
        request_id="request-1",
        trace_id="trace-1",
        result=result,
        latency_ms=10.5,
    )

    assert record.status == "succeeded"
    assert record.query_history == ("original", "rewritten")
    assert record.evidence[0].doc_id == "d1"
    assert not hasattr(record.evidence[0], "text")
    assert record.error_type is None
    assert record.created_at.tzinfo is not None


def test_failure_record_does_not_invent_partial_workflow_output() -> None:
    record = QueryRunRecord.failed(
        request_id="request-1",
        trace_id="trace-1",
        original_query="original",
        latency_ms=3.0,
        error_type="WorkflowError",
    )

    assert record.status == "failed"
    assert record.final_query is None
    assert record.answer is None
    assert record.evidence == ()
    assert record.retrieval_attempts == 0
    assert record.error_type == "WorkflowError"
