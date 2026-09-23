from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.domain.models import (
    ConversationTurn,
    EvidenceAssessment,
    EvidenceDocument,
    QueryRoutingDecision,
    ScoredDocument,
)
from adaptive_rag.domain.protocols import ProviderInvocationError
from adaptive_rag.graph.workflow import (
    AdaptiveRAGConfig,
    AdaptiveRAGWorkflow,
    AdaptiveRAGWorkflowError,
    load_adaptive_rag_config,
)


class FakeRetriever:
    strategy = "test"

    def __init__(self, results_by_query: dict[str, list[ScoredDocument]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        self.calls.append((query, top_k))
        return self.results_by_query[query]


class FakeRouter:
    def __init__(self, decisions: list[QueryRoutingDecision]) -> None:
        self.decisions = decisions
        self.calls: list[str] = []
        self.histories: list[tuple[ConversationTurn, ...]] = []

    def classify(
        self,
        query: str,
        history: Sequence[ConversationTurn],
    ) -> QueryRoutingDecision:
        self.calls.append(query)
        self.histories.append(tuple(history))
        decision = self.decisions[len(self.calls) - 1]
        if decision.standalone_query == "<current-query>":
            return QueryRoutingDecision(decision.route, decision.reason, query)
        return decision


class FakeChatGenerator:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[str] = []
        self.histories: list[tuple[ConversationTurn, ...]] = []

    def chat(
        self,
        query: str,
        history: Sequence[ConversationTurn],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        self.calls.append(query)
        self.histories.append(tuple(history))
        answer = self.answers[len(self.calls) - 1]
        if on_delta is not None:
            on_delta(answer)
        return answer


class FakeGenerator:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        query: str,
        context: str,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        self.calls.append((query, context))
        answer = self.answers[len(self.calls) - 1]
        if on_delta is not None:
            on_delta(answer)
        return answer


class FakeVerifier:
    def __init__(self, assessments: list[EvidenceAssessment]) -> None:
        self.assessments = assessments
        self.calls: list[tuple[str, str, str, tuple[EvidenceDocument, ...]]] = []

    def verify(
        self,
        query: str,
        answer: str,
        context: str,
        evidence: Sequence[EvidenceDocument],
    ) -> EvidenceAssessment:
        self.calls.append((query, answer, context, tuple(evidence)))
        return self.assessments[len(self.calls) - 1]


class FakeRewriter:
    def __init__(self, rewrites: list[str]) -> None:
        self.rewrites = rewrites
        self.calls: list[dict[str, object]] = []

    def rewrite(
        self,
        *,
        original_query: str,
        current_query: str,
        context: str,
        assessment: EvidenceAssessment,
        retrieval_attempt: int,
    ) -> str:
        self.calls.append(
            {
                "original_query": original_query,
                "current_query": current_query,
                "context": context,
                "assessment": assessment,
                "retrieval_attempt": retrieval_attempt,
            }
        )
        return self.rewrites[len(self.calls) - 1]


class FailingGenerator(FakeGenerator):
    def generate(
        self,
        query: str,
        context: str,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        raise ProviderInvocationError("provider detail")


class FailingRouter(FakeRouter):
    def classify(
        self,
        query: str,
        history: Sequence[ConversationTurn],
    ) -> QueryRoutingDecision:
        raise ProviderInvocationError("provider detail")


class FailingChatGenerator(FakeChatGenerator):
    def chat(
        self,
        query: str,
        history: Sequence[ConversationTurn],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        raise ProviderInvocationError("provider detail")


class FailingVerifier(FakeVerifier):
    def verify(
        self,
        query: str,
        answer: str,
        context: str,
        evidence: Sequence[EvidenceDocument],
    ) -> EvidenceAssessment:
        raise ProviderInvocationError("provider detail")


class FailingRewriter(FakeRewriter):
    def rewrite(
        self,
        *,
        original_query: str,
        current_query: str,
        context: str,
        assessment: EvidenceAssessment,
        retrieval_attempt: int,
    ) -> str:
        raise ProviderInvocationError("provider detail")


def _corpus() -> dict[str, CorpusDocument]:
    return {
        "d1": CorpusDocument("d1", "First title", "First body", {}),
        "d2": CorpusDocument("d2", "Second title", "Second body", {}),
    }


def _config(*, maximum_attempts: int = 2) -> AdaptiveRAGConfig:
    return AdaptiveRAGConfig(
        name="test-adaptive-rag",
        retrieval_top_k=2,
        max_context_characters=1_000,
        max_retrieval_attempts=maximum_attempts,
    )


def _workflow(
    *,
    retriever: FakeRetriever,
    router: FakeRouter | None = None,
    chat_generator: FakeChatGenerator | None = None,
    generator: FakeGenerator | None = None,
    verifier: FakeVerifier | None = None,
    rewriter: FakeRewriter | None = None,
    maximum_attempts: int = 2,
) -> AdaptiveRAGWorkflow:
    return AdaptiveRAGWorkflow(
        retriever=retriever,
        corpus=_corpus(),
        router=router
        or FakeRouter(
            [QueryRoutingDecision("scientific_evidence", "needs evidence", "<current-query>")]
        ),
        chat_generator=chat_generator or FakeChatGenerator(["general answer"]),
        generator=generator or FakeGenerator(["answer"] * maximum_attempts),
        verifier=verifier
        or FakeVerifier([EvidenceAssessment(True, "enough")] * maximum_attempts),
        rewriter=rewriter or FakeRewriter(["rewritten query"] * maximum_attempts),
        config=_config(maximum_attempts=maximum_attempts),
    )


def test_accepts_answer_when_first_evidence_is_sufficient() -> None:
    retriever = FakeRetriever(
        {"original query": [ScoredDocument("d1", 0.9, 1)]}
    )
    generator = FakeGenerator(["supported answer"])
    verifier = FakeVerifier([EvidenceAssessment(True, "directly supported")])
    rewriter = FakeRewriter([])
    workflow = _workflow(
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        rewriter=rewriter,
    )

    result = workflow.run("  original query  ")

    assert result.original_query == "original query"
    assert result.response_mode == "scientific_evidence"
    assert result.routing_reason == "needs evidence"
    assert result.final_query == "original query"
    assert result.query_history == ("original query",)
    assert [document.doc_id for document in result.evidence] == ["d1"]
    assert result.answer == "supported answer"
    assert result.evidence_sufficient is True
    assert result.verification_reason == "directly supported"
    assert result.retrieval_attempts == 1
    assert result.termination_reason == "evidence_sufficient"
    assert retriever.calls == [("original query", 2)]
    assert generator.calls[0][0] == "original query"
    assert rewriter.calls == []
    assert len(result.snapshots) == 1
    assert result.snapshots[0].context == generator.calls[0][1]
    assert result.snapshots[0].answer == result.answer
    assert result.snapshots[0].attempt == 1


def test_snapshots_preserve_each_round_and_original_answering_question() -> None:
    generator = FakeGenerator(["first answer", "second answer"])
    workflow = _workflow(
        retriever=FakeRetriever({
            "original": [ScoredDocument("d1", 0.9, 1)],
            "rewritten": [ScoredDocument("d2", 0.8, 1)],
        }),
        generator=generator,
        verifier=FakeVerifier([EvidenceAssessment(False, "gap"), EvidenceAssessment(True, "ok")]),
        rewriter=FakeRewriter(["rewritten"]),
    )
    result = workflow.run("original")
    assert [s.attempt for s in result.snapshots] == [1, 2]
    assert [s.retrieval_query for s in result.snapshots] == ["original", "rewritten"]
    assert [s.query for s in result.snapshots] == ["original", "original"]
    assert [s.context for s in result.snapshots] == [c[1] for c in generator.calls]
    assert [s.answer for s in result.snapshots] == ["first answer", "second answer"]


def test_general_chat_bypasses_every_retrieval_and_evidence_component() -> None:
    retriever = FakeRetriever({})
    router = FakeRouter([QueryRoutingDecision("general_chat", "casual greeting", "Hello")])
    chat_generator = FakeChatGenerator(["Hello! How can I help?"])
    generator = FakeGenerator([])
    verifier = FakeVerifier([])
    rewriter = FakeRewriter([])
    workflow = _workflow(
        retriever=retriever,
        router=router,
        chat_generator=chat_generator,
        generator=generator,
        verifier=verifier,
        rewriter=rewriter,
    )

    result = workflow.run("Hello")

    assert result.response_mode == "general_chat"
    assert result.routing_reason == "casual greeting"
    assert result.answer == "Hello! How can I help?"
    assert result.final_query == "Hello"
    assert result.query_history == ("Hello",)
    assert result.evidence == ()
    assert result.evidence_sufficient is None
    assert result.verification_reason is None
    assert result.retrieval_attempts == 0
    assert result.termination_reason == "general_chat"
    assert router.calls == ["Hello"]
    assert chat_generator.calls == ["Hello"]
    assert retriever.calls == []
    assert generator.calls == []
    assert verifier.calls == []
    assert rewriter.calls == []


def test_general_chat_emits_streaming_answer_callbacks() -> None:
    events: list[str] = []
    workflow = _workflow(
        retriever=FakeRetriever({}),
        router=FakeRouter([QueryRoutingDecision("general_chat", "greeting", "Hello")]),
        chat_generator=FakeChatGenerator(["Hello from the stream"]),
    )

    result = workflow.run(
        "Hello",
        on_answer_start=lambda: events.append("start"),
        on_answer_delta=events.append,
    )

    assert result.answer == "Hello from the stream"
    assert events == ["start", "Hello from the stream"]


def test_scientific_follow_up_uses_history_resolved_query_without_exposing_history() -> None:
    history = (
        ConversationTurn(
            1,
            "Does aspirin reduce cardiovascular risk?",
            "The evidence suggests a qualified benefit.",
            "scientific_evidence",
        ),
    )
    standalone = "What adverse effects does aspirin have?"
    retriever = FakeRetriever({standalone: [ScoredDocument("d1", 0.9, 1)]})
    router = FakeRouter(
        [QueryRoutingDecision("scientific_evidence", "scientific follow-up", standalone)]
    )
    generator = FakeGenerator(["Supported adverse-effects answer"])
    verifier = FakeVerifier([EvidenceAssessment(True, "supported")])
    workflow = _workflow(
        retriever=retriever,
        router=router,
        generator=generator,
        verifier=verifier,
    )

    result = workflow.run("What adverse effects does it have?", history)

    assert router.histories == [history]
    assert retriever.calls == [(standalone, 2)]
    assert generator.calls[0][0] == standalone
    assert verifier.calls[0][0] == standalone
    assert "The evidence suggests a qualified benefit." not in generator.calls[0][1]
    assert result.original_query == "What adverse effects does it have?"
    assert result.final_query == standalone
    assert result.query_history == (standalone,)


def test_rewrites_and_retrieves_again_when_evidence_is_insufficient() -> None:
    retriever = FakeRetriever(
        {
            "original query": [ScoredDocument("d1", 0.9, 1)],
            "more specific query": [ScoredDocument("d2", 0.8, 1)],
        }
    )
    generator = FakeGenerator(["first answer", "second answer"])
    verifier = FakeVerifier(
        [
            EvidenceAssessment(False, "missing comparison"),
            EvidenceAssessment(True, "comparison found"),
        ]
    )
    rewriter = FakeRewriter(["more specific query"])
    workflow = _workflow(
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        rewriter=rewriter,
    )

    result = workflow.run("original query")

    assert result.final_query == "more specific query"
    assert result.query_history == ("original query", "more specific query")
    assert result.answer == "second answer"
    assert result.evidence_sufficient is True
    assert result.retrieval_attempts == 2
    assert result.termination_reason == "evidence_sufficient"
    assert retriever.calls == [
        ("original query", 2),
        ("more specific query", 2),
    ]
    assert [call[0] for call in generator.calls] == ["original query", "original query"]
    assert rewriter.calls[0]["retrieval_attempt"] == 1


def test_scientific_stream_restarts_answer_when_retrieval_is_retried() -> None:
    events: list[str] = []
    workflow = _workflow(
        retriever=FakeRetriever(
            {
                "original": [ScoredDocument("d1", 0.9, 1)],
                "rewritten": [ScoredDocument("d2", 0.8, 1)],
            }
        ),
        generator=FakeGenerator(["draft answer", "final answer"]),
        verifier=FakeVerifier(
            [EvidenceAssessment(False, "gap"), EvidenceAssessment(True, "supported")]
        ),
        rewriter=FakeRewriter(["rewritten"]),
    )

    result = workflow.run(
        "original",
        on_answer_start=lambda: events.append("start"),
        on_answer_delta=events.append,
    )

    assert result.answer == "final answer"
    assert events == ["start", "draft answer", "start", "final answer"]


def test_stops_after_configured_attempt_limit() -> None:
    retriever = FakeRetriever(
        {
            "original query": [ScoredDocument("d1", 0.9, 1)],
            "rewritten query": [ScoredDocument("d2", 0.8, 1)],
        }
    )
    verifier = FakeVerifier(
        [
            EvidenceAssessment(False, "not enough"),
            EvidenceAssessment(False, "still not enough"),
        ]
    )
    rewriter = FakeRewriter(["rewritten query", "must not be used"])
    workflow = _workflow(
        retriever=retriever,
        verifier=verifier,
        rewriter=rewriter,
    )

    result = workflow.run("original query")

    assert result.evidence_sufficient is False
    assert result.retrieval_attempts == 2
    assert result.termination_reason == "max_retrieval_attempts"
    assert len(rewriter.calls) == 1


@pytest.mark.parametrize(
    ("results", "error"),
    [
        ([], "no candidates"),
        ([ScoredDocument("d1", 1.0, 2)], "consecutive"),
        (
            [ScoredDocument("d1", 1.0, 1), ScoredDocument("d1", 0.5, 2)],
            "duplicate",
        ),
        ([ScoredDocument("missing", 1.0, 1)], "unknown document"),
        ([ScoredDocument("d1", float("nan"), 1)], "non-finite"),
    ],
)
def test_invalid_retrieval_results_fail_closed(
    results: list[ScoredDocument],
    error: str,
) -> None:
    workflow = _workflow(retriever=FakeRetriever({"query": results}))

    with pytest.raises(AdaptiveRAGWorkflowError, match=error):
        workflow.run("query")


def test_unchanged_rewrite_fails_closed() -> None:
    workflow = _workflow(
        retriever=FakeRetriever({"query": [ScoredDocument("d1", 1.0, 1)]}),
        verifier=FakeVerifier([EvidenceAssessment(False, "not enough")]),
        rewriter=FakeRewriter([" QUERY "]),
    )

    with pytest.raises(AdaptiveRAGWorkflowError, match="did not change"):
        workflow.run("query")


def test_blank_generated_answer_fails_closed() -> None:
    workflow = _workflow(
        retriever=FakeRetriever({"query": [ScoredDocument("d1", 1.0, 1)]}),
        generator=FakeGenerator(["   "]),
    )

    with pytest.raises(AdaptiveRAGWorkflowError, match="blank answer"):
        workflow.run("query")


def test_generation_provider_failure_is_wrapped_by_workflow_boundary() -> None:
    workflow = _workflow(
        retriever=FakeRetriever({"query": [ScoredDocument("d1", 1.0, 1)]}),
        generator=FailingGenerator([]),
    )

    with pytest.raises(AdaptiveRAGWorkflowError, match="answer generation provider"):
        workflow.run("query")


def test_routing_provider_failure_is_wrapped_by_workflow_boundary() -> None:
    workflow = _workflow(
        retriever=FakeRetriever({}),
        router=FailingRouter([]),
    )

    with pytest.raises(AdaptiveRAGWorkflowError, match="query routing provider"):
        workflow.run("query")


def test_general_chat_provider_failure_is_wrapped_by_workflow_boundary() -> None:
    workflow = _workflow(
        retriever=FakeRetriever({}),
        router=FakeRouter([QueryRoutingDecision("general_chat", "general", "query")]),
        chat_generator=FailingChatGenerator([]),
    )

    with pytest.raises(AdaptiveRAGWorkflowError, match="general chat provider"):
        workflow.run("query")


def test_verification_provider_failure_is_wrapped_by_workflow_boundary() -> None:
    workflow = _workflow(
        retriever=FakeRetriever({"query": [ScoredDocument("d1", 1.0, 1)]}),
        verifier=FailingVerifier([]),
    )

    with pytest.raises(AdaptiveRAGWorkflowError, match="evidence verification provider"):
        workflow.run("query")


def test_rewrite_provider_failure_is_wrapped_by_workflow_boundary() -> None:
    workflow = _workflow(
        retriever=FakeRetriever({"query": [ScoredDocument("d1", 1.0, 1)]}),
        verifier=FakeVerifier([EvidenceAssessment(False, "missing evidence")]),
        rewriter=FailingRewriter([]),
    )

    with pytest.raises(AdaptiveRAGWorkflowError, match="query rewrite provider"):
        workflow.run("query")


def test_blank_input_query_is_rejected_before_graph_execution() -> None:
    retriever = FakeRetriever({})
    workflow = _workflow(retriever=retriever)

    with pytest.raises(ValueError, match="must not be blank"):
        workflow.run("  ")
    assert retriever.calls == []


def test_loads_checked_in_phase_6_config() -> None:
    config = load_adaptive_rag_config(Path("configs/adaptive.yaml"))

    assert config.retrieval_top_k == 10
    assert config.max_retrieval_attempts == 2


def test_config_rejects_unknown_fields_and_unbounded_attempts(tmp_path: Path) -> None:
    invalid_path = tmp_path / "adaptive.yaml"
    invalid_path.write_text(
        "name: invalid\nretrieval_top_k: 10\nmax_context_characters: 1000\n"
        "max_retrieval_attempts: 2\nprovider: hidden\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_adaptive_rag_config(invalid_path)
    with pytest.raises(ValidationError):
        _config(maximum_attempts=6)
