"""Bounded adaptive retrieval workflow built with LangGraph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol, cast

import yaml
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.domain.models import (
    ConversationTurn,
    EvidenceAssessment,
    EvidenceDocument,
    QueryRoutingDecision,
)
from adaptive_rag.domain.protocols import (
    AnswerGenerator,
    EvidenceVerifier,
    GeneralChatGenerator,
    ProviderInvocationError,
    QueryRewriter,
    QueryRouter,
    Retriever,
)
from adaptive_rag.graph.context import build_evidence_context
from adaptive_rag.graph.state import (
    AdaptiveRAGResult,
    AdaptiveRAGState,
    AdaptiveRAGUpdate,
    GenerationSnapshot,
)


class AdaptiveRAGConfig(BaseModel):
    """Validated runtime limits for the Phase 6 workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    retrieval_top_k: int = Field(ge=1)
    max_context_characters: int = Field(ge=256)
    max_retrieval_attempts: int = Field(ge=1, le=5)


class AdaptiveRAGWorkflowError(RuntimeError):
    """Raised when a workflow dependency violates its runtime contract."""


class _CompiledGraph(Protocol):
    def invoke(
        self,
        input: AdaptiveRAGState,
        config: Mapping[str, object] | None = None,
    ) -> AdaptiveRAGState: ...


def load_adaptive_rag_config(path: Path) -> AdaptiveRAGConfig:
    """Load and validate the workflow configuration from YAML."""

    with path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        msg = f"adaptive RAG config must be a mapping: {path}"
        raise ValueError(msg)
    return AdaptiveRAGConfig.model_validate(raw_config)


class AdaptiveRAGWorkflow:
    """Coordinate retrieval, generation, verification, and bounded query rewriting."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        corpus: Mapping[str, CorpusDocument],
        router: QueryRouter,
        chat_generator: GeneralChatGenerator,
        generator: AnswerGenerator,
        verifier: EvidenceVerifier,
        rewriter: QueryRewriter,
        config: AdaptiveRAGConfig,
    ) -> None:
        if not corpus:
            msg = "corpus must not be empty"
            raise ValueError(msg)

        self._retriever = retriever
        self._corpus = corpus
        self._router = router
        self._chat_generator = chat_generator
        self._generator = generator
        self._verifier = verifier
        self._rewriter = rewriter
        self._config = config
        self._graph = self._build_graph()

    def run(
        self,
        query: str,
        history: Sequence[ConversationTurn] = (),
        *,
        on_answer_start: Callable[[], None] | None = None,
        on_answer_delta: Callable[[str], None] | None = None,
    ) -> AdaptiveRAGResult:
        """Run the compiled graph for one user query."""

        normalized_query = query.strip()
        if not normalized_query:
            msg = "query must not be blank"
            raise ValueError(msg)

        initial_state = AdaptiveRAGState(
            snapshots=(),
            original_query=normalized_query,
            resolved_query=normalized_query,
            conversation_history=tuple(history),
            routing_decision=None,
            current_query=normalized_query,
            query_history=(normalized_query,),
            evidence=(),
            context="",
            answer="",
            assessment=None,
            retrieval_attempts=0,
            termination_reason=None,
            on_answer_start=on_answer_start,
            on_answer_delta=on_answer_delta,
        )
        recursion_limit = self._config.max_retrieval_attempts * 6 + 6
        try:
            final_state = self._graph.invoke(
                initial_state,
                config={"recursion_limit": recursion_limit},
            )
        except GraphRecursionError as error:
            msg = "adaptive RAG graph exceeded its recursion limit"
            raise AdaptiveRAGWorkflowError(msg) from error

        assessment = final_state["assessment"]
        routing_decision = final_state["routing_decision"]
        termination_reason = final_state["termination_reason"]
        if routing_decision is None or termination_reason is None:
            msg = "adaptive RAG graph ended without a routed answer"
            raise AdaptiveRAGWorkflowError(msg)
        if routing_decision.route == "scientific_evidence" and assessment is None:
            msg = "scientific RAG route ended without an evidence assessment"
            raise AdaptiveRAGWorkflowError(msg)
        if routing_decision.route == "general_chat" and (
            assessment is not None
            or final_state["retrieval_attempts"] != 0
            or final_state["evidence"]
        ):
            msg = "general chat route unexpectedly used retrieval state"
            raise AdaptiveRAGWorkflowError(msg)

        return AdaptiveRAGResult(
            snapshots=final_state["snapshots"],
            original_query=final_state["original_query"],
            response_mode=routing_decision.route,
            routing_reason=routing_decision.reason,
            final_query=final_state["current_query"],
            query_history=final_state["query_history"],
            evidence=final_state["evidence"],
            answer=final_state["answer"],
            evidence_sufficient=assessment.sufficient if assessment is not None else None,
            verification_reason=assessment.reason if assessment is not None else None,
            retrieval_attempts=final_state["retrieval_attempts"],
            termination_reason=termination_reason,
        )

    def _build_graph(self) -> _CompiledGraph:
        graph = StateGraph(AdaptiveRAGState)
        graph.add_node("classify_query", self._classify_query)
        graph.add_node("general_chat", self._general_chat)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("build_context", self._build_context)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)
        graph.add_node("accept", self._accept)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("exhaust", self._exhaust)

        graph.add_edge(START, "classify_query")
        graph.add_conditional_edges(
            "classify_query",
            self._route_after_classification,
            {
                "scientific_evidence": "retrieve",
                "general_chat": "general_chat",
            },
        )
        graph.add_edge("general_chat", END)
        graph.add_edge("retrieve", "build_context")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", "verify")
        graph.add_conditional_edges(
            "verify",
            self._route_after_verification,
            {
                "accept": "accept",
                "rewrite": "rewrite",
                "exhaust": "exhaust",
            },
        )
        graph.add_edge("accept", END)
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("exhaust", END)
        return cast(_CompiledGraph, graph.compile())

    def _classify_query(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        try:
            decision = self._router.classify(
                state["original_query"],
                state["conversation_history"],
            )
        except ProviderInvocationError as error:
            msg = "query routing provider failed"
            raise AdaptiveRAGWorkflowError(msg) from error
        if not isinstance(decision, QueryRoutingDecision):
            msg = "query router returned an invalid decision"
            raise AdaptiveRAGWorkflowError(msg)
        if decision.route not in {"scientific_evidence", "general_chat"}:
            msg = "query router returned an unsupported route"
            raise AdaptiveRAGWorkflowError(msg)
        if not decision.reason.strip():
            msg = "query router returned a blank reason"
            raise AdaptiveRAGWorkflowError(msg)
        standalone_query = decision.standalone_query.strip()
        if not standalone_query:
            msg = "query router returned a blank standalone query"
            raise AdaptiveRAGWorkflowError(msg)
        if decision.route == "scientific_evidence":
            return AdaptiveRAGUpdate(
                routing_decision=decision,
                resolved_query=standalone_query,
                current_query=standalone_query,
                query_history=(standalone_query,),
            )
        return AdaptiveRAGUpdate(
            routing_decision=decision,
            resolved_query=standalone_query,
        )

    def _route_after_classification(
        self,
        state: AdaptiveRAGState,
    ) -> Literal["scientific_evidence", "general_chat"]:
        decision = state["routing_decision"]
        if decision is None:
            msg = "query routing requires a decision"
            raise AdaptiveRAGWorkflowError(msg)
        return decision.route

    def _general_chat(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        try:
            on_delta = state["on_answer_delta"]
            if on_delta is None:
                answer = self._chat_generator.chat(
                    state["original_query"],
                    state["conversation_history"],
                ).strip()
            else:
                on_start = state["on_answer_start"]
                if on_start is not None:
                    on_start()
                answer = self._chat_generator.chat(
                    state["original_query"],
                    state["conversation_history"],
                    on_delta,
                ).strip()
        except ProviderInvocationError as error:
            msg = "general chat provider failed"
            raise AdaptiveRAGWorkflowError(msg) from error
        if not answer:
            msg = "general chat generator returned a blank answer"
            raise AdaptiveRAGWorkflowError(msg)
        return AdaptiveRAGUpdate(
            answer=answer,
            termination_reason="general_chat",
        )

    def _retrieve(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        results = self._retriever.retrieve(
            state["current_query"],
            self._config.retrieval_top_k,
        )
        if not results:
            msg = "retriever returned no candidates"
            raise AdaptiveRAGWorkflowError(msg)
        if len(results) > self._config.retrieval_top_k:
            msg = "retriever returned more candidates than requested"
            raise AdaptiveRAGWorkflowError(msg)

        evidence: list[EvidenceDocument] = []
        seen_doc_ids: set[str] = set()
        for expected_rank, result in enumerate(results, start=1):
            if result.rank != expected_rank:
                msg = "retriever results must use consecutive one-based ranks"
                raise AdaptiveRAGWorkflowError(msg)
            if result.doc_id in seen_doc_ids:
                msg = f"retriever returned duplicate document: {result.doc_id}"
                raise AdaptiveRAGWorkflowError(msg)
            if not isfinite(result.score):
                msg = f"retriever returned a non-finite score: {result.doc_id}"
                raise AdaptiveRAGWorkflowError(msg)
            document = self._corpus.get(result.doc_id)
            if document is None:
                msg = f"retriever returned unknown document: {result.doc_id}"
                raise AdaptiveRAGWorkflowError(msg)
            seen_doc_ids.add(result.doc_id)
            evidence.append(
                EvidenceDocument(
                    doc_id=document.doc_id,
                    title=document.title,
                    text=document.text,
                    score=result.score,
                    rank=result.rank,
                )
            )

        return AdaptiveRAGUpdate(
            evidence=tuple(evidence),
            retrieval_attempts=state["retrieval_attempts"] + 1,
        )

    def _build_context(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        context = build_evidence_context(
            state["evidence"],
            maximum_characters=self._config.max_context_characters,
        )
        return AdaptiveRAGUpdate(context=context)

    def _generate(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        try:
            on_delta = state["on_answer_delta"]
            if on_delta is None:
                answer = self._generator.generate(
                    state["resolved_query"],
                    state["context"],
                ).strip()
            else:
                on_start = state["on_answer_start"]
                if on_start is not None:
                    on_start()
                answer = self._generator.generate(
                    state["resolved_query"],
                    state["context"],
                    on_delta,
                ).strip()
        except ProviderInvocationError as error:
            msg = "answer generation provider failed"
            raise AdaptiveRAGWorkflowError(msg) from error
        if not answer:
            msg = "answer generator returned a blank answer"
            raise AdaptiveRAGWorkflowError(msg)
        snapshot = GenerationSnapshot(
            query=state["resolved_query"],
            retrieval_query=state["current_query"],
            context=state["context"],
            answer=answer,
            evidence=state["evidence"],
            attempt=state["retrieval_attempts"],
        )
        return AdaptiveRAGUpdate(answer=answer, snapshots=(*state["snapshots"], snapshot))

    def _verify(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        try:
            assessment = self._verifier.verify(
                state["resolved_query"],
                state["answer"],
                state["context"],
                state["evidence"],
            )
        except ProviderInvocationError as error:
            msg = "evidence verification provider failed"
            raise AdaptiveRAGWorkflowError(msg) from error
        if not isinstance(assessment, EvidenceAssessment):
            msg = "evidence verifier returned an invalid assessment"
            raise AdaptiveRAGWorkflowError(msg)
        if not assessment.reason.strip():
            msg = "evidence verifier returned a blank reason"
            raise AdaptiveRAGWorkflowError(msg)
        return AdaptiveRAGUpdate(assessment=assessment)

    def _route_after_verification(
        self,
        state: AdaptiveRAGState,
    ) -> Literal["accept", "rewrite", "exhaust"]:
        assessment = state["assessment"]
        if assessment is None:
            msg = "verification route requires an assessment"
            raise AdaptiveRAGWorkflowError(msg)
        if assessment.sufficient:
            return "accept"
        if state["retrieval_attempts"] >= self._config.max_retrieval_attempts:
            return "exhaust"
        return "rewrite"

    def _accept(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        del state
        return AdaptiveRAGUpdate(termination_reason="evidence_sufficient")

    def _exhaust(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        del state
        return AdaptiveRAGUpdate(termination_reason="max_retrieval_attempts")

    def _rewrite(self, state: AdaptiveRAGState) -> AdaptiveRAGUpdate:
        assessment = state["assessment"]
        if assessment is None:
            msg = "query rewrite requires an assessment"
            raise AdaptiveRAGWorkflowError(msg)

        try:
            rewritten_query = self._rewriter.rewrite(
                original_query=state["resolved_query"],
                current_query=state["current_query"],
                context=state["context"],
                assessment=assessment,
                retrieval_attempt=state["retrieval_attempts"],
            ).strip()
        except ProviderInvocationError as error:
            msg = "query rewrite provider failed"
            raise AdaptiveRAGWorkflowError(msg) from error
        if not rewritten_query:
            msg = "query rewriter returned a blank query"
            raise AdaptiveRAGWorkflowError(msg)
        if _normalize_for_comparison(rewritten_query) == _normalize_for_comparison(
            state["current_query"]
        ):
            msg = "query rewriter did not change the query"
            raise AdaptiveRAGWorkflowError(msg)

        return AdaptiveRAGUpdate(
            current_query=rewritten_query,
            query_history=(*state["query_history"], rewritten_query),
            evidence=(),
            context="",
            answer="",
            assessment=None,
            termination_reason=None,
        )


def _normalize_for_comparison(query: str) -> str:
    return " ".join(query.split()).casefold()
