from collections.abc import Callable, Sequence
from typing import TypeVar, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from adaptive_rag.domain.models import (
    ConversationTurn,
    EvidenceAssessment,
    EvidenceDocument,
    QueryRoutingDecision,
)
from adaptive_rag.generation.adapters import (
    OpenAIAnswerGenerator,
    OpenAIEvidenceVerifier,
    OpenAIGeneralChatGenerator,
    OpenAIQueryRewriter,
    OpenAIQueryRouter,
)

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class FakeChatModel:
    def __init__(self) -> None:
        self.text_messages: list[Sequence[ChatCompletionMessageParam]] = []
        self.structured_messages: list[Sequence[ChatCompletionMessageParam]] = []
        self.structured_payloads: list[dict[str, object]] = []
        self.text_thinking: list[bool | None] = []
        self.structured_thinking: list[bool | None] = []
        self.stream_thinking: list[bool | None] = []

    def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        enable_thinking: bool | None = None,
    ) -> str:
        self.text_messages.append(messages)
        self.text_thinking.append(enable_thinking)
        return "Evidence-bounded answer [document:d1]."

    def complete_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        on_delta: Callable[[str], None],
        *,
        enable_thinking: bool | None = None,
    ) -> str:
        self.text_messages.append(messages)
        self.stream_thinking.append(enable_thinking)
        on_delta("Evidence-bounded ")
        on_delta("answer [document:d1].")
        return "Evidence-bounded answer [document:d1]."

    def complete_structured(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        response_model: type[StructuredOutput],
        *,
        enable_thinking: bool | None = None,
    ) -> StructuredOutput:
        self.structured_messages.append(messages)
        self.structured_thinking.append(enable_thinking)
        payload = self.structured_payloads.pop(0)
        return response_model.model_validate(payload)


def _content(message: ChatCompletionMessageParam) -> str:
    return cast(str, message["content"])


def test_answer_generator_supplies_question_context_and_grounding_rules() -> None:
    model = FakeChatModel()
    generator = OpenAIAnswerGenerator(model)

    answer = generator.generate("scientific question", "[document:d1]\nevidence")

    assert answer == "Evidence-bounded answer [document:d1]."
    messages = model.text_messages[0]
    assert "only the supplied evidence" in _content(messages[0])
    assert "scientific question" in _content(messages[1])
    assert "[document:d1]" in _content(messages[1])


def test_query_router_returns_structured_scientific_evidence_decision() -> None:
    model = FakeChatModel()
    model.structured_payloads.append(
        {
            "route": "scientific_evidence",
            "standalone_query": " Does this treatment reduce mortality? ",
            "reason": " requires study evidence ",
        }
    )
    router = OpenAIQueryRouter(model)

    decision = router.classify("Does this treatment reduce mortality?", ())

    assert decision == QueryRoutingDecision(
        route="scientific_evidence",
        reason="requires study evidence",
        standalone_query="Does this treatment reduce mortality?",
    )
    instructions = _content(model.structured_messages[0][0])
    prompt = _content(model.structured_messages[0][1])
    assert "When uncertain, choose scientific_evidence" in instructions
    assert "only a JSON object" in instructions
    assert "Does this treatment reduce mortality?" in prompt
    assert model.structured_thinking == [False]


def test_query_router_uses_history_to_resolve_follow_up() -> None:
    model = FakeChatModel()
    model.structured_payloads.append(
        {
            "route": "scientific_evidence",
            "standalone_query": "What adverse effects does aspirin have?",
            "reason": "medical follow-up",
        }
    )
    history = (
        ConversationTurn(
            1,
            "Does aspirin reduce cardiovascular risk?",
            "Previous evidence-bounded answer.",
            "scientific_evidence",
        ),
    )

    decision = OpenAIQueryRouter(model).classify("What adverse effects does it have?", history)

    assert decision.standalone_query == "What adverse effects does aspirin have?"
    messages = model.structured_messages[0]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "Does aspirin reduce" in _content(messages[1])
    assert "What adverse effects does it have?" in _content(messages[-1])


def test_general_chat_generator_uses_non_retrieval_instructions() -> None:
    model = FakeChatModel()
    chat = OpenAIGeneralChatGenerator(model)

    history = (
        ConversationTurn(1, "My name is Ada", "Nice to meet you.", "general_chat"),
    )
    answer = chat.chat("What is my name?", history)

    assert answer == "Evidence-bounded answer [document:d1]."
    messages = model.text_messages[0]
    assert "general assistant" in _content(messages[0])
    assert "Do not claim that documents were retrieved" in _content(messages[0])
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "My name is Ada" in _content(messages[1])
    assert "What is my name?" in _content(messages[-1])
    assert model.text_thinking == [False]


def test_general_chat_generator_streams_with_thinking_disabled() -> None:
    model = FakeChatModel()
    deltas: list[str] = []

    answer = OpenAIGeneralChatGenerator(model).chat("Hello", (), deltas.append)

    assert answer == "Evidence-bounded answer [document:d1]."
    assert deltas == ["Evidence-bounded ", "answer [document:d1]."]
    assert model.stream_thinking == [False]


def test_verifier_maps_structured_output_to_domain_assessment() -> None:
    model = FakeChatModel()
    model.structured_payloads.append(
        {"sufficient": False, "reason": " missing comparison "}
    )
    verifier = OpenAIEvidenceVerifier(model)
    evidence = [EvidenceDocument("d1", "title", "body", 0.9, 1)]

    assessment = verifier.verify("question", "answer", "context", evidence)

    assert assessment == EvidenceAssessment(False, "missing comparison")
    instructions = _content(model.structured_messages[0][0])
    prompt = _content(model.structured_messages[0][1])
    assert "only a JSON object" in instructions
    assert "sufficient (boolean)" in instructions
    assert "Retrieved document IDs: d1" in prompt
    assert "answer" in prompt
    assert model.structured_thinking == [False]


def test_rewriter_returns_structured_standalone_query() -> None:
    model = FakeChatModel()
    model.structured_payloads.append({"query": "  targeted scientific query  "})
    rewriter = OpenAIQueryRewriter(model)

    rewritten = rewriter.rewrite(
        original_query="question",
        current_query="current query",
        context="evidence",
        assessment=EvidenceAssessment(False, "missing mechanism"),
        retrieval_attempt=1,
    )

    assert rewritten == "targeted scientific query"
    instructions = _content(model.structured_messages[0][0])
    prompt = _content(model.structured_messages[0][1])
    assert "only a JSON object" in instructions
    assert "query (string)" in instructions
    assert "missing mechanism" in prompt
    assert "Attempt: 1" in prompt
    assert model.structured_thinking == [False]


def test_scientific_answer_streams_without_overriding_thinking_setting() -> None:
    model = FakeChatModel()
    deltas: list[str] = []

    answer = OpenAIAnswerGenerator(model).generate(
        "scientific question",
        "[document:d1]\nevidence",
        deltas.append,
    )

    assert answer == "Evidence-bounded answer [document:d1]."
    assert deltas == ["Evidence-bounded ", "answer [document:d1]."]
    assert model.stream_thinking == [None]
