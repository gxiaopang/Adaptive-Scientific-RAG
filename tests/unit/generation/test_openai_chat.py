from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from openai import OpenAI
from pydantic import BaseModel

from adaptive_rag.generation.openai_chat import LLMInvocationError, OpenAIChatModel


class _Payload(BaseModel):
    value: str


def _model(completions: Mock) -> OpenAIChatModel:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIChatModel(
        api_key="secret",
        base_url="http://llm.test/v1",
        model="test-model",
        timeout_seconds=12,
        max_retries=1,
        temperature=0.2,
        max_tokens=321,
        client=cast(OpenAI, client),
    )


def test_text_completion_uses_configured_chat_parameters() -> None:
    completions = Mock()
    completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" answer ", refusal=None))]
    )
    model = _model(completions)
    messages = [{"role": "user", "content": "question"}]

    assert model.complete(messages) == "answer"

    completions.create.assert_called_once_with(
        model="test-model",
        messages=messages,
        temperature=0.2,
        max_tokens=321,
    )


def test_structured_completion_requests_and_returns_pydantic_model() -> None:
    completions = Mock()
    payload = _Payload(value="parsed")
    completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"value":"parsed"}', refusal=None)
            )
        ]
    )
    model = _model(completions)
    messages = [{"role": "user", "content": "question"}]

    assert model.complete_structured(messages, _Payload) == payload

    completions.create.assert_called_once_with(
        model="test-model",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=321,
    )


def test_text_completion_can_disable_provider_thinking_mode() -> None:
    completions = Mock()
    completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer", refusal=None))]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAIChatModel(
        api_key="secret",
        base_url="http://llm.test/v1",
        model="test-model",
        timeout_seconds=12,
        max_retries=1,
        temperature=0.2,
        max_tokens=321,
        enable_thinking=False,
        client=cast(OpenAI, client),
    )
    messages = [{"role": "user", "content": "question"}]

    assert model.complete(messages) == "answer"
    completions.create.assert_called_once_with(
        model="test-model",
        messages=messages,
        temperature=0.2,
        max_tokens=321,
        extra_body={"enable_thinking": False},
    )


def test_structured_completion_can_disable_provider_thinking_mode() -> None:
    completions = Mock()
    completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"value":"parsed"}', refusal=None)
            )
        ]
    )
    model = _model(completions)
    messages = [{"role": "user", "content": "question"}]

    assert model.complete_structured(
        messages,
        _Payload,
        enable_thinking=False,
    ) == _Payload(value="parsed")
    completions.create.assert_called_once_with(
        model="test-model",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=321,
        extra_body={"enable_thinking": False},
    )


def test_streaming_completion_emits_deltas_and_returns_complete_text() -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.closed = False
            self._chunks = iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=" first"))]
                    ),
                    SimpleNamespace(choices=[]),
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=" answer "))]
                    ),
                ]
            )

        def __iter__(self) -> "FakeStream":
            return self

        def __next__(self) -> object:
            return next(self._chunks)

        def close(self) -> None:
            self.closed = True

    completions = Mock()
    stream = FakeStream()
    completions.create.return_value = stream
    messages = [{"role": "user", "content": "question"}]
    deltas: list[str] = []

    answer = _model(completions).complete_stream(
        messages,
        deltas.append,
        enable_thinking=False,
    )

    assert answer == "first answer"
    assert deltas == [" first", " answer "]
    assert stream.closed is True
    completions.create.assert_called_once_with(
        model="test-model",
        messages=messages,
        temperature=0.2,
        max_tokens=321,
        stream=True,
        extra_body={"enable_thinking": False},
    )


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(content=None, refusal=None),
        SimpleNamespace(content="  ", refusal=None),
        SimpleNamespace(content="ignored", refusal="blocked"),
    ],
)
def test_text_completion_rejects_missing_content_or_refusal(message: object) -> None:
    completions = Mock()
    completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )

    with pytest.raises(LLMInvocationError):
        _model(completions).complete([{"role": "user", "content": "question"}])


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '{"wrong_field":"value"}',
        '{"value":123}',
    ],
)
def test_structured_completion_rejects_invalid_json_or_schema(content: str) -> None:
    completions = Mock()
    completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None))]
    )

    with pytest.raises(LLMInvocationError, match="invalid structured"):
        _model(completions).complete_structured(
            [{"role": "user", "content": "question"}],
            _Payload,
        )


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(content=None, refusal=None),
        SimpleNamespace(content="  ", refusal=None),
        SimpleNamespace(content='{"value":"ignored"}', refusal="blocked"),
    ],
)
def test_structured_completion_rejects_missing_content_or_refusal(message: object) -> None:
    completions = Mock()
    completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )

    with pytest.raises(LLMInvocationError):
        _model(completions).complete_structured(
            [{"role": "user", "content": "return JSON"}],
            _Payload,
        )
