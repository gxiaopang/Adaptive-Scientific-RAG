"""Small synchronous OpenAI Chat Completions boundary."""

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject
from pydantic import BaseModel, ValidationError

from adaptive_rag.domain.protocols import ProviderInvocationError

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class LLMInvocationError(ProviderInvocationError):
    """Raised when an LLM request or response violates the adapter contract."""


class ChatModel(Protocol):
    """Provider-neutral capability consumed by the three workflow adapters."""

    def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        enable_thinking: bool | None = None,
    ) -> str: ...

    def complete_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        on_delta: Callable[[str], None],
        *,
        enable_thinking: bool | None = None,
    ) -> str: ...

    def complete_structured(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        response_model: type[StructuredOutput],
        *,
        enable_thinking: bool | None = None,
    ) -> StructuredOutput: ...


class OpenAIChatModel:
    """Invoke one configured OpenAI-compatible Chat Completions model."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool | None = None,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        if not model.strip():
            raise ValueError("model must not be blank")
        if timeout_seconds <= 0 or max_retries < 0 or max_tokens < 1:
            raise ValueError("LLM timeout, retries, and token limit are invalid")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        enable_thinking: bool | None = None,
    ) -> str:
        """Return a non-empty text completion."""

        try:
            thinking = self._resolve_thinking(enable_thinking)
            if thinking is None:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            else:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    extra_body={"enable_thinking": thinking},
                )
        except (OpenAIError, TypeError, ValueError) as error:
            raise LLMInvocationError("OpenAI Chat Completions request failed") from error

        if not completion.choices:
            raise LLMInvocationError("OpenAI returned no completion choices")
        message = completion.choices[0].message
        if message.refusal:
            raise LLMInvocationError("OpenAI refused the completion request")
        content = message.content
        if content is None or not content.strip():
            raise LLMInvocationError("OpenAI returned an empty completion")
        return content.strip()

    def complete_stream(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        on_delta: Callable[[str], None],
        *,
        enable_thinking: bool | None = None,
    ) -> str:
        """Stream visible answer text while returning the complete validated content."""

        stream = None
        try:
            thinking = self._resolve_thinking(enable_thinking)
            if thinking is None:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    stream=True,
                )
            else:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    stream=True,
                    extra_body={"enable_thinking": thinking},
                )
            parts: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    parts.append(content)
                    on_delta(content)
        except (OpenAIError, TypeError, ValueError) as error:
            raise LLMInvocationError("OpenAI streaming completion request failed") from error
        finally:
            if stream is not None:
                stream.close()
        content = "".join(parts).strip()
        if not content:
            raise LLMInvocationError("OpenAI returned an empty streaming completion")
        return content

    def complete_structured(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        response_model: type[StructuredOutput],
        *,
        enable_thinking: bool | None = None,
    ) -> StructuredOutput:
        """Request portable JSON mode output and validate it locally with Pydantic."""

        try:
            response_format = ResponseFormatJSONObject(type="json_object")
            thinking = self._resolve_thinking(enable_thinking)
            if thinking is None:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format=response_format,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            else:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format=response_format,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    extra_body={"enable_thinking": thinking},
                )
        except (OpenAIError, TypeError, ValueError) as error:
            raise LLMInvocationError("OpenAI structured completion request failed") from error

        if not completion.choices:
            raise LLMInvocationError("OpenAI returned no structured completion choices")
        message = completion.choices[0].message
        if message.refusal:
            raise LLMInvocationError("OpenAI refused the structured completion request")
        content = message.content
        if content is None or not content.strip():
            raise LLMInvocationError("OpenAI returned an empty structured completion")
        try:
            return response_model.model_validate_json(content)
        except ValidationError as error:
            raise LLMInvocationError("OpenAI returned an invalid structured completion") from error

    def close(self) -> None:
        """Release the underlying provider client's connection pool."""

        self._client.close()

    def _resolve_thinking(self, override: bool | None) -> bool | None:
        return self._enable_thinking if override is None else override
