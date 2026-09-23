"""OpenAI-compatible online embedding adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from adaptive_rag.domain.protocols import ProviderInvocationError
from adaptive_rag.providers.http import RetryingJsonClient


class EmbeddingError(ProviderInvocationError):
    """Raised when an embedding response violates the configured contract."""


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int,
        batch_size: int,
        timeout_seconds: float,
        max_retries: int,
        transport: RetryingJsonClient | None = None,
    ) -> None:
        if not provider.strip() or not api_key.strip() or not model.strip():
            raise ValueError("embedding provider, API key, and model must not be blank")
        if dimension < 1 or not 1 <= batch_size <= 10:
            raise ValueError("embedding dimension must be positive and batch size must be 1..10")
        self._provider = provider.strip()
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._model = model.strip()
        self._dimension = dimension
        self._batch_size = batch_size
        self._transport = transport or RetryingJsonClient(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("documents cannot contain blank text")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed(list(texts[start : start + self._batch_size])))
        return vectors

    def encode_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query cannot be blank")
        return self._embed([text])[0]

    def close(self) -> None:
        self._transport.close()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        body = self._transport.post(
            self._url,
            api_key=self._api_key,
            payload={
                "model": self._model,
                "input": texts,
                "dimensions": self._dimension,
                "encoding_format": "float",
            },
        )
        raw_data = body.get("data")
        if not isinstance(raw_data, list) or len(raw_data) != len(texts):
            count = len(raw_data) if isinstance(raw_data, list) else 0
            raise EmbeddingError(
                f"Embedding provider returned {count} vectors for {len(texts)} inputs"
            )
        vectors: list[list[float]] = []
        for expected_index, item in enumerate(raw_data):
            if not isinstance(item, dict) or item.get("index") != expected_index:
                raise EmbeddingError("Embedding provider returned invalid or unordered indexes")
            raw_vector = item.get("embedding")
            if not isinstance(raw_vector, list):
                raise EmbeddingError("Embedding provider returned an invalid vector")
            try:
                vector = [float(cast(float | int | str, value)) for value in raw_vector]
            except (TypeError, ValueError) as error:
                raise EmbeddingError("Embedding provider returned a non-numeric vector") from error
            if len(vector) != self._dimension:
                raise EmbeddingError(
                    f"Embedding provider returned dimension {len(vector)}; "
                    f"expected {self._dimension}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("Embedding provider returned a non-finite vector value")
            vectors.append(vector)
        return vectors
