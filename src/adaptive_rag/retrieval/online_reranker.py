"""Alibaba Model Studio qwen3.7 text rerank adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from adaptive_rag.domain.protocols import ProviderInvocationError
from adaptive_rag.providers.http import RetryingJsonClient


class OnlineRerankerError(ProviderInvocationError):
    """Raised when a rerank response violates the configured contract."""


class QwenTextReranker:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        url: str,
        model: str,
        instruction: str,
        timeout_seconds: float,
        max_retries: int,
        transport: RetryingJsonClient | None = None,
    ) -> None:
        if any(not value.strip() for value in (provider, api_key, url, model, instruction)):
            raise ValueError("reranker configuration values must not be blank")
        self._provider = provider.strip()
        self._api_key = api_key
        self._url = url
        self._model = model.strip()
        self._instruction = instruction.strip()
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

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if not documents:
            return []
        if any(not document.strip() for document in documents):
            raise ValueError("documents cannot contain blank text")
        body = self._transport.post(
            self._url,
            api_key=self._api_key,
            payload={
                "model": self._model,
                "input": {"query": query, "documents": list(documents)},
                "parameters": {
                    "top_n": len(documents),
                    "instruct": self._instruction,
                },
            },
        )
        output = body.get("output")
        raw_results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(raw_results, list) or len(raw_results) != len(documents):
            count = len(raw_results) if isinstance(raw_results, list) else 0
            raise OnlineRerankerError(
                f"Reranker returned {count} results for {len(documents)} documents"
            )
        scores: list[float | None] = [None] * len(documents)
        previous_score = math.inf
        for result in raw_results:
            if not isinstance(result, dict):
                raise OnlineRerankerError("Reranker returned an invalid result")
            index = result.get("index")
            raw_score = result.get("relevance_score")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(scores)
            ):
                raise OnlineRerankerError("Reranker returned an invalid document index")
            if scores[index] is not None:
                raise OnlineRerankerError("Reranker returned a duplicate document index")
            try:
                score = float(cast(float | int | str, raw_score))
            except (TypeError, ValueError) as error:
                raise OnlineRerankerError("Reranker returned a non-numeric score") from error
            if not math.isfinite(score):
                raise OnlineRerankerError("Reranker returned a non-finite score")
            if score > previous_score:
                raise OnlineRerankerError("Reranker results are not sorted by descending score")
            previous_score = score
            scores[index] = score
        if any(score is None for score in scores):
            raise OnlineRerankerError("Reranker omitted one or more document indexes")
        return [cast(float, score) for score in scores]

    def close(self) -> None:
        self._transport.close()
