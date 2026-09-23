"""Small synchronous JSON transport with explicit bounded retry behavior."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import httpx

from adaptive_rag.domain.protocols import ProviderInvocationError


class JsonHttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
        timeout: float,
    ) -> httpx.Response: ...

    def close(self) -> None: ...


class ProviderHttpError(ProviderInvocationError):
    """Raised for a sanitized HTTP transport or status failure."""


class RetryingJsonClient:
    """POST JSON and retry only timeout, rate-limit, and server failures."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_retries: int,
        client: JsonHttpClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("provider timeout and retry count are invalid")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._client = client or httpx.Client()
        self._sleep = sleep

    def post(self, url: str, *, api_key: str, payload: Mapping[str, object]) -> dict[str, Any]:
        if not api_key.strip():
            raise ValueError("api_key must not be blank")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            except httpx.TimeoutException as error:
                if attempt >= self._max_retries:
                    raise ProviderHttpError("Provider request timed out") from error
                self._backoff(attempt)
                continue
            except httpx.HTTPError as error:
                raise ProviderHttpError("Provider HTTP transport failed") from error

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= self._max_retries:
                    raise ProviderHttpError(
                        f"Provider request failed after retries with HTTP {response.status_code}"
                    )
                self._backoff(attempt)
                continue
            if response.status_code >= 400:
                raise ProviderHttpError(
                    f"Provider rejected request with HTTP {response.status_code}"
                )
            try:
                body = response.json()
            except ValueError as error:
                raise ProviderHttpError("Provider returned malformed JSON") from error
            if not isinstance(body, dict):
                raise ProviderHttpError("Provider returned a non-object JSON response")
            return cast(dict[str, Any], body)
        raise AssertionError("retry loop exhausted unexpectedly")

    def close(self) -> None:
        self._client.close()

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(0.5 * (2**attempt), 4.0))
