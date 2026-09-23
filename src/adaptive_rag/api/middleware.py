"""Pure ASGI request correlation and access logging middleware."""

import logging
import re
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from adaptive_rag.core.context import bind_request_context, reset_request_context

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REQUEST_ID_HEADER = "x-request-id"
_TRACE_ID_HEADER = "x-trace-id"


class RequestContextMiddleware:
    """Correlate one HTTP request without storing mutable per-request instance state."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = _resolve_identifier(headers.get(_REQUEST_ID_HEADER))
        trace_id = _resolve_identifier(headers.get(_TRACE_ID_HEADER), fallback=request_id)
        tokens = bind_request_context(request_id=request_id, trace_id=trace_id)
        started_at = perf_counter()
        status_code = 500

        async def send_with_context(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = MutableHeaders(scope=message)
                response_headers[_REQUEST_ID_HEADER] = request_id
                response_headers[_TRACE_ID_HEADER] = trace_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        finally:
            duration_ms = (perf_counter() - started_at) * 1_000
            log_level = logging.ERROR if status_code >= 500 else logging.INFO
            logger.log(
                log_level,
                "http request completed",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "http_method": scope["method"],
                    "http_path": scope["path"],
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 3),
                },
            )
            reset_request_context(tokens)


def _resolve_identifier(value: str | None, *, fallback: str | None = None) -> str:
    if value is not None and _IDENTIFIER_PATTERN.fullmatch(value):
        return value
    if fallback is not None:
        return fallback
    return uuid4().hex
