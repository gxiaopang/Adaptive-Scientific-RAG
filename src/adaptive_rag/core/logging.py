"""Minimal structured logging built on Python's standard library."""

import json
import logging
from datetime import UTC, datetime
from typing import Final

from adaptive_rag.core.context import get_request_context

_HANDLER_MARKER: Final = "_adaptive_rag_json_handler"
_CONTEXT_FIELDS: Final = (
    "request_id",
    "trace_id",
    "query_run_id",
    "http_method",
    "http_path",
    "status_code",
    "duration_ms",
)


class JsonFormatter(logging.Formatter):
    """Serialize the stable application log fields as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field_name in _CONTEXT_FIELDS:
            if hasattr(record, field_name):
                payload[field_name] = getattr(record, field_name)

        request_context = get_request_context()
        if request_context.request_id is not None:
            payload.setdefault("request_id", request_context.request_id)
        if request_context.trace_id is not None:
            payload.setdefault("trace_id", request_context.trace_id)

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | int = "INFO") -> None:
    """Configure one reusable JSON handler on the root logger.

    The function deliberately leaves handlers owned by the hosting process alone.
    Calling it repeatedly updates this application's handler instead of adding a
    duplicate, which is important for tests and development reloads.
    """

    resolved_level = _resolve_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    for handler in root_logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            handler.setLevel(resolved_level)
            handler.setFormatter(JsonFormatter())
            return

    handler = logging.StreamHandler()
    handler.setLevel(resolved_level)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _HANDLER_MARKER, True)
    root_logger.addHandler(handler)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    resolved_level = logging.getLevelName(level.upper())
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unknown log level: {level}")
    return resolved_level
