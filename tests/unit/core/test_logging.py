import json
import logging

import pytest

from adaptive_rag.core.context import bind_request_context, reset_request_context
from adaptive_rag.core.logging import JsonFormatter, configure_logging


def test_json_formatter_emits_expected_fields() -> None:
    record = logging.LogRecord(
        name="adaptive_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="索引准备完成",
        args=(),
        exc_info=None,
    )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "adaptive_rag.test"
    assert payload["message"] == "索引准备完成"
    assert payload["timestamp"].endswith("Z")


def test_configure_logging_is_idempotent() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    try:
        root_logger.handlers.clear()

        configure_logging("INFO")
        configure_logging("DEBUG")

        application_handlers = [
            handler
            for handler in root_logger.handlers
            if getattr(handler, "_adaptive_rag_json_handler", False)
        ]
        assert len(application_handlers) == 1
        assert application_handlers[0].level == logging.DEBUG
        assert root_logger.level == logging.DEBUG
    finally:
        root_logger.handlers[:] = original_handlers
        root_logger.setLevel(original_level)


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("TRACE")


def test_json_formatter_adds_bound_context_and_observability_fields() -> None:
    tokens = bind_request_context(request_id="request-123", trace_id="trace-456")
    try:
        record = logging.LogRecord(
            name="adaptive_rag.api",
            level=logging.INFO,
            pathname=__file__,
            lineno=40,
            msg="http request completed",
            args=(),
            exc_info=None,
        )
        record.http_method = "POST"
        record.http_path = "/api/v1/query"
        record.status_code = 200
        record.duration_ms = 12.5

        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_context(tokens)

    assert payload["request_id"] == "request-123"
    assert payload["trace_id"] == "trace-456"
    assert payload["http_method"] == "POST"
    assert payload["http_path"] == "/api/v1/query"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 12.5
