"""FastAPI application factory for Adaptive Scientific RAG."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adaptive_rag import __version__
from adaptive_rag.api.dependencies import (
    ConversationMemory,
    ReadinessProbe,
    WorkflowRunner,
)
from adaptive_rag.api.middleware import RequestContextMiddleware
from adaptive_rag.api.routes import create_api_router
from adaptive_rag.core.config import Settings, get_settings
from adaptive_rag.core.logging import configure_logging
from adaptive_rag.observability.query_runs import QueryRunRecorder

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    workflow: WorkflowRunner | None = None,
    conversation_memory: ConversationMemory | None = None,
    query_run_recorder: QueryRunRecorder | None = None,
    readiness_probe: ReadinessProbe | None = None,
    shutdown: Callable[[], None] | None = None,
) -> FastAPI:
    """Create an application with explicitly supplied runtime dependencies."""

    resolved_settings = settings if settings is not None else get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                logger.exception("application resource shutdown failed")

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    if resolved_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID"],
        )
    application.add_middleware(RequestContextMiddleware)
    application.include_router(
        create_api_router(
            settings=resolved_settings,
            workflow=workflow,
            conversation_memory=conversation_memory,
            readiness_probe=readiness_probe,
            query_run_recorder=query_run_recorder,
        )
    )
    return application


app = create_app()
