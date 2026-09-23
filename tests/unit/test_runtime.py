from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import FastAPI
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from adaptive_rag.core.config import Settings
from adaptive_rag.domain.models import EmbeddingIndexSnapshot
from adaptive_rag.graph.workflow import AdaptiveRAGWorkflow
from adaptive_rag.memory import ConversationMemoryService
from adaptive_rag.observability.query_runs import QueryRunRecorder
from adaptive_rag.runtime import (
    PostgresResources,
    RuntimeConfigurationError,
    RuntimeReadiness,
    build_openai_adapters,
    build_postgres_dependencies,
    create_runtime_app,
)
from adaptive_rag.storage.postgres.vector_store import PgVectorDenseStore


class FakeRuntimeReadiness:
    def __init__(self, engine: Engine, store: PgVectorDenseStore) -> None:
        self.engine = engine
        self.store = store

    def check(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, *, extension: bool = True, revision: str = "20260922_0002") -> None:
        self.scalars = [extension, revision]

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def scalar(self, statement: object) -> object:
        return self.scalars.pop(0)

    def execute(self, statement: object) -> SimpleNamespace:
        return SimpleNamespace(one=lambda: ("embedding_indexes", "document_embeddings"))


class FakeEngine:
    def __init__(self, connection: FakeConnection | None = None) -> None:
        self.connection = connection or FakeConnection()
        self.disposed = False

    def connect(self) -> FakeConnection:
        return self.connection

    def dispose(self) -> None:
        self.disposed = True


class FakeReadyStore:
    def __init__(self, *, status: str = "ready", count: int = 2) -> None:
        self.status = status
        self.row_count = count
        self.closed = False

    def snapshot(self) -> EmbeddingIndexSnapshot:
        return EmbeddingIndexSnapshot(
            name="index",
            status=cast(object, self.status),
            provider="provider",
            model="model",
            dimension=1024,
            distance_metric="cosine",
            source_checksum="a" * 64,
            doc_ids_checksum="b" * 64,
            expected_records=2,
            actual_records=self.row_count,
            configuration_version="1",
            updated_at=datetime.now(UTC),
        )

    def count(self) -> int:
        return self.row_count

    def close(self) -> None:
        self.closed = True


def test_openai_adapters_require_api_key_and_model() -> None:
    with pytest.raises(RuntimeConfigurationError, match="ASR_LLM_API_KEY"):
        build_openai_adapters(Settings(_env_file=None))
    with pytest.raises(RuntimeConfigurationError, match="ASR_LLM_MODEL"):
        build_openai_adapters(Settings(_env_file=None, llm_api_key="secret"))


def test_openai_adapters_share_one_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adaptive_rag.runtime as runtime

    shared_model = SimpleNamespace()
    constructor_arguments: dict[str, object] = {}

    def fake_model(**kwargs: object) -> object:
        constructor_arguments.update(kwargs)
        return shared_model

    monkeypatch.setattr(runtime, "OpenAIChatModel", fake_model)
    router, chat_generator, generator, verifier, rewriter, model = build_openai_adapters(
        Settings(
            _env_file=None,
            llm_api_key="secret",
            llm_base_url="http://llm.test/v1",
            llm_model="test-model",
        )
    )
    assert router._model is chat_generator._model  # type: ignore[attr-defined]
    assert chat_generator._model is generator._model  # type: ignore[attr-defined]
    assert generator._model is verifier._model  # type: ignore[attr-defined]
    assert verifier._model is rewriter._model  # type: ignore[attr-defined]
    assert rewriter._model is model  # type: ignore[attr-defined]
    assert model is shared_model
    assert constructor_arguments["enable_thinking"] is None


def test_runtime_factory_shares_postgres_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    import adaptive_rag.runtime as runtime

    settings = Settings(_env_file=None, app_env="test")
    workflow = cast(AdaptiveRAGWorkflow, object())
    store = cast(PgVectorDenseStore, object())
    engine = cast(Engine, object())
    sessions = cast(sessionmaker[Session], object())
    memory = cast(ConversationMemoryService, object())
    recorder = cast(QueryRunRecorder, object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(
        runtime,
        "build_postgres_dependencies",
        lambda value: PostgresResources(engine, sessions, memory, recorder),
    )
    monkeypatch.setattr(
        runtime,
        "build_runtime_workflow",
        lambda value, passed_sessions: SimpleNamespace(
            workflow=workflow,
            store=store,
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(runtime, "RuntimeReadiness", FakeRuntimeReadiness)

    def fake_create_app(value: Settings, **kwargs: object) -> FastAPI:
        captured.update(settings=value, **kwargs)
        return FastAPI()

    monkeypatch.setattr(runtime, "create_app", fake_create_app)

    assert isinstance(create_runtime_app(), FastAPI)
    assert captured["workflow"] is workflow
    assert captured["conversation_memory"] is memory
    assert captured["query_run_recorder"] is recorder


def test_postgres_dependencies_require_dsn() -> None:
    with pytest.raises(RuntimeConfigurationError, match="ASR_POSTGRES_DSN"):
        build_postgres_dependencies(Settings(_env_file=None))


def test_readiness_checks_pgvector_metadata_and_closes_engine() -> None:
    engine = FakeEngine()
    store = FakeReadyStore()
    readiness = RuntimeReadiness(
        cast(Engine, engine), cast(PgVectorDenseStore, store)
    )

    readiness.check()
    readiness.close()

    assert store.closed is True
    assert engine.disposed is True


@pytest.mark.parametrize(
    "engine,store",
    [
        (FakeEngine(FakeConnection(extension=False)), FakeReadyStore()),
        (FakeEngine(), FakeReadyStore(status="building")),
        (FakeEngine(), FakeReadyStore(count=1)),
    ],
)
def test_readiness_rejects_incomplete_database(
    engine: FakeEngine, store: FakeReadyStore
) -> None:
    readiness = RuntimeReadiness(
        cast(Engine, engine), cast(PgVectorDenseStore, store)
    )
    with pytest.raises(RuntimeConfigurationError, match="dependencies"):
        readiness.check()
