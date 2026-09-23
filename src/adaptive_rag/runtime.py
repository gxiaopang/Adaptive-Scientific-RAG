"""Explicit production composition for the adaptive RAG application."""

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from adaptive_rag.core.config import Settings, get_settings
from adaptive_rag.data.scifact import load_scifact
from adaptive_rag.data.validation import sha256_file, validate_scifact
from adaptive_rag.domain.protocols import (
    AnswerGenerator,
    EvidenceVerifier,
    GeneralChatGenerator,
    QueryRewriter,
    QueryRouter,
)
from adaptive_rag.embeddings import OpenAICompatibleEmbeddingProvider
from adaptive_rag.generation import (
    OpenAIAnswerGenerator,
    OpenAIChatModel,
    OpenAIEvidenceVerifier,
    OpenAIGeneralChatGenerator,
    OpenAIQueryRewriter,
    OpenAIQueryRouter,
)
from adaptive_rag.graph.workflow import AdaptiveRAGWorkflow, load_adaptive_rag_config
from adaptive_rag.main import create_app
from adaptive_rag.memory import ConversationMemoryService
from adaptive_rag.observability.query_runs import QueryRunRecorder
from adaptive_rag.retrieval.bm25 import BM25Retriever
from adaptive_rag.retrieval.dense import (
    DenseConfig,
    DenseRetriever,
    load_dense_config,
    verify_dense_index,
)
from adaptive_rag.retrieval.hybrid import HybridRetriever, load_hybrid_config
from adaptive_rag.retrieval.online_reranker import QwenTextReranker
from adaptive_rag.retrieval.reranker import (
    RerankerConfig,
    RerankingRetriever,
    load_reranker_config,
)
from adaptive_rag.storage.postgres.repository import (
    SQLAlchemyConversationRepository,
    SQLAlchemyQueryRunRepository,
)
from adaptive_rag.storage.postgres.vector_store import PgVectorDenseStore

_ADAPTIVE_CONFIG = Path("configs/adaptive.yaml")
_DENSE_CONFIG = Path("configs/dense.yaml")
_HYBRID_CONFIG = Path("configs/hybrid.yaml")
_RERANKER_CONFIG = Path("configs/reranker.yaml")
_BM25_INDEX_DIR = Path(".artifacts/indexes/bm25/scifact-v1")
_EXPECTED_ALEMBIC_REVISION = "20260922_0002"


class RuntimeConfigurationError(RuntimeError):
    """Raised before startup when required runtime configuration is missing."""


@dataclass(frozen=True, slots=True)
class PostgresResources:
    """PostgreSQL resources shared by all persistence adapters."""

    engine: Engine
    sessions: sessionmaker[Session]
    conversation_memory: ConversationMemoryService
    query_run_recorder: QueryRunRecorder


@dataclass(frozen=True, slots=True)
class WorkflowResources:
    """Workflow plus the provider clients owned by one runtime composition."""

    workflow: AdaptiveRAGWorkflow
    store: PgVectorDenseStore
    embedder: OpenAICompatibleEmbeddingProvider
    reranker: QwenTextReranker
    chat_model: OpenAIChatModel

    def close(self) -> None:
        self.reranker.close()
        self.embedder.close()
        self.chat_model.close()


class RuntimeReadiness:
    """Probe database schema and dense index state without calling providers."""

    def __init__(self, engine: Engine, store: PgVectorDenseStore) -> None:
        self._engine = engine
        self._store = store

    def check(self) -> None:
        try:
            with self._engine.connect() as connection:
                extension = connection.scalar(
                    text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                )
                revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
                tables = connection.execute(
                    text(
                        "SELECT to_regclass('public.embedding_indexes'), "
                        "to_regclass('public.document_embeddings')"
                    )
                ).one()
            snapshot = self._store.snapshot()
            if not extension or revision != _EXPECTED_ALEMBIC_REVISION or any(
                table is None for table in tables
            ):
                raise RuntimeConfigurationError("PostgreSQL schema is not ready")
            if (
                snapshot.status != "ready"
                or snapshot.expected_records < 1
                or snapshot.actual_records != snapshot.expected_records
                or self._store.count() != snapshot.expected_records
            ):
                raise RuntimeConfigurationError("Dense index metadata is not ready")
        except (SQLAlchemyError, RuntimeError, ValueError) as error:
            raise RuntimeConfigurationError("runtime dependencies are unavailable") from error

    def close(self) -> None:
        try:
            self._store.close()
        finally:
            self._engine.dispose()


def create_runtime_app() -> FastAPI:
    """Build external dependencies explicitly when Uvicorn invokes this factory."""

    settings = get_settings()
    postgres = build_postgres_dependencies(settings)
    resources = build_runtime_workflow(settings, postgres.sessions)
    readiness = RuntimeReadiness(postgres.engine, resources.store)

    def shutdown() -> None:
        try:
            resources.close()
        finally:
            readiness.close()

    return create_app(
        settings,
        workflow=resources.workflow,
        conversation_memory=postgres.conversation_memory,
        query_run_recorder=postgres.query_run_recorder,
        readiness_probe=readiness,
        shutdown=shutdown,
    )


def build_postgres_dependencies(settings: Settings) -> PostgresResources:
    """Create all repositories around one Engine and Session factory."""

    if settings.postgres_dsn is None:
        raise RuntimeConfigurationError("ASR_POSTGRES_DSN is required")
    dsn = settings.postgres_dsn
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    try:
        engine = create_engine(dsn, pool_pre_ping=True)
    except (SQLAlchemyError, ValueError) as error:
        raise RuntimeConfigurationError("unable to initialize PostgreSQL") from error
    sessions = sessionmaker(engine, class_=Session, expire_on_commit=False)
    conversation_repository = SQLAlchemyConversationRepository(sessions)
    return PostgresResources(
        engine=engine,
        sessions=sessions,
        conversation_memory=ConversationMemoryService(
            conversation_repository,
            max_turns=settings.memory_max_turns,
            max_characters=settings.memory_max_characters,
        ),
        query_run_recorder=SQLAlchemyQueryRunRepository(sessions),
    )


def build_runtime_workflow(
    settings: Settings,
    sessions: sessionmaker[Session],
    *,
    training_only: bool = False,
    answer_enable_thinking: bool | None = None,
) -> WorkflowResources:
    """Compose pgvector retrieval and online providers into the graph."""

    router, chat_generator, generator, verifier, rewriter, chat_model = build_openai_adapters(
        settings,
        answer_enable_thinking=answer_enable_thinking,
    )
    if training_only:
        dataset = load_scifact(settings.scifact_data_dir, splits=("train",))
        corpus_checksum = sha256_file(settings.scifact_data_dir / "corpus.jsonl")
    else:
        dataset = load_scifact(settings.scifact_data_dir)
        corpus_checksum = validate_scifact(dataset).checksums["corpus.jsonl"]

    bm25_retriever = BM25Retriever.load(_BM25_INDEX_DIR, mmap=True)
    if bm25_retriever.manifest.corpus_checksum != corpus_checksum:
        raise RuntimeConfigurationError("BM25 index does not match the current SciFact corpus")

    dense_config = load_dense_config(_DENSE_CONFIG)
    embedder = build_embedding_provider(settings, dense_config)
    store = PgVectorDenseStore(
        sessions,
        index_name=dense_config.index_name,
        dimension=dense_config.dimension,
    )
    snapshot = verify_dense_index(
        corpus=dataset.corpus,
        corpus_checksum=corpus_checksum,
        config=dense_config,
        embedder=embedder,
        store=store,
    )
    if bm25_retriever.manifest.doc_ids_checksum != snapshot.doc_ids_checksum:
        raise RuntimeConfigurationError("BM25 and dense indexes contain different documents")
    dense_retriever = DenseRetriever(embedder=embedder, store=store, config=dense_config)

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        config=load_hybrid_config(_HYBRID_CONFIG),
    )
    reranker_config = load_reranker_config(_RERANKER_CONFIG)
    reranker = build_reranker_provider(settings, reranker_config)
    retriever = RerankingRetriever(
        candidate_retriever=hybrid_retriever,
        reranker=reranker,
        corpus=dataset.corpus,
        config=reranker_config,
    )
    return WorkflowResources(
        workflow=AdaptiveRAGWorkflow(
            retriever=retriever,
            corpus=dataset.corpus,
            router=router,
            chat_generator=chat_generator,
            generator=generator,
            verifier=verifier,
            rewriter=rewriter,
            config=load_adaptive_rag_config(_ADAPTIVE_CONFIG),
        ),
        store=store,
        embedder=embedder,
        reranker=reranker,
        chat_model=chat_model,
    )


def build_embedding_provider(
    settings: Settings,
    config: DenseConfig,
) -> OpenAICompatibleEmbeddingProvider:
    """Validate settings against the checked-in dense identity and build the adapter."""

    if settings.embedding_api_key is None:
        raise RuntimeConfigurationError("ASR_EMBEDDING_API_KEY is required")
    if (
        settings.embedding_provider != config.provider
        or settings.embedding_model != config.model_name
        or settings.embedding_dimension != config.dimension
    ):
        raise RuntimeConfigurationError(
            "Embedding settings do not match the checked-in dense index configuration"
        )
    return OpenAICompatibleEmbeddingProvider(
        provider=settings.embedding_provider,
        api_key=settings.embedding_api_key.get_secret_value(),
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )


def build_reranker_provider(settings: Settings, config: RerankerConfig) -> QwenTextReranker:
    if settings.reranker_api_key is None:
        raise RuntimeConfigurationError("ASR_RERANKER_API_KEY is required")
    if (
        settings.reranker_provider != config.provider
        or settings.reranker_model != config.model_name
    ):
        raise RuntimeConfigurationError(
            "Reranker settings do not match the checked-in reranker configuration"
        )
    return QwenTextReranker(
        provider=settings.reranker_provider,
        api_key=settings.reranker_api_key.get_secret_value(),
        url=settings.reranker_url,
        model=settings.reranker_model,
        instruction=settings.reranker_instruction,
        timeout_seconds=settings.reranker_timeout_seconds,
        max_retries=settings.reranker_max_retries,
    )


def build_openai_adapters(
    settings: Settings,
    *,
    answer_enable_thinking: bool | None = None,
) -> tuple[
    QueryRouter,
    GeneralChatGenerator,
    AnswerGenerator,
    EvidenceVerifier,
    QueryRewriter,
    OpenAIChatModel,
]:
    """Create all graph-facing adapters around one shared OpenAI client."""

    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
        raise RuntimeConfigurationError("ASR_LLM_API_KEY is required")
    if settings.llm_model is None:
        raise RuntimeConfigurationError("ASR_LLM_MODEL is required")
    model = OpenAIChatModel(
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        enable_thinking=answer_enable_thinking,
    )
    return (
        OpenAIQueryRouter(model),
        OpenAIGeneralChatGenerator(model),
        OpenAIAnswerGenerator(model),
        OpenAIEvidenceVerifier(model),
        OpenAIQueryRewriter(model),
        model,
    )
