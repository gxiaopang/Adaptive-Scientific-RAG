from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_rag.core.config import Settings, get_settings

_ENVIRONMENT_VARIABLES = (
    "ASR_APP_NAME",
    "ASR_APP_ENV",
    "ASR_LOG_LEVEL",
    "ASR_HOST",
    "ASR_PORT",
    "ASR_SCIFACT_DATA_DIR",
    "ASR_EMBEDDING_API_KEY",
    "ASR_EMBEDDING_BASE_URL",
    "ASR_EMBEDDING_PROVIDER",
    "ASR_EMBEDDING_MODEL",
    "ASR_EMBEDDING_DIMENSION",
    "ASR_EMBEDDING_BATCH_SIZE",
    "ASR_EMBEDDING_TIMEOUT_SECONDS",
    "ASR_EMBEDDING_MAX_RETRIES",
    "ASR_RERANKER_API_KEY",
    "ASR_RERANKER_URL",
    "ASR_RERANKER_PROVIDER",
    "ASR_RERANKER_MODEL",
    "ASR_RERANKER_TIMEOUT_SECONDS",
    "ASR_RERANKER_MAX_RETRIES",
    "ASR_RERANKER_INSTRUCTION",
    "ASR_POSTGRES_DSN",
    "ASR_MEMORY_MAX_TURNS",
    "ASR_MEMORY_MAX_CHARACTERS",
    "ASR_LLM_API_KEY",
    "ASR_LLM_BASE_URL",
    "ASR_LLM_MODEL",
    "ASR_LLM_TIMEOUT_SECONDS",
    "ASR_LLM_MAX_RETRIES",
    "ASR_LLM_TEMPERATURE",
    "ASR_LLM_MAX_TOKENS",
)


@pytest.fixture(autouse=True)
def isolate_settings_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    get_settings.cache_clear()
    for variable in _ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    yield
    get_settings.cache_clear()


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "Adaptive Scientific RAG"
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.scifact_data_dir == Path("scifact")
    assert settings.embedding_api_key is None
    assert settings.embedding_model == "qwen3.7-text-embedding"
    assert settings.embedding_dimension == 1024
    assert settings.embedding_batch_size == 10
    assert settings.reranker_api_key is None
    assert settings.reranker_model == "qwen3.7-text-rerank"
    assert settings.postgres_dsn is None
    assert settings.memory_max_turns == 8
    assert settings.memory_max_characters == 8_000
    assert settings.llm_api_key is None
    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_model is None
    assert settings.llm_timeout_seconds == 60.0
    assert settings.llm_max_retries == 2
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_tokens == 800


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_APP_NAME", "Test RAG")
    monkeypatch.setenv("ASR_APP_ENV", "test")
    monkeypatch.setenv("ASR_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ASR_HOST", "127.0.0.1")
    monkeypatch.setenv("ASR_PORT", "9000")
    monkeypatch.setenv("ASR_SCIFACT_DATA_DIR", "/tmp/scifact-test")
    monkeypatch.setenv("ASR_EMBEDDING_API_KEY", "embedding-secret")
    monkeypatch.setenv("ASR_EMBEDDING_BASE_URL", "http://embedding.test/v1/")
    monkeypatch.setenv("ASR_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("ASR_EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("ASR_EMBEDDING_BATCH_SIZE", "7")
    monkeypatch.setenv("ASR_RERANKER_API_KEY", "reranker-secret")
    monkeypatch.setenv("ASR_RERANKER_URL", "http://reranker.test/rerank/")
    monkeypatch.setenv("ASR_RERANKER_MODEL", "reranker-model")
    monkeypatch.setenv(
        "ASR_POSTGRES_DSN",
        "postgresql+psycopg://rag:secret@postgres/adaptive_rag",
    )
    monkeypatch.setenv("ASR_MEMORY_MAX_TURNS", "6")
    monkeypatch.setenv("ASR_MEMORY_MAX_CHARACTERS", "6000")
    monkeypatch.setenv("ASR_LLM_API_KEY", "test-secret")
    monkeypatch.setenv("ASR_LLM_BASE_URL", "http://llm.test/v1/")
    monkeypatch.setenv("ASR_LLM_MODEL", "test-model")
    monkeypatch.setenv("ASR_LLM_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("ASR_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("ASR_LLM_TEMPERATURE", "0.25")
    monkeypatch.setenv("ASR_LLM_MAX_TOKENS", "512")

    settings = get_settings()

    assert settings.app_name == "Test RAG"
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.scifact_data_dir == Path("/tmp/scifact-test")
    assert settings.embedding_api_key is not None
    assert settings.embedding_api_key.get_secret_value() == "embedding-secret"
    assert settings.embedding_base_url == "http://embedding.test/v1"
    assert settings.embedding_model == "embedding-model"
    assert settings.embedding_batch_size == 7
    assert settings.reranker_api_key is not None
    assert settings.reranker_api_key.get_secret_value() == "reranker-secret"
    assert settings.reranker_url == "http://reranker.test/rerank"
    assert settings.reranker_model == "reranker-model"
    assert settings.postgres_dsn == (
        "postgresql+psycopg://rag:secret@postgres/adaptive_rag"
    )
    assert settings.memory_max_turns == 6
    assert settings.memory_max_characters == 6_000
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "test-secret"
    assert "test-secret" not in repr(settings)
    assert settings.llm_base_url == "http://llm.test/v1"
    assert settings.llm_model == "test-model"
    assert settings.llm_timeout_seconds == 15.0
    assert settings.llm_max_retries == 4
    assert settings.llm_temperature == 0.25
    assert settings.llm_max_tokens == 512


def test_settings_load_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ASR_APP_NAME=Dotenv RAG\nASR_APP_ENV=test\nASR_PORT=8100\n"
        "ASR_LLM_API_KEY=dotenv-secret\nASR_LLM_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.app_name == "Dotenv RAG"
    assert settings.app_env == "test"
    assert settings.port == 8100
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "dotenv-secret"
    assert settings.llm_model == "dotenv-model"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("ASR_APP_ENV", "staging"),
        ("ASR_LOG_LEVEL", "TRACE"),
        ("ASR_PORT", "0"),
        ("ASR_PORT", "65536"),
        ("ASR_EMBEDDING_BATCH_SIZE", "11"),
        ("ASR_EMBEDDING_BASE_URL", "not-a-url"),
        ("ASR_EMBEDDING_TIMEOUT_SECONDS", "0"),
        ("ASR_RERANKER_URL", "not-a-url"),
        ("ASR_RERANKER_MAX_RETRIES", "-1"),
        ("ASR_POSTGRES_DSN", "sqlite:///not-postgres.db"),
        ("ASR_MEMORY_MAX_TURNS", "0"),
        ("ASR_MEMORY_MAX_CHARACTERS", "255"),
        ("ASR_LLM_BASE_URL", "llm.test/v1"),
        ("ASR_LLM_BASE_URL", "ftp://llm.test/v1"),
        ("ASR_LLM_TIMEOUT_SECONDS", "0"),
        ("ASR_LLM_MAX_RETRIES", "-1"),
        ("ASR_LLM_TEMPERATURE", "2.1"),
        ("ASR_LLM_MAX_TOKENS", "0"),
    ],
)
def test_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
