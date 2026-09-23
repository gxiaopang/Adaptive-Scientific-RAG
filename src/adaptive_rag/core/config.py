"""Validated application configuration loaded from defaults, dotenv, and the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnvironment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Process-wide application settings.

    Environment variables use the ``ASR_`` prefix so this project can run beside
    other applications without accidental configuration collisions.
    """

    model_config = SettingsConfigDict(
        env_prefix="ASR_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_name: str = Field(default="Adaptive Scientific RAG", min_length=1)
    app_env: AppEnvironment = "development"
    log_level: LogLevel = "INFO"
    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173",)
    scifact_data_dir: Path = Path("scifact")
    postgres_dsn: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_provider: str = "alibaba-model-studio"
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimension: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=10, ge=1, le=10)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0.0)
    embedding_max_retries: int = Field(default=2, ge=0, le=10)
    reranker_api_key: SecretStr | None = None
    reranker_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    reranker_provider: str = "alibaba-model-studio"
    reranker_model: str = "qwen3.7-text-rerank"
    reranker_timeout_seconds: float = Field(default=30.0, gt=0.0)
    reranker_max_retries: int = Field(default=2, ge=0, le=10)
    reranker_instruction: str = (
        "Given a scientific claim or question, retrieve passages that provide relevant evidence."
    )
    memory_max_turns: int = Field(default=8, ge=1, le=50)
    memory_max_characters: int = Field(default=8_000, ge=256, le=100_000)
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=800, ge=1)

    @field_validator(
        "postgres_dsn",
        "embedding_api_key",
        "reranker_api_key",
        "llm_api_key",
        "llm_model",
        mode="before",
    )
    @classmethod
    def empty_optional_string_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator(
        "embedding_provider",
        "embedding_model",
        "reranker_provider",
        "reranker_model",
        "reranker_instruction",
    )
    @classmethod
    def provider_string_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider configuration values must not be blank")
        return value.strip()

    @field_validator("llm_model")
    @classmethod
    def optional_model_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("llm_model must not be blank")
        return value

    @field_validator("embedding_base_url", "reranker_url", "llm_base_url")
    @classmethod
    def provider_url_is_http(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("provider URLs must be absolute HTTP(S) URLs")
        return value.rstrip("/")

    @field_validator("cors_origins")
    @classmethod
    def cors_origins_are_http(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("cors_origins must contain absolute HTTP(S) origins")
            normalized.append(value.rstrip("/"))
        return tuple(normalized)

    @field_validator("postgres_dsn")
    @classmethod
    def postgres_dsn_uses_supported_driver(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            raise ValueError("postgres_dsn must use PostgreSQL with the psycopg driver")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
