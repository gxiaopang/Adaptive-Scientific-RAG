from fastapi import FastAPI

from adaptive_rag.core.config import Settings
from adaptive_rag.main import app, create_app


def test_module_exports_fastapi_application() -> None:
    assert isinstance(app, FastAPI)
    assert app.title == "Adaptive Scientific RAG"


def test_app_factory_uses_injected_settings() -> None:
    settings = Settings(
        _env_file=None,
        app_name="Injected Test App",
        app_env="test",
        log_level="WARNING",
        host="127.0.0.1",
        port=9001,
    )

    application = create_app(settings)

    assert application.title == "Injected Test App"
    assert application.version == "0.1.0"
    assert application.state.settings is settings


def test_app_configures_only_explicit_cors_origins() -> None:
    application = create_app(
        Settings(
            _env_file=None,
            app_env="test",
            cors_origins=("https://research.example",),
        )
    )

    cors = next(
        middleware
        for middleware in application.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    )
    assert cors.kwargs["allow_origins"] == ["https://research.example"]
    assert cors.kwargs["allow_credentials"] is False


def test_app_registers_current_business_and_deployment_routes() -> None:
    application = create_app(Settings(_env_file=None, app_env="test"))

    assert set(application.openapi()["paths"]) == {
        "/api/v1/query",
        "/api/v1/query/stream",
        "/health",
        "/ready",
    }


def test_app_lifespan_closes_runtime_resources() -> None:
    closed: list[bool] = []
    application = create_app(
        Settings(_env_file=None, app_env="test"),
        shutdown=lambda: closed.append(True),
    )

    from fastapi.testclient import TestClient

    with TestClient(application):
        assert closed == []

    assert closed == [True]
