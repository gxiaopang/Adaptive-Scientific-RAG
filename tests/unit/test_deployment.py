from pathlib import Path

import yaml


def test_compose_declares_pgvector_and_ordered_startup() -> None:
    compose = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"postgres", "bootstrap", "app", "frontend"}
    assert services["postgres"]["image"] == "pgvector/pgvector:0.8.6-pg16-bookworm"
    assert services["bootstrap"]["depends_on"]["postgres"]["condition"] == (
        "service_healthy"
    )
    assert services["app"]["depends_on"]["bootstrap"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["frontend"]["depends_on"]["app"]["condition"] == (
        "service_healthy"
    )
    assert "bm25-index:/app/.artifacts/indexes/bm25" in services["bootstrap"]["volumes"]
    assert set(compose["volumes"]) == {"postgres-data", "bm25-index"}


def test_bootstrap_is_idempotent_and_application_does_not_migrate() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    bootstrap = Path("docker/bootstrap.sh").read_text(encoding="utf-8")
    entrypoint = Path("docker/entrypoint.sh").read_text(encoding="utf-8")

    assert "USER adaptive-rag" in dockerfile
    assert "uv sync --locked --no-dev --no-editable" in dockerfile
    assert "alembic upgrade head" in bootstrap
    assert "if [ ! -f" in bootstrap
    assert "python scripts/build_dense_index.py" in bootstrap
    assert "alembic" not in entrypoint
    assert "exec uvicorn" in entrypoint
    for obsolete in ("CUDA", "nvidia", "HF_HOME", "huggingface"):
        assert obsolete not in dockerfile


def test_frontend_image_builds_static_assets_and_proxies_backend() -> None:
    dockerfile = Path("frontend/Dockerfile").read_text(encoding="utf-8")
    nginx = Path("frontend/nginx.conf").read_text(encoding="utf-8")

    assert "pnpm install --frozen-lockfile" in dockerfile
    assert "pnpm build" in dockerfile
    assert "nginx:1.28.0-alpine" in dockerfile
    assert "proxy_pass http://app:8000" in nginx
    assert "try_files $uri $uri/ /index.html" in nginx


def test_docker_context_excludes_generated_data_and_secrets() -> None:
    ignored = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())

    expected = {".env", ".venv", ".artifacts", "frontend/node_modules", "frontend/dist"}
    assert expected <= ignored
