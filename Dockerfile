FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system adaptive-rag \
    && useradd --system --gid adaptive-rag --create-home adaptive-rag \
    && mkdir -p /app/.artifacts/indexes/bm25 \
    && chown -R adaptive-rag:adaptive-rag /app

WORKDIR /app

COPY --chown=adaptive-rag:adaptive-rag pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=adaptive-rag:adaptive-rag . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
    && chmod +x docker/bootstrap.sh docker/entrypoint.sh

USER adaptive-rag

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"

CMD ["./docker/entrypoint.sh"]
