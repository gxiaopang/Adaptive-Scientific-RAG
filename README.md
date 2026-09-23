# Adaptive Scientific RAG

An evaluation-first scientific retrieval-augmented generation system built around the
SciFact/BEIR benchmark. The production path combines BM25 and pgvector dense retrieval with
reciprocal-rank fusion, online reranking, and a bounded adaptive generation workflow.

## Architecture

- React 19 + TypeScript + Vite research workspace, served by Nginx in production.
- FastAPI backend with explicit dependency injection, a compatible synchronous query API, and an
  NDJSON streaming query API used by the frontend.
- PostgreSQL 16 + pgvector 0.8.6 as the system of record for conversations, run metadata,
  dense-index metadata, document chunks, and vectors.
- Provider-neutral online adapters for embeddings, reranking, and chat completions.
- Qwen `qwen3.7-text-embedding` at 1024 dimensions and `qwen3.7-text-rerank` are the checked-in
  default retrieval identities. Provider credentials and endpoints remain configurable.

The retrieval path is:

```text
question -> (BM25 top 50 || dense top 50) -> RRF -> online rerank top 10
         -> context -> generation -> evidence verification -> bounded rewrite/retry
```

BM25 and dense retrieval execute concurrently. Routing, general chat, evidence verification, and
query rewriting explicitly disable provider thinking; answer generation keeps the configured model
default and streams visible answer tokens to the browser.

General chat is routed around retrieval. Scientific answers return evidence excerpts,
verification status, query history, retrieval attempts, and a termination reason.

## Quick start with Docker Compose

Requirements: Docker with Compose support and valid credentials for the configured embedding,
reranking, and chat providers. No accelerator or locally downloaded model is required.

```bash
cp .env.example .env
# Edit .env: set ASR_EMBEDDING_API_KEY, ASR_RERANKER_API_KEY,
# ASR_LLM_API_KEY, ASR_LLM_MODEL, and any provider endpoint overrides.
docker compose up --build
```

Open the workspace at `http://127.0.0.1:3000`. The API is exposed at
`http://127.0.0.1:8000`.

On the first startup, the one-shot `bootstrap` service applies Alembic migrations, builds the
BM25 index if it is absent, and builds the pgvector index in stable batches. Later startups
reuse compatible indexes; changed documents alone are re-embedded. The service refuses to mix
vectors with a different provider, model, dimension, metric, or indexing version.

Useful checks:

```bash
docker compose ps
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`/health` is process liveness only. `/ready` checks PostgreSQL connectivity, the `vector`
extension, the expected migration and tables, and a complete compatible dense index. It does
not call an external model provider.

Stop services without deleting PostgreSQL data:

```bash
docker compose down
```

Do not add `--volumes` unless an intentional, verified data reset is required.

## Local development

The backend requires Python 3.12 and `uv`; the frontend uses the checked-in pnpm lockfile.

```bash
cp .env.example .env
UV_CACHE_DIR=/tmp/asr-uv-cache uv sync --group dev
docker compose up -d postgres
uv run alembic upgrade head
uv run python scripts/build_bm25_index.py
uv run python scripts/build_dense_index.py
uv run uvicorn adaptive_rag.runtime:create_runtime_app --factory --reload
```

In another terminal:

```bash
cd frontend
corepack pnpm install --frozen-lockfile
corepack pnpm dev
```

The Vite server proxies `/api`, `/health`, and `/ready` to `127.0.0.1:8000`. For an explicit
cross-origin backend, set the public, non-secret `VITE_API_ORIGIN` in a local frontend env file
and add that exact origin to `ASR_CORS_ORIGINS`. Provider keys must never be placed in a Vite
variable or browser source.

## Configuration

`.env.example` is the authoritative safe template. Important groups are:

- `ASR_POSTGRES_DSN` and bounded conversation-memory settings.
- `ASR_EMBEDDING_*`: OpenAI-compatible base URL, key, provider, model, dimension, batch size,
  timeout, and retry count.
- `ASR_RERANKER_*`: Qwen rerank URL, key, provider, model, instruction, timeout, and retries.
- `ASR_LLM_*`: independent OpenAI-compatible chat-completions configuration.
- `ASR_CORS_ORIGINS`: a JSON array of exact development origins; production Nginx uses a
  same-origin proxy and needs no wildcard CORS policy.

The default embedding adapter calls `{ASR_EMBEDDING_BASE_URL}/embeddings` and explicitly requests
1024 dimensions. The reranker uses the Qwen text-rerank request and response shape. Consult the
[official embedding documentation](https://help.aliyun.com/en/model-studio/embedding) and
[official rerank API documentation](https://help.aliyun.com/en/model-studio/text-rerank-api)
when selecting a regional or workspace-specific endpoint.

## API

`POST /api/v1/query` accepts:

```json
{
  "query": "What evidence links a treatment to an outcome?",
  "conversation_id": null
}
```

Reuse the returned `conversation_id` for follow-up questions. An evidence item contains
`doc_id`, `title`, a bounded `excerpt`, `score`, and `rank`. Model output is rendered as plain
text by the frontend.

`POST /api/v1/query/stream` accepts the same JSON request and returns newline-delimited JSON events:
`answer_start`, `answer_delta`, then `result`. A scientific retry can emit another `answer_start`,
which replaces the rejected draft in the UI. Failures after streaming begins are returned as an
`error` event with the same stable status categories as the synchronous endpoint.

The first frontend release deliberately has no authentication, multi-user isolation, conversation
list, or evaluation administration UI. Do not expose it as a multi-tenant public service without
adding those boundaries.

## Evaluation

Validate the immutable SciFact input first:

```bash
uv run python scripts/validate_scifact.py
```

Retrieval evaluation uses the deterministic validation subset derived from training qrels:

```bash
uv run python scripts/evaluate_bm25.py
uv run python scripts/evaluate_dense.py
uv run python scripts/evaluate_hybrid.py
uv run python scripts/evaluate_reranker.py
```

The reported retrieval metrics are Recall@5/10, Precision@5/10, MRR@10, and nDCG@10. Answer
evaluation additionally supports Ragas Answer Relevancy and Faithfulness through opt-in online
providers. Test labels are not used unless final test evaluation is explicitly authorized.
Generated indexes and reports live under `.artifacts/` and are not version-controlled.

## Quality gates

```bash
UV_CACHE_DIR=/tmp/asr-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/asr-uv-cache uv run mypy src
UV_CACHE_DIR=/tmp/asr-uv-cache uv run pytest
UV_CACHE_DIR=/tmp/asr-uv-cache uv run python scripts/validate_scifact.py
UV_CACHE_DIR=/tmp/asr-uv-cache uv run python -c "from adaptive_rag.main import app; print(app.title)"

cd frontend
corepack pnpm lint
corepack pnpm typecheck
corepack pnpm test
corepack pnpm build
```

Database and container changes additionally require `docker compose config`, image builds,
Alembic upgrade/current checks, `/health`, `/ready`, and the opt-in pgvector integration tests.
Live-provider smoke tests require real credentials and cannot be replaced by fake-provider tests.

## Repository map

- `frontend/`: responsive React research workspace and component tests.
- `src/adaptive_rag/api/`: HTTP routes, schemas, middleware, and dependency boundaries.
- `src/adaptive_rag/embeddings/` and `retrieval/online_reranker.py`: online provider adapters.
- `src/adaptive_rag/retrieval/`: BM25, pgvector dense retrieval, RRF, and reranking.
- `src/adaptive_rag/storage/postgres/`: SQLAlchemy repositories and pgvector store.
- `src/adaptive_rag/graph/`: bounded adaptive workflow.
- `migrations/`: Alembic-owned PostgreSQL schema history.
- `scripts/`: validation, index construction, and evaluation entry points.
- `configs/`: checked-in, non-secret retrieval and workflow identities.
- `scifact/`: immutable benchmark input.
