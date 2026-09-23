#!/bin/sh
set -eu

exec uvicorn adaptive_rag.runtime:create_runtime_app \
    --factory \
    --host "${ASR_HOST:-0.0.0.0}" \
    --port "${ASR_PORT:-8000}"
