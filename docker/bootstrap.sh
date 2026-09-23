#!/bin/sh
set -eu

alembic upgrade head

if [ ! -f /app/.artifacts/indexes/bm25/scifact-v1/manifest.json ]; then
    python scripts/build_bm25_index.py
fi

# Dense indexing is restartable: matching metadata is reused and only changed
# chunks are sent to the configured online embedding provider.
python scripts/build_dense_index.py
