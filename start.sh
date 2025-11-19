#!/usr/bin/env bash

set -euo pipefail

# Resolve host/port even when legacy FASTAPI_HOST carried a numeric port.
if [[ "${FASTAPI_HOST:-}" =~ ^[0-9]+$ ]]; then
  HOST="0.0.0.0"
  PORT="${FASTAPI_PORT:-${FASTAPI_HOST}}"
else
  HOST="${FASTAPI_HOST:-0.0.0.0}"
  PORT="${FASTAPI_PORT:-9876}"
fi
WORKERS="${GUNICORN_WORKERS:-1}"
TIMEOUT="${GUNICORN_TIMEOUT:-600}"
LOG_LEVEL="${GUNICORN_LOG_LEVEL:-info}"

exec uv run gunicorn main:app \
  --workers "${WORKERS}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "${HOST}:${PORT}" \
  --timeout "${TIMEOUT}" \
  --log-level "${LOG_LEVEL}"
