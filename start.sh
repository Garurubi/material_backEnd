#!/usr/bin/env bash

set -eu pipefail

# Resolve host/port even when legacy FASTAPI_HOST carried a numeric port.
HOST="${FASTAPI_HOST:-0.0.0.0}"
PORT="${FASTAPI_PORT:-${FASTAPI_HOST:-9876}}"
WORKERS="${GUNICORN_WORKERS:-1}"
TIMEOUT="${GUNICORN_TIMEOUT:-0}"
LOG_LEVEL="${GUNICORN_LOG_LEVEL:-info}"

exec uv run gunicorn main:app \
  --workers "${WORKERS}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "${HOST}:${PORT}" \
  --timeout "${TIMEOUT}" \
  --log-level "${LOG_LEVEL}" \
