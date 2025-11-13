#!/usr/bin/env bash

# Gunicorn 실행 (UvicornWorker 활용)
exec gunicorn app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --log-level info