#!/bin/sh

set -e

echo "Running database migrations..."

uv run alembic upgrade head

echo "Database migrations completed."

echo "Starting FastAPI..."

# On stop, give open connections (e.g. GET /events streams) 5 seconds.
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 5