FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# curl is used by the docker-compose healthcheck. Removing the apt lists in
# the same layer keeps them out of the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

# --no-dev: leave test tools (pytest etc.) out of the image.
RUN uv sync --frozen --no-dev

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

EXPOSE 8000

# Apply database migrations, then start the server. If a migration fails,
# the container stops instead of serving an outdated schema. "exec" makes
# uvicorn the main process so it receives Docker's stop signal directly.
# --no-sync: dependencies were installed at build time; don't re-check them
# on every container start.
# --timeout-graceful-shutdown: on stop, open connections get 5 seconds,
# then they're closed. Without it, an open server-sent events stream
# (GET /events, up to EVENT_STREAM_MAX_SECONDS) keeps the server from
# stopping; the browser reconnects to the new one and resumes.
CMD ["sh", "-c", "uv run --no-sync alembic upgrade head && exec uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 5"]
