# Research Agent

An AI research service: you ask a question, a worker runs an LLM agent that
searches Tavily, arXiv and Wikipedia, and you get back a summarised answer.
Jobs are queued in PostgreSQL, run by workers with leases and retries, and
monitored with Prometheus, Grafana and Alertmanager.

```
Client ──► FastAPI API (:8000) ──► PostgreSQL (jobs, events, runs) ◄── Worker ──► Tavily / arXiv / Wikipedia
                │
                └── /metrics ◄── Prometheus (:9090) ──► Grafana (:3000)
                                       └── alert rules ──► Alertmanager (:9093) ──► email
```

## Running it

Requirements: Docker, and [uv](https://docs.astral.sh/uv/) for local
development.

```bash
cp .env.docker.example .env.docker          # API keys, database URL, worker settings
cp .env.alertmanager.example .env.alertmanager   # alert email settings
printf '%s' 'your-smtp-password' > monitoring/alertmanager/secrets/smtp_password

docker compose up -d --build
```

| Service | URL |
|---|---|
| API (interactive docs) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana (admin / `GRAFANA_ADMIN_PASSWORD`, default `admin`) | http://localhost:3000 |
| Alertmanager | http://localhost:9093 |
| PostgreSQL (from the host) | `localhost:5433` |

The API applies database migrations on startup.

## Frontend

A React dashboard lives in `frontend/` (see `frontend/README.md`):

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

## Development

```bash
uv sync
docker compose up -d postgres     # the tests use the research_test database
uv run pytest                     # run from the project root
```

After changing a model: `uv run alembic revision --autogenerate -m "..."`,
then `uv run alembic upgrade head`.

## Accounts and sign-in

Everything except `GET /health`, `GET /metrics` (Prometheus),
`POST /auth/login` and `POST /auth/logout` needs a signed-in user.
`POST /auth/login` (`{"username": ..., "password": ...}`) sets an httpOnly
session cookie (SameSite=Lax) that the browser sends with every request;
`POST /auth/logout` ends it. Sessions are stored in `user_sessions` (only a
hash of the cookie's value) and last `AUTH_SESSION_HOURS` (default 8). Set
`AUTH_COOKIE_SECURE=true` wherever the app is served over HTTPS.

Requests that change something are refused (403) if they come from a page on
an origin other than the API's own or one in `CORS_ORIGINS`. Repeated failed
sign-ins get 429 for a while (`AUTH_LOGIN_MAX_FAILURES`, per username;
`AUTH_LOGIN_MAX_FAILURES_PER_IP`; within `AUTH_LOGIN_WINDOW_SECONDS`).
A new password or deactivation signs the user out everywhere.

There's no sign-up: an admin creates accounts.

```bash
docker compose exec api uv run --no-sync python -m app.cli create-user alice   # prompts for a password (12+ chars)
docker compose exec api uv run --no-sync python -m app.cli list-users
docker compose exec api uv run --no-sync python -m app.cli set-password alice
docker compose exec api uv run --no-sync python -m app.cli deactivate alice    # can't sign in; open sessions end
```

Or set `AUTH_ADMIN_USERNAME` / `AUTH_ADMIN_PASSWORD` to create the first user
on startup (only when there are no users yet).

## API

All responses are JSON except `/metrics` (Prometheus text). Timestamps are
UTC.

**Research**

| Method | Path | |
|---|---|---|
| POST | `/research` | Create a task (`{"question": "..."}`); queued for a worker, returns 202 |
| GET | `/research?limit=20&offset=0&status=` | Tasks, newest first, paginated: `{items, total, limit, offset}` |
| GET | `/research/{task_id}` | One task: question, status, summary |
| POST | `/research/{task_id}/resume` | Retry a failed task from its last checkpoint |
| GET | `/research/{task_id}/timeline` | Everything that happened, in order (job events, steps, agent runs) |
| GET | `/research/{task_id}/execution` | The task and its jobs |
| GET | `/research/{task_id}/metrics` | Jobs, retries, tool calls, latency, tokens, cost for the task |
| GET | `/research/{task_id}/jobs` | The task's jobs |
| GET | `/research/{task_id}/jobs/{job_id}/events` | One job's events |
| GET | `/research/{task_id}/runs` | Agent runs: tokens, cost, duration |
| GET | `/research/{task_id}/steps` | Tool calls with their results |

**Jobs and workers**

| Method | Path | |
|---|---|---|
| GET | `/jobs/{job_id}` | A job with its attempts and events |
| GET | `/jobs/{job_id}/execution` | Current state, history, and the agent run's tokens, tools and cost |
| GET | `/jobs/{job_id}/timeline` | The job's events as plain text |
| GET | `/workers?hours=24` | Each worker's state: busy, idle or unresponsive |

**System health, metrics and SLOs**

| Method | Path | |
|---|---|---|
| GET | `/health` | Database, workers and job queue: healthy, degraded or unhealthy |
| GET | `/health/jobs` | Job queue health |
| GET | `/research/metrics` | System-wide totals, all time |
| GET | `/research/metrics/system?hours=24` | Detailed metrics for a time window, per tool |
| GET | `/research/metrics/timeseries?range=24h` | Jobs finished and p95 latency per period (24h, 7d, 30d) |
| GET | `/research/slo` | Each SLO over the rolling window: healthy, breached or no_data, with its error budget |
| GET | `/research/slo/error-budget` | Each SLO's error budget in detail: events, burn rate, status |
| GET | `/metrics` | Prometheus metrics |

## SLOs

Defined in `app/core/slo.py`, measured over a rolling window
(`SLO_WINDOW_DAYS`, default 30):

| SLO | Target |
|---|---|
| Research job success | 99% of finished jobs complete |
| Tool success | 98% of tool calls succeed |
| Job latency | p95 under 60 s |
| Tool latency | p95 under 15 s, for each tool |
| API availability | 99.9% (measured by Prometheus) |

An SLO with nothing measured in the window reports `no_data`, never
healthy.

## Monitoring

- **Prometheus** scrapes `api:8000/metrics` every 15 s. Everything is
  computed from the database, so totals survive restarts and cover every
  worker. Config: `monitoring/prometheus.yml`.
- **Alert rules**: `monitoring/alerts/research-agent-alerts.yml`, with unit
  tests in `monitoring/alerts/tests/` (run with
  `promtool test rules`; see the comment at the top of the test file).
- **Grafana** loads the Prometheus data source and the "Research Agent
  Production Dashboard" from `monitoring/grafana`.
- **Alertmanager** emails firing alerts. Settings come from
  `.env.alertmanager` and `monitoring/alertmanager/secrets/smtp_password`,
  both ignored by git. For Gmail, use an App Password.

## Configuration

Settings are environment variables (see `.env.example`); highlights:

| Variable | Default | |
|---|---|---|
| `WORKER_LEASE_SECONDS` | 300 | How long a worker's claim on a job lasts without renewal |
| `WORKER_MAX_ATTEMPTS` | 3 | Attempts before a job fails for good |
| `SLO_WINDOW_DAYS` | 30 | Rolling window for SLOs and error budgets |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | Browser origins allowed to call the API (JSON list) |
