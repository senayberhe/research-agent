from typing import Literal


from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    tavily_api_key: str
    openai_api_key: str

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    environment: Literal[
        "development",
        "production",
        "testing",
    ] = "development"

    # OpenAI model for the agent. Must be a real model name, and should be
    # in app/llm/pricing.py, or run costs are recorded as unknown.
    llm_model: str = "gpt-5.6-terra"

    # Research agent limits. Each can be overridden by an environment
    # variable of the same name in upper case, e.g. AGENT_MAX_TOOL_CALLS=4.
    agent_max_iterations: int = Field(default=6, ge=1)
    agent_max_tool_calls: int = Field(default=8, ge=1)
    agent_tool_timeout_seconds: float = Field(default=30.0, gt=0)
    agent_max_tool_retries: int = Field(default=2, ge=0)
    agent_max_query_length: int = Field(default=500, ge=1)
    agent_max_result_length: int = Field(default=12000, ge=1)
    agent_max_execution_seconds: float = Field(default=300.0, gt=0)

    # Token budget for one agent run (input + output, all LLM calls).
    agent_max_total_tokens: int = Field(default=20_000, ge=1)

    # When the worker starts, requeue work that a crash or restart cut off
    # (it continues from its checkpoint). Off: it is only marked failed, to
    # resume by hand with POST /research/{task_id}/resume.
    agent_resume_on_startup: bool = True

    # Worker: how often to look for a pending job when the queue is empty;
    # how long a worker's claim (lease) on a job lasts before another worker
    # may take the job over, unless it is renewed; and how many times a job
    # may run before it is left failed.
    worker_poll_interval_seconds: float = Field(default=2.0, gt=0)
    worker_lease_seconds: int = Field(default=300, ge=1)
    worker_max_attempts: int = Field(default=3, ge=1)

    # SLOs and error budgets are measured over this rolling window.
    slo_window_days: int = Field(default=30, ge=1)

    # Browser origins allowed to call the API (a frontend on another port
    # or domain). A JSON list, e.g. CORS_ORIGINS='["https://app.example.com"]'.
    # Default: the Vite dev server.
    cors_origins: list[str] = ["http://localhost:5173"]

    # Authentication. AUTH_SECRET_KEY signs the access tokens (JWT, HS256):
    # required, at least 32 characters, and secret (anyone who has it can
    # mint tokens). Generate one with:
    #   python -c "import secrets; print(secrets.token_urlsafe(48))"
    auth_secret_key: SecretStr = Field(min_length=32)
    # How long a login lasts.
    auth_access_token_minutes: int = Field(default=480, ge=1)
    # Optional: created as the first user on startup if there are no users
    # yet (otherwise ignored). Or use: python -m app.cli create-user
    auth_admin_username: str | None = None
    auth_admin_password: SecretStr | None = None

    # Server-sent events (GET /events): how often the stream checks for new
    # job events, and how long one connection lasts before the server
    # closes it (the browser reconnects at once, resuming after the last
    # event it received).
    event_stream_poll_seconds: float = Field(default=1.0, gt=0)
    event_stream_max_seconds: float = Field(default=300.0, gt=0)

    # While a job runs, the worker renews its lease (and writes its health
    # file) this often; must be well under worker_lease_seconds.
    worker_heartbeat_seconds: float = Field(default=30.0, gt=0)

    # On shutdown (docker stop), how long to let the current job finish
    # before cancelling it and returning it to the queue. Keep it below the
    # container's stop_grace_period, or Docker kills the worker first.
    worker_shutdown_timeout_seconds: float = Field(default=240.0, ge=0)

    # The worker's health file, refreshed on every heartbeat and poll; the
    # container health check reads it (python -m app.jobs.healthcheck).
    worker_health_file: str = "/tmp/research-worker-health.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore")


settings = Settings()