"""GET /metrics: operational metrics for the whole research system."""


from pydantic import BaseModel

from app.schemas.jobs import WorkerResponse
from app.schemas.types import UTCDateTime


class SystemMetricsSummary(BaseModel):
    """build_system_metrics: the headline numbers."""

    total_jobs: int

    pending_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int

    total_attempts: int
    retry_count: int

    average_job_duration_ms: float | None

    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int

    average_tool_latency_ms: float | None

    total_agent_runs: int

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    # USD over runs with known pricing (agents.unpriced_runs says how many
    # are left out).
    estimated_cost: float

    # Completed jobs as a percentage of all jobs (0-100).
    success_rate: float


class JobStats(BaseModel):
    total: int
    # pending, running, completed, failed.
    by_status: dict[str, int]
    total_attempts: int
    # Attempts beyond each job's first.
    retry_count: int
    # % of started jobs that needed more than one attempt.
    retry_rate: float | None
    # Finished jobs, over all their attempts.
    average_duration_ms: float | None
    # % of finished jobs that completed.
    success_rate: float | None


class AgentStats(BaseModel):
    total_runs: int
    by_status: dict[str, int]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # USD over runs with known pricing; the true total is higher when
    # unpriced_runs is above 0.
    estimated_cost_usd: float
    unpriced_runs: int
    average_iterations: float | None
    average_tool_calls: float | None
    average_duration_ms: float | None


class ToolCallStats(BaseModel):
    calls: int
    succeeded: int
    failed: int
    success_rate: float | None
    average_latency_ms: float | None
    p95_latency_ms: float | None


class ToolStats(ToolCallStats):
    by_tool: dict[str, ToolCallStats]


class RunningJob(BaseModel):
    job_id: int
    task_id: int
    worker_id: str | None
    attempt: int
    started_at: UTCDateTime | None
    running_for_seconds: float | None
    lease_expires_at: UTCDateTime | None
    lease_expired: bool


class QueueStats(BaseModel):
    """Now, not over the window."""

    pending_jobs: int
    running_jobs: int
    oldest_pending_seconds: float | None
    expired_leases: int


class SystemMetricsResponse(BaseModel):
    # The window jobs, agents, tools and workers cover (until generated_at).
    since: UTCDateTime
    generated_at: UTCDateTime
    summary: SystemMetricsSummary
    jobs: JobStats
    agents: AgentStats
    tools: ToolStats
    workers: list[WorkerResponse]
    running_jobs: list[RunningJob]
    queue: QueueStats


class JobTimeseriesBucket(BaseModel):
    start: UTCDateTime
    completed: int
    failed: int
    # None when no job finished in the bucket.
    p95_latency_seconds: float | None


class JobTimeseriesResponse(BaseModel):
    """Jobs finished per bucket, and their p95 latency."""

    # 24h, 7d or 30d.
    range: str
    bucket_seconds: int
    start: UTCDateTime
    end: UTCDateTime
    buckets: list[JobTimeseriesBucket]
