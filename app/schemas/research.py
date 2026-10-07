from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from app.schemas.types import UTCDateTime


class ResearchRequest(BaseModel):
    question: str


class ResearchResponse(BaseModel):
    id: int
    question: str
    status: str
    summary: str | None = None
    created_at: UTCDateTime

    model_config = {
        "from_attributes": True
    }


class ResearchListResponse(BaseModel):
    """A page of research tasks, newest first."""

    items: list[ResearchResponse]
    # All tasks matching the filter, for paging.
    total: int
    limit: int
    offset: int


class JobEventResponse(BaseModel):
    id: int
    job_id: int
    event_type: str
    message: str | None
    # metadata_json, parsed. Read from JobEvent.details: JobEvent.metadata
    # is SQLAlchemy's table registry, not the event's metadata.
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("details", "metadata"),
    )
    created_at: UTCDateTime

    model_config = {
        "from_attributes": True
    }


class ResearchJobResponse(BaseModel):
    id: int
    task_id: int
    status: str
    # How many times a worker has started it.
    attempts: int
    worker_id: str | None
    lease_expires_at: UTCDateTime | None
    # The latest attempt's error; earlier ones are in its attempt history.
    error: str | None
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    completed_at: UTCDateTime | None

    model_config = {
        "from_attributes": True
    }


class ResearchExecutionResponse(BaseModel):
    task: ResearchResponse
    jobs: list[ResearchJobResponse]


class ExecutionTimelineEvent(BaseModel):
    timestamp: UTCDateTime
    event_type: str
    # Where the event comes from, e.g. "job", "agent" or "tool".
    source: str
    message: str | None
    metadata: dict


class ExecutionTimelineResponse(BaseModel):
    task_id: int
    # Oldest first.
    events: list[ExecutionTimelineEvent]


class ResearchMetricsResponse(BaseModel):
    task_id: int

    total_jobs: int
    completed_jobs: int
    failed_jobs: int

    total_attempts: int
    retry_count: int

    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int

    average_tool_latency_ms: float | None
    # Time workers spent running the task's jobs, over every attempt.
    total_execution_time_ms: float | None

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    # USD; None when a run's model pricing is unknown (any total would be
    # too low).
    estimated_cost: float | None

    # Completed jobs as a percentage of all jobs (0-100).
    success_rate: float


class HealthComponent(BaseModel):
    # healthy, degraded or unhealthy.
    status: str
    message: str


class SystemHealthResponse(BaseModel):
    # The worst of the components: unhealthy > degraded > healthy.
    status: str
    database: HealthComponent
    workers: HealthComponent
    jobs: HealthComponent


class ErrorBudgetResponse(BaseModel):
    """An ErrorBudget (calculate_error_budget). The measured fields are None
    when nothing was measured in the window (no data, not a full budget)."""

    slo_target: float
    actual: float | None
    allowed_failure_rate: float
    actual_failure_rate: float | None
    budget_remaining: float | None
    budget_remaining_percent: float | None
    exhausted: bool


class SLOErrorBudgetResponse(ErrorBudgetResponse):
    """One SLO's budget in the report: the ErrorBudget plus the events
    behind it, the burn rate and a status."""

    slo: str
    # The SLO's own target: a ratio (0-1), or seconds for the latency SLOs
    # (whose slo_target is 0.95: 5% of events may exceed it).
    target: float
    total_events: int
    bad_events: int
    allowed_bad_events: float
    # Share of the budget used (above 1: overspent); None with no data.
    budget_consumed: float | None
    # From the last hour; 1 = on pace to use exactly the budget. None when
    # nothing was measured in the last hour.
    burn_rate_1h: float | None
    # healthy, at_risk (under 25% left), exhausted, or no_data.
    status: str


class ErrorBudgetReportResponse(BaseModel):
    window_days: float
    # The exact window measured, [window_start, window_end], in UTC.
    window_start: UTCDateTime
    window_end: UTCDateTime
    budgets: dict[str, SLOErrorBudgetResponse]


class LatencyPercentilesResponse(BaseModel):
    """Milliseconds; None when nothing was measured."""

    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None


class ToolLatencyResponse(LatencyPercentilesResponse):
    """One tool's latency, judged on its own p95 against the tool latency
    target."""

    # healthy, breached or no_data.
    status: str


class SLOMetricResponse(BaseModel):
    # A ratio (0-1), or seconds for the latency SLOs.
    target: float
    actual: float | None = None
    target_seconds: float | None = None
    # job_latency: the p95 it's judged on. (tool_latency: None; each tool
    # is judged on its own, in by_tool.)
    actual_ms: float | None = None
    # job_latency: p50, p95 and p99 of all jobs.
    percentiles: LatencyPercentilesResponse | None = None
    # tool_latency: each tool's percentiles and status.
    by_tool: dict[str, ToolLatencyResponse] | None = None
    # healthy, breached, no_data (nothing measured in the window), or
    # unknown (not measured here).
    status: str
    # The SLO's error budget over the same window (None for
    # api_availability, which Prometheus measures). Event counts, burn rate
    # and status are in GET /research/slo/error-budget.
    error_budget: ErrorBudgetResponse | None = None


class SLOReportResponse(BaseModel):
    # The exact rolling window measured (SLO_WINDOW_DAYS),
    # [window_start, window_end], in UTC.
    window_days: float
    window_start: UTCDateTime
    window_end: UTCDateTime

    api_availability: SLOMetricResponse
    research_job_success: SLOMetricResponse
    tool_success: SLOMetricResponse
    job_latency: SLOMetricResponse
    tool_latency: SLOMetricResponse


class RunProgressResponse(BaseModel):
    id: int
    # created, running, waiting_for_tool, processing_result, completed or
    # failed.
    status: str
    # True while the run is going (totals from its latest checkpoint).
    live: bool
    iterations: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # None when the model's pricing is unknown.
    estimated_cost_usd: float | None
    error: str | None
    started_at: UTCDateTime
    completed_at: UTCDateTime | None
    elapsed_seconds: float


class ToolCallProgressResponse(BaseModel):
    """One tool call: what was searched, how it went, what came back."""

    id: int
    tool: str
    query: str
    # The agent iteration (LLM call) that asked for it.
    iteration: int
    # pending, running, completed or failed.
    status: str
    # None while running.
    duration_ms: float | None
    started_at: UTCDateTime
    # None while running.
    success: bool | None
    error: str | None
    # The start of what the tool returned (RESULT_PREVIEW_CHARS).
    result_preview: str | None
    result_truncated: bool
    # Characters in the whole result.
    result_length: int


class TaskProgressResponse(BaseModel):
    """A task's progress, for following it live (GET
    /research/{task_id}/progress)."""

    task: ResearchResponse
    # Completed or failed: poll no more (unless resumed).
    finished: bool
    # Failed with a checkpoint to continue from (POST .../resume).
    can_resume: bool
    job: ResearchJobResponse | None
    run: RunProgressResponse | None
    # Each tool call, oldest first.
    tool_calls: list[ToolCallProgressResponse]
    # Everything so far, oldest first (as GET .../timeline).
    events: list[ExecutionTimelineEvent]

