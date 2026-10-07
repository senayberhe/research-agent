from datetime import datetime

from pydantic import BaseModel, computed_field

from app.schemas.research import JobEventResponse, ResearchJobResponse
from app.schemas.types import UTCDateTime


def _seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None

    return round((end - start).total_seconds(), 3)


class JobAttemptResponse(BaseModel):
    attempt_number: int
    worker_id: str | None
    outcome: str
    error: str | None
    started_at: UTCDateTime
    ended_at: UTCDateTime | None

    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        return _seconds_between(self.started_at, self.ended_at)

    model_config = {
        "from_attributes": True
    }


class JobEventDetailResponse(JobEventResponse):
    """A JobEventResponse plus the commonly used metadata pulled out, and a
    line for people."""

    attempt: int | None = None
    worker_id: str | None = None
    # For tool_started / tool_completed / tool_failed.
    tool: str | None = None
    duration_ms: float | None = None
    # E.g. "Tavily, 842 ms" (see describe_job_event).
    summary: str | None = None


class JobListItemResponse(ResearchJobResponse):
    # The job's research task's question.
    question: str


class JobListResponse(BaseModel):
    """A page of jobs, newest first."""

    items: list[JobListItemResponse]
    total: int
    limit: int
    offset: int


class JobResponse(ResearchJobResponse):
    attempt_history: list[JobAttemptResponse] = []
    # Everything that happened to the job, oldest first.
    events: list[JobEventDetailResponse] = []


class AgentRunResponse(BaseModel):
    id: int
    task_id: int
    status: str
    iteration_count: int
    tool_call_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    error: str | None
    started_at: UTCDateTime
    completed_at: UTCDateTime | None

    # From the first start to the end. A resumed run keeps its first
    # started_at, so this includes time spent waiting to be retried; each
    # attempt's own duration is in the job's attempt_history.
    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        return _seconds_between(self.started_at, self.completed_at)

    model_config = {
        "from_attributes": True
    }


class ToolResultResponse(BaseModel):
    success: bool
    error: str | None
    created_at: UTCDateTime

    model_config = {
        "from_attributes": True
    }


class StepResponse(BaseModel):
    id: int
    tool: str
    query: str
    iteration: int
    status: str
    duration_ms: float | None
    # Content is left out (it can be long); success and error per result.
    results: list[ToolResultResponse] = []

    model_config = {
        "from_attributes": True
    }


class WorkerResponse(BaseModel):
    worker_id: str
    # busy, unresponsive or idle (see get_worker_activity).
    state: str
    current_job_id: int | None
    lease_expires_at: UTCDateTime | None
    last_attempt_started_at: UTCDateTime
    last_attempt_ended_at: UTCDateTime | None
    # Attempts in the time window, per outcome.
    attempts: dict[str, int]


# -------------------------------------------------------------------
# GET /jobs/{job_id}/execution
# -------------------------------------------------------------------


class JobStateResponse(BaseModel):
    status: str
    attempts: int
    worker_id: str | None
    lease_expires_at: UTCDateTime | None
    error: str | None
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    completed_at: UTCDateTime | None

    model_config = {
        "from_attributes": True
    }


class TokensResponse(BaseModel):
    input: int
    output: int
    total: int


class CostResponse(BaseModel):
    # None when the model's pricing is unknown.
    estimated_usd: float | None


class ToolsResponse(BaseModel):
    # Tool calls the model asked for, including invalid or over-limit ones.
    requested: int
    # Searches that actually ran (from the checkpoint).
    executed: int | None
    calls: list[StepResponse]


class CheckpointResponse(BaseModel):
    saved_at: UTCDateTime | None
    iteration: int | None
    executed_tool_calls: int | None
    # Messages in the saved conversation (the conversation isn't sent).
    conversation_items: int | None
    valid: bool
    error: str | None

    model_config = {
        "from_attributes": True
    }


class AgentExecutionResponse(BaseModel):
    id: int
    # The run's state: created, running, waiting_for_tool,
    # processing_result, completed or failed.
    status: str
    started_at: UTCDateTime
    completed_at: UTCDateTime | None
    error: str | None
    iterations: int
    tokens: TokensResponse
    cost: CostResponse
    tools: ToolsResponse
    checkpoint: CheckpointResponse | None

    # Like AgentRunResponse.duration_seconds: includes time waiting between
    # attempts if the run was resumed.
    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        return _seconds_between(self.started_at, self.completed_at)


class JobExecutionResponse(BaseModel):
    job_id: int
    task_id: int
    current_state: JobStateResponse
    history: list[JobEventDetailResponse]
    # None if the job hasn't started an agent run yet.
    agent_run: AgentExecutionResponse | None
