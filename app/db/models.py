import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import DateTime, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    # Current UTC time without tzinfo, matching the DateTime columns.
    # Replaces datetime.utcnow(), which is deprecated.
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass

class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRunStatus(str, Enum):
    # Row created; the agent hasn't started.
    CREATED = "created"
    # Calling the LLM (also: a failed run being resumed).
    RUNNING = "running"
    # The model asked for searches; they are running.
    WAITING_FOR_TOOL = "waiting_for_tool"
    # Searches finished; their results are being saved and added to the
    # conversation.
    PROCESSING_RESULT = "processing_result"
    COMPLETED = "completed"
    # Can be resumed from its checkpoint, back into RUNNING.
    FAILED = "failed"



class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    task_id: Mapped[int] = mapped_column(
        ForeignKey("research_tasks.id"),
        nullable=False,
        index=True,
    )

    status: Mapped[AgentRunStatus] = mapped_column(
        String(50),
        default=AgentRunStatus.CREATED,
        nullable=False,
    )

    iteration_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )

    tool_call_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )

    # Token usage and estimated cost over all LLM calls in the run.
    # server_default: existing rows get 0 when the columns are added.
    input_tokens: Mapped[int] = mapped_column(
        default=0,
        server_default="0",
        nullable=False,
    )

    output_tokens: Mapped[int] = mapped_column(
        default=0,
        server_default="0",
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        default=0,
        server_default="0",
        nullable=False,
    )

    # USD; NULL when the model's pricing is unknown (see app/llm/pricing.py).
    estimated_cost_usd: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    # The agent's latest checkpoint (AgentCheckpoint.to_dict()): round,
    # counters, token usage and the conversation so far. NULL until the
    # first round finishes.
    state: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # When state was last saved. A RUNNING run whose checkpoint is old has
    # probably died (used by crash recovery).
    state_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    task: Mapped["ResearchTask"] = relationship(
        back_populates="agent_runs",
    )


class ResearchTask(Base):
    __tablename__ = "research_tasks"

    id: Mapped[int] = mapped_column(
        primary_key = True,
        autoincrement = True
    )

    question: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    status: Mapped[TaskStatus] = mapped_column(
        String(50),
        default=TaskStatus.PENDING,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
            DateTime,
            default=utc_now,
            nullable=False
        )

    # Who started it (None for tasks from before accounts existed, or whose
    # user was removed).
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Loaded with the task (one join), so created_by_username can be read
    # without a lazy load (not allowed with async sessions).
    created_by: Mapped["User | None"] = relationship(lazy="joined")

    @property
    def created_by_username(self) -> str | None:
        return self.created_by.username if self.created_by is not None else None



    steps: Mapped[list["ResearchStep"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan"
    )

    agent_runs: Mapped[list["AgentRun"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan"
    )

    # ResearchJob lives in app/jobs/models.py (imported at the end of this
    # file).
    jobs: Mapped[list["ResearchJob"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan"
    )



class ResearchStep(Base):
    __tablename__ = "research_steps"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    task_id: Mapped[int] = mapped_column(
        ForeignKey("research_tasks.id"),
        nullable=False,
        index=True,
    )

    tool: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # server_default: existing rows get 1 when the column is added.
    iteration: Mapped[int] = mapped_column(
        default=1,
        server_default="1",
        nullable=False,
    )

    status: Mapped[StepStatus] = mapped_column(
        String(50),
        default=StepStatus.PENDING,
        nullable=False,
    )

    duration_ms: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    # When the agent started the search (used to order the execution
    # timeline).
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    task: Mapped["ResearchTask"] = relationship(
        back_populates="steps"
    )

    results: Mapped[list["ResearchResult"]] = relationship(
        back_populates="step",
        cascade="all, delete-orphan",
    )



class ResearchResult(Base):
    __tablename__ = "research_results"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True
    )

    step_id: Mapped[int] = mapped_column(
        ForeignKey("research_steps.id"),
        nullable=False,
        index=True
    )

    tool: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    success: Mapped[bool] = mapped_column(
        default=True,
        nullable=False
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False
    )

    step: Mapped["ResearchStep"] = relationship(
        back_populates="results"
    )


class UserRole(str, Enum):
    """What a user may do (app/core/permissions.py): viewers read research
    and jobs; researchers also start and resume research; operators also
    see analytics (system metrics, SLOs, workers); admins also manage
    users."""

    VIEWER = "viewer"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    ADMIN = "admin"


class User(Base):
    """Someone who can sign in. Created by an admin (python -m app.cli
    create-user, or the Users page), never by sign-up."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    # Stored lowercase; unique.
    username: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
        unique=True,
        index=True,
    )

    # argon2 (app/core/security.py); never the password itself.
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # New accounts get the least access; an admin grants more.
    role: Mapped[UserRole] = mapped_column(
        String(20),
        default=UserRole.VIEWER,
        server_default=UserRole.VIEWER.value,
        nullable=False,
    )

    # A deactivated user can't sign in, and their sessions end.
    is_active: Mapped[bool] = mapped_column(
        default=True,
        server_default="true",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )


class UserSession(Base):
    """One sign-in: the browser holds a random session id in an httpOnly
    cookie; only its SHA-256 is stored here, so a copy of the table can't
    be used to sign in. Ends when it expires, or when revoked (sign out,
    deactivation, a new password)."""

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # SHA-256 of the cookie's value, hex.
    token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    user: Mapped[User] = relationship(lazy="joined")


class JobEvent(Base):
    """One thing that happened to a job: the job's timeline. JobAttempt
    summarises each run; events also cover what happens between runs
    (created, requeued) and inside them (each search).

    event_type is a JobEventType value (app/jobs/models.py). Details that
    only some events have (the worker, attempt, tool, duration) are in
    metadata_json; the properties below read them. Written by
    app.jobs.events.record_job_event.
    """

    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    job_id: Mapped[int] = mapped_column(
        ForeignKey("research_jobs.id"),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    # E.g. the error for FAILED, or why the job was released.
    message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # A JSON object, e.g. {"attempt": 1, "tool": "tavily",
    # "duration_ms": 842.4}. Named metadata_json because Base already has a
    # "metadata" attribute (the table definitions).
    metadata_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # utc_now instead of datetime.utcnow, which is deprecated; same value.
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
        index=True,
    )

    job: Mapped["ResearchJob"] = relationship(
        back_populates="events",
    )

    @property
    def details(self) -> dict[str, Any]:
        """metadata_json, parsed; {} if there is none or it isn't valid
        JSON (like app.jobs.event_service.parse_event_metadata)."""

        if not self.metadata_json:
            return {}

        try:
            return json.loads(self.metadata_json)
        except json.JSONDecodeError:
            return {}

    @property
    def attempt(self) -> int | None:
        """ResearchJob.attempts when it happened (0 before the first
        claim)."""

        return self.details.get("attempt")

    @property
    def worker_id(self) -> str | None:
        return self.details.get("worker_id")

    @property
    def tool(self) -> str | None:
        """For the tool_* events: which tool."""

        return self.details.get("tool")

    @property
    def duration_ms(self) -> float | None:
        """For tool_completed / tool_failed: how long the search took,
        including retries."""

        return self.details.get("duration_ms")


# Registers ResearchJob with Base, so that anything loading these models
# (the app, Alembic, the tests) also knows the research_jobs table that
# ResearchTask.jobs points to. At the end of the file because
# app/jobs/models.py imports Base and utc_now from here, and a module import
# (not "from ... import ResearchJob") so it also works when app.jobs.models
# is the one imported first.
import app.jobs.models  # noqa: E402, F401


def __getattr__(name: str):
    """Lets the job models (JobStatus, ResearchJob, AttemptOutcome,
    JobAttempt, JobEventType; defined in app/jobs/models.py) be imported from here too,
    like the other models:

        from app.db.models import JobStatus, ResearchJob

    Looked up on first use rather than imported above, because a plain
    "from app.jobs.models import ..." here fails when app.jobs.models is the
    module being imported first (circular import).
    """

    if name in (
        "AttemptOutcome",
        "JobAttempt",
        "JobEventType",
        "JobStatus",
        "ResearchJob",
    ):
        return getattr(app.jobs.models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
