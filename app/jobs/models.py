from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

# The project's one Base, so research_jobs is in the same metadata as the
# other tables (Alembic sees it, and the foreign key to research_tasks
# resolves).
from app.db.models import Base, utc_now

if TYPE_CHECKING:
    from app.db.models import ResearchTask


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchJob(Base):
    __tablename__ = "research_jobs"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    task_id: Mapped[int] = mapped_column(
        ForeignKey("research_tasks.id"),
        nullable=False,
        index=True,
    )

    status: Mapped[JobStatus] = mapped_column(
        String(50),
        default=JobStatus.PENDING,
        nullable=False,
        index=True,
    )

    attempts: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )

    worker_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    # The worker's claim on a RUNNING job: a token identifying this claim,
    # and when it runs out unless the worker renews it. A job whose lease
    # has expired belongs to a worker that died, and can be taken over.
    lease_token: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # utc_now instead of datetime.utcnow, which is deprecated; same value.
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    task: Mapped["ResearchTask"] = relationship(
        back_populates="jobs",
    )

    # Everything that has happened to the job, oldest first (see JobEvent,
    # in app/db/models.py).
    events: Mapped[list["JobEvent"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobEvent.id",
    )


class AttemptOutcome(str, Enum):
    # A worker is running it now.
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # The worker died or stopped renewing its lease; the job was recovered.
    LEASE_EXPIRED = "lease_expired"
    # The worker shut down and handed the job back to the queue.
    RELEASED = "released"


class JobAttempt(Base):
    """One run of a job by a worker. research_jobs only holds the latest
    attempt (its error is cleared on retry); these rows keep every one."""

    __tablename__ = "job_attempts"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    job_id: Mapped[int] = mapped_column(
        ForeignKey("research_jobs.id"),
        nullable=False,
        index=True,
    )

    # 1 for the first run; matches ResearchJob.attempts at the time.
    attempt_number: Mapped[int] = mapped_column(
        nullable=False,
    )

    worker_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    outcome: Mapped[AttemptOutcome] = mapped_column(
        String(50),
        default=AttemptOutcome.RUNNING,
        nullable=False,
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

    # NULL while the attempt is running.
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )


class JobEventType(str, Enum):
    CREATED = "created"
    # A worker claimed it (an attempt started).
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    # Put back in the queue after failing (requeue_job).
    REQUEUED = "requeued"
    # Its worker died or stopped renewing the lease; recovered.
    LEASE_EXPIRED = "lease_expired"
    # Its worker shut down and handed it back to the queue.
    RELEASED = "released"
    # The agent started a search; tool is set.
    TOOL_STARTED = "tool_started"
    # A search finished; tool and duration_ms are set.
    TOOL_COMPLETED = "tool_completed"
    # A search failed (after its retries); message is the error.
    TOOL_FAILED = "tool_failed"
