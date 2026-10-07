import uuid

from datetime import datetime, timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import utc_now
from app.jobs.events import record_job_event
from app.jobs.models import (
    AttemptOutcome,
    JobAttempt,
    JobEventType,
    JobStatus,
    ResearchJob,
)
from app.services.state_machine import InvalidStateTransition


# Where each job status may go next. A failed job is retried either
# directly (FAILED -> RUNNING) or by putting it back in the queue for any
# worker to claim (FAILED -> PENDING, see requeue_job). Each run counts as an
# attempt. COMPLETED is final.
#
# RUNNING -> PENDING: the worker's lease expired (it died), so the job goes
# back in the queue for another worker (see recover_stale_jobs).
JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.RUNNING, JobStatus.FAILED},
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.PENDING,
    },
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: {JobStatus.RUNNING, JobStatus.PENDING},
}


def _transition_job(job: ResearchJob, new_status: JobStatus) -> None:

    # status is a plain string column, so a loaded row holds "pending"
    # rather than JobStatus.PENDING; JobStatus(...) accepts both.
    current = JobStatus(job.status)

    if new_status not in JOB_TRANSITIONS[current]:
        raise InvalidStateTransition(
            f"Research job {job.id} can't go from "
            f"{current.value} to {new_status.value}"
        )

    job.status = new_status


async def _start_attempt(
    db: AsyncSession,
    job: ResearchJob,
    started_at: datetime,
) -> None:
    """Records the attempt mark_job_running just began."""

    db.add(
        JobAttempt(
            job_id=job.id,
            attempt_number=job.attempts,
            worker_id=job.worker_id,
            outcome=AttemptOutcome.RUNNING,
            started_at=started_at,
        )
    )


async def _end_attempt(
    db: AsyncSession,
    job_id: int,
    outcome: AttemptOutcome,
    error: str | None = None,
) -> None:
    """Records how the job's running attempt ended. Does nothing if none is
    running (e.g. a PENDING job failed, or one claimed before attempts were
    recorded)."""

    await db.execute(
        update(JobAttempt)
        .where(
            JobAttempt.job_id == job_id,
            JobAttempt.ended_at.is_(None),
        )
        .values(
            outcome=outcome,
            error=error,
            ended_at=utc_now(),
        )
    )


async def create_research_job(
        db: AsyncSession,
        task_id: int,
) -> ResearchJob:
    job = ResearchJob(
        task_id=task_id,
        status=JobStatus.PENDING,
        attempts=0,
    )
    db.add(job)
    # Gives the job its id for the event.
    await db.flush()
    await record_job_event(
        db,
        job.id,
        JobEventType.CREATED,
        metadata={"attempt": 0},
    )
    await db.commit()
    await db.refresh(job)
    return job


async def get_research_job(
        db: AsyncSession,
        job_id: int,
) -> ResearchJob | None:
    result = await db.execute(
        select(ResearchJob).where(ResearchJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_latest_job_for_task(
        db: AsyncSession,
        task_id: int,
) -> ResearchJob | None:
    return await db.scalar(
        select(ResearchJob)
        .where(ResearchJob.task_id == task_id)
        .order_by(ResearchJob.id.desc())
        .limit(1)
    )


async def get_next_pending_job(
        db: AsyncSession,
        lock: bool = False,
) -> ResearchJob | None:
    """The oldest PENDING job (the next one a worker would claim), or None.

    By default it only reads: two workers calling it at the same time get
    the same job. With lock=True (what claim_next_job uses) the row is
    locked FOR UPDATE SKIP LOCKED until the transaction ends: other workers
    skip it and get the next pending job instead, so no two can take the
    same one.
    """

    query = (
        select(ResearchJob)
        .where(
            ResearchJob.status == JobStatus.PENDING
        )
        # id breaks ties between jobs created at the same instant.
        .order_by(ResearchJob.created_at.asc(), ResearchJob.id.asc())
        .limit(1)
    )

    if lock:
        query = query.with_for_update(skip_locked=True)

    result = await db.execute(query)

    return result.scalars().first()


async def find_running_jobs(
        db: AsyncSession,
) -> list[ResearchJob]:
    result = await db.execute(
        select(ResearchJob)
        .where(ResearchJob.status == JobStatus.RUNNING)
        .order_by(ResearchJob.id)
    )
    return list(result.scalars().all())


async def claim_next_job(
        db: AsyncSession,
        worker_id: str,
) -> ResearchJob | None:
    """Takes the oldest PENDING job for this worker and marks it RUNNING,
    or returns None if there is none.

    Safe with several workers: the job is read with lock=True, so a job one
    worker is claiming is skipped by the others, and each job is claimed
    exactly once. (Reading it without the lock and then marking it running
    is not safe: two workers can both read it before either marks it.)
    """

    job = await get_next_pending_job(db, lock=True)

    if job is None:
        return None

    await mark_job_running(
        db=db,
        job=job,
        worker_id=worker_id,
    )

    # Ends the transaction, which releases the row lock with the job
    # already RUNNING. (mark_job_running commits too; this keeps the end of
    # the claim explicit.)
    await db.commit()

    return job


async def mark_job_running(
    db: AsyncSession,
    job: ResearchJob,
    worker_id: str,
) -> ResearchJob:
    """Claims the job for this worker with a new lease. Flushes; the
    caller commits (claim_next_job does)."""

    _transition_job(job, JobStatus.RUNNING)

    job.worker_id = worker_id

    # A new token for every claim (including a retry of the same job), so a
    # worker can tell whether the job is still its own: if another worker
    # took it over, the token has changed.
    job.lease_token = str(uuid.uuid4())

    # One clock reading, so the lease and started_at agree.
    now = utc_now()

    job.lease_expires_at = (
        now
        + timedelta(
            seconds=settings.worker_lease_seconds
        )
    )

    job.attempts += 1

    job.started_at = now

    # A retry starts clean: the previous attempt's error and end time are
    # gone from the job (they stay in its JobAttempt row).
    job.error = None
    job.completed_at = None

    await _start_attempt(db, job, started_at=now)

    await record_job_event(
        db,
        job.id,
        JobEventType.CLAIMED,
        metadata={
            "worker_id": worker_id,
            "attempt": job.attempts,
        },
    )

    await db.flush()

    return job


async def _lock_if_still_ours(
    db: AsyncSession,
    job: ResearchJob,
    lease_token: str,
) -> ResearchJob | None:
    """The job, locked, if it is still RUNNING under this lease token;
    otherwise None (the lease expired and the job was requeued or taken
    over by another worker).

    The lock (FOR UPDATE) makes check-then-update safe: stale-job recovery
    can't requeue the job between this check and the caller's update.
    """

    result = await db.execute(
        select(ResearchJob)
        .where(
            ResearchJob.id == job.id,
            ResearchJob.status == JobStatus.RUNNING,
            ResearchJob.lease_token == lease_token,
        )
        .with_for_update()
        # Read the current row, not what this session last saw.
        .execution_options(populate_existing=True)
    )

    return result.scalar_one_or_none()


async def mark_job_completed(
    db: AsyncSession,
    job: ResearchJob,
    lease_token: str,
) -> bool:
    """Marks the job COMPLETED if this worker still holds its lease.
    Returns False (and changes nothing) if it doesn't. Flushes; the caller
    commits."""

    current_job = await _lock_if_still_ours(db, job, lease_token)

    if current_job is None:
        return False

    _transition_job(current_job, JobStatus.COMPLETED)

    await _end_attempt(db, current_job.id, AttemptOutcome.COMPLETED)

    await record_job_event(
        db,
        current_job.id,
        JobEventType.COMPLETED,
        metadata={
            "worker_id": current_job.worker_id,
            "attempt": current_job.attempts,
        },
    )

    current_job.completed_at = utc_now()

    current_job.lease_token = None

    current_job.lease_expires_at = None

    await db.flush()

    return True


async def mark_job_failed(
    db: AsyncSession,
    job: ResearchJob,
    error: str,
    lease_token: str | None = None,
) -> bool:
    """Marks the job FAILED. With a lease_token (a worker failing its own
    job), only if that worker still holds the lease; returns False and
    changes nothing if it doesn't. Without one (recovery, which knows the
    worker is gone), unconditionally. Flushes; the caller commits."""

    if lease_token is not None:
        current_job = await _lock_if_still_ours(db, job, lease_token)

        if current_job is None:
            return False
    else:
        current_job = job

    _transition_job(current_job, JobStatus.FAILED)

    await _end_attempt(db, current_job.id, AttemptOutcome.FAILED, error)

    await record_job_event(
        db,
        current_job.id,
        JobEventType.FAILED,
        message=error,
        metadata={
            "worker_id": current_job.worker_id,
            "attempt": current_job.attempts,
        },
    )

    current_job.error = error

    current_job.completed_at = utc_now()

    current_job.lease_token = None

    current_job.lease_expires_at = None

    await db.flush()

    return True


async def requeue_job(
        db: AsyncSession,
        job: ResearchJob,
) -> ResearchJob:
    """Puts a failed job back in the queue (PENDING) for a worker to claim
    and run again. Its attempt count is kept."""
    _transition_job(job, JobStatus.PENDING)
    await record_job_event(
        db,
        job.id,
        JobEventType.REQUEUED,
        metadata={"attempt": job.attempts},
    )
    # Back in the queue: no worker holds it, so no lease either.
    job.worker_id = None
    job.lease_token = None
    job.lease_expires_at = None
    job.error = None
    job.started_at = None
    job.completed_at = None
    await db.commit()
    await db.refresh(job)
    return job


STALE_LEASE_ERROR = (
    "Previous worker lease expired; "
    "job returned to pending."
)

MAX_ATTEMPTS_ERROR = (
    "Job exceeded maximum retry attempts."
)


async def recover_stale_jobs(
    db: AsyncSession,
) -> int:
    """Finds RUNNING jobs whose lease has expired (their worker died or
    stopped renewing) and puts them back in the queue, or fails them if
    they have used up WORKER_MAX_ATTEMPTS. Returns how many were recovered.

    The dead worker's agent run is marked FAILED (keeping its checkpoint)
    and its task FAILED, so the next worker resumes the task from the
    checkpoint instead of finding it half-started.

    Safe with several workers: rows are locked FOR UPDATE SKIP LOCKED, so
    two workers recovering at once never handle the same job. Flushes; the
    caller commits.
    """

    # Imported here: these services import app.jobs indirectly.
    from app.services.agent_run_service import fail_unfinished_runs_for_task

    now = utc_now()

    result = await db.execute(
        select(ResearchJob)
        .where(
            ResearchJob.status == JobStatus.RUNNING,
            or_(
                ResearchJob.lease_expires_at < now,
                # No lease at all: claimed before leases existed (every
                # claim sets one now), so nothing will ever renew it.
                ResearchJob.lease_expires_at.is_(None),
            ),
        )
        .order_by(ResearchJob.id)
        .with_for_update(
            skip_locked=True,
        )
    )

    jobs = result.scalars().all()

    recovered = 0

    for job in jobs:

        await fail_unfinished_runs_for_task(
            db=db,
            task_id=job.task_id,
            error=STALE_LEASE_ERROR,
        )

        await _end_attempt(
            db,
            job.id,
            AttemptOutcome.LEASE_EXPIRED,
            STALE_LEASE_ERROR,
        )

        await record_job_event(
            db,
            job.id,
            JobEventType.LEASE_EXPIRED,
            message=STALE_LEASE_ERROR,
            metadata={
                "worker_id": job.worker_id,
                "attempt": job.attempts,
            },
        )

        if job.attempts >= settings.worker_max_attempts:

            _transition_job(job, JobStatus.FAILED)

            await record_job_event(
                db,
                job.id,
                JobEventType.FAILED,
                message=MAX_ATTEMPTS_ERROR,
                metadata={"attempt": job.attempts},
            )

            job.error = MAX_ATTEMPTS_ERROR

            job.completed_at = now

        else:

            _transition_job(job, JobStatus.PENDING)

            job.worker_id = None

            job.error = STALE_LEASE_ERROR

        # Either way no worker holds it any more.
        job.lease_token = None

        job.lease_expires_at = None

        recovered += 1

    await db.flush()

    return recovered


async def get_queue_health(
    db: AsyncSession,
) -> dict:
    """Numbers for monitoring the job queue: jobs per status, how long the
    oldest pending job has waited, RUNNING jobs whose lease has expired
    (their worker died; recovered on the next poll), and the workers that
    hold a live lease."""

    from sqlalchemy import func

    now = utc_now()

    counts = {status.value: 0 for status in JobStatus}

    result = await db.execute(
        select(ResearchJob.status, func.count())
        .group_by(ResearchJob.status)
    )

    for status, count in result.all():
        counts[JobStatus(status).value] = count

    oldest_pending = await db.scalar(
        select(func.min(ResearchJob.created_at))
        .where(ResearchJob.status == JobStatus.PENDING)
    )

    expired_leases = await db.scalar(
        select(func.count())
        .select_from(ResearchJob)
        .where(
            ResearchJob.status == JobStatus.RUNNING,
            or_(
                ResearchJob.lease_expires_at < now,
                ResearchJob.lease_expires_at.is_(None),
            ),
        )
    )

    active_workers = (
        await db.execute(
            select(ResearchJob.worker_id)
            .where(
                ResearchJob.status == JobStatus.RUNNING,
                ResearchJob.lease_expires_at >= now,
            )
            .distinct()
            .order_by(ResearchJob.worker_id)
        )
    ).scalars().all()

    return {
        "jobs": counts,
        "oldest_pending_seconds": (
            round((now - oldest_pending).total_seconds(), 1)
            if oldest_pending is not None
            else None
        ),
        "expired_leases": expired_leases,
        "active_workers": list(active_workers),
    }


async def renew_job_lease(
    db: AsyncSession,
    job_id: int,
    lease_token: str,
) -> bool:
    """Extends the lease by WORKER_LEASE_SECONDS from now, if the job is
    still RUNNING under this token. Returns False if it isn't (the lease
    expired and the job was recovered or taken over). Commits.

    A single UPDATE with the token in its WHERE clause, so the check and the
    renewal can't be split by stale-job recovery.
    """

    result = await db.execute(
        update(ResearchJob)
        .where(
            ResearchJob.id == job_id,
            ResearchJob.status == JobStatus.RUNNING,
            ResearchJob.lease_token == lease_token,
        )
        .values(
            lease_expires_at=(
                utc_now()
                + timedelta(seconds=settings.worker_lease_seconds)
            ),
        )
    )

    await db.commit()

    return result.rowcount == 1


async def release_job(
    db: AsyncSession,
    job: ResearchJob,
    lease_token: str,
    reason: str,
) -> bool:
    """Gives a job this worker still holds straight back to the queue
    (RUNNING -> PENDING), e.g. on shutdown, instead of leaving it for its
    lease to expire. Its run is failed (checkpoint kept) and its task too,
    so the next worker resumes it. Returns False if the job is no longer
    this worker's. Flushes; the caller commits.
    """

    from app.services.agent_run_service import fail_unfinished_runs_for_task

    current_job = await _lock_if_still_ours(db, job, lease_token)

    if current_job is None:
        return False

    await fail_unfinished_runs_for_task(
        db=db,
        task_id=current_job.task_id,
        error=reason,
    )

    _transition_job(current_job, JobStatus.PENDING)

    await _end_attempt(db, current_job.id, AttemptOutcome.RELEASED, reason)

    await record_job_event(
        db,
        current_job.id,
        JobEventType.RELEASED,
        message=reason,
        metadata={
            "worker_id": current_job.worker_id,
            "attempt": current_job.attempts,
        },
    )

    current_job.worker_id = None
    current_job.lease_token = None
    current_job.lease_expires_at = None
    current_job.error = reason

    await db.flush()

    return True


async def get_jobs_for_task(
    db: AsyncSession,
    task_id: int,
) -> list[ResearchJob]:
    """The task's jobs, oldest first."""

    result = await db.execute(
        select(ResearchJob)
        .where(
            ResearchJob.task_id == task_id
        )
        .order_by(
            ResearchJob.created_at.asc(),
            # Breaks ties between jobs created at the same instant.
            ResearchJob.id.asc(),
        )
    )

    return list(result.scalars().all())


async def list_job_attempts(
    db: AsyncSession,
    job_id: int,
) -> list[JobAttempt]:
    """Every run of the job, oldest first."""

    result = await db.execute(
        select(JobAttempt)
        .where(JobAttempt.job_id == job_id)
        .order_by(JobAttempt.attempt_number, JobAttempt.id)
    )

    return list(result.scalars().all())


async def get_worker_activity(
    db: AsyncSession,
    since: datetime,
) -> list[dict]:
    """What each worker that has run a job since `since` is doing, from its
    job attempts: "busy" (running a job under a live lease), "unresponsive"
    (running a job whose lease has expired: it probably died) or "idle"
    (its last attempt ended; it may also have stopped). Workers that haven't
    claimed a job in that time aren't listed: the database never hears
    from them (their health file is local to their container)."""

    from sqlalchemy import func
    from sqlalchemy.dialects.postgresql import distinct_on

    now = utc_now()

    # Each worker's latest attempt, with its job's lease.
    latest = (
        await db.execute(
            select(JobAttempt, ResearchJob.lease_expires_at)
            .join(ResearchJob, ResearchJob.id == JobAttempt.job_id)
            .where(
                JobAttempt.worker_id.is_not(None),
                JobAttempt.started_at >= since,
            )
            .order_by(
                JobAttempt.worker_id,
                JobAttempt.started_at.desc(),
                JobAttempt.id.desc(),
            )
            .ext(distinct_on(JobAttempt.worker_id))
        )
    ).all()

    outcome_counts = (
        await db.execute(
            select(JobAttempt.worker_id, JobAttempt.outcome, func.count())
            .where(
                JobAttempt.worker_id.is_not(None),
                JobAttempt.started_at >= since,
            )
            .group_by(JobAttempt.worker_id, JobAttempt.outcome)
        )
    ).all()

    counts: dict[str, dict[str, int]] = {}

    for worker_id, outcome, count in outcome_counts:
        counts.setdefault(worker_id, {})[AttemptOutcome(outcome).value] = count

    workers = []

    for attempt, lease_expires_at in latest:

        running = AttemptOutcome(attempt.outcome) == AttemptOutcome.RUNNING

        if not running:
            state = "idle"
        elif lease_expires_at is not None and lease_expires_at >= now:
            state = "busy"
        else:
            state = "unresponsive"

        workers.append(
            {
                "worker_id": attempt.worker_id,
                "state": state,
                "current_job_id": attempt.job_id if running else None,
                "lease_expires_at": lease_expires_at if running else None,
                "last_attempt_started_at": attempt.started_at,
                "last_attempt_ended_at": attempt.ended_at,
                "attempts": counts.get(attempt.worker_id, {}),
            }
        )

    return workers


async def list_jobs(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
    status: JobStatus | None = None,
) -> tuple[list[tuple[ResearchJob, str]], int]:
    """A page of jobs with their task's question, newest first, and how
    many there are in all (optionally only those with this status)."""

    from sqlalchemy import func

    from app.db.models import ResearchTask

    query = select(ResearchJob, ResearchTask.question).join(
        ResearchTask,
        ResearchTask.id == ResearchJob.task_id,
    )
    count = select(func.count()).select_from(ResearchJob)

    if status is not None:
        query = query.where(ResearchJob.status == status)
        count = count.where(ResearchJob.status == status)

    rows = (
        await db.execute(
            query
            .order_by(ResearchJob.created_at.desc(), ResearchJob.id.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()

    return [(job, question) for job, question in rows], await db.scalar(count)

