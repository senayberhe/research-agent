"""Research worker: claims pending research jobs and runs them.

    uv run python -m app.jobs.worker

For each job: claim it (PENDING -> RUNNING, safe with several workers), run
the task's research workflow (a new task) or resume its failed run from the
checkpoint (a retried task), which runs the ResearchAgent and records the
AgentRun, then mark the job COMPLETED or FAILED.

While it runs (ResearchWorker.run):

- Heartbeat: every WORKER_HEARTBEAT_SECONDS it renews the current job's
  lease and refreshes its health file. If the lease was lost (taken over),
  it cancels the job rather than keep spending tokens on it.
- Recovery: before each claim, jobs whose lease expired (their worker died)
  go back to the queue, so several workers can run safely.
- Graceful shutdown: on SIGTERM / SIGINT it stops claiming, lets the current
  job finish for up to WORKER_SHUTDOWN_TIMEOUT_SECONDS, then cancels it and
  hands it straight back to the queue (resumed from its checkpoint).
"""

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.db.models import ResearchTask, TaskStatus, utc_now
from app.jobs.models import JobStatus
from app.jobs.service import (
    claim_next_job,
    get_research_job,  # process_job loads the claimed job with it
    mark_job_completed,
    mark_job_failed,
    recover_stale_jobs,
    release_job,
    renew_job_lease,
)
from app.services.agent_run_service import get_latest_agent_run
from app.services.workflow_service import (
    execute_research_workflow,
    resume_research_workflow,
)

logger = logging.getLogger(__name__)


def default_worker_id() -> str:
    # Unique per process, and says where it runs: "<hostname>-<pid>".
    return f"{socket.gethostname()}-{os.getpid()}"


async def _run_task(
    db: AsyncSession,
    task: ResearchTask,
    job_id: int,
) -> ResearchTask:
    """Runs the research for a job's task: from the start for a new task,
    or resumed from its failed run's checkpoint for a retried one."""

    status = TaskStatus(task.status)

    if status == TaskStatus.PENDING:
        return await execute_research_workflow(
            db=db,
            task=task,
            job_id=job_id,
        )

    if status == TaskStatus.FAILED:

        run = await get_latest_agent_run(db=db, task_id=task.id)

        # Raises ValueError (the reason) if there is nothing to resume.
        return await resume_research_workflow(
            db=db,
            task=task,
            agent_run=run,
            job_id=job_id,
        )

    raise ValueError(
        f"Research task {task.id} is {status.value}; nothing to run."
    )


async def _failure_reason(
    db: AsyncSession,
    task: ResearchTask,
) -> str:

    run = await get_latest_agent_run(db=db, task_id=task.id)

    if run is not None and run.error:
        return run.error

    return (
        f"Research task {task.id} ended as {TaskStatus(task.status).value}."
    )


async def process_job(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: int,
) -> JobStatus:
    """Runs one claimed (RUNNING) job and marks it COMPLETED or FAILED."""

    async with session_factory() as db:

        job = await get_research_job(db=db, job_id=job_id)
        task = await db.get(ResearchTask, job.task_id)

        # This claim's lease: the result is recorded only if the job is
        # still ours (the lease didn't expire and get taken over meanwhile).
        lease_token = job.lease_token

        logger.info(
            "Running job job_id=%s task_id=%s attempt=%s worker_id=%s",
            job.id,
            job.task_id,
            job.attempts,
            job.worker_id,
        )

        # The workflow handles agent errors itself (it records them on the
        # AgentRun and fails the task), so an exception here is something
        # else: a task that can't be resumed, a database error...
        try:
            task = await _run_task(db=db, task=task, job_id=job.id)

            error = (
                None
                if TaskStatus(task.status) == TaskStatus.COMPLETED
                else await _failure_reason(db=db, task=task)
            )

        except Exception as exc:
            logger.exception("Job failed job_id=%s", job_id)
            await db.rollback()
            error = str(exc) or type(exc).__name__

        # The workflow committed and may have rolled back, which expires the
        # job object; reload it before changing it.
        await db.refresh(job)

        if error is None:
            recorded = await mark_job_completed(
                db=db,
                job=job,
                lease_token=lease_token,
            )
        else:
            recorded = await mark_job_failed(
                db=db,
                job=job,
                error=error,
                lease_token=lease_token,
            )

        await db.commit()

        if not recorded:
            # The lease expired and the job was requeued or taken over; the
            # job's state now belongs to whoever has it.
            await db.refresh(job)

            logger.warning(
                "Worker lost its lease on job job_id=%s (now %s); result "
                "not recorded",
                job.id,
                job.status,
            )

            return JobStatus(job.status)

        logger.info(
            "Finished job job_id=%s status=%s error=%s",
            job.id,
            job.status,
            job.error,
        )

        return JobStatus(job.status)


SHUTDOWN_RELEASE_ERROR = (
    "Worker shut down before the job finished; job returned to pending."
)


async def run_once(
    session_factory: async_sessionmaker[AsyncSession],
    worker_id: str,
) -> bool:
    """Claims and runs the next pending job. Returns False if there was
    none. (The same as ResearchWorker.process_next_job.)"""

    return await ResearchWorker(
        worker_id=worker_id,
        session_factory=session_factory,
    ).process_next_job()


async def run_worker(
    session_factory: async_sessionmaker[AsyncSession],
    worker_id: str,
    stop: asyncio.Event,
    poll_interval_seconds: float,
    recover: bool = True,
) -> None:
    """Runs jobs until stop is set. (The same as ResearchWorker.run.)"""

    await ResearchWorker(
        worker_id=worker_id,
        session_factory=session_factory,
        poll_interval_seconds=poll_interval_seconds,
    ).run(stop=stop, recover=recover)


@dataclass
class _CurrentJob:
    job_id: int
    lease_token: str
    task: asyncio.Task
    # Set by the heartbeat when renewing found the job taken over.
    lease_lost: bool = False


class ResearchWorker:
    """A worker with its own id and database sessions.

        worker = ResearchWorker()
        await worker.process_next_job()   # one job, if there is one
        await worker.run()                # until SIGTERM / SIGINT
    """

    def __init__(
        self,
        worker_id: str | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        poll_interval_seconds: float | None = None,
        heartbeat_seconds: float | None = None,
        shutdown_timeout_seconds: float | None = None,
        health_file: str | None = None,
    ):
        self.worker_id = worker_id or default_worker_id()

        # Looked up here (not as a default argument) so tests can point
        # AsyncSessionLocal at the test database.
        self.session_factory = session_factory or AsyncSessionLocal

        self.poll_interval_seconds = _or_setting(
            poll_interval_seconds,
            settings.worker_poll_interval_seconds,
        )
        self.heartbeat_seconds = _or_setting(
            heartbeat_seconds,
            settings.worker_heartbeat_seconds,
        )
        self.shutdown_timeout_seconds = _or_setting(
            shutdown_timeout_seconds,
            settings.worker_shutdown_timeout_seconds,
        )
        self.health_file = health_file or settings.worker_health_file

        self._status = "starting"
        self.jobs_completed = 0
        self.jobs_failed = 0

        self._current: _CurrentJob | None = None

        # Set by stop(); run() returns once it is set (after the current
        # job finishes or the shutdown timeout runs out).
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """Asks the worker to shut down gracefully: no new jobs; the current
        one gets up to shutdown_timeout_seconds to finish. Safe to call from
        a signal handler."""

        if self._status != "stopped":
            self._status = "stopping"

        self._stop.set()

    @property
    def status(self) -> str:
        """starting, idle, busy, stopping or stopped."""

        return self._status

    # -------------------------------------------------------------------
    # One job
    # -------------------------------------------------------------------

    async def process_next_job(self) -> bool:
        """Claims and runs the next pending job; False if there was none."""

        claimed = await self._claim()

        if claimed is None:
            return False

        await self._run_claimed(*claimed, stop=None)

        return True

    async def _claim(self) -> tuple[int, str] | None:

        # The claim gets its own short session: it commits as soon as the
        # job is RUNNING, which releases the row lock.
        async with self.session_factory() as db:

            job = await claim_next_job(
                db=db,
                worker_id=self.worker_id,
            )

            if job is None:
                return None

            logger.info(
                "Worker %s claimed job %s",
                self.worker_id,
                job.id,
            )

            return job.id, job.lease_token

    async def _run_claimed(
        self,
        job_id: int,
        lease_token: str,
        stop: asyncio.Event | None,
    ) -> None:
        """Runs a claimed job as its own task, so the heartbeat (lease lost)
        or a shutdown timeout can cancel it."""

        job_task = asyncio.create_task(
            process_job(session_factory=self.session_factory, job_id=job_id)
        )

        current = _CurrentJob(job_id, lease_token, job_task)
        self._current = current
        # stop() may have been called while the job was being claimed.
        if self._status != "stopping":
            self._status = "busy"

        try:
            if stop is None:
                status = await job_task
            else:
                status = await self._wait_or_drain(current, stop)

            if status == JobStatus.COMPLETED:
                self.jobs_completed += 1
            elif status == JobStatus.FAILED:
                self.jobs_failed += 1

        except asyncio.CancelledError:
            # Cancelled by the heartbeat: the job is someone else's now.
            # Anything else (the worker itself being cancelled, a killed
            # process) is passed on; the job's lease will expire.
            if not current.lease_lost:
                raise

            logger.warning(
                "Stopped job job_id=%s: its lease was taken over",
                job_id,
            )

        finally:
            self._current = None
            if self._status == "busy":
                self._status = "idle"

    async def _wait_or_drain(
        self,
        current: _CurrentJob,
        stop: asyncio.Event,
    ) -> JobStatus | None:
        """Waits for the job. If stop is set meanwhile, gives it up to
        shutdown_timeout_seconds more, then cancels it and returns it to
        the queue."""

        stop_wait = asyncio.create_task(stop.wait())

        try:
            done, _ = await asyncio.wait(
                {current.task, stop_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stop_wait.cancel()

        if current.task in done:
            return current.task.result()

        self._status = "stopping"
        self._write_health()

        logger.info(
            "Shutdown requested; letting job job_id=%s finish (up to %ss)",
            current.job_id,
            self.shutdown_timeout_seconds,
        )

        try:
            # shield: the timeout must not cancel the job by itself; the
            # job is cancelled below, deliberately, and then released.
            return await asyncio.wait_for(
                asyncio.shield(current.task),
                timeout=self.shutdown_timeout_seconds,
            )
        except TimeoutError:
            pass

        current.task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await current.task

        await self._release(current)

        return None

    async def _release(self, current: _CurrentJob) -> None:

        async with self.session_factory() as db:

            job = await get_research_job(db=db, job_id=current.job_id)

            released = job is not None and await release_job(
                db=db,
                job=job,
                lease_token=current.lease_token,
                reason=SHUTDOWN_RELEASE_ERROR,
            )

            await db.commit()

        logger.warning(
            "Shutdown timeout: job job_id=%s %s",
            current.job_id,
            "returned to the queue" if released else "was no longer ours",
        )

    # -------------------------------------------------------------------
    # The loop
    # -------------------------------------------------------------------

    async def run(
        self,
        stop: asyncio.Event | None = None,
        recover: bool = True,
    ) -> None:
        """Runs jobs until stop() is called (or the given stop event is set);
        recovers jobs with expired leases before each claim unless recover
        is false.

        Signals are not handled here: the entry point (app/jobs/main.py)
        calls stop() on SIGTERM / SIGINT.
        """

        if stop is not None:
            self._stop = stop

        stop = self._stop

        logger.info("Worker starting worker_id=%s", self.worker_id)

        if not stop.is_set():
            self._status = "idle"

        heartbeat = asyncio.create_task(self._heartbeat_loop())

        try:
            while not stop.is_set():

                claimed = None

                try:
                    if recover:
                        await self._recover_stale_jobs()

                    claimed = await self._claim()

                except Exception:
                    # E.g. the database is briefly unreachable: keep the
                    # worker alive and try again after the poll interval.
                    logger.exception(
                        "Worker loop error worker_id=%s",
                        self.worker_id,
                    )

                if claimed is not None:
                    await self._run_claimed(*claimed, stop=stop)
                    continue

                # Nothing to do: wait, but wake up at once if asked to stop.
                try:
                    await asyncio.wait_for(
                        stop.wait(),
                        timeout=self.poll_interval_seconds,
                    )
                except TimeoutError:
                    pass

        finally:
            heartbeat.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

            self._status = "stopped"
            self._write_health()

            logger.info("Worker stopped worker_id=%s", self.worker_id)

    async def _recover_stale_jobs(self) -> None:

        async with self.session_factory() as db:

            recovered = await recover_stale_jobs(db)

            await db.commit()

        if recovered:
            logger.warning(
                "Recovered %d jobs with expired leases",
                recovered,
            )

    # -------------------------------------------------------------------
    # Heartbeat and health
    # -------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:

        while True:
            await self.heartbeat()
            await asyncio.sleep(self.heartbeat_seconds)

    async def heartbeat(self) -> None:
        """Renews the current job's lease and refreshes the health file.
        Never raises: a failed beat is logged and retried next time."""

        current = self._current

        if current is not None and not current.task.done():

            try:
                async with self.session_factory() as db:
                    renewed = await renew_job_lease(
                        db=db,
                        job_id=current.job_id,
                        lease_token=current.lease_token,
                    )

                if not renewed:
                    # Recovered and possibly claimed by another worker:
                    # stop working on it.
                    logger.warning(
                        "Lost the lease on job job_id=%s; cancelling it",
                        current.job_id,
                    )
                    current.lease_lost = True
                    current.task.cancel()

            except Exception:
                logger.exception(
                    "Heartbeat could not renew the lease on job job_id=%s",
                    current.job_id,
                )

        self._write_health()

    def _write_health(self) -> None:

        current = self._current

        health = {
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "status": self._status,
            "job_id": current.job_id if current is not None else None,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "updated_at": utc_now().isoformat() + "Z",
        }

        try:
            # Written to a temporary file and renamed, so the health check
            # never reads a half-written file.
            temporary = f"{self.health_file}.tmp"

            with open(temporary, "w") as file:
                json.dump(health, file)

            os.replace(temporary, self.health_file)

        except OSError:
            logger.exception("Could not write health file %s", self.health_file)


def _or_setting(value, default):
    return value if value is not None else default


async def main() -> None:

    from app.core.logging import configure_logging

    configure_logging()

    worker = ResearchWorker()

    # docker stop / Ctrl+C: finish (or hand back) the current job, then exit.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stop)

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
