"""Server-sent events: job events worth notifying someone about, streamed
as they happen (GET /events).

Each message's id is the job_events row id. EventSource sends the last id
it received when it reconnects (Last-Event-ID), and the stream resumes
after it, so nothing is sent twice and nothing is missed across
reconnects. A new connection without one starts at the newest event: it
gets what happens from now on, not the history.

Ids are assigned when a row is inserted, but rows become visible when their
transaction commits, which can be slightly out of order across workers. So
the stream only sends events at least SETTLE_SECONDS old: by then any lower
id has committed too.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import JobEvent, ResearchTask, utc_now
from app.jobs.event_service import parse_event_metadata
from app.jobs.models import JobEventType, ResearchJob


# The events a person wants to hear about.
NOTIFY_EVENT_TYPES = (
    JobEventType.COMPLETED.value,
    JobEventType.FAILED.value,
    JobEventType.TOOL_FAILED.value,
    JobEventType.REQUEUED.value,
    JobEventType.LEASE_EXPIRED.value,
)

SETTLE_SECONDS = 1.0

# A comment line every so often keeps proxies from closing an idle stream.
KEEPALIVE_SECONDS = 15.0

# At most this many events per poll (a backlog is sent over several).
BATCH_SIZE = 100


async def latest_event_id(db: AsyncSession) -> int:
    return await db.scalar(select(func.coalesce(func.max(JobEvent.id), 0)))


async def events_after(db: AsyncSession, after_id: int) -> list[dict]:
    """Notifiable events with an id above after_id, settled, oldest first,
    with their task and question."""

    rows = (
        await db.execute(
            select(JobEvent, ResearchJob.task_id, ResearchTask.question)
            .join(ResearchJob, ResearchJob.id == JobEvent.job_id)
            .join(ResearchTask, ResearchTask.id == ResearchJob.task_id)
            .where(
                JobEvent.id > after_id,
                JobEvent.event_type.in_(NOTIFY_EVENT_TYPES),
                JobEvent.created_at <= utc_now() - timedelta(seconds=SETTLE_SECONDS),
            )
            .order_by(JobEvent.id)
            .limit(BATCH_SIZE)
        )
    ).all()

    return [
        {
            "id": event.id,
            "job_id": event.job_id,
            "task_id": task_id,
            "question": question,
            "event_type": event.event_type,
            "message": event.message,
            "metadata": parse_event_metadata(event),
            "created_at": event.created_at.isoformat() + "Z",
        }
        for event, task_id, question in rows
    ]


def format_message(event: dict) -> str:
    """One SSE message: id, event name, JSON data."""

    return (
        f"id: {event['id']}\n"
        "event: job_event\n"
        f"data: {json.dumps(event)}\n\n"
    )


async def stream_job_events(
    session_factory: async_sessionmaker[AsyncSession],
    last_event_id: int | None,
    poll_seconds: float,
    max_seconds: float,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[str]:
    """SSE messages until max_seconds have passed or the client goes."""

    async with session_factory() as db:
        after = (
            last_event_id
            if last_event_id is not None
            else await latest_event_id(db)
        )

    # Tell EventSource how long to wait before reconnecting, and where it
    # is (so even an idle stream resumes from the right place).
    yield f"retry: 2000\nid: {after}\n\n"

    started = last_keepalive = time.monotonic()

    while time.monotonic() - started < max_seconds:

        if await is_disconnected():
            return

        async with session_factory() as db:
            events = await events_after(db, after)

        for event in events:
            yield format_message(event)
            after = event["id"]

        if not events and time.monotonic() - last_keepalive >= KEEPALIVE_SECONDS:
            yield ": keepalive\n\n"
            last_keepalive = time.monotonic()

        if len(events) < BATCH_SIZE:
            await asyncio.sleep(poll_seconds)
