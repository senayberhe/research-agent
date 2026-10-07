import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobEvent, JobEventType


async def record_job_event(
    db: AsyncSession,
    job_id: int,
    event_type: str,
    message: str | None = None,
    metadata: dict | None = None,
) -> JobEvent:
    """Adds an event to the job's timeline and flushes; the caller commits.

    event_type is a JobEventType or its value ("failed"); anything else
    raises ValueError. metadata is stored as JSON in metadata_json (read
    back through JobEvent.details).

    Added by job_id rather than job.events.append, which would lazy-load
    the collection (not allowed with async sessions).
    """

    event = JobEvent(
        job_id=job_id,
        event_type=JobEventType(event_type).value,
        message=message,
        metadata_json=(
            json.dumps(metadata)
            if metadata is not None
            else None
        ),
    )

    db.add(event)

    await db.flush()

    return event


async def record_tool_event(
    db: AsyncSession,
    job_id: int,
    event_type: str,
    tool: str,
    query: str,
    success: bool | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
) -> JobEvent:
    """A tool_started / tool_completed / tool_failed event: the tool and
    query, plus (once the search has finished) whether it succeeded, how
    long it took and its error."""

    metadata = {
        "tool": tool,
        "query": query,
    }

    if success is not None:
        metadata["success"] = success

    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms

    if error is not None:
        metadata["error"] = error

    return await record_job_event(
        db=db,
        job_id=job_id,
        event_type=event_type,
        message=f"{tool} tool event.",
        metadata=metadata,
    )
