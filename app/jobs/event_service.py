import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import JobEvent, JobEventType


async def get_job_events(
    db: AsyncSession,
    job_id: int,
) -> list[JobEvent]:
    """The job's timeline, oldest first."""

    result = await db.execute(
        select(JobEvent)
        .where(JobEvent.job_id == job_id)
        # id breaks ties between events recorded at the same instant.
        .order_by(JobEvent.created_at.asc(), JobEvent.id.asc())
    )

    return list(result.scalars().all())


def parse_event_metadata(
    event: JobEvent,
) -> dict:
    """The event's metadata_json as a dict; {} if there is none or it
    isn't valid JSON (a bad row shouldn't break reading the timeline)."""

    if not event.metadata_json:
        return {}

    try:
        return json.loads(
            event.metadata_json
        )
    except json.JSONDecodeError:
        return {}


# How tool names (ResearchTool.name) are shown in a timeline.
TOOL_DISPLAY_NAMES = {
    "arxiv": "arXiv",
    "tavily": "Tavily",
    "wikipedia": "Wikipedia",
}


def describe_job_event(event: JobEvent) -> str | None:
    """The detail shown after an event's type, e.g. "Tavily, 842 ms" for a
    finished search, or None if there's nothing to add ("created")."""

    event_type = JobEventType(event.event_type)
    metadata = parse_event_metadata(event)

    tool = metadata.get("tool")

    if tool is not None:

        detail = TOOL_DISPLAY_NAMES.get(tool, tool)

        duration_ms = metadata.get("duration_ms")

        if duration_ms is not None:
            detail += f", {duration_ms:.0f} ms"

        error = metadata.get("error")

        if event_type == JobEventType.TOOL_FAILED and error:
            detail += f": {error}"

        return detail

    if event_type == JobEventType.CLAIMED:
        return metadata.get("worker_id")

    return event.message


def format_job_timeline(events: list[JobEvent]) -> str:
    """The events as text, one per line; a blank line after each finished
    search groups it with its start:

        tool_started     → Tavily
        tool_completed   → Tavily, 842 ms

        completed
    """

    lines = []

    for event in events:

        event_type = JobEventType(event.event_type)
        detail = describe_job_event(event)

        lines.append(
            f"{event_type.value:<16} → {detail}"
            if detail
            else event_type.value
        )

        if event_type in (JobEventType.TOOL_COMPLETED, JobEventType.TOOL_FAILED):
            lines.append("")

    return "\n".join(lines).strip() + "\n"
