from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Re-exported: tests override it from here.
from app.api.dependencies import get_session_factory, get_streaming_user
from app.core.config import settings
from app.services.event_stream_service import stream_job_events


# The stream needs a signed-in user too: the frontend's fetch sends the
# session cookie (frontend/src/eventStream.ts). Checked with a short database
# session of its own (get_streaming_user), not the request's.
router = APIRouter(
    tags=["Events"],
    dependencies=[Depends(get_streaming_user)],
)


def _event_id(value: str | int | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


@router.get(
    "/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_events(
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    after: int | None = Query(default=None, ge=0),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """Server-sent events: job completed, job failed, tool failed, job
    retried, as they happen. Each message's id is the job event's id; on
    reconnect, EventSource sends the last one (Last-Event-ID) and the
    stream resumes after it, so nothing is repeated or missed. Without
    one (or ?after=), it starts from now."""

    return StreamingResponse(
        stream_job_events(
            session_factory=session_factory,
            last_event_id=_event_id(last_event_id) if last_event_id else after,
            poll_seconds=settings.event_stream_poll_seconds,
            max_seconds=settings.event_stream_max_seconds,
            is_disconnected=request.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Don't let a proxy buffer the stream.
            "X-Accel-Buffering": "no",
        },
    )
