"""Server-sent job events (GET /events) and the jobs list (GET /jobs)."""

import json
from datetime import timedelta

import pytest
import pytest_asyncio

from app.api.events import get_session_factory
from app.core.config import settings
from app.db.models import JobEvent, JobStatus, ResearchJob, utc_now
from app.main import app
from app.services.event_stream_service import (
    events_after,
    format_message,
    latest_event_id,
    stream_job_events,
)
from app.tests.conftest import TestSessionLocal
from app.tests.test_research_api import client
from app.tests.test_workflow_service import create_task


@pytest.fixture(autouse=True)
def short_streams(monkeypatch):
    # The stream reads through its own sessions: the test database's. And
    # a short connection, so a test can read the whole response.
    app.dependency_overrides[get_session_factory] = lambda: TestSessionLocal
    monkeypatch.setattr(settings, "event_stream_poll_seconds", 0.05)
    monkeypatch.setattr(settings, "event_stream_max_seconds", 0.4)

    yield

    app.dependency_overrides.pop(get_session_factory, None)


async def add_events(*event_types, age_seconds=5.0, question="What is RAG?"):
    """A job with these events, created age_seconds ago (settled). Returns
    (task id, job id, event ids)."""

    async with TestSessionLocal() as db:
        task = await create_task(db)
        task.question = question

        job = ResearchJob(task_id=task.id, status=JobStatus.RUNNING, attempts=1)
        db.add(job)
        await db.flush()

        events = [
            JobEvent(
                job_id=job.id,
                event_type=event_type,
                message=f"{event_type} message",
                metadata_json=json.dumps({"tool": "tavily", "error": "timed out"}),
                created_at=utc_now() - timedelta(seconds=age_seconds),
            )
            for event_type in event_types
        ]
        db.add_all(events)
        await db.commit()

        return task.id, job.id, [event.id for event in events]


def parse_stream(body: str) -> list[dict]:
    """The SSE messages in a response body: [{id, event, data}]."""

    messages = []

    for block in body.split("\n\n"):
        fields = {}

        for line in block.splitlines():
            if line.startswith(":") or ":" not in line:
                continue
            name, value = line.split(":", 1)
            fields[name] = value.strip()

        if "event" in fields:
            messages.append(
                {
                    "id": int(fields["id"]),
                    "event": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )

    return messages


async def never_disconnected():
    return False


async def collect(last_event_id, max_seconds=0.3):
    return [
        message
        async for message in stream_job_events(
            session_factory=TestSessionLocal,
            last_event_id=last_event_id,
            poll_seconds=0.05,
            max_seconds=max_seconds,
            is_disconnected=never_disconnected,
        )
    ]


# -------------------------
# What's sent
# -------------------------


@pytest.mark.asyncio
async def test_only_notifiable_events(clean_tables):

    task_id, job_id, ids = await add_events(
        "created",
        "claimed",
        "tool_started",
        "tool_completed",
        "tool_failed",
        "requeued",
        "lease_expired",
        "completed",
        "failed",
        question="How does RAG work?",
    )

    async with TestSessionLocal() as db:
        events = await events_after(db, 0)

    assert [event["event_type"] for event in events] == [
        "tool_failed",
        "requeued",
        "lease_expired",
        "completed",
        "failed",
    ]

    tool_failed = events[0]

    assert tool_failed["task_id"] == task_id
    assert tool_failed["job_id"] == job_id
    assert tool_failed["question"] == "How does RAG work?"
    assert tool_failed["metadata"] == {"tool": "tavily", "error": "timed out"}
    assert tool_failed["created_at"].endswith("Z")


@pytest.mark.asyncio
async def test_unsettled_events_wait(clean_tables):

    # Just written: a lower id might not have committed yet, so wait.
    await add_events("completed", age_seconds=0)

    async with TestSessionLocal() as db:
        assert await events_after(db, 0) == []


def test_message_format():

    message = format_message({"id": 42, "event_type": "completed"})

    assert message == (
        'id: 42\nevent: job_event\ndata: {"id": 42, "event_type": "completed"}\n\n'
    )


# -------------------------
# Resuming without duplicates
# -------------------------


@pytest.mark.asyncio
async def test_new_connection_starts_from_now(clean_tables):

    # History before connecting isn't replayed.
    await add_events("completed", "failed")

    messages = parse_stream("".join(await collect(last_event_id=None)))

    assert messages == []


@pytest.mark.asyncio
async def test_reconnect_resumes_after_the_last_event(clean_tables):

    _, _, ids = await add_events("completed", "tool_failed", "failed")

    # First connection got as far as the first event...
    first = parse_stream("".join(await collect(last_event_id=0)))
    assert [message["id"] for message in first] == ids

    # ...a reconnect from ids[0] gets only what came after, once each.
    resumed = parse_stream("".join(await collect(last_event_id=ids[0])))
    assert [message["id"] for message in resumed] == ids[1:]

    # From the last one: nothing repeated.
    assert parse_stream("".join(await collect(last_event_id=ids[-1]))) == []


@pytest.mark.asyncio
async def test_stream_tells_the_client_where_it_is(clean_tables):

    _, _, ids = await add_events("completed")

    async with TestSessionLocal() as db:
        latest = await latest_event_id(db)

    chunks = await collect(last_event_id=None)

    # A retry delay, and the current position, even with no events: a
    # reconnect then resumes from here, not from the beginning.
    assert chunks[0] == f"retry: 2000\nid: {latest}\n\n"
    assert latest == ids[-1]


@pytest.mark.asyncio
async def test_stream_stops_when_the_client_goes(clean_tables):

    calls = 0

    async def gone():
        nonlocal calls
        calls += 1
        return True

    chunks = [
        chunk
        async for chunk in stream_job_events(
            session_factory=TestSessionLocal,
            last_event_id=0,
            poll_seconds=0.05,
            max_seconds=60,
            is_disconnected=gone,
        )
    ]

    # Just the opening line, then it stops (not after 60 seconds).
    assert len(chunks) == 1
    assert calls == 1


# -------------------------
# GET /events
# -------------------------


@pytest.mark.asyncio
async def test_events_endpoint(clean_tables):

    _, _, ids = await add_events("completed", "failed")

    response = client.get("/events?after=0")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"

    messages = parse_stream(response.text)

    assert [message["id"] for message in messages] == ids
    assert {message["event"] for message in messages} == {"job_event"}
    assert messages[0]["data"]["event_type"] == "completed"


@pytest.mark.asyncio
async def test_events_endpoint_resumes_from_last_event_id(clean_tables):

    _, _, ids = await add_events("completed", "tool_failed", "failed")

    # What EventSource sends on reconnect.
    response = client.get("/events", headers={"Last-Event-ID": str(ids[0])})

    assert [message["id"] for message in parse_stream(response.text)] == ids[1:]


@pytest.mark.asyncio
async def test_events_endpoint_from_now(clean_tables):

    await add_events("completed")

    response = client.get("/events")

    assert parse_stream(response.text) == []


@pytest.mark.asyncio
async def test_events_endpoint_ignores_a_bad_last_event_id(clean_tables):

    await add_events("completed")

    # Not a number: treated as a new connection (from now).
    response = client.get("/events", headers={"Last-Event-ID": "abc"})

    assert response.status_code == 200
    assert parse_stream(response.text) == []


def test_events_endpoint_allows_the_frontend_origin():

    response = client.get(
        "/events",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )


# -------------------------
# GET /jobs
# -------------------------


@pytest_asyncio.fixture
async def jobs():
    """Three tasks' jobs: completed, failed after a retry, running."""

    created = []

    async with TestSessionLocal() as db:

        for number, (status, attempts) in enumerate(
            [
                (JobStatus.COMPLETED, 1),
                (JobStatus.FAILED, 2),
                (JobStatus.RUNNING, 1),
            ]
        ):
            task = await create_task(db)
            task.question = f"Question {number}"

            job = ResearchJob(
                task_id=task.id,
                status=status,
                attempts=attempts,
                error="Tavily timed out" if status == JobStatus.FAILED else None,
                created_at=utc_now() - timedelta(minutes=10 - number),
            )
            db.add(job)
            await db.flush()
            created.append(job.id)

        await db.commit()

    return created


@pytest.mark.asyncio
async def test_list_jobs_newest_first(clean_tables, jobs):

    body = client.get("/jobs").json()

    assert body["total"] == 3
    assert [job["id"] for job in body["items"]] == list(reversed(jobs))

    failed = body["items"][1]

    assert failed["question"] == "Question 1"
    assert failed["status"] == "failed"
    assert failed["attempts"] == 2
    assert failed["error"] == "Tavily timed out"
    assert failed["created_at"].endswith("Z")


@pytest.mark.asyncio
async def test_list_jobs_by_status_and_pages(clean_tables, jobs):

    failed = client.get("/jobs?status=failed").json()

    assert failed["total"] == 1
    assert failed["items"][0]["id"] == jobs[1]

    page = client.get("/jobs?limit=1&offset=1").json()

    assert page["total"] == 3
    assert [job["id"] for job in page["items"]] == [jobs[1]]

    assert client.get("/jobs?status=unknown").status_code == 422
    assert client.get("/jobs?limit=0").status_code == 422


# -------------------------
# The stream doesn't hold a session open
# -------------------------


@pytest.mark.real_auth
@pytest.mark.asyncio
async def test_stream_sign_in_check_closes_its_session(clean_tables, monkeypatch):
    """Regression: the sign-in check used the request's session, which
    FastAPI keeps open for the whole streaming response: each open tab held
    a pooled connection idle in a transaction (minutes), with a lock on
    users that blocked migrations. Now the check's own session is closed
    before streaming starts."""

    from contextlib import asynccontextmanager

    from app.api.dependencies import get_session_factory as dependency
    from app.services import event_stream_service
    from app.services.user_service import create_user

    open_sessions = 0
    open_when_streaming = []

    @asynccontextmanager
    async def tracking_session():
        nonlocal open_sessions
        open_sessions += 1
        try:
            async with TestSessionLocal() as session:
                yield session
        finally:
            open_sessions -= 1

    async def tracking_get_db():
        async with tracking_session() as session:
            yield session

    from app.db.database import get_db

    previous_get_db = app.dependency_overrides.get(get_db)
    app.dependency_overrides[dependency] = lambda: tracking_session
    # Request sessions count too: the old bug was one of these left open.
    app.dependency_overrides[get_db] = tracking_get_db

    real_latest = event_stream_service.latest_event_id

    async def latest_event_id(db):
        # The stream's first query: only its own session should be open.
        open_when_streaming.append(open_sessions)
        return await real_latest(db)

    monkeypatch.setattr(event_stream_service, "latest_event_id", latest_event_id)

    async with TestSessionLocal() as db:
        await create_user(db, "streamer", "correct horse battery staple")

    # Signs in: the client keeps the session cookie, as a browser does.
    client.post(
        "/auth/login",
        json={"username": "streamer", "password": "correct horse battery staple"},
    )

    try:
        response = client.get("/events")
    finally:
        app.dependency_overrides[get_db] = previous_get_db

    assert response.status_code == 200
    assert open_when_streaming == [1]
    assert open_sessions == 0
