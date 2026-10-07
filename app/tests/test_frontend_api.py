"""What a browser frontend needs from the API: CORS for its origin, and a
paginated list of research tasks."""

from datetime import timedelta

import pytest

from app.db.models import ResearchTask, TaskStatus, utc_now
from app.tests.conftest import TestSessionLocal
from app.tests.test_research_api import client


# -------------------------
# CORS
# -------------------------


def test_cors_allows_the_frontend_origin():

    response = client.options(
        "/research",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )


def test_cors_on_a_normal_request():

    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )


def test_cors_refuses_other_origins():

    response = client.get(
        "/health",
        headers={"Origin": "https://evil.example.com"},
    )

    assert "access-control-allow-origin" not in response.headers


def test_cors_origins_are_configurable(monkeypatch):

    from app.core.config import Settings

    monkeypatch.setenv(
        "CORS_ORIGINS",
        '["https://app.example.com", "http://localhost:3001"]',
    )

    assert Settings().cors_origins == [
        "https://app.example.com",
        "http://localhost:3001",
    ]


# -------------------------
# GET /research
# -------------------------


async def add_tasks(statuses: list[TaskStatus]) -> list[int]:
    """Tasks created a minute apart, oldest first; returns their ids."""

    start = utc_now() - timedelta(hours=1)

    async with TestSessionLocal() as db:

        tasks = [
            ResearchTask(
                question=f"Question {number}",
                status=status,
                created_at=start + timedelta(minutes=number),
            )
            for number, status in enumerate(statuses)
        ]

        db.add_all(tasks)
        await db.commit()

        return [task.id for task in tasks]


@pytest.mark.asyncio
async def test_list_tasks_newest_first(clean_tables):

    ids = await add_tasks([TaskStatus.COMPLETED] * 3)

    body = client.get("/research").json()

    assert body["total"] == 3
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert [task["id"] for task in body["items"]] == list(reversed(ids))

    assert set(body["items"][0]) == {
        "id",
        "question",
        "status",
        "summary",
        "created_at",
        "created_by",
    }


@pytest.mark.asyncio
async def test_list_tasks_pages(clean_tables):

    ids = await add_tasks([TaskStatus.COMPLETED] * 5)
    newest_first = list(reversed(ids))

    first = client.get("/research?limit=2").json()
    second = client.get("/research?limit=2&offset=2").json()
    last = client.get("/research?limit=2&offset=4").json()

    assert [task["id"] for task in first["items"]] == newest_first[:2]
    assert [task["id"] for task in second["items"]] == newest_first[2:4]
    assert [task["id"] for task in last["items"]] == newest_first[4:]

    # The total is for all pages.
    assert first["total"] == second["total"] == last["total"] == 5


@pytest.mark.asyncio
async def test_list_tasks_by_status(clean_tables):

    await add_tasks(
        [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.COMPLETED]
    )

    body = client.get("/research?status=completed").json()

    assert body["total"] == 2
    assert {task["status"] for task in body["items"]} == {"completed"}


@pytest.mark.asyncio
async def test_list_tasks_empty(clean_tables):

    assert client.get("/research").json() == {
        "items": [],
        "total": 0,
        "limit": 20,
        "offset": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=101", "offset=-1", "status=unknown"],
)
async def test_list_tasks_rejects_bad_parameters(clean_tables, query):

    assert client.get(f"/research?{query}").status_code == 422


# -------------------------
# Timestamps are explicit UTC
# -------------------------


@pytest.mark.asyncio
async def test_api_timestamps_are_utc(clean_tables):
    """Every timestamp ends in Z, so a browser's new Date() doesn't read it
    as local time."""

    import re

    from app.services.research_service import create_research_task

    async with TestSessionLocal() as db:
        task = await create_research_task(db=db, question="What is RAG?")

    timestamp = re.compile(r'"\d{4}-\d{2}-\d{2}T[\d:.]+(Z|[+-]\d{2}:\d{2})?"')

    for path in (
        "/research",
        f"/research/{task.id}",
        f"/research/{task.id}/jobs",
        f"/research/{task.id}/timeline",
        "/research/slo",
    ):
        text = client.get(path).text

        found = timestamp.findall(text)
        assert found, path

        for match in timestamp.finditer(text):
            assert match.group(0).endswith('Z"'), (path, match.group(0))


def test_utc_type_keeps_aware_times_and_converts_others():

    from datetime import UTC, datetime, timedelta, timezone

    from app.schemas.types import _to_utc_iso

    assert _to_utc_iso(datetime(2026, 10, 7, 12, 0)) == "2026-10-07T12:00:00Z"
    assert _to_utc_iso(datetime(2026, 10, 7, 12, 0, tzinfo=UTC)) == (
        "2026-10-07T12:00:00Z"
    )
    # Another offset is converted to UTC.
    plus_two = timezone(timedelta(hours=2))
    assert _to_utc_iso(datetime(2026, 10, 7, 14, 0, tzinfo=plus_two)) == (
        "2026-10-07T12:00:00Z"
    )


def test_cors_allows_patch_for_user_management():
    """The Users page changes roles with PATCH: the browser's preflight
    must allow it (a missing method fails only in a real browser)."""

    response = client.options(
        "/users/1",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )

    assert response.status_code == 200
    assert "PATCH" in response.headers["access-control-allow-methods"]

