import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from sqlalchemy.pool import NullPool

from app.db.models import Base


# Tests use their own database, never the dev database (research_db).
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres"
    "@localhost:5433/research_test",
)

# Safety check: setup_database drops every table, so refuse to run against
# anything that isn't clearly a test database.
if not make_url(TEST_DATABASE_URL).database.endswith("_test"):
    raise RuntimeError(
        f"TEST_DATABASE_URL must point at a *_test database, "
        f"got {make_url(TEST_DATABASE_URL).database!r}"
    )


# NullPool: every session opens its own connection. Pooled asyncpg
# connections are tied to the event loop that created them, and TestClient
# and pytest-asyncio each run tests in different loops.
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    poolclass=NullPool,
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest_asyncio.fixture(
    scope="session",
    loop_scope="session",
    autouse=True,
)
async def setup_database():
    # Fail fast with a clear message instead of a long asyncpg traceback
    # when the Postgres container isn't running.
    try:
        async with test_engine.connect():
            pass
    except OSError:
        pytest.exit(
            "Can't reach the test database at "
            f"{make_url(TEST_DATABASE_URL).render_as_string(hide_password=True)}. "
            "Is Postgres running? Start it with: docker compose up -d postgres",
            returncode=1,
        )

    # Fresh tables for the whole test run, dropped at the end.
    async with test_engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.drop_all
        )
        await connection.run_sync(
            Base.metadata.create_all
        )

    yield

    async with test_engine.begin() as connection:
        await connection.run_sync(
            Base.metadata.drop_all
        )


@pytest_asyncio.fixture
async def db_session():
    # Run the whole test inside one outer transaction. With
    # "create_savepoint", the code under test can still call commit()
    # (it only releases a savepoint), and everything is rolled back at
    # the end, so each test starts with empty tables.
    async with test_engine.connect() as connection:

        transaction = await connection.begin()

        session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )

        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()


@pytest_asyncio.fixture
async def clean_tables():
    # For tests that commit through their own sessions (e.g. TestClient
    # requests), which a rollback can't undo: empty every table afterwards.
    yield

    table_names = ", ".join(
        table.name
        for table in Base.metadata.sorted_tables
    )

    async with test_engine.begin() as connection:
        await connection.execute(
            text(f"TRUNCATE {table_names} RESTART IDENTITY CASCADE")
        )


@pytest.fixture(autouse=True)
def no_agent_retry_backoff(monkeypatch):
    # The agent waits 1s, then 2s between tool retries; skip the waits in
    # tests so failing-tool tests don't sleep.
    from app.agents.research_agent import ResearchAgent

    monkeypatch.setattr(ResearchAgent, "RETRY_BACKOFF_SECONDS", 0)


@pytest.fixture(autouse=True)
def signed_in(request):
    """API tests run as a signed-in user, unless marked real_auth (those
    exercise the real session checks: app/tests/test_auth.py)."""

    from app.api.dependencies import get_current_user, get_streaming_user
    from app.db.models import User
    from app.main import app

    if request.node.get_closest_marker("real_auth"):
        yield
        return

    user = User(id=1, username="test-user", is_active=True, role="admin")
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_streaming_user] = lambda: user

    yield

    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_streaming_user, None)


@pytest.fixture(autouse=True)
def fresh_sign_in_state():
    """Each test starts signed out (the shared TestClient keeps cookies)
    and with no failed sign-ins counted."""

    from app.services.login_throttle import throttle
    from app.tests.test_research_api import client

    client.cookies.clear()
    throttle.reset()

    yield

    client.cookies.clear()
    throttle.reset()
