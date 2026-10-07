"""Authentication: passwords, sessions, login, the protected API, and the
account CLI. Marked real_auth: no signed-in test user, real session checks."""

import asyncio
import io
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update

from app.api.dependencies import SESSION_COOKIE
from app.core.config import settings
from app.core.security import hash_password, hash_session_token, verify_password
from app.db.models import User, UserSession
from app.main import app
from app.services.user_service import UserError, create_user
from app.tests.conftest import TestSessionLocal
from app.tests.test_research_api import client


pytestmark = pytest.mark.real_auth

PASSWORD = "correct horse battery staple"


async def add_user(username="alice", password=PASSWORD, active=True, role="viewer") -> User:
    async with TestSessionLocal() as db:
        user = await create_user(db, username, password, role)
        if not active:
            user.is_active = False
            await db.commit()
        return user


def login(username="alice", password=PASSWORD):
    return client.post("/auth/login", json={"username": username, "password": password})


def sign_in(username="alice", password=PASSWORD) -> str:
    """Signs in and returns the session id, leaving the client's cookie jar
    empty (requests then say which session they use: cookie())."""

    response = login(username, password)
    assert response.status_code == 200, response.text
    token = response.cookies[SESSION_COOKIE]
    client.cookies.clear()
    return token


def cookie(token: str) -> dict:
    return {"Cookie": f"{SESSION_COOKIE}={token}"}


# -------------------------
# Passwords and session ids
# -------------------------


def test_password_hashing():

    hashed = hash_password(PASSWORD)

    assert hashed.startswith("$argon2id$")
    assert PASSWORD not in hashed
    # Salted: the same password hashes differently each time.
    assert hashed != hash_password(PASSWORD)

    assert verify_password(PASSWORD, hashed) is True
    assert verify_password("wrong password!!", hashed) is False
    # No such user: still False (and the work is still done).
    assert verify_password(PASSWORD, None) is False


# -------------------------
# Login
# -------------------------


@pytest.mark.asyncio
async def test_login(clean_tables):

    await add_user()

    response = login()

    assert response.status_code == 200

    body = response.json()

    assert body["user"]["username"] == "alice"
    assert body["expires_at"].endswith("Z")
    # The session id is only in the cookie, never in the body.
    assert "access_token" not in body

    set_cookie = response.headers["set-cookie"]
    assert set_cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert f"Max-Age={settings.auth_session_hours * 3600}" in set_cookie

    token = response.cookies[SESSION_COOKIE]

    # The sign-in is recorded, and only the session id's hash is stored.
    async with TestSessionLocal() as db:
        user = await db.scalar(select(User).where(User.username == "alice"))
        session = await db.scalar(select(UserSession))
    assert user.last_login_at is not None
    assert session.user_id == user.id
    assert session.token_hash == hash_session_token(token)
    assert token not in session.token_hash


@pytest.mark.asyncio
async def test_login_username_is_case_insensitive(clean_tables):

    await add_user("Alice")

    assert login("  ALICE ").status_code == 200


@pytest.mark.asyncio
async def test_wrong_password_and_unknown_user_get_the_same_answer(clean_tables):

    await add_user()

    wrong_password = login(password="not the password")
    unknown_user = login(username="mallory")

    for response in (wrong_password, unknown_user):
        assert response.status_code == 401
        assert response.json() == {"detail": "Incorrect username or password."}
        assert "set-cookie" not in response.headers


@pytest.mark.asyncio
async def test_deactivated_user_cant_sign_in(clean_tables):

    await add_user(active=False)

    assert login().status_code == 401


def test_login_validates_its_body():

    assert client.post("/auth/login", json={"username": "alice"}).status_code == 422
    assert client.post("/auth/login", json={"username": "", "password": "x"}).status_code == 422


@pytest.mark.asyncio
async def test_password_is_never_stored(clean_tables):

    await add_user()

    async with TestSessionLocal() as db:
        user = await db.scalar(select(User))

    assert PASSWORD not in user.password_hash


# -------------------------
# The current user and sessions
# -------------------------


@pytest.mark.asyncio
async def test_me_with_the_session_cookie(clean_tables):

    await add_user()
    login()

    # The client sends back the cookie it was given, as a browser does.
    response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.json()["username"] == "alice"
    assert "password_hash" not in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Cookie": f"{SESSION_COOKIE}=not-a-session"},
        {"Cookie": f"{SESSION_COOKIE}="},
        {"Cookie": "some_other_cookie=abc"},
        # The old way in: no longer accepted.
        {"Authorization": "Bearer something"},
    ],
)
async def test_me_without_a_current_session(clean_tables, headers):

    response = client.get("/auth/me", headers=headers)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_expired_session_is_refused(clean_tables):

    await add_user()
    token = sign_in()

    async with TestSessionLocal() as db:
        await db.execute(
            update(UserSession).values(expires_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1))
        )
        await db.commit()

    response = client.get("/auth/me", headers=cookie(token))

    assert response.status_code == 401
    assert "ended" in response.json()["detail"]


@pytest.mark.asyncio
async def test_logout_ends_the_session(clean_tables):

    await add_user()
    token = sign_in()
    other = sign_in()

    response = client.post("/auth/logout", headers=cookie(token))

    assert response.status_code == 204
    # The browser is told to drop the cookie.
    assert f'{SESSION_COOKIE}=""' in response.headers["set-cookie"]
    assert "Max-Age=0" in response.headers["set-cookie"]

    # A copy of the cookie stops working too; other sessions don't.
    assert client.get("/auth/me", headers=cookie(token)).status_code == 401
    assert client.get("/auth/me", headers=cookie(other)).status_code == 200


def test_logout_when_not_signed_in():

    assert client.post("/auth/logout").status_code == 204


@pytest.mark.asyncio
async def test_deactivating_a_user_ends_their_sessions(clean_tables):

    from app.services.user_service import set_active

    await add_user()
    token = sign_in()

    assert client.get("/auth/me", headers=cookie(token)).status_code == 200

    async with TestSessionLocal() as db:
        await set_active(db, "alice", False)

    # Reactivated: the old session stays ended.
    async with TestSessionLocal() as db:
        await set_active(db, "alice", True)

    assert client.get("/auth/me", headers=cookie(token)).status_code == 401


@pytest.mark.asyncio
async def test_a_new_password_ends_existing_sessions(clean_tables):

    from app.services.user_service import set_password

    await add_user()
    token = sign_in()

    async with TestSessionLocal() as db:
        await set_password(db, "alice", "a brand new password")

    assert client.get("/auth/me", headers=cookie(token)).status_code == 401


@pytest.mark.asyncio
async def test_long_expired_sessions_are_cleared_at_sign_in(clean_tables):

    await add_user()
    sign_in()

    async with TestSessionLocal() as db:
        await db.execute(
            update(UserSession).values(expires_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2))
        )
        await db.commit()

    sign_in()

    async with TestSessionLocal() as db:
        sessions = (await db.scalars(select(UserSession))).all()

    assert len(sessions) == 1


# -------------------------
# Sign-in throttling
# -------------------------


@pytest.mark.asyncio
async def test_repeated_failed_sign_ins_are_throttled(clean_tables, monkeypatch):

    monkeypatch.setattr(settings, "auth_login_max_failures", 3)

    await add_user()

    for _ in range(3):
        assert login(password="not the password").status_code == 401

    # Locked out: even the right password waits (and isn't checked).
    response = login()

    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0

    # Other usernames aren't affected.
    await add_user("bob")
    assert login("bob").status_code == 200


@pytest.mark.asyncio
async def test_a_successful_sign_in_resets_the_count(clean_tables, monkeypatch):

    monkeypatch.setattr(settings, "auth_login_max_failures", 3)

    await add_user()

    for _ in range(2):
        login(password="not the password")
    assert login().status_code == 200

    for _ in range(2):
        login(password="not the password")
    assert login().status_code == 200


def test_throttle_window_passes():

    from app.services.login_throttle import LoginThrottle

    now = [1000.0]
    throttle = LoginThrottle(clock=lambda: now[0])

    for _ in range(settings.auth_login_max_failures):
        throttle.failed("alice", "1.2.3.4")

    assert throttle.retry_after("alice", "1.2.3.4") is not None

    now[0] += settings.auth_login_window_seconds + 1

    assert throttle.retry_after("alice", "1.2.3.4") is None
    # Nothing kept once the window has passed.
    assert throttle.failures == {}


def test_throttle_limits_one_address_across_usernames(monkeypatch):

    from app.services.login_throttle import LoginThrottle

    monkeypatch.setattr(settings, "auth_login_max_failures_per_ip", 4)
    throttle = LoginThrottle()

    for name in ("a", "b", "c", "d"):
        throttle.failed(name, "1.2.3.4")

    assert throttle.retry_after("e", "1.2.3.4") is not None
    assert throttle.retry_after("e", "5.6.7.8") is None


# -------------------------
# Cross-site requests (CSRF)
# -------------------------


@pytest.mark.asyncio
async def test_changes_from_another_origin_are_refused(clean_tables):

    await add_user(role="researcher")
    token = sign_in()

    response = client.post(
        "/research",
        json={"question": "What is RAG?"},
        headers={**cookie(token), "Origin": "https://evil.example"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Cross-origin request refused."}

    # The frontend's own origin is fine.
    response = client.post(
        "/research",
        json={"question": "What is RAG?"},
        headers={**cookie(token), "Origin": "http://localhost:5173"},
    )

    assert response.status_code == 202


def test_sign_in_from_another_origin_is_refused():

    response = client.post(
        "/auth/login",
        json={"username": "alice", "password": PASSWORD},
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 403


def test_cors_allows_credentials_for_the_frontend():

    response = client.options(
        "/research",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PATCH",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-credentials"] == "true"


# -------------------------
# Protected endpoints
# -------------------------


# The only endpoints anyone may call without signing in.
PUBLIC = {
    ("GET", "/"),
    ("GET", "/health"),  # container health checks
    ("GET", "/metrics"),  # Prometheus scraping
    ("POST", "/auth/login"),
    ("POST", "/auth/logout"),  # ends a session if there is one
}


def every_route():
    """Every endpoint and method, from the app's OpenAPI schema (generated
    from the same routes the app serves)."""

    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            yield method.upper(), path


def test_every_endpoint_but_the_public_ones_needs_a_session():
    """Walks every route the app has, so a new endpoint added without
    protection fails here."""

    checked = []

    for method, path in every_route():

        if (method, path) in PUBLIC:
            continue

        url = path.replace("{task_id}", "1").replace("{job_id}", "1")
        response = client.request(method, url, json={"question": "What is RAG?"})

        assert response.status_code == 401, f"{method} {path} -> {response.status_code}"
        checked.append(f"{method} {path}")

    # Sanity: the protected surface really was walked.
    assert len(checked) >= 25
    assert "GET /events" in checked
    assert "POST /research" in checked
    assert "GET /jobs" in checked


def test_public_endpoints_work_without_a_session():

    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/").status_code == 200


@pytest.mark.asyncio
async def test_protected_endpoints_work_with_a_session(clean_tables):

    # An operator may also start research and see analytics (other roles:
    # test_rbac.py).
    await add_user(role="operator")
    login()

    for path in ("/research", "/jobs", "/research/slo", "/research/metrics"):
        assert client.get(path).status_code == 200, path

    created = client.post("/research", json={"question": "What is RAG?"})

    assert created.status_code == 202


def test_401_responses_carry_cors_headers():
    """So the frontend can read a 401 and send the user to sign in."""

    response = client.get("/research", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-credentials"] == "true"


# -------------------------
# Accounts: the CLI and the initial admin
# -------------------------


@pytest.fixture
def cli(monkeypatch):
    """Runs app.cli against the test database, with the password piped."""

    import app.cli

    monkeypatch.setattr(app.cli, "AsyncSessionLocal", TestSessionLocal)

    async def run(*argv, password=PASSWORD):
        monkeypatch.setattr("sys.stdin", io.StringIO(password + "\n"))
        # The CLI runs its own event loop (asyncio.run), as from a shell:
        # give it a thread of its own.
        return await asyncio.to_thread(app.cli.main, list(argv))

    return run


@pytest.mark.asyncio
async def test_cli_manages_accounts(clean_tables, cli, capsys):

    assert await cli("create-user", "Bob") == 0
    assert "Created user 'bob'" in capsys.readouterr().out

    assert login("bob").status_code == 200

    # Duplicate, and a too-short password.
    assert await cli("create-user", "bob") == 1
    assert "already exists" in capsys.readouterr().err
    assert await cli("create-user", "carol", password="short") == 1
    assert "at least 12" in capsys.readouterr().err

    assert await cli("set-password", "bob", password="a brand new password") == 0
    assert login("bob").status_code == 401
    assert login("bob", "a brand new password").status_code == 200

    assert await cli("deactivate", "bob") == 0
    assert login("bob", "a brand new password").status_code == 401

    assert await cli("list-users") == 0
    listed = capsys.readouterr().out
    assert "bob" in listed and "deactivated" in listed

    assert await cli("activate", "bob") == 0
    assert login("bob", "a brand new password").status_code == 200

    assert await cli("set-password", "nobody") == 1


@pytest.mark.asyncio
async def test_initial_admin_is_created_only_when_there_are_no_users(
    clean_tables,
    monkeypatch,
):
    from pydantic import SecretStr

    import app.db.database
    from app.main import create_initial_admin

    monkeypatch.setattr(app.db.database, "AsyncSessionLocal", TestSessionLocal)
    monkeypatch.setattr(settings, "auth_admin_username", "admin")
    monkeypatch.setattr(settings, "auth_admin_password", SecretStr("an admin password"))

    await create_initial_admin()

    assert login("admin", "an admin password").status_code == 200

    # Already has users: a changed env password doesn't overwrite anything.
    monkeypatch.setattr(settings, "auth_admin_password", SecretStr("a different password"))
    await create_initial_admin()

    assert login("admin", "an admin password").status_code == 200
    assert login("admin", "a different password").status_code == 401


@pytest.mark.asyncio
async def test_user_service_rejects_bad_input(clean_tables):

    async with TestSessionLocal() as db:
        with pytest.raises(UserError):
            await create_user(db, "   ", PASSWORD)
        with pytest.raises(UserError):
            await create_user(db, "dave", "short")
