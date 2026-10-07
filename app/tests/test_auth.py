"""Authentication: passwords, tokens, login, the protected API, and the
account CLI. Marked real_auth: no signed-in test user, real token checks."""

import asyncio
import io
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from sqlalchemy import select

from app.core.config import settings
from app.core.security import (
    ALGORITHM,
    InvalidToken,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.db.models import User
from app.main import app
from app.services.user_service import UserError, create_user
from app.tests.conftest import TestSessionLocal
from app.tests.test_research_api import client


pytestmark = pytest.mark.real_auth

PASSWORD = "correct horse battery staple"


async def add_user(username="alice", password=PASSWORD, active=True) -> User:
    async with TestSessionLocal() as db:
        user = await create_user(db, username, password)
        if not active:
            user.is_active = False
            await db.commit()
        return user


def login(username="alice", password=PASSWORD):
    return client.post("/auth/login", json={"username": username, "password": password})


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# -------------------------
# Passwords and tokens
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


def test_token_round_trip():

    token, expires = create_access_token(42)

    assert decode_access_token(token) == 42
    assert expires > datetime.now(UTC) + timedelta(minutes=settings.auth_access_token_minutes - 1)


def test_expired_token_is_rejected():

    long_ago = datetime.now(UTC) - timedelta(days=2)
    token, _ = create_access_token(42, now=long_ago)

    with pytest.raises(InvalidToken):
        decode_access_token(token)


def test_tampered_token_is_rejected():

    token, _ = create_access_token(42)
    header, payload, signature = token.split(".")

    # Someone edits the payload to be user 1: the signature no longer
    # matches.
    forged_payload = jwt.utils.base64url_encode(b'{"sub":"1","exp":9999999999}').decode()

    with pytest.raises(InvalidToken):
        decode_access_token(f"{header}.{forged_payload}.{signature}")


def test_token_signed_with_another_key_is_rejected():

    token = jwt.encode(
        {"sub": "42", "exp": 9999999999},
        "some-other-secret-key-that-is-long-enough",
        algorithm=ALGORITHM,
    )

    with pytest.raises(InvalidToken):
        decode_access_token(token)


def test_unsigned_token_is_rejected():

    # alg "none": no signature at all.
    token = jwt.encode({"sub": "42", "exp": 9999999999}, None, algorithm="none")

    with pytest.raises(InvalidToken):
        decode_access_token(token)


def test_token_without_expiry_is_rejected():

    token = jwt.encode(
        {"sub": "42"},
        settings.auth_secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )

    with pytest.raises(InvalidToken):
        decode_access_token(token)


# -------------------------
# Login
# -------------------------


@pytest.mark.asyncio
async def test_login(clean_tables):

    await add_user()

    response = login()

    assert response.status_code == 200

    body = response.json()

    assert body["token_type"] == "bearer"
    assert body["user"]["username"] == "alice"
    assert body["expires_at"].endswith("Z")
    assert decode_access_token(body["access_token"]) == body["user"]["id"]

    # The sign-in is recorded.
    async with TestSessionLocal() as db:
        user = await db.scalar(select(User).where(User.username == "alice"))
    assert user.last_login_at is not None


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
        assert response.headers["www-authenticate"] == "Bearer"


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
# The current user
# -------------------------


@pytest.mark.asyncio
async def test_me_with_a_valid_token(clean_tables):

    await add_user()
    token = login().json()["access_token"]

    response = client.get("/auth/me", headers=bearer(token))

    assert response.status_code == 200
    assert response.json()["username"] == "alice"
    assert "password_hash" not in response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer not-a-token"},
        {"Authorization": "Basic YWxpY2U6cGFzc3dvcmQ="},
        {"Authorization": "Bearer"},
    ],
)
async def test_me_without_a_valid_token(clean_tables, headers):

    response = client.get("/auth/me", headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_expired_token_is_refused(clean_tables):

    user = await add_user()
    token, _ = create_access_token(user.id, now=datetime.now(UTC) - timedelta(days=2))

    response = client.get("/auth/me", headers=bearer(token))

    assert response.status_code == 401
    assert "expired" in response.json()["detail"]


@pytest.mark.asyncio
async def test_deactivating_a_user_ends_their_sessions(clean_tables):

    from app.services.user_service import set_active

    await add_user()
    token = login().json()["access_token"]

    assert client.get("/auth/me", headers=bearer(token)).status_code == 200

    async with TestSessionLocal() as db:
        await set_active(db, "alice", False)

    # The same, still-unexpired token no longer works.
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


@pytest.mark.asyncio
async def test_token_for_a_deleted_user_is_refused(clean_tables):

    token, _ = create_access_token(999_999)

    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


# -------------------------
# Protected endpoints
# -------------------------


# The only endpoints anyone may call without signing in.
PUBLIC = {
    ("GET", "/"),
    ("GET", "/health"),  # container health checks
    ("GET", "/metrics"),  # Prometheus scraping
    ("POST", "/auth/login"),
}


def every_route():
    """Every endpoint and method, from the app's OpenAPI schema (generated
    from the same routes the app serves)."""

    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            yield method.upper(), path


def test_every_endpoint_but_the_public_ones_needs_a_token():
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


def test_public_endpoints_work_without_a_token():

    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/").status_code == 200


@pytest.mark.asyncio
async def test_protected_endpoints_work_with_a_token(clean_tables):

    await add_user()
    token = login().json()["access_token"]

    for path in ("/research", "/jobs", "/research/slo", "/research/metrics"):
        assert client.get(path, headers=bearer(token)).status_code == 200, path

    created = client.post("/research", json={"question": "What is RAG?"}, headers=bearer(token))

    assert created.status_code == 202


def test_401_responses_carry_cors_headers():
    """So the frontend can read a 401 and send the user to sign in."""

    response = client.get("/research", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


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
