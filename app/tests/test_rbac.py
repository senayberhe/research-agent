"""Role-based access control: viewers read research and jobs, researchers
also start and resume research, operators also see analytics (system
metrics, SLOs, workers), admins also manage users. Real sessions."""

import asyncio
import io

import pytest

from app.api.dependencies import SESSION_COOKIE
from app.core.permissions import Permission, has_permission, permissions_for
from app.db.models import UserRole
from app.main import app
from app.services.user_service import UserError, create_user, set_role
from app.tests.conftest import TestSessionLocal
from app.tests.test_research_api import client


pytestmark = pytest.mark.real_auth

PASSWORD = "correct horse battery staple"


async def add_user(username: str, role: str, active: bool = True) -> int:
    async with TestSessionLocal() as db:
        user = await create_user(db, username, PASSWORD, role)
        if not active:
            user.is_active = False
            await db.commit()
        return user.id


def token_for(username: str) -> dict:
    """Signs in; the headers that send that user's session cookie (the
    client's own cookie jar is left empty, so several users can be signed
    in at once)."""

    response = client.post("/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    token = response.cookies[SESSION_COOKIE]
    client.cookies.clear()
    return {"Cookie": f"{SESSION_COOKIE}={token}"}


# -------------------------
# The role -> permission table
# -------------------------


def test_role_permissions():

    assert permissions_for("viewer") == {Permission.VIEW}
    assert permissions_for("researcher") == {
        Permission.VIEW,
        Permission.RESEARCH_CREATE,
        Permission.RESEARCH_RESUME,
    }
    assert permissions_for("operator") == {
        Permission.VIEW,
        Permission.RESEARCH_CREATE,
        Permission.RESEARCH_RESUME,
        Permission.ANALYTICS_VIEW,
    }
    assert permissions_for("admin") == set(Permission)

    # Each role has everything the one below it has.
    assert (
        permissions_for("viewer")
        < permissions_for("researcher")
        < permissions_for("operator")
        < permissions_for("admin")
    )

    assert not has_permission("viewer", Permission.RESEARCH_CREATE)
    assert not has_permission("researcher", Permission.ANALYTICS_VIEW)
    assert not has_permission("operator", Permission.USERS_MANAGE)


# -------------------------
# Every endpoint, every role
# -------------------------


# Which endpoints need more than VIEW. Everything else any signed-in user
# may call.
NEEDS = {
    ("POST", "/research"): Permission.RESEARCH_CREATE,
    ("POST", "/research/{task_id}/resume"): Permission.RESEARCH_RESUME,
    # Analytics: system-wide metrics, SLOs, workers, queue health.
    ("GET", "/research/metrics"): Permission.ANALYTICS_VIEW,
    ("GET", "/research/metrics/system"): Permission.ANALYTICS_VIEW,
    ("GET", "/research/metrics/timeseries"): Permission.ANALYTICS_VIEW,
    ("GET", "/research/slo"): Permission.ANALYTICS_VIEW,
    ("GET", "/research/slo/error-budget"): Permission.ANALYTICS_VIEW,
    ("GET", "/workers"): Permission.ANALYTICS_VIEW,
    ("GET", "/health/jobs"): Permission.ANALYTICS_VIEW,
    ("GET", "/users"): Permission.USERS_MANAGE,
    ("POST", "/users"): Permission.USERS_MANAGE,
    ("PATCH", "/users/{user_id}"): Permission.USERS_MANAGE,
}

# Not role-checked. (Signing out would also end the session the walk uses.)
PUBLIC = {
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("POST", "/auth/login"),
    ("POST", "/auth/logout"),
}


def every_endpoint():
    for path, operations in app.openapi()["paths"].items():
        for method in operations:
            if (method.upper(), path) not in PUBLIC:
                yield method.upper(), path


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["viewer", "researcher", "operator", "admin"])
async def test_every_endpoint_for_every_role(clean_tables, monkeypatch, role):
    """403 exactly where the role lacks the permission; never elsewhere.
    A new endpoint that needs a permission but doesn't check it fails here
    (or must be added to NEEDS)."""

    from app.api.events import get_session_factory
    from app.core.config import settings

    # GET /events streams: keep it short, on the test database.
    app.dependency_overrides[get_session_factory] = lambda: TestSessionLocal
    monkeypatch.setattr(settings, "event_stream_max_seconds", 0.2)

    try:
        await add_user("someone", role)
        headers = token_for("someone")

        for method, path in every_endpoint():
            url = path.replace("{task_id}", "999").replace("{job_id}", "999").replace("{user_id}", "999")
            response = client.request(
                method,
                url,
                headers=headers,
                json={"question": "What is RAG?", "username": "new-user", "password": PASSWORD},
            )

            allowed = has_permission(role, NEEDS.get((method, path), Permission.VIEW))

            if allowed:
                assert response.status_code not in (401, 403), (
                    f"{role}: {method} {path} -> {response.status_code}"
                )
            else:
                assert response.status_code == 403, (
                    f"{role}: {method} {path} -> {response.status_code}"
                )
    finally:
        app.dependency_overrides.pop(get_session_factory, None)


@pytest.mark.asyncio
async def test_forbidden_says_why(clean_tables):

    await add_user("val", "viewer")

    response = client.post("/research", json={"question": "What is RAG?"}, headers=token_for("val"))

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Your role (viewer) can't start research. Researchers, operators and admins can."
    )


# -------------------------
# What each role sees and does
# -------------------------


@pytest.mark.asyncio
async def test_me_includes_role_and_permissions(clean_tables):

    await add_user("rae", "researcher")

    me = client.get("/auth/me", headers=token_for("rae")).json()

    assert me["role"] == "researcher"
    assert me["permissions"] == ["research:create", "research:resume", "view"]

    login = client.post("/auth/login", json={"username": "rae", "password": PASSWORD}).json()
    assert login["user"]["role"] == "researcher"


@pytest.mark.asyncio
async def test_research_records_who_started_it(clean_tables):

    await add_user("rae", "researcher")
    headers = token_for("rae")

    created = client.post("/research", json={"question": "What is RAG?"}, headers=headers)

    assert created.status_code == 202
    assert created.json()["created_by"] == "rae"

    task_id = created.json()["id"]

    # Everyone sees all research, and who started it.
    await add_user("val", "viewer")
    viewer = token_for("val")

    listed = client.get("/research", headers=viewer).json()["items"]
    assert [(t["id"], t["created_by"]) for t in listed] == [(task_id, "rae")]

    assert client.get(f"/research/{task_id}", headers=viewer).json()["created_by"] == "rae"
    assert client.get(f"/research/{task_id}/progress", headers=viewer).json()["task"]["created_by"] == "rae"


@pytest.mark.asyncio
async def test_role_change_applies_immediately(clean_tables):

    await add_user("admin", "admin")
    user_id = await add_user("val", "viewer")
    val = token_for("val")

    assert client.post("/research", json={"question": "Q"}, headers=val).status_code == 403

    # Promoted while signed in: the same session can now start research.
    client.patch(f"/users/{user_id}", json={"role": "researcher"}, headers=token_for("admin"))

    assert client.post("/research", json={"question": "Q"}, headers=val).status_code == 202


# -------------------------
# Managing users (admins)
# -------------------------


@pytest.mark.asyncio
async def test_admin_manages_users(clean_tables):

    await add_user("admin", "admin")
    admin = token_for("admin")

    # Create: viewer unless a role is given.
    created = client.post("/users", json={"username": "Bob", "password": PASSWORD}, headers=admin)
    assert created.status_code == 201
    assert created.json()["username"] == "bob"
    assert created.json()["role"] == "viewer"
    assert created.json()["is_active"] is True
    assert "password_hash" not in created.json()

    bob = created.json()["id"]

    # Change role, deactivate, reset password.
    assert client.patch(f"/users/{bob}", json={"role": "researcher"}, headers=admin).json()["role"] == "researcher"
    assert client.patch(f"/users/{bob}", json={"is_active": False}, headers=admin).json()["is_active"] is False
    assert client.post("/auth/login", json={"username": "bob", "password": PASSWORD}).status_code == 401

    client.patch(f"/users/{bob}", json={"is_active": True, "password": "a whole new password"}, headers=admin)
    assert client.post("/auth/login", json={"username": "bob", "password": "a whole new password"}).status_code == 200

    users = client.get("/users", headers=admin).json()
    assert [(u["username"], u["role"]) for u in users] == [("admin", "admin"), ("bob", "researcher")]


@pytest.mark.asyncio
async def test_user_management_validation(clean_tables):

    await add_user("admin", "admin")
    admin = token_for("admin")

    assert client.post("/users", json={"username": "admin", "password": PASSWORD}, headers=admin).status_code == 409
    assert client.post("/users", json={"username": "x", "password": "short"}, headers=admin).status_code == 409
    assert client.post("/users", json={"username": "x", "password": PASSWORD, "role": "owner"}, headers=admin).status_code == 422
    assert client.patch("/users/999999", json={"role": "viewer"}, headers=admin).status_code == 404


@pytest.mark.asyncio
async def test_admin_cant_lock_themselves_out(clean_tables):

    admin_id = await add_user("admin", "admin")
    admin = token_for("admin")

    for change in ({"role": "viewer"}, {"is_active": False}):
        response = client.patch(f"/users/{admin_id}", json=change, headers=admin)
        assert response.status_code == 409
        assert response.json()["detail"] == "You can't change your own role or deactivate yourself."

    # Their own password is fine.
    assert client.patch(f"/users/{admin_id}", json={"password": "another long password"}, headers=admin).status_code == 200


@pytest.mark.asyncio
async def test_the_last_admin_always_stays(clean_tables):

    first = await add_user("first", "admin")
    second = await add_user("second", "admin")
    headers = token_for("first")

    # Two admins: one may demote the other...
    assert client.patch(f"/users/{second}", json={"role": "researcher"}, headers=headers).status_code == 200

    # ...but the last one can't be demoted or deactivated by any route.
    async with TestSessionLocal() as db:
        with pytest.raises(UserError, match="last active admin"):
            await set_role(db, "first", "viewer")

    from app.services.user_service import set_active

    async with TestSessionLocal() as db:
        with pytest.raises(UserError, match="last active admin"):
            await set_active(db, "first", False)

    # A deactivated admin doesn't count as one.
    third = await add_user("third", "admin", active=False)
    async with TestSessionLocal() as db:
        with pytest.raises(UserError, match="last active admin"):
            await set_role(db, "first", "viewer")

    assert first and third


# -------------------------
# The CLI
# -------------------------


@pytest.mark.asyncio
async def test_cli_roles(clean_tables, monkeypatch, capsys):

    import app.cli

    monkeypatch.setattr(app.cli, "AsyncSessionLocal", TestSessionLocal)

    async def cli(*argv):
        monkeypatch.setattr("sys.stdin", io.StringIO(PASSWORD + "\n"))
        return await asyncio.to_thread(app.cli.main, list(argv))

    assert await cli("create-user", "ann") == 0
    assert "(id 1, viewer)" in capsys.readouterr().out

    assert await cli("create-user", "ben", "--role", "admin") == 0
    assert await cli("set-role", "ann", "researcher") == 0
    assert "'ann' is now researcher" in capsys.readouterr().out

    assert await cli("set-role", "ann", "owner") == 1
    assert "Unknown role" in capsys.readouterr().err

    # ben is the only admin.
    assert await cli("set-role", "ben", "viewer") == 1
    assert "last active admin" in capsys.readouterr().err

    assert await cli("list-users") == 0
    listed = capsys.readouterr().out
    assert "ann" in listed and "researcher" in listed and "admin" in listed


def test_roles_are_the_three_documented():

    assert [role.value for role in UserRole] == ["viewer", "researcher", "operator", "admin"]


# -------------------------
# Analytics: operators and admins
# -------------------------


ANALYTICS = [
    "/research/metrics",
    "/research/metrics/system",
    "/research/metrics/timeseries",
    "/research/slo",
    "/research/slo/error-budget",
    "/workers",
    "/health/jobs",
]


@pytest.mark.asyncio
async def test_researcher_can_use_research_but_not_analytics(clean_tables):

    await add_user("rae", "researcher")
    headers = token_for("rae")

    # Research: list, start, read, its own metrics.
    assert client.get("/research", headers=headers).status_code == 200
    created = client.post("/research", json={"question": "What is RAG?"}, headers=headers)
    assert created.status_code == 202
    task_id = created.json()["id"]
    for path in (f"/research/{task_id}", f"/research/{task_id}/progress", f"/research/{task_id}/metrics", "/jobs"):
        assert client.get(path, headers=headers).status_code == 200, path

    # Analytics: refused, saying who may.
    for path in ANALYTICS:
        response = client.get(path, headers=headers)
        assert response.status_code == 403, path
        assert response.json()["detail"] == (
            "Your role (researcher) can't view analytics. Operators and admins can."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["operator", "admin"])
async def test_operators_and_admins_see_analytics_and_slos(clean_tables, role):

    await add_user("ops", role)
    headers = token_for("ops")

    for path in ANALYTICS:
        assert client.get(path, headers=headers).status_code == 200, path

    slo = client.get("/research/slo", headers=headers).json()
    assert {"research_job_success", "tool_success", "job_latency", "tool_latency"} <= set(slo)


@pytest.mark.asyncio
async def test_operator_cant_manage_users(clean_tables):

    await add_user("ops", "operator")
    headers = token_for("ops")

    assert client.get("/users", headers=headers).status_code == 403
    assert client.post("/users", json={"username": "x", "password": PASSWORD}, headers=headers).status_code == 403


@pytest.mark.asyncio
async def test_admin_can_make_someone_an_operator(clean_tables):

    await add_user("admin", "admin")
    rae = await add_user("rae", "researcher")
    rae_headers = token_for("rae")

    assert client.get("/research/slo", headers=rae_headers).status_code == 403

    response = client.patch(f"/users/{rae}", json={"role": "operator"}, headers=token_for("admin"))
    assert response.json()["role"] == "operator"

    # Same session, new role: analytics now open.
    assert client.get("/research/slo", headers=rae_headers).status_code == 200
    assert client.get("/auth/me", headers=rae_headers).json()["permissions"] == [
        "analytics:view",
        "research:create",
        "research:resume",
        "view",
    ]

