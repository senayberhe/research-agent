from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, computed_field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import SESSION_COOKIE, get_current_user
from app.core.config import settings
from app.core.permissions import permissions_for
from app.db.database import get_db
from app.db.models import User
from app.schemas.types import UTCDateTime
from app.services.login_throttle import throttle
from app.services.session_service import create_session, revoke_session
from app.services.user_service import authenticate, normalize_username


router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=1, max_length=1024)


class UserResponse(BaseModel):
    id: int
    username: str
    # viewer, researcher, operator or admin.
    role: str
    created_at: UTCDateTime
    last_login_at: UTCDateTime | None

    model_config = {
        "from_attributes": True
    }

    @computed_field
    @property
    def permissions(self) -> list[str]:
        """What this role may do (app/core/permissions.py), for the UI."""

        return sorted(p.value for p in permissions_for(self.role))


class SessionResponse(BaseModel):
    expires_at: UTCDateTime
    user: UserResponse


def _client_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post(
    "/login",
    response_model=SessionResponse,
)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Checks a username and password and starts a session: sets the
    session cookie (httpOnly, so page scripts can't read it), which the
    browser then sends with every request."""

    username = normalize_username(body.username)
    client = _client_address(request)

    # Checked first: a locked-out guess costs no password hashing.
    wait = throttle.retry_after(username, client)

    if wait is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed sign-ins. Try again in {wait} seconds.",
            headers={"Retry-After": str(wait)},
        )

    user = await authenticate(db, username, body.password)

    if user is None:
        throttle.failed(username, client)
        # The same answer whatever was wrong: no hint which usernames exist.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )

    throttle.succeeded(username)

    token, expires_at = await create_session(db, user)

    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.auth_session_hours * 3600,
        httponly=True,
        secure=settings.auth_cookie_secure,
        # Not sent on requests other sites start (with the Origin check in
        # app/main.py: CSRF protection).
        samesite="lax",
        path="/",
    )

    return {"expires_at": expires_at, "user": user}


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def logout(
    response: Response,
    token: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    db: AsyncSession = Depends(get_db),
):
    """Ends this session (the cookie stops working, here and anywhere it
    was copied to) and clears the cookie. Fine to call when not signed
    in."""

    if token:
        await revoke_session(db, token)

    response.delete_cookie(
        SESSION_COOKIE,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )


@router.get(
    "/me",
    response_model=UserResponse,
)
async def me(user: User = Depends(get_current_user)):
    """The signed-in user (and how the frontend checks for a session)."""

    return user
