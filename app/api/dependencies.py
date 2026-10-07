from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyCookie
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.permissions import WHAT, WHO_CAN, Permission, has_permission
from app.db.database import AsyncSessionLocal, get_db
from app.db.models import User
from app.services.session_service import user_for_session


SESSION_COOKIE = "research_agent_session"

# The session id, from the httpOnly cookie POST /auth/login sets.
# auto_error=False: our own 401 (not FastAPI's 403) when it's missing.
_session_cookie = APIKeyCookie(name=SESSION_COOKIE, auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )


async def user_from_session_cookie(db: AsyncSession, token: str | None) -> User:
    """The user a session cookie belongs to; 401 if there isn't a current
    one (missing, expired, signed out, or the account was deactivated)."""

    if not token:
        raise _unauthorized("Not signed in.")

    user = await user_for_session(db, token)

    if user is None:
        raise _unauthorized("Your session has ended. Sign in again.")

    return user


async def get_current_user(
    token: str | None = Depends(_session_cookie),
    db: AsyncSession = Depends(get_db),
) -> User:
    """The signed-in user, from the session cookie. 401 if there's no
    current session."""

    return await user_from_session_cookie(db, token)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """For code that opens its own short sessions instead of the request's
    (the event stream). A dependency, so tests can point it elsewhere."""

    return AsyncSessionLocal


async def get_streaming_user(
    token: str | None = Depends(_session_cookie),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
) -> User:
    """get_current_user for a long-lived response (GET /events): checks the
    session cookie in a database session of its own, closed before the response starts.

    With get_db, the request's session would stay open, mid-transaction,
    for the whole stream (minutes): one pooled connection per open tab,
    holding a lock on users that blocks schema changes. The stream does its
    own short queries instead.
    """

    async with session_factory() as db:
        user = await user_from_session_cookie(db, token)
        # Keep the object usable after the session closes.
        db.expunge(user)

    return user


def require(permission: Permission):
    """A dependency: the signed-in user, if their role allows `permission`;
    otherwise 403 (signed in, but not allowed), saying who is."""

    async def check(user: User = Depends(get_current_user)) -> User:

        if not has_permission(user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Your role ({user.role}) can't {WHAT[permission]}. "
                    f"{WHO_CAN[permission]} can."
                ),
            )

        return user

    return check
