from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import InvalidToken, decode_access_token
from app.db.database import get_db
from app.db.models import User


# auto_error=False: our own 401 (FastAPI's default for a missing header is
# 403, which is the wrong code for "not signed in").
_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """The signed-in user, from "Authorization: Bearer <token>". 401 if the
    token is missing, invalid or expired, or its user no longer exists or
    has been deactivated."""

    if credentials is None:
        raise _unauthorized("Not signed in.")

    try:
        user_id = decode_access_token(credentials.credentials)
    except InvalidToken:
        raise _unauthorized("Your session is invalid or has expired. Sign in again.")

    user = await db.get(User, user_id)

    if user is None or not user.is_active:
        raise _unauthorized("Your account is no longer active.")

    return user
