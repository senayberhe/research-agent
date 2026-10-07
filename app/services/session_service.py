"""Sign-in sessions (app/db/models.py UserSession): created at login,
looked up on every request, ended by sign out, expiry, deactivation or a
new password."""

from datetime import datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import hash_session_token, new_session_token
from app.db.models import User, UserSession, utc_now


async def create_session(db: AsyncSession, user: User) -> tuple[str, datetime]:
    """A new session for the user: (the id for the cookie, when it
    expires). Also clears out long-expired sessions."""

    now = utc_now()
    token = new_session_token()
    expires_at = now + timedelta(hours=settings.auth_session_hours)

    # Kept a day past expiry, then dropped (a cheap sweep at each sign-in).
    await db.execute(
        delete(UserSession).where(UserSession.expires_at < now - timedelta(days=1))
    )

    db.add(
        UserSession(
            user_id=user.id,
            token_hash=hash_session_token(token),
            created_at=now,
            expires_at=expires_at,
        )
    )
    await db.commit()

    return token, expires_at


async def user_for_session(db: AsyncSession, token: str) -> User | None:
    """The user a session id belongs to, if the session is current (not
    expired or revoked) and the account is active."""

    session = await db.scalar(
        select(UserSession).where(
            UserSession.token_hash == hash_session_token(token),
            UserSession.revoked_at.is_(None),
            UserSession.expires_at > utc_now(),
        )
    )

    if session is None or not session.user.is_active:
        return None

    return session.user


async def revoke_session(db: AsyncSession, token: str) -> None:
    """Sign out: this session only."""

    await db.execute(
        update(UserSession)
        .where(
            UserSession.token_hash == hash_session_token(token),
            UserSession.revoked_at.is_(None),
        )
        .values(revoked_at=utc_now())
    )
    await db.commit()


async def revoke_user_sessions(db: AsyncSession, user_id: int) -> None:
    """Every session the user has (deactivated, or a new password). Leaves
    committing to the caller."""

    await db.execute(
        update(UserSession)
        .where(
            UserSession.user_id == user_id,
            UserSession.revoked_at.is_(None),
        )
        .values(revoked_at=utc_now())
    )
