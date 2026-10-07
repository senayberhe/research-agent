"""Users: created by an admin (python -m app.cli), authenticated by
username and password."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.db.models import User, utc_now


MIN_PASSWORD_LENGTH = 12


class UserError(ValueError):
    """A user can't be created or changed (the message says why)."""


def normalize_username(username: str) -> str:
    return username.strip().lower()


def check_password_strength(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserError(
            f"Passwords need at least {MIN_PASSWORD_LENGTH} characters."
        )


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    return await db.scalar(
        select(User).where(User.username == normalize_username(username))
    )


async def count_users(db: AsyncSession) -> int:
    return await db.scalar(select(func.count()).select_from(User))


async def create_user(db: AsyncSession, username: str, password: str) -> User:
    username = normalize_username(username)

    if not username:
        raise UserError("A username is required.")

    check_password_strength(password)

    if await get_user_by_username(db, username) is not None:
        raise UserError(f"User {username!r} already exists.")

    user = User(username=username, password_hash=hash_password(password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


async def authenticate(db: AsyncSession, username: str, password: str) -> User | None:
    """The user, if the username and password are right and the account is
    active; None otherwise (without saying which part was wrong)."""

    user = await get_user_by_username(db, username)

    password_ok = verify_password(
        password,
        user.password_hash if user is not None else None,
    )

    if user is None or not password_ok or not user.is_active:
        return None

    user.last_login_at = utc_now()
    await db.commit()

    return user


async def set_password(db: AsyncSession, username: str, password: str) -> User:
    user = await get_user_by_username(db, username)

    if user is None:
        raise UserError(f"No user {normalize_username(username)!r}.")

    check_password_strength(password)
    user.password_hash = hash_password(password)
    await db.commit()

    return user


async def set_active(db: AsyncSession, username: str, active: bool) -> User:
    user = await get_user_by_username(db, username)

    if user is None:
        raise UserError(f"No user {normalize_username(username)!r}.")

    user.is_active = active
    await db.commit()

    return user


async def list_users(db: AsyncSession) -> list[User]:
    return list((await db.execute(select(User).order_by(User.id))).scalars().all())
