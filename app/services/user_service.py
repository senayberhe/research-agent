"""Users: created by an admin (python -m app.cli), authenticated by
username and password."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.db.models import User, UserRole, utc_now
from app.services.session_service import revoke_user_sessions


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


async def create_user(
    db: AsyncSession,
    username: str,
    password: str,
    role: UserRole | str = UserRole.VIEWER,
) -> User:
    username = normalize_username(username)
    role = _role(role)

    if not username:
        raise UserError("A username is required.")

    check_password_strength(password)

    if await get_user_by_username(db, username) is not None:
        raise UserError(f"User {username!r} already exists.")

    user = User(username=username, password_hash=hash_password(password), role=role)
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
    # Whoever had the old password is signed out too.
    await revoke_user_sessions(db, user.id)
    await db.commit()

    return user


def _role(role: UserRole | str) -> UserRole:
    try:
        return UserRole(role)
    except ValueError:
        roles = ", ".join(r.value for r in UserRole)
        raise UserError(f"Unknown role {role!r}: use one of {roles}.") from None


async def count_active_admins(db: AsyncSession) -> int:
    return await db.scalar(
        select(func.count())
        .select_from(User)
        .where(User.role == UserRole.ADMIN, User.is_active.is_(True))
    )


async def _check_not_last_admin(db: AsyncSession, user: User) -> None:
    """Refuses a change that would leave no active admin (nobody could
    manage users any more)."""

    if (
        user.role == UserRole.ADMIN
        and user.is_active
        and await count_active_admins(db) <= 1
    ):
        raise UserError(
            f"{user.username!r} is the last active admin: make someone else "
            "an admin first."
        )


async def update_user(
    db: AsyncSession,
    user: User,
    role: UserRole | str | None = None,
    is_active: bool | None = None,
    password: str | None = None,
) -> User:
    """Changes a user's role, active state and/or password (None: leave
    it). Never leaves the system without an active admin."""

    if role is not None:
        role = _role(role)
        if role != UserRole.ADMIN:
            await _check_not_last_admin(db, user)
        user.role = role

    if is_active is not None:
        if not is_active:
            await _check_not_last_admin(db, user)
        user.is_active = is_active
        if not is_active:
            await revoke_user_sessions(db, user.id)

    if password is not None:
        check_password_strength(password)
        user.password_hash = hash_password(password)
        await revoke_user_sessions(db, user.id)

    await db.commit()
    await db.refresh(user)

    return user


async def set_active(db: AsyncSession, username: str, active: bool) -> User:
    user = await get_user_by_username(db, username)

    if user is None:
        raise UserError(f"No user {normalize_username(username)!r}.")

    return await update_user(db, user, is_active=active)


async def set_role(db: AsyncSession, username: str, role: UserRole | str) -> User:
    user = await get_user_by_username(db, username)

    if user is None:
        raise UserError(f"No user {normalize_username(username)!r}.")

    return await update_user(db, user, role=role)


async def list_users(db: AsyncSession) -> list[User]:
    return list((await db.execute(select(User).order_by(User.id))).scalars().all())
