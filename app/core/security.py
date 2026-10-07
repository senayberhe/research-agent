"""Passwords and access tokens.

Passwords are hashed with argon2 (pwdlib), never stored. Access tokens are
JWTs signed with AUTH_SECRET_KEY (HS256): "sub" is the user's id, "exp"
when it stops working. A token can't be changed without the key, so the
API trusts its contents once the signature and expiry check out (and the
user still exists and is active: see app/api/dependencies.py).
"""

from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from app.core.config import settings


ALGORITHM = "HS256"

_hasher = PasswordHash.recommended()

# Checked against when a username doesn't exist, so a wrong username takes
# as long as a wrong password (no telling which usernames exist by timing).
_DUMMY_HASH = _hasher.hash("not-a-real-password")


class InvalidToken(Exception):
    """Missing, malformed, tampered with, or expired."""


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """True if the password matches. With no hash (no such user), still
    does the work, and returns False."""

    if password_hash is None:
        _hasher.verify(password, _DUMMY_HASH)
        return False

    return _hasher.verify(password, password_hash)


def create_access_token(user_id: int, now: datetime | None = None) -> tuple[str, datetime]:
    """(token, when it expires)."""

    now = now or datetime.now(UTC)
    expires = now + timedelta(minutes=settings.auth_access_token_minutes)

    token = jwt.encode(
        {
            "sub": str(user_id),
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
        },
        settings.auth_secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )

    return token, expires


def decode_access_token(token: str) -> int:
    """The user id in a valid token; raises InvalidToken otherwise."""

    try:
        claims = jwt.decode(
            token,
            settings.auth_secret_key.get_secret_value(),
            # Only this algorithm: a token can't pick a weaker one.
            algorithms=[ALGORITHM],
            options={"require": ["sub", "exp"]},
        )
        return int(claims["sub"])

    except (jwt.InvalidTokenError, ValueError) as exc:
        raise InvalidToken(str(exc)) from exc
