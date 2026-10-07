"""Passwords and session ids.

Passwords are hashed with argon2 (pwdlib), never stored. A sign-in is a
random session id (256 bits) in an httpOnly cookie, which page scripts
can't read; the database keeps only its SHA-256 (app/db/models.py
UserSession), and app/api/dependencies.py looks it up on every request.
"""

import hashlib
import secrets

from pwdlib import PasswordHash


_hasher = PasswordHash.recommended()

# Checked against when a username doesn't exist, so a wrong username takes
# as long as a wrong password (no telling which usernames exist by timing).
_DUMMY_HASH = _hasher.hash("not-a-real-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """True if the password matches. With no hash (no such user), still
    does the work, and returns False."""

    if password_hash is None:
        _hasher.verify(password, _DUMMY_HASH)
        return False

    return _hasher.verify(password, password_hash)


def new_session_token() -> str:
    """A new session id, for the cookie."""

    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """What's stored for a session id. A fast hash is enough: the id is
    random and long, so it can't be guessed from its hash."""

    return hashlib.sha256(token.encode()).hexdigest()
