from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.security import create_access_token
from app.db.database import get_db
from app.db.models import User
from app.schemas.types import UTCDateTime
from app.services.user_service import authenticate


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
    created_at: UTCDateTime
    last_login_at: UTCDateTime | None

    model_config = {
        "from_attributes": True
    }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: UTCDateTime
    user: UserResponse


@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Exchanges a username and password for an access token. Send it as
    "Authorization: Bearer <token>" on every other request."""

    user = await authenticate(db, request.username, request.password)

    if user is None:
        # The same answer whatever was wrong: no hint which usernames exist.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, expires_at = create_access_token(user.id)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires_at,
        "user": user,
    }


@router.get(
    "/me",
    response_model=UserResponse,
)
async def me(user: User = Depends(get_current_user)):
    """The signed-in user (and a quick way to check a token)."""

    return user

