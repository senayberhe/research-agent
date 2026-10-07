from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import UserResponse
from app.api.dependencies import require
from app.core.permissions import Permission
from app.db.database import get_db
from app.db.models import User, UserRole
from app.services.user_service import (
    UserError,
    create_user,
    list_users,
    update_user,
)


# Managing users is for admins only.
router = APIRouter(
    prefix="/users",
    tags=["Users"],
    dependencies=[Depends(require(Permission.USERS_MANAGE))],
)


class AdminUserResponse(UserResponse):
    is_active: bool


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=150)
    password: str = Field(min_length=1, max_length=1024)
    role: UserRole = UserRole.VIEWER


class UpdateUserRequest(BaseModel):
    """Fields to change; omitted ones stay as they are."""

    role: UserRole | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=1, max_length=1024)


def _bad_request(error: UserError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))


@router.get("", response_model=list[AdminUserResponse])
async def get_users(db: AsyncSession = Depends(get_db)):
    """Every account, oldest first."""

    return await list_users(db)


@router.post(
    "",
    response_model=AdminUserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_user(request: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    """Creates an account (viewer unless another role is given)."""

    try:
        return await create_user(db, request.username, request.password, request.role)
    except UserError as error:
        raise _bad_request(error) from error


@router.patch("/{user_id}", response_model=AdminUserResponse)
async def change_user(
    user_id: int,
    request: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require(Permission.USERS_MANAGE)),
):
    """Changes a user's role, active state or password. An admin can't
    change their own role or deactivate themselves (no locking yourself
    out), and the last active admin always stays."""

    user = await db.get(User, user_id)

    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if user.id == admin.id and (
        (request.role is not None and request.role != user.role)
        or request.is_active is False
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You can't change your own role or deactivate yourself.",
        )

    try:
        return await update_user(
            db,
            user,
            role=request.role,
            is_active=request.is_active,
            password=request.password,
        )
    except UserError as error:
        raise _bad_request(error) from error
