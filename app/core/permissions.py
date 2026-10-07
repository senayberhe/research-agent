"""Role-based access control: what each role may do.

    Viewer      view research and jobs
    Researcher  + start and resume research
    Operator    + analytics: system metrics, SLOs, error budgets, workers
    Admin       + manage users and roles

Checked on the server for every request (app/api/dependencies.py require),
from the user's current role in the database, so a role change applies at
once. The frontend hides what a role can't do, but that's only courtesy.
"""

from enum import Enum

from app.db.models import UserRole


class Permission(str, Enum):
    VIEW = "view"
    RESEARCH_CREATE = "research:create"
    RESEARCH_RESUME = "research:resume"
    ANALYTICS_VIEW = "analytics:view"
    USERS_MANAGE = "users:manage"


ROLE_PERMISSIONS: dict[UserRole, frozenset[Permission]] = {
    UserRole.VIEWER: frozenset({Permission.VIEW}),
    UserRole.RESEARCHER: frozenset(
        {
            Permission.VIEW,
            Permission.RESEARCH_CREATE,
            Permission.RESEARCH_RESUME,
        }
    ),
    UserRole.OPERATOR: frozenset(
        {
            Permission.VIEW,
            Permission.RESEARCH_CREATE,
            Permission.RESEARCH_RESUME,
            Permission.ANALYTICS_VIEW,
        }
    ),
    UserRole.ADMIN: frozenset(Permission),
}


def permissions_for(role: str | UserRole) -> frozenset[Permission]:
    return ROLE_PERMISSIONS[UserRole(role)]


def has_permission(role: str | UserRole, permission: Permission) -> bool:
    return permission in permissions_for(role)


# For error messages: "Researchers and admins can start research."
WHO_CAN = {
    Permission.VIEW: "Everyone signed in",
    Permission.RESEARCH_CREATE: "Researchers, operators and admins",
    Permission.RESEARCH_RESUME: "Researchers, operators and admins",
    Permission.ANALYTICS_VIEW: "Operators and admins",
    Permission.USERS_MANAGE: "Admins",
}

WHAT = {
    Permission.VIEW: "view this",
    Permission.RESEARCH_CREATE: "start research",
    Permission.RESEARCH_RESUME: "resume research",
    Permission.ANALYTICS_VIEW: "view analytics",
    Permission.USERS_MANAGE: "manage users",
}
