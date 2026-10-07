import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import configure_logging
from app.db.database import get_db
from app.schemas.research import SystemHealthResponse
from app.services.health_service import build_system_health
from app.api.auth import router as auth_router
from app.api.events import router as events_router
from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.api.metrics import router as metrics_router
from app.api.research import router as research_router


logger = logging.getLogger(__name__)


async def create_initial_admin() -> None:
    """AUTH_ADMIN_USERNAME / AUTH_ADMIN_PASSWORD, if set, become the first
    user, but only while there are no users (never overwritten)."""

    from app.db.database import AsyncSessionLocal
    from app.services.user_service import UserError, count_users, create_user

    if not (settings.auth_admin_username and settings.auth_admin_password):
        return

    try:
        async with AsyncSessionLocal() as db:
            if await count_users(db) == 0:
                user = await create_user(
                    db,
                    settings.auth_admin_username,
                    settings.auth_admin_password.get_secret_value(),
                )
                logger.info("Created the initial admin user %r", user.username)
    except UserError as error:
        logger.error("Couldn't create the initial admin user: %s", error)
    except Exception:
        logger.exception("Couldn't create the initial admin user")

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info(
        "Application startup beginning"
    )

    # Tables are managed by Alembic, not created here. After changing a model:
    #   uv run alembic revision --autogenerate -m "..."
    #   uv run alembic upgrade head

    # Research runs in the worker (app/jobs/worker.py), not here; so does
    # recovering work that a crash or restart cut off.

    await create_initial_admin()

    logger.info("Application startup completed")

    yield

    logger.info(
        "Application shutdown beginning"
    )

app = FastAPI(
    title="Research Agent API",
    description="Ai-powered research workflow API",
    version="1.0.0",
    lifespan=lifespan
)
# Lets the frontend (CORS_ORIGINS) call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(health_router)
app.include_router(research_router)
app.include_router(jobs_router)
app.include_router(metrics_router)
app.include_router(events_router)


@app.get("/")
async def root():
    return {
        "message": "Research Agent API is running"
    }


@app.get(
    "/health",
    response_model=SystemHealthResponse,
)
async def health(
    db: AsyncSession = Depends(get_db),
):
    """Database, workers and job queue (see build_system_health). Always
    200: a degraded or unhealthy system is reported in the body, so the
    container healthcheck doesn't restart the API over a database outage."""

    return await build_system_health(
        db=db,
    )