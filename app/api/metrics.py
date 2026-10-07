from datetime import timedelta

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Response,
)
from prometheus_client import CONTENT_TYPE_LATEST
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.prometheus import render_prometheus_metrics
from app.db.database import get_db
from app.db.models import utc_now
from app.schemas.metrics import SystemMetricsResponse
from app.services.prometheus_service import collect_prometheus_snapshot
from app.services.system_metrics_service import build_system_monitoring


# Metrics come in two forms:
#
# - JSON for applications, under /research/metrics: the flat system
#   summary (/research/metrics, in app/api/research.py), the detailed view
#   below (/research/metrics/system) and per task
#   (/research/{task_id}/metrics).
# - Prometheus text for monitoring systems at /metrics (below): the only
#   scrape target, computed from the database.

router = APIRouter(
    tags=["Metrics"],
)


@router.get(
    "/research/metrics/system",
    response_model=SystemMetricsResponse,
    # JSON metrics need a signed-in user; /metrics (Prometheus) is public
    # so it can be scraped.
    dependencies=[Depends(get_current_user)],
)
async def get_system_metrics(
    hours: float = Query(default=24, gt=0),
    db: AsyncSession = Depends(get_db),
):
    """Operational metrics for the whole system over the last `hours`: job
    counts, retry and success rates and durations; agent runs, tokens and
    cost; tool success rates and latency; worker activity; plus the jobs
    running now and the queue."""

    return await build_system_monitoring(
        db=db,
        since=utc_now() - timedelta(hours=hours),
    )


@router.get(
    "/metrics",
    response_class=Response,
    responses={200: {"content": {CONTENT_TYPE_LATEST: {}}}},
)
async def get_prometheus_metrics(
    db: AsyncSession = Depends(get_db),
):
    """Everything for Prometheus to scrape, computed from the database:
    job, tool and agent activity (counters and histograms) and the current
    state of the queue. The metrics are listed in app/core/prometheus.py.
    """

    return Response(
        content=render_prometheus_metrics(
            await collect_prometheus_snapshot(db)
        ),
        media_type=CONTENT_TYPE_LATEST,
    )

