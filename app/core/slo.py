"""Service level objectives: what healthy production behavior means.

Each SLO is a target for a service level indicator (SLI) measured over a
rolling window (app/services/slo_service.py). Ratio targets (0-1) leave an
error budget of 1 - target, e.g. 99% job success allows 1% failures; latency
targets are p95 thresholds in seconds.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from app.core.config import settings


# SLOs are judged over a rolling window (SLO_WINDOW_DAYS, default 30), so an
# old incident stops counting once it's this far in the past.
SLO_WINDOW_DAYS = settings.slo_window_days
SLO_WINDOW = timedelta(days=SLO_WINDOW_DAYS)


def window_bounds(
    window_start: datetime,
    window_end: datetime,
) -> dict:
    """The window as the API reports it: the exact instants the
    measurements used, as explicit UTC (the database stores naive UTC)."""

    return {
        "window_days": (window_end - window_start).total_seconds() / 86400,
        "window_start": window_start.replace(tzinfo=UTC),
        "window_end": window_end.replace(tzinfo=UTC),
    }


@dataclass(frozen=True)
class SLO:
    name: str
    target: float
    description: str


API_AVAILABILITY = SLO(
    name="api_availability",
    target=0.999,
    description="API should be available at least 99.9% of the time.",
)

RESEARCH_JOB_SUCCESS = SLO(
    name="research_job_success",
    target=0.99,
    description="At least 99% of research jobs should complete successfully.",
)

TOOL_SUCCESS = SLO(
    name="tool_success",
    target=0.98,
    description="At least 98% of research tool calls should succeed.",
)

JOB_LATENCY = SLO(
    name="job_latency",
    target=60.0,
    description="Research job p95 latency should remain below 60 seconds.",
)

TOOL_LATENCY = SLO(
    name="tool_latency",
    target=15.0,
    description="Research tool p95 latency should remain below 15 seconds.",
)

WORKER_AVAILABILITY = SLO(
    name="worker_availability",
    target=0.999,
    description="Workers should be available at least 99.9% of the time.",
)
