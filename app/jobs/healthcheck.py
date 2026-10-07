"""Worker health check, for Docker's HEALTHCHECK:

    python -m app.jobs.healthcheck

Exit code 0 if the worker's health file (WORKER_HEALTH_FILE) was refreshed
within the last 3 heartbeats, 1 otherwise. The worker refreshes it on every
heartbeat, also while a job runs, so a stale file means the worker has died
or is stuck.
"""

import json
import os
import sys
import time

from app.core.config import settings

# Missed heartbeats allowed before the worker counts as unhealthy.
MISSED_HEARTBEATS = 3


def check_health(
    path: str,
    max_age_seconds: float,
    now: float | None = None,
) -> tuple[bool, str]:
    """(healthy, reason) for the health file at path."""

    try:
        modified = os.stat(path).st_mtime
    except FileNotFoundError:
        return False, f"no health file at {path} (worker not started?)"

    try:
        with open(path) as file:
            health = json.load(file)
    except (OSError, ValueError) as exc:
        return False, f"unreadable health file: {exc}"

    age = (now if now is not None else time.time()) - modified

    if age > max_age_seconds:
        return False, (
            f"health file is {age:.0f}s old (limit {max_age_seconds:.0f}s): "
            "worker is stuck or dead"
        )

    if health.get("status") == "stopped":
        return False, "worker has stopped"

    return True, (
        f"worker {health.get('worker_id')} is {health.get('status')} "
        f"(job {health.get('job_id')}, updated {age:.0f}s ago)"
    )


def main() -> int:

    healthy, reason = check_health(
        path=settings.worker_health_file,
        max_age_seconds=settings.worker_heartbeat_seconds * MISSED_HEARTBEATS,
    )

    print(("healthy: " if healthy else "unhealthy: ") + reason)

    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
