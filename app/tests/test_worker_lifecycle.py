"""ResearchWorker.status through start, stop and shutdown."""

import pytest

from app.jobs.worker import ResearchWorker


@pytest.mark.asyncio
async def test_worker_run_ends_cleanly(tmp_path):

    worker = ResearchWorker(
        worker_id="worker-1",
        poll_interval_seconds=0.01,
        health_file=str(tmp_path / "worker-health.json"),
    )

    worker.stop()

    await worker.run()

    assert worker.status == "stopped"


@pytest.mark.asyncio
async def test_worker_stop_changes_status():

    worker = ResearchWorker(
        worker_id="worker-1",
    )

    worker._status = "running"

    worker.stop()

    assert worker.status == "stopping"
