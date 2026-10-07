import asyncio
import logging
import os
import signal
import uuid

from app.core.logging import configure_logging
from app.jobs.worker import ResearchWorker

logger = logging.getLogger(__name__)


async def main():

    # Without this nothing is logged: no "Worker starting", no job results.
    configure_logging()

    worker_id = os.getenv("WORKER_ID")

    if not worker_id:
        worker_id = (
            f"research-worker-{uuid.uuid4().hex[:8]}"
        )

    worker = ResearchWorker(
        worker_id=worker_id,
    )

    loop = asyncio.get_running_loop()

    def handle_shutdown_signal():
        logger.info(
            "Shutdown signal received by %s",
            worker_id,
        )

        worker.stop()

    loop.add_signal_handler(
        signal.SIGTERM,
        handle_shutdown_signal,
    )

    loop.add_signal_handler(
        signal.SIGINT,
        handle_shutdown_signal,
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
