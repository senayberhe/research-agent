import logging
import sys

from app.core.config import settings


def configure_logging():
    level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    # Only pass handlers=: basicConfig raises ValueError if stream= is
    # given as well.
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
