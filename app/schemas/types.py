"""Shared field types for the API's response schemas."""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import PlainSerializer


def _to_utc_iso(value: datetime) -> str:
    # The database stores naive UTC; say so explicitly, so clients (e.g. a
    # browser's new Date()) don't read the time as local.
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


# A datetime sent as ISO 8601 UTC with a "Z", e.g. "2026-10-07T12:00:00Z".
UTCDateTime = Annotated[
    datetime,
    PlainSerializer(_to_utc_iso, return_type=str, when_used="json"),
]
