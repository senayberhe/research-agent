"""Slows down password guessing: after too many failed sign-ins for one
username, or from one client address, within a window, POST /auth/login
answers 429 until the window has passed.

Kept in memory: fine for the single API process this app runs (a restart
forgets the counts). Several API processes would need a shared store
(Redis, or a table).
"""

import time
from collections import defaultdict, deque

from app.core.config import settings


class LoginThrottle:

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.failures: dict[str, deque[float]] = defaultdict(deque)

    def _recent(self, key: str) -> deque[float]:
        times = self.failures[key]
        cutoff = self.clock() - settings.auth_login_window_seconds

        while times and times[0] <= cutoff:
            times.popleft()

        if not times:
            # Don't keep an entry for every username ever tried.
            del self.failures[key]
            return deque()

        return times

    def retry_after(self, username: str, client: str) -> int | None:
        """Seconds until another attempt is allowed, or None if it is."""

        limits = (
            (f"user:{username}", settings.auth_login_max_failures),
            (f"ip:{client}", settings.auth_login_max_failures_per_ip),
        )

        waits = []

        for key, limit in limits:
            times = self._recent(key)
            if len(times) >= limit:
                oldest_that_counts = times[-limit]
                waits.append(oldest_that_counts + settings.auth_login_window_seconds - self.clock())

        return max(1, round(max(waits))) if waits else None

    def failed(self, username: str, client: str) -> None:
        now = self.clock()
        self.failures[f"user:{username}"].append(now)
        self.failures[f"ip:{client}"].append(now)

    def succeeded(self, username: str) -> None:
        # The right password resets the username's count (not the
        # address's: one address guessing many usernames stays limited).
        self.failures.pop(f"user:{username}", None)

    def reset(self) -> None:
        self.failures.clear()


throttle = LoginThrottle()
