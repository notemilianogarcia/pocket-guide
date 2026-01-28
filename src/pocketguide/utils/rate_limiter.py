"""Simple synchronous rate limiter for API calls."""

import time


class RateLimiter:
    """Simple token bucket rate limiter.

    Enforces a maximum requests per minute (RPM) limit.
    """

    def __init__(self, rpm: int):
        """Initialize rate limiter.

        Args:
            rpm: Maximum requests per minute
        """
        self.rpm = rpm
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_request_time: float | None = None

    def wait_if_needed(self) -> None:
        """Wait if needed to respect rate limit."""
        if self.rpm <= 0:
            # No rate limiting
            return

        now = time.time()

        if self.last_request_time is not None:
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)

        self.last_request_time = time.time()
