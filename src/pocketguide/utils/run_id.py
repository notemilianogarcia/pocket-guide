"""
Run ID generation utilities for deterministic and reproducible evaluation runs.
"""

from datetime import datetime


def make_run_id(now: datetime | None = None) -> str:
    """
    Generate a run ID in format YYYYMMDD_HHMMSS.

    Args:
        now: Optional datetime to use. If None, uses current time.
            This parameter allows tests to supply fixed timestamps.

    Returns:
        Run ID string in format YYYYMMDD_HHMMSS (e.g., "20260128_143052")

    Examples:
        >>> from datetime import datetime
        >>> dt = datetime(2026, 1, 28, 14, 30, 52)
        >>> make_run_id(dt)
        '20260128_143052'
        >>> # For production use (current time)
        >>> make_run_id()  # doctest: +SKIP
        '20260128_143052'
    """
    if now is None:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")
