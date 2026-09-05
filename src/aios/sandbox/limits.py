"""Shared hard limits for sandbox execution."""

# PostgreSQL ``make_interval`` and the sandbox backend both receive timeout
# horizons derived from persisted data. Keep that data bounded to a generous,
# explicit 100-year ceiling so resolution, execution, and sweep agree.
MAX_BASH_TIMEOUT_SECONDS = 3_155_760_000


def bound_bash_timeout_seconds(value: int) -> int:
    """Clamp a validated positive timeout to the shared persistence/backend bound."""
    return min(value, MAX_BASH_TIMEOUT_SECONDS)
