"""Scheduled tasks: per-session cron-fired bash commands.

A session owns a list of scheduled tasks; each entry is a cron-fired
command that runs in the session's sandbox without waking the model.
The model wakes only if the bash command explicitly escalates by
POST-ing a user-role event back to the session.

See issue #636 for the substrate rationale.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from croniter import croniter
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SCHEDULED_TASKS_PER_SESSION = 32

DEFAULT_TIMEOUT_SECONDS = 300
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 3600

DEFAULT_MAX_OUTPUT_BYTES = 65536
MIN_MAX_OUTPUT_BYTES = 1024
MAX_MAX_OUTPUT_BYTES = 1_048_576  # 1 MiB

MAX_COMMAND_CHARS = 16_384
MAX_SCHEDULE_CHARS = 128
MAX_NAME_CHARS = 64

ScheduledTaskStatus = Literal["ok", "error", "timeout", "skipped"]


def _validate_cron_expression(value: str) -> str:
    if not croniter.is_valid(value):
        raise ValueError(f"invalid cron expression: {value!r}")
    return value


class ScheduledTaskCreate(BaseModel):
    """Request body for adding a scheduled task to a session.

    Also accepted in :class:`SessionCreate.scheduled_tasks` for initial
    attachment at session creation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=MAX_NAME_CHARS,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
        description="Stable user-chosen identifier; unique per session.",
    )
    schedule: str = Field(
        min_length=1,
        max_length=MAX_SCHEDULE_CHARS,
        description="Standard 5-field cron expression in UTC.",
    )
    command: str = Field(
        min_length=1,
        max_length=MAX_COMMAND_CHARS,
        description="Bash command run in the session's sandbox at each fire.",
    )
    enabled: bool = True
    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        ge=MIN_TIMEOUT_SECONDS,
        le=MAX_TIMEOUT_SECONDS,
    )
    max_output_bytes: int = Field(
        default=DEFAULT_MAX_OUTPUT_BYTES,
        ge=MIN_MAX_OUTPUT_BYTES,
        le=MAX_MAX_OUTPUT_BYTES,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schedule")
    @classmethod
    def _validate_schedule(cls, v: str) -> str:
        return _validate_cron_expression(v)


class ScheduledTaskUpdate(BaseModel):
    """PATCH body for updating a scheduled task.

    ``name`` cannot be changed — it is the addressable identifier within
    a session. All other fields are optional; omitted fields are left
    unchanged. Toggling ``enabled`` true→false clears ``next_fire``;
    false→true recomputes it from now.
    """

    model_config = ConfigDict(extra="forbid")

    schedule: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SCHEDULE_CHARS,
    )
    command: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_COMMAND_CHARS,
    )
    enabled: bool | None = None
    timeout_seconds: int | None = Field(
        default=None,
        ge=MIN_TIMEOUT_SECONDS,
        le=MAX_TIMEOUT_SECONDS,
    )
    max_output_bytes: int | None = Field(
        default=None,
        ge=MIN_MAX_OUTPUT_BYTES,
        le=MAX_MAX_OUTPUT_BYTES,
    )
    metadata: dict[str, Any] | None = None

    @field_validator("schedule")
    @classmethod
    def _validate_schedule(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_cron_expression(v)


class ScheduledTaskEcho(BaseModel):
    """Read view of a scheduled task as echoed on ``Session.scheduled_tasks``.

    Runtime fields (``last_fire_at`` / ``last_fire_status`` /
    ``consecutive_failures``) reflect the most recent fire outcome.
    ``running_since`` is internal scheduler bookkeeping and is not
    exposed here.
    """

    id: str
    name: str
    schedule: str
    command: str
    enabled: bool
    timeout_seconds: int
    max_output_bytes: int
    next_fire: datetime | None
    last_fire_at: datetime | None
    last_fire_status: ScheduledTaskStatus | None
    consecutive_failures: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


def compute_next_fire(schedule: str, from_time: datetime) -> datetime:
    """Compute the next cron-fire time strictly after ``from_time``.

    ``from_time`` should be timezone-aware (UTC); the returned datetime
    is in the same timezone. The expression must be a valid 5-field
    cron string (validated by :func:`_validate_cron_expression`).
    """
    it = croniter(schedule, from_time)
    next_time = it.get_next(datetime)
    assert isinstance(next_time, datetime)
    return next_time


def validate_scheduled_tasks(tasks: list[ScheduledTaskCreate]) -> None:
    """Cross-item invariants for the initial scheduled_tasks list on
    :class:`SessionCreate`.

    The per-session cap is enforced by ``Field(max_length=...)`` on the
    list field; this function only checks for duplicate names so the
    caller sees a 422 instead of a 500-from-IntegrityError.
    """
    seen: set[str] = set()
    for task in tasks:
        if task.name in seen:
            raise ValueError(f"duplicate scheduled task name {task.name!r}")
        seen.add(task.name)
