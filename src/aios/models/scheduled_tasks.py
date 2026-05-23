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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    Each row carries either a cron ``schedule`` (recurring) or a
    ``fire_at`` absolute time (one-shot — self-deletes after firing).
    Exactly one must be set; enforced by both a Pydantic ``model_validator``
    here and a DB CHECK constraint.

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
    schedule: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SCHEDULE_CHARS,
        description=(
            "Standard 5-field cron expression in UTC for recurring rows. "
            "Mutually exclusive with ``fire_at``."
        ),
    )
    fire_at: datetime | None = Field(
        default=None,
        description=(
            "Absolute UTC time for one-shot rows; the row self-deletes "
            "after firing. Mutually exclusive with ``schedule``."
        ),
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
    def _validate_schedule(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_cron_expression(v)

    @field_validator("fire_at")
    @classmethod
    def _validate_fire_at_tz_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError(
                "fire_at must be timezone-aware (e.g. ISO 8601 with a `Z` or "
                "explicit offset) — naive datetimes are ambiguous against the "
                "`timestamptz` column"
            )
        return v

    @model_validator(mode="after")
    def _validate_trigger_xor(self) -> ScheduledTaskCreate:
        if (self.schedule is None) == (self.fire_at is None):
            raise ValueError("exactly one of `schedule` (cron) or `fire_at` (one-shot) must be set")
        return self


class ScheduledTaskUpdate(BaseModel):
    """PATCH body for updating a scheduled task.

    ``name`` cannot be changed — it is the addressable identifier within
    a session. All other fields are optional; omitted fields are left
    unchanged. Toggling ``enabled`` true→false clears ``next_fire``;
    false→true recomputes it from now.

    Updates can adjust the trigger by setting ``schedule`` (cron) or
    ``fire_at`` (one-shot) — but not both in the same PATCH; the DB
    CHECK constraint enforces the XOR invariant after the merged write.
    """

    model_config = ConfigDict(extra="forbid")

    schedule: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_SCHEDULE_CHARS,
    )
    fire_at: datetime | None = Field(
        default=None,
        description="Update the one-shot fire time. Mutually exclusive with `schedule`.",
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

    @field_validator("fire_at")
    @classmethod
    def _validate_fire_at_tz_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError(
                "fire_at must be timezone-aware (e.g. ISO 8601 with a `Z` "
                "or explicit offset) — naive datetimes are ambiguous against "
                "the `timestamptz` column"
            )
        return v

    @model_validator(mode="after")
    def _reject_both_triggers_in_one_patch(self) -> ScheduledTaskUpdate:
        if self.schedule is not None and self.fire_at is not None:
            raise ValueError(
                "PATCH may set at most one of `schedule` or `fire_at` — they're "
                "mutually exclusive on the row. (Cross-merge violations after "
                "the PATCH lands are caught by the service layer's XOR check "
                "against the merged row state.)"
            )
        return self


class ScheduledTaskEcho(BaseModel):
    """Read view of a scheduled task as echoed on ``Session.scheduled_tasks``.

    Exactly one of ``schedule`` (cron) or ``fire_at`` (one-shot) is set.
    Runtime fields (``last_fire_at`` / ``last_fire_status`` /
    ``consecutive_failures``) reflect the most recent fire outcome.
    ``running_since`` is internal scheduler bookkeeping and is not
    exposed here.
    """

    id: str
    name: str
    schedule: str | None
    fire_at: datetime | None
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


def compute_initial_next_fire(
    schedule: str | None, fire_at: datetime | None, now: datetime
) -> datetime | None:
    """Return the initial ``next_fire`` for a freshly-created row.

    For a cron row (``schedule`` set): the next slot strictly after
    ``now``. For a one-shot row (``fire_at`` set): the absolute time
    verbatim — even if ``fire_at`` is in the past, the scheduler will
    treat it as due immediately and fire-and-delete on the next tick.
    """
    if fire_at is not None:
        return fire_at
    if schedule is not None:
        return compute_next_fire(schedule, now)
    raise ValueError("either schedule or fire_at must be provided")


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
