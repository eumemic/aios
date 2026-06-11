"""Triggers: per-session timers carrying a ``source`` (what fires them) and
an ``action`` (what runs at fire time).

A session owns a list of triggers. The ``source`` decides *when* a trigger
fires — a ``cron`` schedule (recurring) or a ``one_shot`` ``fire_at``
(self-deletes after firing). The ``action`` decides *what* happens — a
``sandbox_command`` (a bash command in the session's sandbox, no model
wake) or a ``wake_owner`` (a user-role message delivered to the owning
session, waking the model). The two axes are orthogonal: any source with
any action.

Growth rule (load-bearing): a new behavior is always a new ``source`` or a
new ``action`` *kind*, never a flag on an existing one. Adding a kind is a
DB CHECK constraint swap (zero row rewrites) plus a Pydantic union member.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from croniter import CroniterBadDateError, croniter
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

MAX_TRIGGERS_PER_SESSION = 32  # carried verbatim from MAX_SCHEDULED_TASKS_PER_SESSION

DEFAULT_TIMEOUT_SECONDS = 300
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 3600

DEFAULT_MAX_OUTPUT_BYTES = 65536
MIN_MAX_OUTPUT_BYTES = 1024
MAX_MAX_OUTPUT_BYTES = 1_048_576  # 1 MiB

MAX_COMMAND_CHARS = 16_384
MAX_SCHEDULE_CHARS = 128
MAX_NAME_CHARS = 64
MAX_WAKE_CONTENT_CHARS = 16_384  # ≈ today's implicit bound (content rode inside command)

CRON_OCCURRENCE_HORIZON_YEARS = 1

TriggerFireStatus = Literal["ok", "error", "timeout", "skipped"]  # was ScheduledTaskStatus


def _validate_cron_expression(value: str) -> str:
    """WRITE-PATH ONLY (``TriggerCreate`` / ``TriggerUpdate``) — never on read models.

    Grammar-validates the expression, then requires at least one real
    occurrence within :data:`CRON_OCCURRENCE_HORIZON_YEARS`. Grammar-valid
    expressions with no real fire in the horizon (e.g. ``0 0 30 2 *``) are
    rejected at create/update instead of sitting silently dead — croniter
    raises ``CroniterBadDateError`` when no match exists within
    ``max_years_between_matches``.
    """
    if not croniter.is_valid(value):
        raise ValueError(f"invalid cron expression: {value!r}")
    try:
        croniter(
            value,
            datetime.now(UTC),
            max_years_between_matches=CRON_OCCURRENCE_HORIZON_YEARS,
        ).get_next(datetime)
    except CroniterBadDateError:
        raise ValueError(
            f"cron expression {value!r} produces no occurrence within the next "
            f"{CRON_OCCURRENCE_HORIZON_YEARS} year(s)"
        ) from None
    return value


# ─── source: what fires the trigger ──────────────────────────────────────────


class CronSource(BaseModel):
    """Recurring source: a standard 5-field cron expression in UTC.

    Structure-only here — grammar/occurrence checks live on the write
    models (``TriggerCreate`` / ``TriggerUpdate``) so the read path accepts
    every row the write path ever accepted (see module docstring + §2.2 of
    the design contract). A future additive ``timezone`` field (IANA name;
    absent = UTC) lands without touching this model or the DB CHECK.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["cron"] = "cron"
    schedule: str = Field(min_length=1, max_length=MAX_SCHEDULE_CHARS)


class OneShotSource(BaseModel):
    """One-shot source: fires once at an absolute UTC time, then self-deletes."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["one_shot"] = "one_shot"
    fire_at: datetime

    @field_validator("fire_at")
    @classmethod
    def _validate_fire_at_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "fire_at must be timezone-aware (e.g. ISO 8601 with a `Z` or explicit "
                "offset) — naive datetimes are ambiguous against the `timestamptz` column"
            )
        return v


TriggerSource = Annotated[CronSource | OneShotSource, Field(discriminator="kind")]


# ─── action: what runs at fire time ──────────────────────────────────────────


class SandboxCommandAction(BaseModel):
    """Run a bash command in the session's sandbox WITHOUT waking the model.

    Verbatim today's scheduled-task behavior. Defaults for
    ``timeout_seconds`` / ``max_output_bytes`` are materialized at write
    time so the stored row is self-describing (the runner carries no
    defaults knowledge).
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["sandbox_command"] = "sandbox_command"
    command: str = Field(min_length=1, max_length=MAX_COMMAND_CHARS)
    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS
    )
    max_output_bytes: int = Field(
        default=DEFAULT_MAX_OUTPUT_BYTES, ge=MIN_MAX_OUTPUT_BYTES, le=MAX_MAX_OUTPUT_BYTES
    )


class SandboxCommandActionReplace(SandboxCommandAction):
    """Update-side variant: optional-at-create fields are REQUIRED, so a
    partial action 422s instead of silently resetting stored values to
    defaults. (Create keeps the defaults for tool ergonomics.)"""

    timeout_seconds: int = Field(ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS)
    max_output_bytes: int = Field(ge=MIN_MAX_OUTPUT_BYTES, le=MAX_MAX_OUTPUT_BYTES)


class WakeOwnerAction(BaseModel):
    """Deliver ``content`` as a user-role message to the trigger's OWNING
    session, waking the model — the self-wake primitive on a timer.

    No ``session_id``: the target is implicitly ``owner_session_id`` (NOT
    NULL this slice). An explicit-target wake is a separate future kind,
    not a field on this one (avoids the uncapped cross-session wake that an
    unconstrained ``session_id`` would be — see the ``wake_session`` tool's
    depth/rate caps).
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["wake_owner"] = "wake_owner"
    content: str = Field(min_length=1, max_length=MAX_WAKE_CONTENT_CHARS)


TriggerAction = Annotated[SandboxCommandAction | WakeOwnerAction, Field(discriminator="kind")]
TriggerActionReplace = Annotated[
    SandboxCommandActionReplace | WakeOwnerAction, Field(discriminator="kind")
]

# Module-level adapters for the query/runner read path. Structure-only —
# they must accept every row the write path ever accepted, so they carry NO
# cron occurrence check (that is write-side only; a legally-persisted rare
# cron must still read back).
TRIGGER_SOURCE_ADAPTER: TypeAdapter[CronSource | OneShotSource] = TypeAdapter(TriggerSource)
TRIGGER_ACTION_ADAPTER: TypeAdapter[SandboxCommandAction | WakeOwnerAction] = TypeAdapter(
    TriggerAction
)


# ─── request / response models ───────────────────────────────────────────────


class TriggerCreate(BaseModel):
    """Request body for adding a trigger to a session.

    Carries a ``source`` (cron / one_shot) and an ``action``
    (sandbox_command / wake_owner). Also accepted in
    :class:`SessionCreate.triggers` for initial attachment at session
    creation.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        min_length=1,
        max_length=MAX_NAME_CHARS,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
        description="Stable user-chosen identifier; unique per session.",
    )
    source: TriggerSource
    action: TriggerAction
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_cron_write_path(self) -> TriggerCreate:
        if isinstance(self.source, CronSource):
            _validate_cron_expression(self.source.schedule)
        return self


class TriggerUpdate(BaseModel):
    """Update body. ``source`` / ``action`` are replaced WHOLESALE when
    provided (a cron↔one-shot or sandbox↔wake conversion is just a
    different object). ``None`` = leave alone; there is no clear-to-null
    (both columns are NOT NULL). The next_fire / cap / past-fire_at
    business rules are enforced in the service layer (§2.4).
    """

    model_config = ConfigDict(extra="forbid")
    source: TriggerSource | None = None
    action: TriggerActionReplace | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_cron_write_path(self) -> TriggerUpdate:
        if isinstance(self.source, CronSource):
            _validate_cron_expression(self.source.schedule)
        return self


class TriggerEcho(BaseModel):
    """Read view of a trigger as echoed on ``Session.triggers``.

    Runtime fields (``last_fire_at`` / ``last_fire_status`` /
    ``consecutive_failures``) reflect the most recent fire outcome.
    ``running_since`` is internal scheduler bookkeeping and is not exposed
    here.
    """

    id: str
    name: str
    source: TriggerSource
    action: TriggerAction
    enabled: bool
    next_fire: datetime | None
    last_fire_at: datetime | None
    last_fire_status: TriggerFireStatus | None
    consecutive_failures: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class TriggerRunEcho(BaseModel):
    """Read view of one ``trigger_runs`` row — a single fire of a trigger.

    ``trigger_context`` echoes the firing source (``cron`` / ``one_shot`` /
    ``run_completion``); ``event`` carries the per-event context for
    ``run_completion`` fires (``{run_id, workflow_id, status}``) and is
    ``None`` for timer fires. ``status`` is an open string on read (rows
    written by future writers must always read back); the current writer
    vocabulary is ``pending``/``running``/``ok``/``error``/``timeout``/
    ``skipped``. ``result_id`` is the prefixed id of the resource the fire
    created (a ``wfr_…`` run today), or ``None`` when the fire produced no
    resource or failed. ``created_at`` is when the fire INTENT was created
    (the match transaction for event fires; insert time for timer rows);
    ``started_at`` is when execution claimed it.
    """

    id: str
    trigger_id: str
    trigger_context: str
    event: dict[str, Any] | None
    status: str
    result_id: str | None
    error_summary: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def compute_next_fire(schedule: str, from_time: datetime) -> datetime:
    """Compute the next cron-fire time strictly after ``from_time``.

    ``from_time`` should be timezone-aware (UTC); the returned datetime is
    in the same timezone. Uses croniter's default occurrence horizon — NOT
    the 1-year write-time horizon (steady-state advance must never fail on
    a legally-persisted rare cron).
    """
    it = croniter(schedule, from_time)
    next_time = it.get_next(datetime)
    assert isinstance(next_time, datetime)
    return next_time


def compute_initial_next_fire(source: CronSource | OneShotSource, now: datetime) -> datetime:
    """Return the initial ``next_fire`` for a freshly-enabled row.

    Cron: the next slot strictly after ``now``. One-shot: ``fire_at``
    verbatim — even if in the past, the scheduler treats it as due
    immediately and fire-and-deletes on the next tick (today's semantic).
    """
    if isinstance(source, OneShotSource):
        return source.fire_at
    return compute_next_fire(source.schedule, now)


def validate_triggers(triggers: list[TriggerCreate]) -> None:
    """Cross-item invariants for the initial ``triggers`` list on
    :class:`SessionCreate`.

    The per-session cap is enforced by ``Field(max_length=...)`` on the
    list field; this function only checks for duplicate names so the caller
    sees a 422 instead of a 500-from-IntegrityError.
    """
    seen: set[str] = set()
    for trigger in triggers:
        if trigger.name in seen:
            raise ValueError(f"duplicate trigger name {trigger.name!r}")
        seen.add(trigger.name)
