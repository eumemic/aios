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

import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, get_args

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
MAX_INPUT_TEMPLATE_BYTES = 16_384  # compact-JSON serialized; enforced WRITE-PATH ONLY (see below)

# External-event ingress: cap on the inbound webhook body, enforced at the
# ingress before any parse or DB touch (cheapest-first ordering — a malformed
# or oversized probe must never reach the resolver or the carrier INSERT).
MAX_INGEST_EVENT_BYTES = 65536

# 8 = the maximum gap between consecutive fires of any valid 5-field cron. A
# leap-day schedule (`0 0 29 2 *`) skips the non-leap century year — 2096 fires,
# 2100 does not (not divisible by 400), 2104 fires — an 8-year gap. The horizon
# must cover that worst case so a genuinely-recurring schedule is accepted; a
# smaller value (the original 1) false-rejected leap-day crons as "no occurrence"
# in ~3 of every 4 years. Truly-impossible crons (`0 0 30 2 *`) never match at
# any horizon, so raising it does not let them slip through.
CRON_OCCURRENCE_HORIZON_YEARS = 8

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


RunTerminalStatus = Literal["completed", "errored", "cancelled"]
# The single source for every restatement of the terminal-status vocabulary
# (the statuses default, the tool-schema enums). A unit test ties it to
# ``models.workflows.TERMINAL_RUN_STATUSES`` — the upstream truth the
# completion matcher fires on — so the two cannot drift silently.
RUN_TERMINAL_STATUSES: tuple[RunTerminalStatus, ...] = get_args(RunTerminalStatus)


class RunCompletionSource(BaseModel):
    """Reactive source: fires once per terminal completion of any run of the
    watched workflow whose status is in ``statuses``.

    No ``next_fire`` — never scheduled by the tick (the claim/MIN queries'
    ``next_fire IS NOT NULL`` predicate plus the DB guard make a reactive row
    unschedulable by construction); fires are dispatched from the watched
    run's completion transaction instead. The watch is account-scoped: the
    trigger is only ever handed run data its owner could already read via the
    account-scoped run reads.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["run_completion"] = "run_completion"
    workflow_id: str
    # Materialized at write (the row is self-describing; the matcher carries no
    # defaults knowledge). Fire-on-everything is the no-silent-omission default;
    # narrow explicitly (e.g. ["errored"]) for failure-only pipelines.
    statuses: list[RunTerminalStatus] = Field(
        default_factory=lambda: list(RUN_TERMINAL_STATUSES),
        min_length=1,
    )


class ExternalEventSource(BaseModel):
    """Reactive source: fires from an authenticated inbound webhook ingress.

    No wire fields beyond ``kind`` — the inbound HTTP body IS the event, and
    the per-trigger ingest secret is server-minted (NOT a wire field; returned
    plaintext-once on create and stored only as a SHA-256 hash). Like
    ``run_completion`` this is unschedulable by the tick (``next_fire``
    permanently NULL); fires are dispatched from the ingress edge instead of
    a run-completion transaction. It carries no defaulted fields, so (unlike
    ``RunCompletionSource``) it needs no ``*Replace`` subclass.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["external_event"] = "external_event"


class RunCompletionSourceReplace(RunCompletionSource):
    """Update-side variant (§2.2 Replace rule): ``statuses`` is REQUIRED, so a
    partial source on update 422s instead of silently resetting a narrowed
    filter back to all-three. (The first SOURCE member with a defaulted field,
    hence the first ``TriggerSourceReplace`` union.)"""

    statuses: list[RunTerminalStatus] = Field(min_length=1)


TriggerSource = Annotated[
    CronSource | OneShotSource | RunCompletionSource | ExternalEventSource,
    Field(discriminator="kind"),
]
TriggerSourceReplace = Annotated[
    CronSource | OneShotSource | RunCompletionSourceReplace | ExternalEventSource,
    Field(discriminator="kind"),
]


# ─── action: what runs at fire time ──────────────────────────────────────────


class SandboxCommandAction(BaseModel):
    """Run a bash command in the session's sandbox WITHOUT waking the model.

    Defaults for ``timeout_seconds`` / ``max_output_bytes`` are materialized
    at write time so the stored row is self-describing (the runner carries no
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


class WakeSessionAction(BaseModel):
    """Deliver ``content`` as a user-role message to an EXPLICITLY-NAMED
    same-account session, waking it. The cross-session twin of wake_owner:
    runs through ``services.wake.deliver_cross_session_wake`` with the FIRING
    TRIGGER as the lineage root, so the wake_session depth/per-pair-rate caps
    apply (a fire→wake→fire cascade terminates in bounded steps).

    ``target_session_id`` is NOT resolved in the model — it is account-data,
    checked at fire time (same as ``workflow_id``). A cross-account / archived
    / cap-breaching target surfaces as a fire ``error`` (no silent drop), not a
    write-time rejection. Single named target only: no topics, no
    competing-consumers, no fan-out (issue #1280).
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["wake_session"] = "wake_session"
    target_session_id: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=MAX_WAKE_CONTENT_CHARS)


class WorkflowAction(BaseModel):
    """Launch a run of ``workflow_id`` at fire time — deterministic, no model
    wake.

    The run's input is ALWAYS the envelope ``{"trigger": <firing context>,
    "input": <input_template verbatim>}`` — no placeholder substitution. A
    workflow built to be triggered reads ``input["trigger"]`` for the firing
    context (for run_completion fires: the completing run's id, status,
    output, and error kind) and ``input["input"]`` for this template.

    ``environment_id`` is deliberately NOT a field: the run always binds to
    the owner session's environment, resolved at write time into the
    first-class ``triggers.environment_id`` column (sessions' environment is
    immutable, so write-time freeze equals fire-time resolution). Anything on
    this model is agent-reachable through the ``trigger_create`` tool — a
    caller-chosen environment would bypass the same-stance refusal on the
    ``call_workflow`` builtin.

    ``workflow_version``: ``None`` = run the workflow's CURRENT version at
    each fire (float); an integer is a DRIFT ASSERTION — it must equal the
    workflow's current version at write, and a fire whose workflow has since
    been edited records an error instead of running the unreviewed script
    (workflows have no version-history table: a pin cannot resolve an old
    script, only refuse a new one).

    This member is STRUCTURE-ONLY: the ``input_template`` size bound lives on
    the write models — see :func:`_validate_input_template_bound` for why a
    read-side byte bound is unsafe.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["workflow"] = "workflow"
    workflow_id: str = Field(min_length=1)
    workflow_version: int | None = Field(default=None, ge=1)
    version: int | None = Field(default=None, ge=1)
    input_template: Any = None  # arbitrary JSON, any type; null = no payload
    vault_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_version_and_assertion(self) -> WorkflowAction:
        # ``version`` (selector) and ``workflow_version`` (drift assertion) are
        # orthogonal (F1): the only surface where both can be set is this trigger
        # union, and asserting the head has not moved WHILE pinning a historical
        # version to re-run is self-contradictory -- reject it here, at the write
        # edge. (``WfRunCreate`` has no ``expected_version`` field, so its body
        # needs no analogous guard.)
        if self.version is not None and self.workflow_version is not None:
            raise ValueError(
                "set at most one of `version` (re-run a historical version) or "
                "`workflow_version` (assert the current version has not drifted); "
                "they are orthogonal and contradictory together"
            )
        return self


class WorkflowActionReplace(WorkflowAction):
    """Update-side variant (§2.2): optional-at-create fields are REQUIRED, so
    a partial action 422s instead of silently flipping a pin to float, nulling
    the template, or dropping vault bindings. Explicit null/[] are explicit."""

    workflow_version: int | None = Field(ge=1)
    version: int | None = Field(ge=1)
    input_template: Any
    vault_ids: list[str]


TriggerAction = Annotated[
    SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction,
    Field(discriminator="kind"),
]
TriggerActionReplace = Annotated[
    SandboxCommandActionReplace | WakeOwnerAction | WakeSessionAction | WorkflowActionReplace,
    Field(discriminator="kind"),
]

# Module-level adapters for the query/runner read path. Structure-only —
# they must accept every row the write path ever accepted, so they carry NO
# cron occurrence check and NO input_template byte bound (write-side only;
# see _validate_input_template_bound).
TRIGGER_SOURCE_ADAPTER: TypeAdapter[
    CronSource | OneShotSource | RunCompletionSource | ExternalEventSource
] = TypeAdapter(TriggerSource)
TRIGGER_ACTION_ADAPTER: TypeAdapter[
    SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction
] = TypeAdapter(TriggerAction)


def _validate_input_template_bound(action: Any) -> None:
    """WRITE-PATH ONLY (``TriggerCreate`` / ``TriggerUpdate``) — never on read
    models or the query-layer TypeAdapters.

    Serialized-byte bounds are not jsonb-round-trip stable (numeric
    normalization can expand a written template ~50x: ``1e+308`` is 6 chars on
    the wire and 309 digits back from jsonb), so this must never run on read.
    ``allow_nan=False`` measures exactly what jsonb will accept — a
    NaN/Infinity template becomes a 422 here instead of a 500 at INSERT.
    """
    if not isinstance(action, WorkflowAction):
        return
    n = len(
        json.dumps(
            action.input_template, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    )
    if n > MAX_INPUT_TEMPLATE_BYTES:
        raise ValueError(f"input_template serializes to {n} bytes (max {MAX_INPUT_TEMPLATE_BYTES})")


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
    def _validate_write_path(self) -> TriggerCreate:
        if isinstance(self.source, CronSource):
            _validate_cron_expression(self.source.schedule)
        _validate_input_template_bound(self.action)
        return self


class TriggerUpdate(BaseModel):
    """Update body. ``source`` / ``action`` are replaced WHOLESALE when
    provided (a cron↔one-shot or sandbox↔wake conversion is just a
    different object) — via the Replace union variants, whose
    optional-at-create fields are required so a partial object 422s instead
    of silently re-defaulting. ``None`` = leave alone; there is no
    clear-to-null (both columns are NOT NULL). The next_fire / cap /
    past-fire_at business rules are enforced in the service layer (§2.4).
    """

    model_config = ConfigDict(extra="forbid")
    source: TriggerSourceReplace | None = None
    action: TriggerActionReplace | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_write_path(self) -> TriggerUpdate:
        if isinstance(self.source, CronSource):
            _validate_cron_expression(self.source.schedule)
        _validate_input_template_bound(self.action)
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


class TriggerCreated(TriggerEcho):
    """Create/update response — the trigger echo plus a one-time ``ingest_token``.

    Subclasses :class:`TriggerEcho` so every existing read-field caller keeps
    working; adds ``ingest_token``, the plaintext ingest secret surfaced
    EXACTLY ONCE for ``external_event`` sources (mint at create, re-mint on a
    source-replace TO ``external_event`` = rotation), ``None`` otherwise. The
    plaintext is never persisted and can never be re-read — losing it means
    rotating via ``update_trigger``. The full ingress URL
    (``POST /v1/triggers/ingest/{ingest_token}``) is derivable client-side and
    is deliberately not stored.
    """

    ingest_token: str | None = None


class AccountTriggerEcho(BaseModel):
    """Account-wide read view of a trigger — the liveness-audit projection.

    Backs the account-scoped ``list_account_triggers`` read tool (#1673), the
    filed blocking precondition for the ops-agent O7 trigger-liveness auditor.
    Unlike :class:`TriggerEcho` (session-scoped, echoed on ``Session.triggers``)
    this carries the ``owner_session_id`` — the account-wide sweep needs to name
    *which* session owns each trigger, and correlate a zombie
    (``enabled=true, next_fire=NULL``) back to its cron.

    ``source_kind`` is the discriminator text only (``cron`` / ``one_shot`` /
    ``run_completion`` / ``external_event``) — the auditor branches on it to
    classify a trigger as schedulable (``cron`` → ``next_fire`` MUST be non-null)
    vs reactive/one-shot (``run_completion`` / ``external_event`` → exempt from
    the non-null invariant). The full source spec is deliberately not projected:
    the liveness predicate needs the kind, not the schedule payload.
    """

    id: str
    name: str
    owner_session_id: str
    source_kind: str
    enabled: bool
    next_fire: datetime | None
    last_fire_status: TriggerFireStatus | None
    consecutive_failures: int


class TriggerRunEcho(BaseModel):
    """Read view of one ``trigger_runs`` row — a single fire of a trigger.

    ``trigger_context`` echoes the firing source (``cron`` / ``one_shot`` /
    ``run_completion`` / ``external_event``); ``event`` carries the per-event
    context for ``run_completion`` fires (``{run_id, workflow_id, status}``)
    and the arbitrary inbound jsonb body for ``external_event`` fires (no shape
    change — ``event`` is already arbitrary jsonb), and is ``None`` for timer
    fires. ``status`` is an open string on read (rows
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


def compute_initial_next_fire(source: TriggerSource, now: datetime) -> datetime | None:
    """Return the initial ``next_fire`` for a freshly-enabled row.

    Cron: the next slot strictly after ``now``. One-shot: ``fire_at``
    verbatim — even if in the past, the scheduler treats it as due
    immediately and fire-and-deletes on the next tick (today's semantic).
    run_completion: ``None`` — reactive rows are unschedulable by the tick BY
    PREDICATE (the claim/MIN queries' ``next_fire IS NOT NULL``) and by the
    ``triggers_run_completion_no_next_fire`` DB guard; their fires dispatch
    from the watched run's completion transaction.
    """
    if isinstance(source, RunCompletionSource | ExternalEventSource):
        return None
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
