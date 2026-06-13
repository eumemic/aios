"""The schedule_wake tool — ask the harness to wake this session at a future time.

Thin sugar over :mod:`aios.services.triggers`: creates a one-shot trigger
whose ``source`` is ``{kind: one_shot, fire_at}`` and whose ``action`` is
``{kind: wake_owner, content}``. At the configured time the scheduler claims
the row and the runner delivers ``content`` as a user-role message to this
session IN-WORKER (append + ``defer_wake``) — no sandbox, no broker secret,
and no fire-time 404 if ``wake_self`` isn't declared on the agent (the old
``sandbox_command`` + ``tool wake_self`` path's failure mode).

Accepts either ``delay_seconds`` (relative) or ``at`` (absolute, ISO 8601 or
natural language via ``dateparser``). Hard-fails unparseable strings — the
agent retries through the session log, which is the design (see
:doc:`CLAUDE.md`).

The harness retry-backoff path keeps using ``defer_wake`` directly because
that's a job-execution concern, not a user-visible trigger (no row in
``triggers``, no per-account cap, no visibility in ``aios sessions triggers
list``).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import dateparser
from pytz.exceptions import UnknownTimeZoneError

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.models.triggers import OneShotSource, TriggerCreate, WakeOwnerAction
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.tools.registry import registry


class ScheduleWakeArgumentError(AiosError):
    error_type = "schedule_wake_argument_error"
    status_code = 400


SCHEDULE_WAKE_DESCRIPTION = (
    "Schedule a future wake of this session, with a short reason that will be "
    "delivered as a user-role message at fire time. Exactly one of "
    "`delay_seconds` (relative) or `at` (absolute) must be set; `at` accepts "
    "ISO 8601 (e.g. `2026-05-23T15:00:00Z`) or natural language (e.g. "
    "`tomorrow at 9am`, `in 30 minutes`). For natural language, pass `tz` "
    "(IANA name, e.g. `America/Los_Angeles`) for interpretation context — "
    "defaults to UTC. The wake is implemented as a one-shot trigger; you can "
    "list / cancel it via the `trigger_*` tools by name."
)

SCHEDULE_WAKE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "delay_seconds": {
            "type": "integer",
            "minimum": 1,
            "description": "Seconds from now until the wake fires. Mutually exclusive with `at`.",
        },
        "at": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Absolute fire time. ISO 8601 with offset, or natural language "
                "resolved via `dateparser`. Mutually exclusive with `delay_seconds`."
            ),
        },
        "tz": {
            "type": "string",
            "minLength": 1,
            "description": (
                "IANA timezone for interpreting `at` when it lacks an explicit "
                "offset. Ignored for `delay_seconds`. Defaults to UTC."
            ),
        },
        "reason": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Short note delivered as a user-role message at wake time so you "
                "remember why you scheduled this."
            ),
        },
    },
    "required": ["reason"],
    "additionalProperties": False,
}


def _resolve_fire_at(arguments: dict[str, Any]) -> datetime:
    """Resolve the arguments to an absolute UTC fire time.

    Validates the XOR between ``delay_seconds`` and ``at``, parses ``at``
    via :mod:`dateparser` (which handles ISO 8601 natively too, so we
    don't need a separate ISO branch), clamps the result to be strictly
    in the future, and rejects fire times beyond
    ``Settings.schedule_wake_max_delay_seconds`` to bound the agent's
    set-and-forget surface.
    """
    delay_seconds = arguments.get("delay_seconds")
    at = arguments.get("at")
    tz = arguments.get("tz")
    now = datetime.now(UTC)
    max_delay = get_settings().schedule_wake_max_delay_seconds

    have_delay = delay_seconds is not None
    have_at = at is not None
    if have_delay == have_at:
        raise ScheduleWakeArgumentError("exactly one of `delay_seconds` or `at` must be provided")

    if have_delay:
        if not isinstance(delay_seconds, int) or delay_seconds < 1:
            raise ScheduleWakeArgumentError("delay_seconds must be a positive integer")
        if delay_seconds > max_delay:
            raise ScheduleWakeArgumentError(
                f"delay_seconds={delay_seconds} exceeds the max allowed ({max_delay}s "
                f"≈ {max_delay // 86400} days). Use a shorter delay or split the "
                "schedule into a recurring cron trigger if you need a long horizon."
            )
        if tz is not None:
            raise ScheduleWakeArgumentError("`tz` is not valid with `delay_seconds`")
        try:
            return now + timedelta(seconds=delay_seconds)
        except OverflowError:
            # `int` accepts arbitrarily large values; timedelta does not.
            # Caught here to surface a typed error rather than a 500.
            raise ScheduleWakeArgumentError(
                f"delay_seconds={delay_seconds} overflows the datetime range"
            ) from None

    assert have_at
    if not isinstance(at, str) or not at:
        raise ScheduleWakeArgumentError("at must be a non-empty string")
    if tz is not None and (not isinstance(tz, str) or not tz):
        raise ScheduleWakeArgumentError("tz must be a non-empty string when provided")

    try:
        parsed: datetime | None = dateparser.parse(
            at,
            settings={
                "TIMEZONE": tz or "UTC",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",
            },
        )
    except UnknownTimeZoneError:
        # An unresolvable `tz` makes dateparser (via pytz) raise this. It is
        # NOT an AiosError, so without this guard the tool dispatcher would
        # classify it as a server fault and evict the session container for
        # what is purely a bad model input. Convert it to the client-class
        # rejection, symmetric with the other argument errors (and mirroring
        # the OverflowError handling on the delay path). We defer to
        # dateparser's own verdict rather than pre-validating with a stdlib
        # ZoneInfo check — dateparser accepts names ZoneInfo rejects (the
        # case-insensitive 'utc', fixed-offset 'UTC+5'), so pre-validation
        # would over-reject valid inputs.
        raise ScheduleWakeArgumentError(
            f"unknown timezone {tz!r} — use an IANA name like 'America/New_York' or 'UTC'"
        ) from None
    if parsed is None:
        raise ScheduleWakeArgumentError(
            f"could not parse `at` value {at!r} — use ISO 8601 (e.g. "
            "'2026-05-23T15:00:00Z') or natural language (e.g. 'tomorrow at 9am'). "
            "Pass `tz` for natural-language times in a specific timezone."
        )
    parsed_utc = parsed.astimezone(UTC)
    if parsed_utc <= now:
        raise ScheduleWakeArgumentError(
            f"resolved fire time {parsed_utc.isoformat()} is not in the future"
        )
    if (parsed_utc - now).total_seconds() > max_delay:
        raise ScheduleWakeArgumentError(
            f"resolved fire time {parsed_utc.isoformat()} is more than "
            f"{max_delay}s ({max_delay // 86400} days) in the future, which "
            "exceeds the max allowed. Use a shorter offset or split into a "
            "recurring cron trigger if you need a long horizon."
        )
    return parsed_utc


def _wake_content(reason: str) -> str:
    """The user-role marker delivered at fire time.

    Byte-identical to the string the prior ``sandbox_command`` path passed
    to ``tool wake_self`` — the new ``wake_owner`` action delivers it
    directly via the in-worker self-delivery path (#818).
    """
    return f"[Your scheduled wake fired. Reason: {reason}]"


def _make_trigger_name() -> str:
    """Generate a unique-per-session name for the auto-created trigger.

    Uses a short ULID-ish suffix that fits the ``[A-Za-z0-9_-]*`` name
    constraint enforced by :class:`TriggerCreate`.
    """
    return f"wake-{uuid.uuid4().hex[:12]}"


async def schedule_wake_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    reason = arguments.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ScheduleWakeArgumentError("reason must be a non-empty string")

    fire_at = _resolve_fire_at(arguments)

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)

    spec = TriggerCreate(
        name=_make_trigger_name(),
        source=OneShotSource(fire_at=fire_at),
        action=WakeOwnerAction(content=_wake_content(reason)),
        # Tag for human-friendly rendering in the CLI's triggers listing
        # (`wake: <reason>` instead of the underlying action).
        metadata={"kind": "wake", "reason": reason},
    )

    echo = await triggers_service.add_trigger(pool, session_id, spec, account_id=account_id)
    return {
        "scheduled": True,
        "trigger_id": echo.id,
        "name": echo.name,
        "fire_at": fire_at.isoformat(),
        "reason": reason,
    }


def _register() -> None:
    registry.register(
        name="schedule_wake",
        description=SCHEDULE_WAKE_DESCRIPTION,
        parameters_schema=SCHEDULE_WAKE_PARAMETERS_SCHEMA,
        handler=schedule_wake_handler,
        transport="agent_tool",
    )


_register()
