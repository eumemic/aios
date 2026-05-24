"""The schedule_wake tool — ask the harness to wake this session at a future time.

Thin wrapper over :mod:`aios.services.scheduled_tasks`: creates a one-shot
``scheduled_tasks`` row whose bash command invokes the ``tool wake_self``
CLI, which delivers a user-role message to the session via the broker
(authenticating through the sandbox's ``TOOL_BROKER_URL`` /
``TOOL_BROKER_SECRET`` env vars, so no secret lands in the command string).
At the configured time, the scheduler claims the row, the runner fires the
bash, the broker appends the marker, and the model wakes naturally.

Accepts either ``delay_seconds`` (relative) or ``at`` (absolute, ISO 8601
or natural language via ``dateparser``). Hard-fails unparseable strings —
the agent retries through the session log, which is the design (see
:doc:`CLAUDE.md`).

Replaces the prior ``defer_wake(delay_seconds=…)``-based implementation;
the harness retry-backoff path keeps using ``defer_wake`` directly
because that's a job-execution concern, not a user-visible scheduled
task (no row in ``session_scheduled_tasks``, no per-account cap, no
visibility in ``aios scheduled-tasks list``).
"""

from __future__ import annotations

import json
import shlex
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import dateparser

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.models.scheduled_tasks import ScheduledTaskCreate
from aios.services import scheduled_tasks as scheduled_tasks_service
from aios.services import sessions as sessions_service
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
    "defaults to UTC. The wake is implemented as a one-shot scheduled task; "
    "you can list / cancel it via the `schedule_task_*` tools by name."
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
                "schedule into a recurring cron task if you need a long horizon."
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

    parsed: datetime | None = dateparser.parse(
        at,
        settings={
            "TIMEZONE": tz or "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
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
            "recurring cron task if you need a long horizon."
        )
    return parsed_utc


def _build_wake_bash(reason: str) -> str:
    """Generate the bash one-liner the scheduled fire will execute.

    Invokes the in-sandbox ``tool wake_self`` CLI with a JSON-encoded
    ``content`` argument. This keeps the broker secret out of the
    command string and uses the canonical self-wake primitive (#703).
    The JSON payload is shell-escaped so arbitrary characters in the
    reason can't break out of the argument.
    """
    payload = json.dumps(
        {"content": f"[Your scheduled wake fired. Reason: {reason}]"},
        ensure_ascii=False,
    )
    # Requires `wake_self` to be declared on the agent; an undeclared tool
    # yields a fire-time 404 surfaced through the scheduled-task runner.
    return f"tool wake_self {shlex.quote(payload)}"


def _make_task_name() -> str:
    """Generate a unique-per-session name for the auto-created task.

    Uses a short ULID-ish suffix that fits the ``[A-Za-z0-9_-]*`` name
    constraint enforced by :class:`ScheduledTaskCreate`.
    """
    return f"wake-{uuid.uuid4().hex[:12]}"


async def schedule_wake_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    reason = arguments.get("reason")
    if not isinstance(reason, str) or not reason:
        raise ScheduleWakeArgumentError("reason must be a non-empty string")

    fire_at = _resolve_fire_at(arguments)

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)

    spec = ScheduledTaskCreate(
        name=_make_task_name(),
        fire_at=fire_at,
        command=_build_wake_bash(reason),
        # `tool wake_self` finishes in seconds; tight bound keeps stuck fires from
        # tying up worker slots.
        timeout_seconds=30,
        # Tag for human-friendly rendering in the CLI's
        # scheduled-tasks listing (`wake: <reason>` instead of the
        # full bash command).
        metadata={"kind": "wake", "reason": reason},
    )

    echo = await scheduled_tasks_service.add_task(pool, session_id, spec, account_id=account_id)
    return {
        "scheduled": True,
        "task_id": echo.id,
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
