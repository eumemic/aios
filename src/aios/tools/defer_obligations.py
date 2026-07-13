"""``defer_obligations`` — an ACTIVE-WAIT: hold this turn open, un-nudged (#1533).

An agent that owes an obligation (an open awaited ``request_opened`` edge) and
knows it must wait before answering has no clean way to do so: ending the turn
while owing and otherwise idle trips the quiescence guard
(``services/sessions.py::append_assistant_and_guard_quiescence``) — a nudge per
turn, then auto-``no_return`` past ``REQUEST_NUDGE_BUDGET``.

The fix is deliberately NOT a new mechanism. ``defer_obligations`` is an
ordinary in-flight tool call — exactly like an awaited ``call_workflow`` or a
slow ``bash`` — that simply takes ``duration_seconds`` to return. While it is
open the session is ACTIVE (``open_tool_call_count > 0``), so:

* **obligations stay open + un-nudged by inaction** — the guard's gate-1
  (``tool_calls`` present) returns before the nudge loop is even consulted, and
  its gate-3 (``derive_session_status != "idle"``) holds while the call is
  in flight. Session-wide by construction: nothing is done to any obligation.
  NO per-obligation ``request_id`` (rejected by the settled design).
* **early-resolve-on-stimulus is inherited** — the handler parks on the
  session's own event channel (the ``_await_session`` triad:
  :func:`open_listen_for_events` + :func:`await_completion`) with ``duration``
  as the await timeout; any inbound stimulus (``append_event`` bumps
  ``last_stimulus_seq`` and NOTIFYs the channel) resolves the wait early.
* **crash-recovery is inherited** — ``resumable=True`` puts the tool in
  ``registry.resumable_tool_names()``; a ghosted defer re-derives its servicer
  via ``find_parked_servicer``, finds none (a defer writes no servicer edge),
  and lands in the retryable ``launch_lost`` branch. Truncating the wait is
  correct because a wait is not a side effect: the obligations are still open
  and the model can simply re-defer on resume.

Bound validation reuses :func:`aios.tools.schedule_wake._resolve_fire_at`
verbatim (the ``>= 1`` floor, the ``schedule_wake_max_delay_seconds`` ceiling,
the typed :class:`~aios.tools.schedule_wake.ScheduleWakeArgumentError`) — one
bound, never a second nudge budget.

Registers ``transport="agent_tool"`` (model-only), arg model ``extra="forbid"``
(a smuggled ``caller``/``account_id`` is rejected before the handler runs),
identity = the harness-supplied executing session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.db import queries
from aios.db.listen import open_listen_for_events
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services.await_completion import await_completion
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry
from aios.tools.schedule_wake import _resolve_fire_at


class _DeferObligationsArgs(BaseModel):
    """Session-wide: ONE duration, no ``request_id`` (per-obligation targeting is
    rejected by the settled #1533 design). ``extra="forbid"`` so a smuggled
    ``caller``/``account_id``/``request_id`` is rejected before the handler runs;
    the bounds themselves are enforced by the shared ``schedule_wake`` resolver,
    not re-implemented here."""

    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(
        description=(
            "How long to hold this turn open, in seconds (>= 1, bounded by the "
            "configured schedule_wake maximum). The wait returns early if any "
            "new message arrives first."
        ),
    )


async def defer_obligations_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    """Hold this tool call in flight for up to ``duration_seconds``.

    Mirrors :func:`aios.services.sessions.await_session` (the ``_await_session``
    triad), swapping the done-predicate to "an inbound stimulus arrived":

    1. Parse + clamp via the shared ``schedule_wake._resolve_fire_at`` (the
       ``>= 1`` floor, the ceiling, the typed error) and convert the absolute
       fire time back to a remaining-seconds timeout.
    2. Subscribe to the session's own event channel BEFORE reading the baseline
       (the LISTEN-before-read invariant: a stimulus landing between the
       baseline read and the park is already queued, so it can't be missed).
    3. Capture ``baseline = last_stimulus_seq`` — inbound stimuli only, so the
       wait wakes on a real message, not the span bookkeeping this very step
       writes (the channel NOTIFYs on *every* append).
    4. ``await_completion`` until ``last_stimulus_seq > baseline`` or the
       remaining duration elapses.
    """
    try:
        args = _DeferObligationsArgs.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc

    # Shared bound validation (reused verbatim, NOT re-implemented): the >= 1
    # floor, the schedule_wake_max_delay_seconds ceiling, and the typed
    # ScheduleWakeArgumentError all come from schedule_wake's resolver.
    fire_at = _resolve_fire_at({"delay_seconds": args.duration_seconds})
    remaining = (fire_at - datetime.now(UTC)).total_seconds()

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)

    async def _read_last_stimulus_seq() -> int:
        async with pool.acquire() as conn:
            wm = await queries.read_session_watermarks(conn, session_id, account_id=account_id)
            return wm[1] if wm is not None else 0  # last_stimulus_seq

    # ``on_connected=None`` deliberately OMITS the #81 subscriber advisory lock:
    # this park consumes only the stimulus watermark, never streaming deltas —
    # same stance as ``await_session`` / ``open_listen_for_run_events``.
    subscription = await open_listen_for_events(
        get_settings().db_url, session_id, on_connected=None
    )
    try:
        baseline = await _read_last_stimulus_seq()
        last_seq = await await_completion(
            subscription.queue,
            read_state=_read_last_stimulus_seq,
            is_done=lambda seq: seq > baseline,
            timeout_seconds=remaining,
        )
    finally:
        subscription.terminate()

    return {
        "deferred": True,
        "resolved": "stimulus" if last_seq > baseline else "duration_elapsed",
    }


DEFER_OBLIGATIONS_DESCRIPTION = (
    "Hold your current turn open for up to `duration_seconds` without being "
    "nudged about obligations you still owe. Returns early the moment any new "
    "message arrives — including a wake you scheduled yourself. `defer` means "
    "hold, not abandon: your obligations stay open and you answer them when "
    "this returns (`resolved` is `duration_elapsed` or `stimulus`; on a "
    "stimulus, read the new message and act). If the worker restarts mid-wait, "
    "the wait ends early with a retryable note."
)


def _register() -> None:
    # ``resumable=True``: the handler is a pure await (no side effects, no
    # servicer edge), so a crashed in-flight defer is safe for the ghost-repair
    # sweep's resumable branch — ``find_parked_servicer`` finds no edge and the
    # call lands in the retryable ``launch_lost`` result (#1533 crash semantic).
    registry.register(
        name="defer_obligations",
        description=DEFER_OBLIGATIONS_DESCRIPTION,
        parameters_schema=_DeferObligationsArgs.model_json_schema(),
        handler=defer_obligations_handler,
        transport="agent_tool",
        resumable=True,
    )


_register()
