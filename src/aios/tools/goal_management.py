"""Explicit goal-management model tools (#1508) over the obligation/quiescence gate.

A **self-goal** is a session's first-class definition-of-done that it is then held
to until met: a self-referential **awaited** obligation (#1123 ``request_opened``
with ``caller={kind:"session", id:<this session>}``, ``awaited=true``) that the
quiescence guard refuses to let the session quiesce past. The *gating* already
exists and works — an open awaited obligation drives the quiescence-guard
nudge / ``no_return`` loop (``db/queries/sessions.py`` + ``harness/context.py``
``_agent_owes_response``), and at the quiescence attempt the nudge surfaces the
outstanding obligation + its acceptance contract (#1514; ``harness/obligations.py``
keeps only the shared formatting helpers since the per-step tail block was removed).
The ONLY way to open a self-goal used to be the
cryptic ``call_session(session_id=<its own id>, input=…)`` awaited self-call
(#1414) — undiscoverable, awkward, easy to forget.

These four tools are the **explicit, first-class surface** over that EXISTING
mechanism (they do NOT reinvent the gate):

* ``create_goal(goal, acceptance_criteria?)`` — open a self-goal: write the same
  self-referential awaited edge a ``call_session(self)`` opens (reusing the #1414
  self-goal path via ``sessions_service.invoke``), so the quiescence guard holds
  the session to it automatically. Returns a ``goal_id`` (the ``request_id`` of
  the opened edge). Enforces the per-session open-goal admission cap
  (``Settings.session_open_goals_max``) with a clear error on exceed.
* ``list_goals()`` — enumerate the session's OPEN self-goals (the open-obligation
  set filtered to self-caller edges), each with ``goal_id``, ``goal`` text, and
  ``age``.
* ``complete_goal(goal_id, evidence?)`` — close a goal as DONE (the ``return``
  arm via ``respond_to_request``); ``evidence`` is recorded on the response value.
* ``fail_goal(goal_id, reason)`` — close a goal as abandoned (the ``error`` arm),
  with a reason.

Deliberately **no silent ``update_goal``** — the value is goals that *don't move*
(the "invent a 7% tolerance mid-flight" anti-pattern). Revision is an explicit
``fail_goal`` + ``create_goal``.

Unlike the parking ``call_*`` builtins, ``create_goal`` does NOT park on the
edge — a self-goal's servicer IS the session, so parking would deadlock. The tool
opens the edge and returns immediately; the quiescence guard (not a park) is what
holds the session. ``complete_goal``/``fail_goal`` write the ``request_response``
half via the shared ``respond_to_request`` core (the same writer ``return``/
``error`` use), closing the obligation so the session may quiesce.

All register ``transport="agent_tool"`` (model-only; the CLI broker refuses them).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.db import queries
from aios.harness import runtime
from aios.harness.obligations import _format_age
from aios.models.sessions import Obligation
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry
from aios.tools.workflow_completion import respond_to_request

# ─── argument models ─────────────────────────────────────────────────────────


class _CreateGoalArgs(BaseModel):
    """``create_goal`` arguments. ``extra="forbid"`` so a smuggled
    ``caller``/``account_id`` (the trusted self-identity is the executing session
    the harness supplies, never a field here) is rejected before the handler runs."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        min_length=1,
        description="The definition-of-done you are pinning — what 'done' means for "
        "this goal. You will be held to it (you cannot go idle until you "
        "complete_goal or fail_goal it).",
    )
    acceptance_criteria: str | None = Field(
        default=None,
        description="Optional explicit acceptance criteria — the concrete, checkable "
        "conditions that must hold for the goal to count as met.",
    )


class _ListGoalsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _CompleteGoalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(
        description="The id of the open goal to close as done (from create_goal or list_goals)."
    )
    evidence: str | None = Field(
        default=None,
        description="Optional evidence that the goal's acceptance criteria are met "
        "(recorded on the closing response).",
    )


class _FailGoalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(
        description="The id of the open goal to abandon (from create_goal or list_goals)."
    )
    reason: str = Field(
        min_length=1,
        description="Why the goal is being abandoned rather than completed.",
    )


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


# ─── shared helpers ──────────────────────────────────────────────────────────


def _is_self_goal(obligation: Obligation, *, session_id: str) -> bool:
    """A self-goal is a self-referential awaited obligation: a ``session`` caller
    that is the session ITSELF (#1414). The other open obligations (``api`` /
    ``run`` / a peer ``session`` caller) are caller-assigned tasks, not self-goals,
    and are NOT enumerated or closeable through the goal surface."""
    return obligation.caller_kind == "session" and obligation.caller_id == session_id


async def _open_self_goals(pool: Any, session_id: str, *, account_id: str) -> list[Obligation]:
    """The session's still-open self-goals, oldest-first — the open-obligation set
    (``get_open_obligations``) filtered to self-caller edges."""
    async with pool.acquire() as conn:
        obligations = await queries.get_open_obligations(conn, session_id, account_id=account_id)
    return [o for o in obligations if _is_self_goal(o, session_id=session_id)]


# ─── handlers ────────────────────────────────────────────────────────────────


async def create_goal_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    """Open a self-goal — the same self-referential awaited edge a
    ``call_session(self)`` opens (#1414), via ``sessions_service.invoke`` with
    ``target_kind="session"`` and ``caller={kind:"session", id:<this session>}``.

    Enforces the per-session open-goal admission cap BEFORE writing the edge (a
    clear tool error on exceed — no obligation opened). Does NOT park (the servicer
    is the session itself; parking would deadlock) — the quiescence guard holds the
    session to the obligation. Returns the opened edge's ``request_id`` as
    ``goal_id``.
    """
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CreateGoalArgs, arguments)

    # Admission cap: count THIS session's currently-open self-goals (concurrency,
    # not a lifetime budget — complete_goal/fail_goal free slots). Read-then-write
    # under the same pool; a benign race past the cap only over-admits by a small
    # constant (the session_open_goals_max ceiling still bounds the quiescence-nudge
    # that lists every outstanding obligation + its acceptance contract, #1514).
    cap = get_settings().session_open_goals_max
    open_goals = await _open_self_goals(pool, session_id, account_id=account_id)
    if len(open_goals) >= cap:
        return ToolResult(
            content=(
                f"open-goal cap reached ({len(open_goals)}/{cap}): close an existing goal "
                "with complete_goal or fail_goal before creating another."
            ),
            is_error=True,
        )

    # The goal text + optional acceptance criteria become the request input (the
    # definition-of-done the obligation carries; the tail block renders a preview).
    goal_input: dict[str, Any] = {"goal": args.goal}
    if args.acceptance_criteria is not None:
        goal_input["acceptance_criteria"] = args.acceptance_criteria

    handle = await sessions_service.invoke(
        pool,
        account_id=account_id,
        target_kind="session",
        target=session_id,  # the target IS this session — a self-goal (#1414).
        input=goal_input,
        caller={"kind": "session", "id": session_id},
    )
    return {
        "goal_id": handle.request_id,
        "goal": args.goal,
        "status": "open",
    }


async def list_goals_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    """Enumerate the session's OPEN self-goals (oldest-first), each with
    ``goal_id``, ``goal`` text (the request summary), and a terse ``age``."""
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    _parse(_ListGoalsArgs, arguments)  # reject smuggled keys
    open_goals = await _open_self_goals(pool, session_id, account_id=account_id)
    now = datetime.now(UTC)
    goals = [
        {
            "goal_id": o.request_id,
            "goal": o.summary or "",
            "age": _format_age(o.opened_at, now),
        }
        for o in open_goals
    ]
    return {"goals": goals}


async def _close_goal(
    session_id: str,
    *,
    goal_id: str,
    is_error: bool,
    result: Any,
    error: dict[str, Any] | None,
) -> dict[str, Any] | ToolResult:
    """Close a self-goal by writing the ``request_response`` half via the shared
    :func:`respond_to_request` core (the same exactly-once writer ``return``/
    ``error`` use). Rejects a ``goal_id`` that isn't one of THIS session's open
    self-goals — so the goal surface can't be used to answer a caller-assigned
    (``api``/``run``/peer-``session``) obligation, which must go through
    ``return``/``error``.
    """
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    open_goals = await _open_self_goals(pool, session_id, account_id=account_id)
    if goal_id not in {o.request_id for o in open_goals}:
        return ToolResult(
            content=(
                "no open goal with that goal_id — list_goals shows your open goals "
                "(only self-goals are closeable here)."
            ),
            is_error=True,
        )
    status = await respond_to_request(
        pool,
        session_id,
        request_id=goal_id,
        is_error=is_error,
        result=result,
        error=error,
    )
    if status not in ("responded", "duplicate"):
        # The pre-check already filtered to open self-goals; a non-success here is a
        # genuine race (the obligation closed underneath us). Surface it so the model
        # re-reads with list_goals.
        return ToolResult(
            content=f"could not close goal {goal_id} ({status}) — it may already be closed.",
            is_error=True,
        )
    return {"goal_id": goal_id, "status": "failed" if is_error else "completed"}


async def complete_goal_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    """Close a goal as DONE (the ``return`` arm). ``evidence`` is recorded on the
    closing response value."""
    args = _parse(_CompleteGoalArgs, arguments)
    result: dict[str, Any] = {"completed": True}
    if args.evidence is not None:
        result["evidence"] = args.evidence
    return await _close_goal(
        session_id, goal_id=args.goal_id, is_error=False, result=result, error=None
    )


async def fail_goal_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    """Close a goal as ABANDONED (the ``error`` arm), with a reason."""
    args = _parse(_FailGoalArgs, arguments)
    return await _close_goal(
        session_id,
        goal_id=args.goal_id,
        is_error=True,
        result=None,
        error={"message": args.reason},
    )


# ─── descriptions + registration ─────────────────────────────────────────────

CREATE_GOAL_DESCRIPTION = (
    "Self-assign a goal: pin a definition-of-done up front that you are then held "
    "to. Opens a self-goal that the quiescence guard refuses to let you go idle "
    "past — you stay re-prompted until you complete_goal or fail_goal it. `goal` is "
    "what 'done' means; optionally add explicit `acceptance_criteria`. Returns a "
    "`goal_id`. This is the highest-leverage way to actually close a task: declare "
    "done up front, then you can't quiesce until it's met."
)
LIST_GOALS_DESCRIPTION = (
    "List your open self-goals (the goals you've pinned with create_goal that you "
    "haven't yet completed or failed), each with its goal_id, goal text, and age."
)
COMPLETE_GOAL_DESCRIPTION = (
    "Close one of your open goals as DONE — its definition-of-done is met. `goal_id` "
    "is the id from create_goal or list_goals; optionally record `evidence` that the "
    "acceptance criteria are satisfied. Closing the last open goal lets you quiesce."
)
FAIL_GOAL_DESCRIPTION = (
    "Abandon one of your open goals — you will not meet its definition-of-done. "
    "`goal_id` is the id from create_goal or list_goals; `reason` says why. Use this "
    "(then optionally create_goal a revised goal) instead of silently moving the "
    "goalposts: goals don't move, revision is explicit."
)


def _register() -> None:
    registry.register(
        name="create_goal",
        description=CREATE_GOAL_DESCRIPTION,
        parameters_schema=_CreateGoalArgs.model_json_schema(),
        handler=create_goal_handler,
        transport="agent_tool",
    )
    registry.register(
        name="list_goals",
        description=LIST_GOALS_DESCRIPTION,
        parameters_schema=_ListGoalsArgs.model_json_schema(),
        handler=list_goals_handler,
        transport="agent_tool",
    )
    registry.register(
        name="complete_goal",
        description=COMPLETE_GOAL_DESCRIPTION,
        parameters_schema=_CompleteGoalArgs.model_json_schema(),
        handler=complete_goal_handler,
        transport="agent_tool",
    )
    registry.register(
        name="fail_goal",
        description=FAIL_GOAL_DESCRIPTION,
        parameters_schema=_FailGoalArgs.model_json_schema(),
        handler=fail_goal_handler,
        transport="agent_tool",
    )


_register()
