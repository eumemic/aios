"""Explicit goal-management model tools (#1508) over the obligation/quiescence gate.

Terminology of record (unify-obligations epic #1516, child #1523)
================================================================
A **GOAL is the reflexive call**: a self-referential **awaited** obligation whose
``caller == servicer == self`` (#1123 ``request_opened`` with
``caller={kind:"session", id:<this session>}``, ``awaited=true``, target = the same
session). It is a *label* for that one shape of obligation, **not a subsystem** —
there is no separate goal store, gate, or lifecycle. The quiescence guard refuses
to let a session quiesce past any open awaited obligation, self-goals included.

Because a goal is just an obligation with a self-caller, the **uniform,
source-agnostic obligation verbs** already cover its whole lifecycle, and they —
not goal-named duplicates — are the canonical verbs of record:

* **OPEN** a goal → ``create_goal`` (the one reflexive-OPEN verb; see below).
* **CLOSE** a goal → ``return``/``error`` (servicer authority, schema-validated):
  ``return(request_id=<goal_id>, value=…)`` completes it (the session's persisted
  ``output_schema`` is enforced servicer-side by ``return``'s own schema gate,
  ``workflow_completion._enforce_output_schema`` → ``_validate_value``, so a
  non-conforming value is rejected with ``output_schema_violation`` and the goal
  stays open) and ``error(request_id=<goal_id>, message=…)`` abandons it.
* **LIST** open goals → ``list_obligations`` (origin=self) and/or ``list_calls``
  (origin=self) — the general open-obligation enumeration filtered to self-caller
  edges.
* **DROP/CANCEL** a goal → ``cancel_call`` (caller authority, by ``tool_call_id``;
  epic child #5). There is no ``cancel_goal``.

``create_goal`` is therefore the ONLY goal-named verb in the surface. There is **no**
``set_goal``, **no** ``cancel_goal``, **no** ``complete_goal``/``fail_goal``
(retired #1518/#1525), **no** ``list_goals`` long-term (folds into
``list_obligations``/``list_calls``, child #3), and **no** silent ``update_goal``.
Revision stays explicit: ``error`` the goal, then ``create_goal`` a revised one (the
value is goals that *don't move* — the "invent a 7% tolerance mid-flight"
anti-pattern).

Declined: #1414 ``set_goal``/``cancel_goal``
--------------------------------------------
Open issue #1414 ("set_goal / cancel_goal: self-issued awaited-request 'goals' for
always-on agents") proposed a ``set_goal``/``cancel_goal`` naming. The
chairman-ratified decision in epic #1516 is to **keep ``create_goal`` and NOT rename
to ``set_goal``**, and to **NOT add ``cancel_goal``** — dropping a goal is the
uniform ``cancel_call`` (child #5). This module is the terminology of record for that
decision so #1414's naming is not re-litigated; #1414 may be closed as
resolved-by-#1516 by the seat/chairman.

The mechanism (unchanged)
-------------------------
The *gating* already exists and works — an open awaited obligation drives the
quiescence-guard nudge / ``no_return`` loop (``db/queries/sessions.py`` +
``harness/context.py`` ``_agent_owes_response``) and renders in the "Open
obligations" tail block (``harness/obligations.py``). The ONLY way to open a
self-goal used to be the cryptic ``call_session(session_id=<its own id>, input=…)``
awaited self-call (#1414) — undiscoverable, awkward, easy to forget; ``create_goal``
is the explicit, first-class surface over that SAME mechanism (it does NOT reinvent
the gate):

* ``create_goal(goal, output_schema)`` — open a self-goal: write the same
  self-referential awaited edge a ``call_session(self)`` opens (reusing the #1414
  self-goal path via ``sessions_service.invoke``), so the quiescence guard holds
  the session to it automatically. ``output_schema`` is **REQUIRED** (#1512): a JSON
  Schema expressing the checkable acceptance criteria as the shape the completion
  ``value`` must satisfy, persisted on the ``request_opened`` frame the same way
  ``call_*`` carry ``output_schema``. There is no schemaless goal. Returns a
  ``goal_id`` (the ``request_id`` of the opened edge). Enforces the per-session
  open-goal admission cap (``Settings.session_open_goals_max``) with a clear error
  on exceed.
Unlike the parking ``call_*`` builtins, ``create_goal`` does NOT park on the
edge — a self-goal's servicer IS the session, so parking would deadlock. The tool
opens the edge and returns immediately; the quiescence guard (not a park) is what
holds the session.

All register ``transport="agent_tool"`` (model-only; the CLI broker refuses them).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.config import get_settings
from aios.db import queries
from aios.harness import runtime
from aios.models.sessions import Obligation
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry

# ─── argument models ─────────────────────────────────────────────────────────


class _CreateGoalArgs(BaseModel):
    """``create_goal`` arguments. ``extra="forbid"`` so a smuggled
    ``caller``/``account_id`` (the trusted self-identity is the executing session
    the harness supplies, never a field here) is rejected before the handler runs."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        min_length=1,
        description="The definition-of-done you are pinning — what 'done' means for "
        "this goal. You will be held to it (you cannot go idle until you close it "
        "with `return` or `error`, using its goal_id as the request_id).",
    )
    output_schema: dict[str, Any] = Field(
        description="REQUIRED JSON Schema expressing the concrete, checkable "
        "acceptance criteria as the shape the completion `value` must satisfy. "
        "Closing the goal with `return` validates its `value` against this schema (a "
        "non-conforming value is rejected with output_schema_violation), so 'done' is "
        "a contract fixed up front, not prose. There is no schemaless goal — every "
        "goal declares a checkable completion contract.",
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
    and are NOT enumerated through the goal surface."""
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
    ``goal_id`` (the same id passed as ``request_id`` to ``return``/``error`` to
    close it).
    """
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CreateGoalArgs, arguments)

    # Admission cap: count THIS session's currently-open self-goals (concurrency,
    # not a lifetime budget — closing a goal with return/error frees slots).
    # Read-then-write under the same pool; a benign race past the cap only
    # over-admits by a small constant, which the obligations-tail render cap
    # (MAX_RENDERED_OBLIGATIONS) still bounds.
    cap = get_settings().session_open_goals_max
    open_goals = await _open_self_goals(pool, session_id, account_id=account_id)
    if len(open_goals) >= cap:
        return ToolResult(
            content=(
                f"open-goal cap reached ({len(open_goals)}/{cap}): close an existing goal "
                "with `return` or `error` (using its goal_id as request_id) before "
                "creating another."
            ),
            is_error=True,
        )

    # The goal text becomes the request input (the definition-of-done preview the
    # tail block renders); the REQUIRED output_schema becomes the completion contract,
    # persisted on the same request_opened frame the way call_* carry output_schema
    # (#1512) — `return` validates its value against it servicer-side.
    goal_input: dict[str, Any] = {"goal": args.goal}

    handle = await sessions_service.invoke(
        pool,
        account_id=account_id,
        target_kind="session",
        target=session_id,  # the target IS this session — a self-goal (#1414).
        input=goal_input,
        output_schema=args.output_schema,
        caller={"kind": "session", "id": session_id},
    )
    return {
        "goal_id": handle.request_id,
        "goal": args.goal,
        "status": "open",
    }


# ─── descriptions + registration ─────────────────────────────────────────────

CREATE_GOAL_DESCRIPTION = (
    "Self-assign a goal: pin a definition-of-done up front that you are then held "
    "to. Opens a self-goal that the quiescence guard refuses to let you go idle "
    "past — you stay re-prompted until you close it. `goal` is what 'done' means in "
    "words; `output_schema` (REQUIRED, a JSON Schema) is the checkable completion "
    "contract — the shape the closing value must satisfy. Returns a `goal_id`. Close "
    "the goal with the general answer verbs using that goal_id as the request_id: "
    "`return(request_id=<goal_id>, value=...)` when done (the value is validated "
    "against output_schema — a non-conforming value is rejected and the goal stays "
    "open), or `error(request_id=<goal_id>, message=...)` to abandon it. List your "
    "open goals with `list_obligations`/`list_calls` (origin=self) and drop one with "
    "`cancel_call` (by its tool_call_id) — there is no separate goal-list or "
    "goal-cancel verb. This is the highest-leverage way to actually close a task: "
    "declare a checkable 'done' up front, then you can't quiesce until a conforming "
    "result proves it."
)


def _register() -> None:
    registry.register(
        name="create_goal",
        description=CREATE_GOAL_DESCRIPTION,
        parameters_schema=_CreateGoalArgs.model_json_schema(),
        handler=create_goal_handler,
        transport="agent_tool",
    )


_register()
