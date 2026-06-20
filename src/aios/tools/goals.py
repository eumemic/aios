"""Self-issued **goals** for always-on agents — ``set_goal`` / ``cancel_goal`` (#1414).

A **goal** is the reflexive fixpoint of the deliver-kernel request/quiescence
machinery: a self-issued **awaited** ``request_opened`` edge where
``caller == servicer == this session``. The "must ``return()`` before you may
quiesce" obligation is the *existing* quiescence guard, unchanged. Purpose:
**continuous always-on operation** — invert "agent idle most of the time" into
"agent working toward its standing goals". A goal appears in Issue B's
obligations block with a ``self`` origin and re-wakes a do-nothing session via
the Issue A quiescence-guard nudge (the always-on heartbeat), bounded by A's
consecutive-inaction limit → ``no_return`` closes the edge → the session idles.

Two model-only tools (``transport=agent_tool``; the CLI broker refuses them):

* ``set_goal`` — the **no-park** writer. Ungated (granted in ``agent.tools`` like
  ``invoke``) so the model can set its first goal while owing nothing. Writes the
  self-edge via the service ``set_goal`` (``Ask(ExistingSession)`` with
  ``caller={kind:session, id:self}`` and the idempotent first-writer-wins edge
  write) but **without** parking — a park is an open tool-call, so it would keep
  the session non-idle and the quiescence guard would never fire (self-deadlock).
  Returns ``{goal_id: request_id}``.

* ``cancel_goal`` — structured retraction. Gated under ``owes_request`` (injected
  by ``compute_step_prelude`` exactly like ``return``/``error``). A genuinely
  distinct **third** terminal kind: ``return`` = completed, ``error`` =
  couldn't-fulfill (free-text ``{message}``), ``cancel_goal`` = retract-own-
  intention (``{kind:cancelled, by:self}``). **SECURITY:** verifies the obligation
  is a real self-goal (``caller=={kind:session, id:self}``) BEFORE writing, so a
  peer-invoke or workflow-child obligation can never be stamped ``{by:self}``.

The trusted caller id is ALWAYS the harness-supplied executing ``session_id``,
NEVER model input: both arg models are ``additionalProperties:false`` so an
injected ``caller``/``session_id``/``target`` is rejected before the handler runs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.errors import RateLimitedError
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, openai_tool_entry, registry

SET_GOAL_TOOL_NAME = "set_goal"
CANCEL_GOAL_TOOL_NAME = "cancel_goal"

SET_GOAL_DESCRIPTION = (
    "Set a standing goal for yourself: a self-issued request you owe a result to "
    "(via `return`/`error`). Use this to keep working continuously toward an "
    "objective instead of going idle — the goal stays open (and is surfaced back "
    "to you each step) until you answer or cancel it. Returns `{goal_id}` — the "
    "id you echo to `return`/`error`/`cancel_goal`."
)
CANCEL_GOAL_DESCRIPTION = (
    "Retract one of your own standing goals you no longer intend to pursue. "
    "`goal_id` is the id `set_goal` returned (also shown in your obligations). "
    "Distinct from `error` (which reports you couldn't fulfill a request): this "
    "records that you deliberately gave up the intention. Only works on goals you "
    "set yourself — it will not cancel a request someone else asked of you."
)


class _SetGoalArgs(BaseModel):
    """``set_goal`` arguments — the goal text plus an optional result schema.

    ``extra=\"forbid\"``: the servicer is implicitly *self* and the trusted caller
    id is the harness-supplied executing session — never a field here. An injected
    ``caller``/``session_id``/``target`` is rejected before the handler runs.
    """

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(description="The standing objective to work toward (a description).")
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON Schema the goal's eventual `return` value must satisfy.",
    )


class _CancelGoalArgs(BaseModel):
    """``cancel_goal`` arguments — just the ``goal_id`` to retract."""

    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(description="The id of your own goal to retract (from `set_goal`).")


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


async def cancel_goal_response(
    pool: Any,
    session_id: str,
    *,
    account_id: str,
    goal_id: str,
    by: str,
) -> str:
    """Verify ``goal_id`` is a genuine self-goal of ``session_id`` then retract it.

    The shared core behind the model ``cancel_goal`` tool (``by=\"self\"``) and the
    operator-cancel endpoint (``by=\"operator\"``). **SECURITY (mandatory):** reads
    the trusted ``get_request_caller`` edge and refuses unless the caller is exactly
    ``{kind:session, id:session_id}`` — so a peer-invoke (``caller={session, id:
    OTHER}``) or workflow-child (``caller={run}``) obligation is NEVER stamped with
    ``{kind:cancelled}``. On a verified self-goal, writes the response edge via
    ``respond_to_request`` with ``error={kind:\"cancelled\", by:<by>}`` — a genuinely
    distinct *third* terminal kind (vs ``return``'s value and ``error``'s free-text
    ``{message}``). Closing the edge drops the goal from B's obligations block.

    Returns ``\"cancelled\"`` | ``\"not_self_goal\"`` (caller is not a verified
    self-goal — the request is unknown to this session OR owned by someone else) |
    ``\"unknown_request\"`` (the response write found no open request).
    """
    from aios.db import queries
    from aios.tools.workflow_completion import respond_to_request

    async with pool.acquire() as conn:
        caller = await queries.get_request_caller(conn, session_id, request_id=goal_id)
    if caller != {"kind": "session", "id": session_id}:
        # Not a self-goal: an unknown id, or a peer-invoke / workflow-child
        # obligation. Fail closed BEFORE any write so a non-self edge can never be
        # stamped {by:self} / {by:operator}.
        return "not_self_goal"
    status = await respond_to_request(
        pool,
        session_id,
        request_id=goal_id,
        is_error=True,
        result=None,
        error={"kind": "cancelled", "by": by},
    )
    if status in ("responded", "duplicate"):
        return "cancelled"
    return "unknown_request"


# ─── handlers ────────────────────────────────────────────────────────────────


async def set_goal_handler(
    session_id: str, arguments: dict[str, Any], tool_call_id: str | None = None
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_SetGoalArgs, arguments)
    if not tool_call_id:
        # The deterministic id is keyed on the originating tool_call_id; without it
        # we can't guarantee exactly-once. The model path always supplies it.
        raise ToolBail("set_goal requires a tool_call_id (internal: not threaded)")
    request_id = sessions_service.goal_request_id(session_id, tool_call_id)
    try:
        goal_id = await sessions_service.set_goal(
            pool,
            session_id,
            account_id=account_id,
            request_id=request_id,
            goal=args.goal,
            output_schema=args.output_schema,
        )
    except RateLimitedError as exc:
        return ToolResult(content=exc.to_message(), is_error=True)
    return {"goal_id": goal_id}


async def cancel_goal_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CancelGoalArgs, arguments)
    status = await cancel_goal_response(
        pool, session_id, account_id=account_id, goal_id=args.goal_id, by="self"
    )
    if status == "not_self_goal":
        return ToolResult(content="goal_id is not one of your goals", is_error=True)
    if status == "unknown_request":
        return ToolResult(
            content="no open goal with that goal_id — it may already be answered or cancelled",
            is_error=True,
        )
    return {"status": "cancelled", "goal_id": args.goal_id}


def cancel_goal_tool_spec() -> dict[str, Any]:
    """The chat-completions tool entry for ``cancel_goal`` — injected into a
    session's tool list by ``compute_step_prelude`` whenever it owes a request
    (the same ``owes_request`` gate as ``return``/``error``)."""
    return openai_tool_entry(registry.get(CANCEL_GOAL_TOOL_NAME))


def _register() -> None:
    registry.register(
        name=SET_GOAL_TOOL_NAME,
        description=SET_GOAL_DESCRIPTION,
        parameters_schema=_SetGoalArgs.model_json_schema(),
        handler=set_goal_handler,
        transport="agent_tool",
        wants_tool_call_id=True,
    )
    registry.register(
        name=CANCEL_GOAL_TOOL_NAME,
        description=CANCEL_GOAL_DESCRIPTION,
        parameters_schema=_CancelGoalArgs.model_json_schema(),
        handler=cancel_goal_handler,
        transport="agent_tool",
    )


_register()
