"""Model-facing call-edge verbs — ``cancel_call`` + ``list_calls`` (#1428).

The in-band counterpart of the operator cancel plane (``POST /v1/tasks/{task_id}/cancel``):
a session's own model can see and command the durable ``call_*`` tasks it is awaiting,
keyed on the launching ``tool_call_id`` — the handle the model already holds, stamped on the
servicer edge by ``invoke_session._caller`` and re-derivable via ``find_parked_servicer`` (#1431).
The operator plane keys on ``task_id`` (the servicer id); the model plane keys on ``tool_call_id``.
Isolation rides on the edge's ``caller.id`` (pinned to THIS session by ``find_parked_servicer``),
never on ``tool_call_id`` uniqueness — a forged/guessed id can't reach another session's task.

The model surface uses call-edge terminology; the internal services remain task-named
(``tasks_service.cancel_task`` / ``list_open_tasks``) because they also back the operator task
plane. Persisted ``stop_task``/``list_tasks`` ToolSpecs are handled by the retirement lifecycle.

``cancel_call`` supersedes the retired ``cancel_run`` model tool: ``cancel_run`` reached only runs,
whereas ``cancel_call`` reaches a session servicer too — and via ``cancel_task``'s session
cascade, the servicer's whole awaited subtree. It is the sole model cancel verb (the local
``cancel`` tool-task-detach was deleted in #1458).

**Identity is load-bearing** (the F1 invariant, mirrored from ``invoke_session`` /
``workflow_management``): the trusted *caller* is the harness-supplied executing ``session_id``
(``invoke_builtin(session_id, …)``), NEVER model input. The arg models are ``extra="forbid"`` so an
injected ``caller``/``account_id``/``session_id`` is rejected before the handler runs. Both register
``transport="agent_tool"`` (model-only; the CLI broker refuses them).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from aios.db import queries
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services import tasks as tasks_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry

# ─── argument models ─────────────────────────────────────────────────────────


class _CancelCallArgs(BaseModel):
    """``cancel_call`` arguments — just the ``tool_call_id`` of the task to stop.

    ``extra="forbid"``: the trusted caller is the executing session, never a field here — an
    injected ``caller``/``session_id`` is rejected before the handler runs.
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str = Field(
        description=(
            "The tool_call_id of one of YOUR open call_session/call_agent/call_workflow tasks "
            "(as returned by list_calls)."
        )
    )


class _ListCallsArgs(BaseModel):
    """``list_calls`` arguments — none. ``extra="forbid"`` rejects any smuggled key."""

    model_config = ConfigDict(extra="forbid")


def _parse[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc


# ─── handlers ────────────────────────────────────────────────────────────────


async def cancel_call_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = _parse(_CancelCallArgs, arguments)
    # Resolve the servicer this tool_call_id is awaiting. find_parked_servicer pins
    # caller.id=session_id, so a forged/foreign tool_call_id resolves to None here — a clean
    # model-visible error, never another session's task. (For a run servicer, caller.id IS the
    # launcher, so cancel_task's launcher guard is satisfied by construction.)
    async with pool.acquire() as conn:
        handle = await queries.find_parked_servicer(
            conn,
            caller_session_id=session_id,
            tool_call_id=args.tool_call_id,
            account_id=account_id,
        )
        if handle is None:
            return ToolResult(
                content=f"no open call for tool_call_id {args.tool_call_id!r}", is_error=True
            )
        servicer_kind, servicer_id, request_id, _output_schema = handle
        # Already terminal? Then there is nothing to cancel — report it plainly rather than
        # seeding a cancel that resolves a no-op (kills the confusing "requested stop → ok").
        if servicer_kind == "session":
            assert request_id is not None  # a session servicer always carries a request_id
            resp = await queries.derive_response(
                conn, servicer_id, account_id=account_id, request_id=request_id
            )
        else:
            resp = await wf_queries.derive_run_response(conn, servicer_id, account_id=account_id)
        if resp is not None:
            return {"ok": "already resolved"}
    # The conn is released; seed the cancel on the servicer (cancel_task acquires its own).
    # It harvests under its own step lock and answers the parked call ``cancelled`` (an
    # independent tool_result — by design). canceller_session_id gates the run arm's launcher
    # guard (no-op for the session arm).
    await tasks_service.cancel_task(
        pool,
        servicer_kind=servicer_kind,
        servicer_id=servicer_id,
        request_id=request_id,
        account_id=account_id,
        canceller_session_id=session_id,
    )
    return {"ok": "cancel requested"}


async def list_calls_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    _parse(_ListCallsArgs, arguments)  # no fields; enforces extra="forbid" (reject smuggled keys)
    calls = await tasks_service.list_open_tasks(pool, session_id=session_id, account_id=account_id)
    return {
        "calls": [
            {
                **call.model_dump(mode="json"),
                "origin": (
                    "self" if call.kind == "session" and call.target == session_id else call.kind
                ),
            }
            for call in calls
        ]
    }


# ─── descriptions + registration ─────────────────────────────────────────────

CANCEL_CALL_DESCRIPTION = (
    "Stop (cancel) one of your in-flight tasks — a call_session / call_agent / call_workflow "
    "you launched and are still awaiting — by its tool_call_id (the id of that pending tool "
    "call, as listed by list_calls). The task and everything it is itself awaiting are "
    "cancelled, and the original call then resolves as cancelled. Use this to abandon a call "
    "you no longer need or one that is stuck. Reports 'already resolved' if the call finished "
    "before the stop landed, or an error if no open call matches that tool_call_id."
)
LIST_CALLS_DESCRIPTION = (
    "List your open outgoing calls — the call_session / call_agent / call_workflow calls you have "
    "launched that are still running and awaiting an answer. Each entry carries the "
    "tool_call_id (pass it to cancel_call to cancel), the servicer kind (session or run), the "
    "target id, and when it was opened. A point-in-time snapshot; answered or cancelled calls "
    "drop off."
)


def _register() -> None:
    registry.register(
        name="cancel_call",
        description=CANCEL_CALL_DESCRIPTION,
        parameters_schema=_CancelCallArgs.model_json_schema(),
        handler=cancel_call_handler,
        transport="agent_tool",
    )
    registry.register(
        name="list_calls",
        description=LIST_CALLS_DESCRIPTION,
        parameters_schema=_ListCallsArgs.model_json_schema(),
        handler=list_calls_handler,
        transport="agent_tool",
    )


_register()
