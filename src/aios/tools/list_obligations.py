"""``list_obligations`` — the model-facing INCOMING obligations view (#1522).

The incoming twin of ``list_calls`` (the outgoing view, child #4) and the
**source-agnostic** replacement for ``list_goals`` (child #3, which only listed
the self-caller subset). Where ``list_goals`` filtered the open-obligation set to
self-goals, ``list_obligations`` lists **every** open awaited obligation the
session owes a response to — regardless of caller kind (``api`` / ``session`` /
``run``, and a self-goal where ``origin=self``).

Each entry carries ``request_id``, ``caller_kind``, ``origin`` (incl. ``self``),
``summary``, ``age``, and the **acceptance contract** ``output_schema`` (the JSON
Schema the closing ``return`` value must satisfy). It draws those entries from the
ONE shared owed-read-model renderer (:func:`aios.harness.obligations.render_owed_entry`)
that the quiescence-attempt surfacing also feeds from — so the "what you owe +
its contract" formatting lives in exactly one place and the two consumers can't
drift.

Registers ``transport="agent_tool"`` (model-only; the CLI broker refuses it), arg
model ``extra="forbid"``, identity = the harness-supplied executing session (never
a smuggled ``caller``/``account_id`` field).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from aios.db import queries
from aios.harness import runtime
from aios.harness.obligations import render_owed_entry
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry


class _ListObligationsArgs(BaseModel):
    """``list_obligations`` takes no arguments. ``extra="forbid"`` so a smuggled
    ``caller``/``account_id`` (the trusted identity is the executing session the
    harness supplies, never a field here) is rejected before the handler runs."""

    model_config = ConfigDict(extra="forbid")


async def list_obligations_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any] | ToolResult:
    """Enumerate the session's OPEN obligations (oldest-first), source-agnostic.

    Reads the widened owed-read-model (:func:`queries.get_open_obligations`, now
    carrying ``output_schema``) and projects each open edge through the shared
    :func:`render_owed_entry` formatter: ``request_id``, ``caller_kind``,
    ``origin`` (``api``/``session``/``run``/``self``), ``summary``, ``age``, and
    the bounded ``output_schema`` contract. Lists ALL caller kinds — a caller's
    api/run/peer task AND a #1414 self-goal alike (``origin`` distinguishes them).
    """
    try:
        _ListObligationsArgs.model_validate(arguments)  # reject smuggled keys
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc

    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    async with pool.acquire() as conn:
        obligations = await queries.get_open_obligations(conn, session_id, account_id=account_id)
    now = datetime.now(UTC)
    return {
        "obligations": [render_owed_entry(o, session_id=session_id, now=now) for o in obligations]
    }


LIST_OBLIGATIONS_DESCRIPTION = (
    "List every open obligation you owe a response to — your INCOMING tasks, "
    "source-agnostic (a caller's api/run/peer-session request AND any self-goal "
    "you pinned with create_goal, where origin=self). Each entry has its "
    "request_id, caller_kind, origin, summary, age, and output_schema (the "
    "acceptance contract the answer must satisfy, when one is set). Answer each "
    "with return(request_id=<request_id>, value=...) when done (the value is "
    "validated against output_schema), or error(request_id=<request_id>, "
    "message=...) to abandon it. You cannot go idle while any obligation is open."
)


def _register() -> None:
    registry.register(
        name="list_obligations",
        description=LIST_OBLIGATIONS_DESCRIPTION,
        parameters_schema=_ListObligationsArgs.model_json_schema(),
        handler=list_obligations_handler,
        transport="agent_tool",
    )


_register()
