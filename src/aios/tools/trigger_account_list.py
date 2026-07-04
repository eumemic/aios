"""The ``list_account_triggers`` tool — the account-wide trigger-read (#1673).

The account-scoped analog of the session-scoped ``trigger_list``: it enumerates
EVERY trigger in the caller's account (across all its sessions), not just one
session's own. This is the filed blocking precondition for the eumemic-company
ops-agent O7 trigger-liveness auditor — until it lands, trigger-liveness is
emitted as a NAMED ``cannot-determine`` residue every sweep, never a predicate
that pretends to run.

The auditor needs to sweep every enabled trigger across its account
(dev-pipeline cron, reconciler cron, reaper, telemetry-observer, future
sentinels — each on its own session) and read each one's ``next_fire``, to catch
the #925 zombie class: an ``enabled=true, next_fire=NULL`` cron row is invisible
to the scheduler (which filters ``next_fire IS NOT NULL``) and never fires, and
nothing re-arms it on its own.

**Dual surface (mirrors ``list_runs`` #1396).** Registered ``transport="agent_tool"``
so an operator-trust session model can call it (account-scoped to the executing
session's account), AND added to :data:`aios.workflows.run_tools.RUN_TOOLS` so a
workflow RUN may call it — the run path dispatches account-scoped to
``run.account_id`` (see ``aios.workflows.run_tools._read_account_triggers``). The
ops-agent is a workflow run, so the run path is the load-bearing one; the session
surface is parity.

Each returned trigger carries ``{id, name, owner_session_id, source_kind,
enabled, next_fire, last_fire_status, consecutive_failures}`` — the
liveness-audit projection (:class:`aios.models.triggers.AccountTriggerEcho`).
``owner_session_id`` lets the sweep name which session owns a zombie;
``source_kind`` lets it classify schedulable (``cron`` → ``next_fire`` must be
non-null) vs reactive/one-shot (``run_completion`` / ``external_event`` →
exempt).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from aios.harness import runtime
from aios.services import sessions as sessions_service
from aios.services import triggers as triggers_service
from aios.tools.input import tool_input
from aios.tools.registry import registry

LIST_ACCOUNT_TRIGGERS_DESCRIPTION = (
    "List every trigger across the entire account (all sessions), each with its "
    "id, name, owner_session_id, source kind, enabled state, next_fire, last fire "
    "status, and consecutive failure count. Account-wide — unlike trigger_list, "
    "which lists only THIS session's triggers. Use it to audit trigger liveness: a "
    "schedulable (cron) trigger that is enabled but has a null next_fire is a "
    "zombie the scheduler will never fire. Defaults to enabled triggers only; set "
    "enabled_only=false for every trigger on non-archived sessions."
)


class _ListAccountTriggersArgs(BaseModel):
    """``list_account_triggers`` arguments.

    There is deliberately no ``account_id`` / ``session_id`` field — the account
    derives from the trusted executing session (session surface) or the run's own
    account (run surface); it is never model input. ``extra="forbid"`` rejects an
    injected id before the handler runs.
    """

    model_config = ConfigDict(extra="forbid")

    enabled_only: bool = True


async def list_account_triggers_handler(
    session_id: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    args = tool_input(_ListAccountTriggersArgs, arguments)
    echoes = await triggers_service.list_account_triggers(
        pool, account_id=account_id, enabled_only=args.enabled_only
    )
    return {"triggers": [e.model_dump(mode="json") for e in echoes]}


def _register() -> None:
    registry.register(
        name="list_account_triggers",
        description=LIST_ACCOUNT_TRIGGERS_DESCRIPTION,
        parameters_schema=_ListAccountTriggersArgs.model_json_schema(),
        handler=list_account_triggers_handler,
        transport="agent_tool",
    )


_register()
