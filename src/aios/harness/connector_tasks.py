"""Procrastinate tasks for the connector RPC plane.

The API process can't talk to connector subprocesses directly — they're
owned by the worker via :class:`~aios.harness.connector_supervisor.ConnectorSubprocessRegistry`.
The four ``/v1/connectors/...`` admin endpoints flow through procrastinate:

* The router generates a ULID ``call_id`` and ``LISTEN``s on
  ``connector_result_<call_id>`` (LISTEN-before-action invariant from
  :mod:`aios.db.listen`).
* It enqueues one of these tasks with the ``call_id`` as a kwarg.
* The worker runs the task, dispatches into the supervisor, then
  ``pg_notify``'s the result envelope on the same channel.
* The router awaits NOTIFY (60s ceiling) and returns.

Lock semantics:

* ``harness.connector_call`` — ``lock="connector:{connector_name}"``
  so concurrent tool calls against the same connector serialize.  Set
  at defer time, not on the decorator: procrastinate stores decorator
  lock arguments verbatim with no template substitution (same reason
  ``harness.wake_session`` configures its lock per-call).
* ``harness.connector_status`` / ``harness.connector_tools`` —
  ``queueing_lock`` only (call_id-scoped, dedupes API retries).  No
  mutual-exclusion lock; reads are cheap and don't conflict.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import asyncpg

from aios.harness import runtime
from aios.harness.procrastinate_app import app
from aios.logging import get_logger

if TYPE_CHECKING:
    pass


log = get_logger("aios.harness.connector_tasks")

CONNECTOR_QUEUE = "connectors"
_TOOLS_TIMEOUT_S = 30.0


def _result_channel(call_id: str) -> str:
    return f"connector_result_{call_id}"


async def _notify_result(pool: asyncpg.Pool[Any], call_id: str, payload: dict[str, Any]) -> None:
    """``pg_notify`` the API-side LISTENer with the JSON-encoded result envelope."""
    async with pool.acquire() as conn:
        await conn.execute(
            "SELECT pg_notify($1, $2)",
            _result_channel(call_id),
            json.dumps(payload),
        )


@app.task(name="harness.connector_status", queue=CONNECTOR_QUEUE, retry=False, pass_context=False)
async def connector_status(call_id: str, name: str | None = None) -> None:
    """Snapshot connector state and notify the API.

    ``name=None`` returns every connector's snapshot (used by
    ``GET /v1/connectors``).  ``name=<n>`` filters to a single
    connector (used by ``GET /v1/connectors/:name/accounts`` and the
    aggregate row in ``aios connector list`` when the operator targets
    one connector).  Unknown ``name`` notifies an ``error`` envelope so
    the router can return 404 without timing out.
    """
    pool = runtime.require_pool()
    registry = runtime.connector_subprocess_registry
    if registry is None:
        await _notify_result(pool, call_id, {"error": "worker not initialized"})
        return
    if name is None:
        snapshot = registry.snapshot_all()
        await _notify_result(pool, call_id, {"connectors": snapshot})
        return
    state = registry.state(name)
    if state is None:
        await _notify_result(pool, call_id, {"error": f"connector {name!r} not enabled"})
        return
    await _notify_result(pool, call_id, {"connector": state.snapshot()})


@app.task(name="harness.connector_tools", queue=CONNECTOR_QUEUE, retry=False, pass_context=False)
async def connector_tools(call_id: str, name: str) -> None:
    """List the named connector's tools and notify the API.

    Round-trips to the subprocess (``session.list_tools()``) so a
    crashed connector surfaces as ``error`` rather than a stale cache.
    The worker's supervisor handles the not-ready / circuit-open
    cases, mapping each to a distinct error string the operator can
    read.
    """
    pool = runtime.require_pool()
    registry = runtime.connector_subprocess_registry
    if registry is None:
        await _notify_result(pool, call_id, {"error": "worker not initialized"})
        return

    from aios.harness.connector_supervisor import (
        CircuitOpen,
        ConnectorNotEnabled,
        ConnectorNotReady,
    )

    try:
        session = await asyncio.wait_for(registry.get_session(name), timeout=_TOOLS_TIMEOUT_S)
    except ConnectorNotEnabled:
        await _notify_result(pool, call_id, {"error": f"connector {name!r} not enabled"})
        return
    except CircuitOpen:
        await _notify_result(pool, call_id, {"error": f"connector {name!r} circuit open"})
        return
    except (ConnectorNotReady, TimeoutError):
        await _notify_result(pool, call_id, {"error": f"connector {name!r} not ready"})
        return

    try:
        result = await asyncio.wait_for(session.list_tools(), timeout=_TOOLS_TIMEOUT_S)
    except Exception as err:
        log.warning("connector_tools.list_failed", connector=name, exc_info=True)
        await _notify_result(
            pool, call_id, {"error": f"connector transport error: {type(err).__name__}: {err}"}
        )
        return

    tools_payload = [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in result.tools
    ]
    await _notify_result(pool, call_id, {"tools": tools_payload})


@app.task(name="harness.connector_call", queue=CONNECTOR_QUEUE, retry=False, pass_context=False)
async def connector_call(
    call_id: str,
    name: str,
    tool: str,
    arguments: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """Dispatch a tool call into the connector subprocess and notify the API."""
    pool = runtime.require_pool()
    registry = runtime.connector_subprocess_registry
    if registry is None:
        await _notify_result(pool, call_id, {"error": "worker not initialized"})
        return
    result = await registry.dispatch_call(name, tool, arguments, meta=meta)
    await _notify_result(pool, call_id, result)


async def defer_connector_call(
    *,
    call_id: str,
    name: str,
    tool: str,
    arguments: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """Enqueue a ``connector_call`` job with per-connector + per-call locking."""
    deferrer = app.configure_task(
        "harness.connector_call",
        lock=f"connector:{name}",
        queueing_lock=f"connector:call:{call_id}",
    )
    await deferrer.defer_async(
        call_id=call_id,
        name=name,
        tool=tool,
        arguments=arguments,
        meta=meta,
    )


async def defer_connector_status(*, call_id: str, name: str | None = None) -> None:
    """Enqueue a ``connector_status`` snapshot job."""
    deferrer = app.configure_task(
        "harness.connector_status",
        queueing_lock=f"connector:status:{call_id}",
    )
    await deferrer.defer_async(call_id=call_id, name=name)


async def defer_connector_tools(*, call_id: str, name: str) -> None:
    """Enqueue a ``connector_tools`` round-trip job."""
    deferrer = app.configure_task(
        "harness.connector_tools",
        queueing_lock=f"connector:tools:{call_id}",
    )
    await deferrer.defer_async(call_id=call_id, name=name)
