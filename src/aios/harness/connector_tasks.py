"""Procrastinate tasks for the connector RPC plane.

The API process can't talk to connector subprocesses directly — they're
owned by the worker via :class:`~aios.harness.connector_supervisor.ConnectorSubprocessRegistry`.
The ``/v1/connectors/...`` admin endpoints flow through procrastinate:

* The router generates a ULID ``call_id`` and ``LISTEN``s on
  ``connector_result_<call_id>`` (LISTEN-before-action invariant from
  :mod:`aios.db.listen`).
* It enqueues one of these tasks with the ``call_id`` as a kwarg.
* The worker runs the task, dispatches into the supervisor, then
  ``pg_notify``'s the result envelope on the same channel.
* The router awaits NOTIFY (60s ceiling) and returns.

The ``_`` instance sentinel (:data:`DEFAULT_INSTANCE_SENTINEL`) lets
the CLI omit ``<instance>`` without a separate resolve round-trip; the
worker resolves it via :meth:`ConnectorSubprocessRegistry.resolve_default_instance`.

Lock semantics:

* ``harness.connector_call`` —
  ``lock="connector:{connector}:{instance}"`` so concurrent tool calls
  against the same instance serialize but distinct instances run in
  parallel.  Set at defer time, not on the decorator: procrastinate
  stores decorator lock arguments verbatim with no template substitution
  (same reason ``harness.wake_session`` configures its lock per-call).
* ``harness.connector_status`` / ``harness.connector_tools`` — no
  procrastinate locks.  Reads are cheap, don't conflict, and the
  per-call NOTIFY channel routes results back to the right LISTENer.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg

from aios.harness import runtime
from aios.harness.procrastinate_app import app
from aios.logging import get_logger

log = get_logger("aios.harness.connector_tasks")

CONNECTOR_QUEUE = "connectors"
_TOOLS_TIMEOUT_S = 30.0

# Re-export so worker-side callers (e.g. tests) that already import
# from this module don't need to switch to ``aios.config``.  The
# canonical home is ``aios.config`` because the CLI imports it without
# triggering this module's procrastinate-app load.
from aios.config import DEFAULT_INSTANCE_SENTINEL  # noqa: E402  (re-export)


def _result_channel(call_id: str) -> str:
    return f"connector_result_{call_id}"


def _resolve_instance_or_error(
    registry: Any, connector: str, instance: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Translate the ``_`` sentinel to a concrete instance name.

    Returns ``(resolved_instance, None)`` on success, or
    ``(None, error_envelope)`` if the sentinel can't be resolved
    (no instances enabled, or multiple instances and operator must
    pick one explicitly).
    """
    if instance != DEFAULT_INSTANCE_SENTINEL:
        return instance, None
    resolved = registry.resolve_default_instance(connector)
    if resolved is not None:
        return resolved, None
    states = registry.states_for_connector(connector)
    if not states:
        return None, {
            "error": f"connector {connector!r} not enabled",
            "code": "not_enabled",
        }
    return None, {
        "error": (
            f"connector {connector!r} has multiple instances "
            f"({', '.join(s.instance for s in states)}); specify one explicitly"
        ),
        "code": "ambiguous_instance",
    }


async def _notify_result(pool: asyncpg.Pool[Any], call_id: str, payload: dict[str, Any]) -> None:
    """``pg_notify`` the API-side LISTENer with the JSON-encoded result envelope."""
    async with pool.acquire() as conn:
        await conn.execute(
            "SELECT pg_notify($1, $2)",
            _result_channel(call_id),
            json.dumps(payload),
        )


@app.task(name="harness.connector_status", queue=CONNECTOR_QUEUE, retry=False, pass_context=False)
async def connector_status(
    call_id: str,
    connector: str | None = None,
    instance: str | None = None,
) -> None:
    """Snapshot connector state and notify the API.

    Three call shapes:

    * ``connector=None, instance=None``: every instance's snapshot (used
      by ``GET /v1/connectors``).
    * ``connector=<c>, instance=None``: every instance of one connector
      type (used by ``GET /v1/connectors/<c>``).
    * ``connector=<c>, instance=<i>``: single-instance state (used by
      ``GET /v1/connectors/<c>/<i>`` and ``aios connector get``).

    Unknown pairs notify an ``error`` envelope so the router can return
    404 without timing out.
    """
    pool = runtime.require_pool()
    registry = runtime.connector_subprocess_registry
    if registry is None:
        await _notify_result(
            pool, call_id, {"error": "worker not initialized", "code": "not_ready"}
        )
        return
    if connector is None:
        snapshot = registry.snapshot_all()
        await _notify_result(pool, call_id, {"connectors": snapshot})
        return
    if instance is None:
        states = registry.states_for_connector(connector)
        if not states:
            await _notify_result(
                pool,
                call_id,
                {"error": f"connector {connector!r} not enabled", "code": "not_enabled"},
            )
            return
        await _notify_result(pool, call_id, {"connectors": [s.snapshot() for s in states]})
        return
    resolved, err = _resolve_instance_or_error(registry, connector, instance)
    if err is not None:
        await _notify_result(pool, call_id, err)
        return
    assert resolved is not None
    state = registry.state(connector, resolved)
    if state is None:
        await _notify_result(
            pool,
            call_id,
            {
                "error": f"connector instance {connector}:{resolved} not enabled",
                "code": "not_enabled",
            },
        )
        return
    await _notify_result(pool, call_id, {"connector": state.snapshot()})


@app.task(name="harness.connector_tools", queue=CONNECTOR_QUEUE, retry=False, pass_context=False)
async def connector_tools(call_id: str, connector: str, instance: str) -> None:
    """List one instance's tools and notify the API.

    Round-trips to the subprocess (``session.list_tools()``) so a
    crashed connector surfaces as a fresh transport error rather than
    a stale cache.
    """
    pool = runtime.require_pool()
    registry = runtime.connector_subprocess_registry
    if registry is None:
        await _notify_result(
            pool, call_id, {"error": "worker not initialized", "code": "not_ready"}
        )
        return

    from aios.harness.connector_supervisor import (
        CircuitOpen,
        ConnectorNotEnabled,
        ConnectorNotReady,
        instance_label,
    )

    resolved, err = _resolve_instance_or_error(registry, connector, instance)
    if err is not None:
        await _notify_result(pool, call_id, err)
        return
    assert resolved is not None
    label = instance_label(connector, resolved)
    try:
        session = await asyncio.wait_for(
            registry.get_session(connector, resolved), timeout=_TOOLS_TIMEOUT_S
        )
    except ConnectorNotEnabled:
        await _notify_result(
            pool,
            call_id,
            {"error": f"connector instance {label!r} not enabled", "code": "not_enabled"},
        )
        return
    except CircuitOpen:
        await _notify_result(
            pool,
            call_id,
            {"error": f"connector instance {label!r} circuit open", "code": "circuit_open"},
        )
        return
    except (ConnectorNotReady, TimeoutError):
        await _notify_result(
            pool,
            call_id,
            {"error": f"connector instance {label!r} not ready", "code": "not_ready"},
        )
        return

    try:
        result = await asyncio.wait_for(session.list_tools(), timeout=_TOOLS_TIMEOUT_S)
    except Exception as err:
        log.warning(
            "connector_tools.list_failed",
            connector=connector,
            instance=resolved,
            exc_info=True,
        )
        await _notify_result(
            pool,
            call_id,
            {
                "error": f"connector transport error: {type(err).__name__}: {err}",
                "code": "transport_error",
            },
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
    connector: str,
    instance: str,
    tool: str,
    arguments: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """Dispatch a tool call into the named instance and notify the API."""
    pool = runtime.require_pool()
    registry = runtime.connector_subprocess_registry
    if registry is None:
        await _notify_result(
            pool, call_id, {"error": "worker not initialized", "code": "not_ready"}
        )
        return
    resolved, err = _resolve_instance_or_error(registry, connector, instance)
    if err is not None:
        await _notify_result(pool, call_id, err)
        return
    assert resolved is not None
    result = await registry.dispatch_call(connector, resolved, tool, arguments, meta=meta)
    await _notify_result(pool, call_id, result)


async def defer_connector_call(
    *,
    call_id: str,
    connector: str,
    instance: str,
    tool: str,
    arguments: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> None:
    """Enqueue a ``connector_call`` job with per-instance mutual exclusion.

    Only ``lock`` is set — ``queueing_lock`` would dedup pending jobs
    sharing a value, but every API request mints a fresh ``call_id``
    ULID so the per-call queueing_lock would never fire.  The
    correctness story is the per-call NOTIFY channel: each result
    routes back to exactly the LISTENer that minted its ``call_id``.
    """
    deferrer = app.configure_task(
        "harness.connector_call",
        lock=f"connector:{connector}:{instance}",
    )
    await deferrer.defer_async(
        call_id=call_id,
        connector=connector,
        instance=instance,
        tool=tool,
        arguments=arguments,
        meta=meta,
    )


async def defer_connector_status(
    *,
    call_id: str,
    connector: str | None = None,
    instance: str | None = None,
) -> None:
    """Enqueue a ``connector_status`` snapshot job."""
    deferrer = app.configure_task("harness.connector_status")
    await deferrer.defer_async(call_id=call_id, connector=connector, instance=instance)


async def defer_connector_tools(*, call_id: str, connector: str, instance: str) -> None:
    """Enqueue a ``connector_tools`` round-trip job."""
    deferrer = app.configure_task("harness.connector_tools")
    await deferrer.defer_async(call_id=call_id, connector=connector, instance=instance)
