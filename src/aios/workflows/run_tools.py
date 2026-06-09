"""Run-side tool execution — a workflow run calls its declared network/credential tools.

**Park-and-harvest.** The step opens a ``tool`` frontier (journals ``call_started``),
launches a fire-and-forget **worker** task here, and parks the run. The task runs the tool
and, on completion, writes a ``tool_result`` ``wf_run_signals`` row + wakes the run; the next
step's pre-replay harvest folds the signal into a ``call_result``. This is the ``agent()`` /
``gate()`` shape — the run never holds its lock/slot while a tool runs.

**Owner-minimal.** ``web_search`` / ``web_fetch`` handlers are owner-agnostic (called with an
empty owner id). ``http_request`` reuses the shared :func:`aios.tools.http_request._do_http_request`
core, fed the run's *snapshotted* ``http_servers`` and a run-scoped credential resolver
(:func:`resolve_auth_for_target_url_run`). ``invoke_builtin`` is untouched.

**Surface gating.** A tool is callable only if it is in the slice-2 builtin set AND declared in
the run's snapshotted ``tools``. An undeclared/unknown tool is a *recoverable* ``{"error": …}``
value the script branches on — not a run-terminal error (matching ``agent()``'s "errors resolve"
and ``http_request``'s own error contract).

**At-least-once.** A hard worker crash mid-task leaves no signal; the periodic sweep re-wakes the
run and the step re-dispatches. Safe for idempotent tools (``web_*`` / GET); a non-idempotent
``http_request`` (POST/DELETE) may double-execute — the same exposure the session tool path has.
The ``idempotent`` flag that would surface ``may_have_completed`` instead of re-dispatching is a
deferred follow-up.
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.logging import get_logger
from aios.mcp.client import resolve_auth_for_target_url_run
from aios.models.workflows import WfRun
from aios.services.wake import defer_run_wake
from aios.tools.http_request import _do_http_request, _find_server, _match_route
from aios.tools.invoke import validate_arguments
from aios.tools.registry import registry
from aios.tools.web_fetch import web_fetch_handler
from aios.tools.web_search import web_search_handler

log = get_logger("aios.workflows.run_tools")

# The network/credential builtins a run may call directly (slice 2). Sandbox tools
# (bash/read/…) and authed-MCP / search_events are out of scope (filesystem horizon /
# later slices).
RUN_TOOLS: frozenset[str] = frozenset({"web_search", "web_fetch", "http_request"})

# Per-worker in-flight tool tasks, keyed (run_id, call_key). Gates *launching* (so a
# sibling-triggered re-wake — e.g. parallel([tool(), agent()]) — doesn't double-dispatch a
# still-running tool); never gates *harvesting* (the signal in the DB is the truth).
_INFLIGHT: dict[tuple[str, str], asyncio.Task[None]] = {}


def has_inflight(run_id: str, call_key: str) -> bool:
    """True iff a live tool task for ``(run_id, call_key)`` is running on this worker."""
    task = _INFLIGHT.get((run_id, call_key))
    return task is not None and not task.done()


def launch_tool_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    tool_name: str,
    tool_input: Any,
) -> None:
    """Launch the worker task for a freshly-opened tool frontier (no-op if already live)."""
    key = (run.id, call_key)
    if has_inflight(*key):
        return
    _INFLIGHT[key] = asyncio.create_task(
        _run_tool_task(pool, run, call_key=call_key, tool_name=tool_name, tool_input=tool_input)
    )


async def _run_tool_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    tool_name: str,
    tool_input: Any,
) -> None:
    """Run one tool, write its ``tool_result`` signal, and wake the run (always-signals)."""
    try:
        try:
            result = await invoke_run_tool(
                run=run, account_id=run.account_id, tool_name=tool_name, tool_input=tool_input
            )
        except Exception as exc:  # backstop — invoke_run_tool returns dicts, never raises
            log.exception("run_tool.unexpected", run_id=run.id, tool=tool_name)
            result = {"error": f"tool {tool_name!r} failed: {type(exc).__name__}: {exc}"}
        try:
            async with pool.acquire() as conn:
                await wf_queries.insert_run_signal(
                    conn, run_id=run.id, call_key=call_key, kind="tool_result", result=result
                )
            await defer_run_wake(run.id)
        except Exception:
            # The tool ran but persisting/waking failed (DB blip, pool exhaustion). No signal
            # → the run stalls 'suspended' until the periodic sweep re-wakes it and the harvest
            # re-dispatches. Log so the stall is diagnosable, not a silent unretrieved-task warning.
            log.exception(
                "run_tool.signal_failed", run_id=run.id, call_key=call_key, tool=tool_name
            )
    finally:
        # CancelledError (worker shutdown) propagates here with no signal written — the
        # periodic sweep re-wakes the run and the step re-dispatches.
        _INFLIGHT.pop((run.id, call_key), None)


async def invoke_run_tool(
    *, run: WfRun, account_id: str, tool_name: str, tool_input: Any
) -> dict[str, Any]:
    """Dispatch one declared tool for a run. Always returns a dict (success or ``{"error": …}``);
    never raises — gating, validation, and handler errors all surface as recoverable values."""
    if tool_name not in RUN_TOOLS:
        return {"error": f"tool {tool_name!r} is not callable from a workflow run"}
    if tool_name not in {t.type for t in run.tools if t.enabled}:
        return {"error": f"tool {tool_name!r} is not in the workflow's declared tools"}

    args = tool_input if isinstance(tool_input, dict) else {}
    schema_error = validate_arguments(args, registry.get(tool_name).parameters_schema)
    if schema_error is not None:
        return {"error": schema_error}

    if tool_name == "http_request":
        # A route an operator marked ``always_ask`` requires per-call human confirmation —
        # which a run has no channel for. Deny rather than execute unconfirmed, so a run is
        # never *more* privileged than a session on the identical declared surface.
        server = _find_server(run.http_servers, str(args.get("server_ref", "")))
        route = _match_route(server, str(args.get("path", ""))) if server is not None else None
        if (
            route is not None
            and route.permission_policy is not None
            and (route.permission_policy.type == "always_ask")
        ):
            return {
                "error": (
                    "this route is marked always_ask (per-call confirmation), which a "
                    "workflow run cannot satisfy"
                )
            }
        pool = runtime.require_pool()
        crypto_box = runtime.require_crypto_box()

        async def resolve_auth(base_url: str) -> tuple[str | None, dict[str, str]]:
            return await resolve_auth_for_target_url_run(
                pool, crypto_box, run.id, base_url, account_id=account_id
            )

        return await _do_http_request(
            servers=run.http_servers, arguments=args, resolve_auth=resolve_auth
        )
    if tool_name == "web_search":
        return await web_search_handler("", args)
    return await web_fetch_handler("", args)
