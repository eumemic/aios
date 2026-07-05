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
from collections.abc import Awaitable
from typing import Any

import asyncpg

from aios.db.queries import workflows as wf_queries
from aios.errors import AiosError
from aios.harness import runtime
from aios.jobs.app import defer_run_wake
from aios.logging import get_logger
from aios.mcp.client import resolve_auth_for_target_url_run
from aios.models.workflows import WfRun
from aios.services import triggers as triggers_service
from aios.services import workflows as wf_service
from aios.tools.http_request import _do_http_request, _find_server, _match_route, _split_query
from aios.tools.invoke import ToolBail, validate_arguments
from aios.tools.registry import registry
from aios.tools.trigger_account_list import _ListAccountTriggersArgs
from aios.tools.web_fetch import web_fetch_handler
from aios.tools.web_search import web_search_handler
from aios.tools.workflow_management import (
    _RUN_ECHO_EXCLUDE,
    _GetRunArgs,
    _ListRunsArgs,
)
from aios.workflows.idempotency_key import (
    AIOS_IDEMPOTENCY_KEY_SENTINEL,
    idempotency_key,
)

log = get_logger("aios.workflows.run_tools")


def _substitute_idempotency_sentinel(
    args: dict[str, Any], run_id: str, call_key: str
) -> dict[str, Any]:
    """Return ``args`` with any ``Idempotency-Key`` header whose value is the opt-in
    sentinel replaced by the real per-call token. The header key match is
    case-insensitive (HTTP header names are); the value match is exact, so a literal
    author-supplied key is never clobbered. Copies on substitution so the caller's
    ``tool_input`` (journaled / replay-visible) stays verbatim; returns ``args``
    unchanged when nothing opted in."""
    headers = args.get("headers")
    if not isinstance(headers, dict) or not any(
        k.lower() == "idempotency-key" and v == AIOS_IDEMPOTENCY_KEY_SENTINEL
        for k, v in headers.items()
    ):
        return args
    token = idempotency_key(run_id, call_key)
    new_headers = {
        k: (token if k.lower() == "idempotency-key" and v == AIOS_IDEMPOTENCY_KEY_SENTINEL else v)
        for k, v in headers.items()
    }
    return {**args, "headers": new_headers}


# The builtins a run may call directly. The network/credential trio run on the
# worker (:func:`invoke_run_tool`); ``bash`` runs in the run's provisioned sandbox
# (:mod:`aios.workflows.run_sandbox`) — the step routes by the tool's execution
# class. ``list_runs`` / ``get_run`` are the run-journal READ pair (#1396): they run
# on the worker too, dispatched to the workflow service ACCOUNT-SCOPED to ``run.account_id``
# (never a launcher-session filter, never cross-account) — the substrate the standing
# immune-system dead-men (``gate_reaper`` #1386, ``telemetry_observer`` #1326) read to
# correlate the GitHub blackboard against which runs are live. The other sandbox builtins
# (read/write/edit/glob/grep) and authed-MCP / search_events stay out of scope (later
# slices), so a ``tool('read')`` is a recoverable not-callable value at the run frontier.
RUN_TOOLS: frozenset[str] = frozenset(
    {
        "web_search",
        "web_fetch",
        "http_request",
        "bash",
        "list_runs",
        "get_run",
        "list_account_triggers",
    }
)

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
                run=run,
                call_key=call_key,
                account_id=run.account_id,
                tool_name=tool_name,
                tool_input=tool_input,
            )
        except Exception as exc:  # backstop — invoke_run_tool returns dicts, never raises
            log.exception("run_tool.unexpected", run_id=run.id, tool=tool_name)
            result = {"error": f"tool {tool_name!r} failed: {type(exc).__name__}: {exc}"}
        try:
            async with pool.acquire() as conn:
                await wf_queries.insert_run_signal(
                    conn, run_id=run.id, call_key=call_key, kind="tool_result", result=result
                )
            # batch: tool results are a high-frequency wake source, like child
            # completions — a burst coalesces into one re-drive when the window is on.
            await defer_run_wake(run.id, batch=True)
        except Exception:
            # The tool ran but persisting/waking failed (DB blip, pool exhaustion). If the
            # signal committed and only the wake failed, the sweep's unharvested-signal clause
            # re-wakes within a tick; with no signal at all, the stale-tool clause re-wakes at
            # the staleness horizon and the harvest re-dispatches. Log so the stall is
            # diagnosable, not a silent unretrieved-task warning.
            log.exception(
                "run_tool.signal_failed", run_id=run.id, call_key=call_key, tool=tool_name
            )
    finally:
        # CancelledError (worker shutdown) propagates here with no signal written — the
        # periodic sweep re-wakes the run and the step re-dispatches.
        _INFLIGHT.pop((run.id, call_key), None)


def gate_run_tool(run: WfRun, tool_name: str) -> dict[str, Any] | None:
    """The run-tool surface gate, shared by the worker (:func:`invoke_run_tool`) and
    sandbox (:func:`aios.workflows.run_sandbox._execute`) executors so the exact
    error strings live in ONE place.

    A tool is callable from a run only if it is in the run-callable set
    (:data:`RUN_TOOLS`) AND declared+enabled in the run's snapshotted ``tools``.
    Returns the recoverable ``{"error": …}`` value to surface to the script when
    either check fails (the script branches on it — gating is never run-terminal),
    or ``None`` when the call is admitted.
    """
    if tool_name not in RUN_TOOLS:
        return {"error": f"tool {tool_name!r} is not callable from a workflow run"}
    if tool_name not in {t.type for t in run.tools if t.enabled}:
        return {"error": f"tool {tool_name!r} is not in the workflow's declared tools"}
    return None


async def _read_run_journal(
    *, account_id: str, tool_name: str, args: dict[str, Any]
) -> dict[str, Any]:
    """The run-journal READ pair — ``list_runs`` / ``get_run`` from inside a run (#1396).

    Dispatches to the workflow service ACCOUNT-SCOPED to the calling run's account, never
    a launcher-session filter, never cross-account:

      * ``list_runs`` — ``launcher_session_id=None`` (a run has no launching session; the
        only meaningful scope is the whole account). The ``account_wide`` arg is therefore
        a no-op here — a run is *always* account-scoped, never narrower; we accept it for
        schema parity with the session tool and ignore it. The run sees its OWN account's
        runs and only those, because ``account_id=account_id`` is the run's own account
        (``run.account_id``, threaded by :func:`invoke_run_tool`) — the isolation boundary
        is the same one the http-credential resolver and the get_run NotFound scope enforce.
      * ``get_run`` — the account-scoped single-run read. A cross-account / missing id
        raises :class:`NotFoundError` in the query layer; we catch it (and any client-class
        :class:`AiosError`) and return it as a recoverable ``{"error": …}`` value, so the
        contract "``invoke_run_tool`` never raises; errors are values the script branches on"
        holds — and a run can never read another account's run.

    The returned shape MATCHES the session tools (``{"runs": [...]}`` / the full WfRun
    dict, both with the heavy script/surface blobs trimmed by ``_RUN_ECHO_EXCLUDE``), so a
    workflow's reader code is identical whether the same tool is called from a session or a
    run.
    """
    pool = runtime.require_pool()
    try:
        if tool_name == "list_runs":
            parsed = _ListRunsArgs.model_validate(args)
            runs = await wf_service.list_runs(
                pool,
                account_id=account_id,
                limit=parsed.limit,
                after=parsed.after,
                workflow_id=parsed.workflow_id,
                status=parsed.status,
                parent_run_id=parsed.parent_run_id,
                # A run is account-scoped, never launcher-scoped: a run has no launching
                # session to filter on. account_wide is accepted for schema parity and
                # has no effect (the run is always account-wide within its OWN account).
                launcher_session_id=None,
            )
            return {"runs": [r.model_dump(mode="json", exclude=_RUN_ECHO_EXCLUDE) for r in runs]}
        # get_run
        parsed_get = _GetRunArgs.model_validate(args)
        run = await wf_service.get_run(pool, parsed_get.run_id, account_id=account_id)
        return run.model_dump(mode="json", exclude=_RUN_ECHO_EXCLUDE)
    except AiosError as exc:
        # A denied/not-found read (e.g. a cross-account get_run id) is a recoverable value
        # the script branches on — never a run-terminal raise (matching the agent()/
        # http_request "errors resolve" contract). The 4xx isolation boundary holds: the
        # query is account-scoped, so a foreign id reads NotFound, never another account's
        # run.
        return {"error": str(exc)}


async def _read_account_triggers(*, account_id: str, args: dict[str, Any]) -> dict[str, Any]:
    """The account-wide trigger read — ``list_account_triggers`` from inside a run (#1673).

    Dispatches to the triggers service ACCOUNT-SCOPED to the calling run's account, never
    session-scoped, never cross-account (mirroring :func:`_read_run_journal`): the run sees
    EVERY trigger in its OWN account and only those, because ``account_id=account_id`` is the
    run's own account (``run.account_id``, threaded by :func:`invoke_run_tool`) — the same
    isolation boundary the http-credential resolver and the run-journal reads enforce.

    This is the substrate the ops-agent O7 trigger-liveness auditor reads: it enumerates every
    enabled trigger across the account (each sentinel on its own session) and checks each
    ``next_fire`` for the #925 zombie class. The returned shape MATCHES the session tool
    (``{"triggers": [...]}`` of :class:`~aios.models.triggers.AccountTriggerEcho` dicts), so a
    workflow's reader code is identical whether the tool is called from a session or a run.
    """
    pool = runtime.require_pool()
    try:
        parsed = _ListAccountTriggersArgs.model_validate(args)
        echoes = await triggers_service.list_account_triggers(
            pool, account_id=account_id, enabled_only=parsed.enabled_only
        )
        return {"triggers": [e.model_dump(mode="json") for e in echoes]}
    except AiosError as exc:
        # Recoverable value the script branches on — never a run-terminal raise (the
        # agent()/http_request "errors resolve" contract). The read is account-scoped, so a
        # run can never read another account's triggers.
        return {"error": str(exc)}


async def invoke_run_tool(
    *, run: WfRun, call_key: str, account_id: str, tool_name: str, tool_input: Any
) -> dict[str, Any]:
    """Dispatch one declared tool for a run. Always returns a dict (success or ``{"error": …}``);
    never raises — gating, validation, and handler errors all surface as recoverable values.

    ``call_key`` is the run-frontier key for this call; with ``run.id`` it derives the
    per-call idempotency token (:func:`aios.workflows.idempotency_key.idempotency_key`)
    the ``http_request`` branch substitutes for an author's sentinel header."""
    if (err := gate_run_tool(run, tool_name)) is not None:
        return err

    args = tool_input if isinstance(tool_input, dict) else {}
    schema_error = validate_arguments(args, registry.get(tool_name).parameters_schema)
    if schema_error is not None:
        return {"error": schema_error}

    if tool_name in ("list_runs", "get_run"):
        return await _read_run_journal(account_id=account_id, tool_name=tool_name, args=args)
    if tool_name == "list_account_triggers":
        return await _read_account_triggers(account_id=account_id, args=args)
    if tool_name == "http_request":
        # A route an operator marked ``always_ask`` requires per-call human confirmation —
        # which a run has no channel for. Deny rather than execute unconfirmed, so a run is
        # never *more* privileged than a session on the identical declared surface.
        server = _find_server(run.http_servers, str(args.get("server_ref", "")))
        # Match on the path PORTION (a query string is matched/decided inside
        # _do_http_request); a query-bearing path must not slip past the always_ask
        # pre-check by glob-mismatching the route it actually targets.
        path_only = _split_query(str(args.get("path", "")))[0]
        route = (
            _match_route(server, path_only, str(args.get("method", "")))
            if server is not None
            else None
        )
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

        # Idempotency opt-in (#830): if the author wrote the sentinel as an
        # ``Idempotency-Key`` header value, substitute the real per-call token worker-side
        # — mirroring the bash path's ``$AIOS_IDEMPOTENCY_KEY`` env opt-in ("pass it OR
        # knowingly accept at-least-once"). A call that doesn't ask keeps at-least-once;
        # a literal author value is the author's own key, left intact. Substituting on a
        # shallow copy keeps the script-visible / journaled ``tool_input`` verbatim.
        args = _substitute_idempotency_sentinel(args, run.id, call_key)

        return await _values(
            _do_http_request(
                servers=run.http_servers, arguments=args, resolve_auth=resolve_auth
            )
        )
    if tool_name == "web_search":
        return await _values(web_search_handler("", args))
    return await _values(web_fetch_handler("", args))


async def _values(coro: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    """Translate a shared handler's typed failure back into a value for the run path.

    The shared network handlers (``_do_http_request`` / ``web_search`` / ``web_fetch``)
    now signal expected failure by raising a client-class :class:`AiosError` or
    :class:`~aios.tools.invoke.ToolBail` (#1680: one typed failure channel on the session
    path). The workflow-run contract is the opposite — ``invoke_run_tool`` "always returns a
    dict (success or ``{"error": …}``); never raises" — because a run journals failures as
    replay-deterministic values, not exceptions. This is the one seam that bridges the two:
    it re-materializes those two typed arms as ``{"error": msg}``, exactly the dict shape the
    handlers used to return. A server-class (``>= 500``) ``AiosError`` or any other exception
    is a genuine fault, left to propagate to the ``invoke_run_tool`` caller's backstop
    (:func:`run_declared_tool`), which already turns a raise into an ``{"error": …}`` value —
    so the "never raises" contract holds either way. Mirrors ``wf_script_host``'s existing
    value↔exception translation for ``agent()``.
    """
    try:
        return await coro
    except ToolBail as bail:
        return {"error": bail.message, **bail.detail}
    except AiosError as exc:
        if exc.status_code >= 500:
            raise
        return {"error": exc.to_message()}
