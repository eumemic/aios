"""Run-side sandbox execution — a workflow run runs a shell command in its own
ephemeral sandbox (#988).

The structural twin of :mod:`aios.workflows.run_tools`, but for the ``sandbox()``
capability instead of ``tool()``: the step opens a ``sandbox`` frontier (journals
``call_started``), launches a fire-and-forget **worker** task here, and parks the
run. The task provisions (or reuses) the run's scratch sandbox, runs the command,
and on completion writes a ``sandbox_result`` ``wf_run_signals`` row + wakes the
run; the next step's pre-replay harvest folds the signal into a ``call_result``.
The run never holds its lock/slot while a command runs.

**Signal envelope (asymmetric with the tool path — deliberate).** Success writes
``result={"ok": bash_dict}`` where ``bash_dict`` is the bash-tool result
(``{exit_code, stdout, stderr, timed_out, truncated}``); an *infrastructure*
failure writes ``result={"error": {"capability": "sandbox", "code", "message"}}``.
The one-level ``ok``/``error`` wrap is what lets the harvest tell "the command
ran" (a value the script branches on, even a nonzero exit) from "the sandbox
failed" (a :class:`~aios.workflows.wf_script_host.SandboxError` thrown at the
await). A command that ran and hit its own ``timeout_s`` is a SUCCESSFUL result
carrying ``timed_out=True`` — not an error.

**At-least-once.** A hard worker crash mid-task leaves no signal; the periodic
sweep re-wakes the run and the step re-dispatches. Sound because the run sandbox
is ephemeral scratch — a re-executed command runs against a fresh (or reused)
scratch container, never corrupting durable state (the #784/#795 dimension).
"""

from __future__ import annotations

import asyncio
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.logging import get_logger
from aios.models.workflows import WfRun
from aios.sandbox.backends.base import SandboxBackendError
from aios.services.wake import defer_run_wake

log = get_logger("aios.workflows.run_sandbox")

# Per-worker in-flight sandbox tasks, keyed (run_id, call_key). Its OWN map,
# separate from run_tools' — a distinct concern with a distinct lifecycle. Gates
# *launching* (so a sibling-triggered re-wake doesn't double-dispatch a still-
# running command); never gates *harvesting* (the signal in the DB is the truth).
_INFLIGHT: dict[tuple[str, str], asyncio.Task[None]] = {}

# Strong refs for the fire-and-forget teardown tasks (see ``teardown_run_sandbox``):
# asyncio only weak-refs tasks, so without this the release task could be GC'd
# before it finishes destroying the container.
_TEARDOWN_TASKS: set[asyncio.Task[None]] = set()


def has_inflight(run_id: str, call_key: str) -> bool:
    """True iff a live sandbox task for ``(run_id, call_key)`` is running on this worker."""
    task = _INFLIGHT.get((run_id, call_key))
    return task is not None and not task.done()


def launch_sandbox_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    command: str,
    timeout_s: float | None,
) -> None:
    """Launch the worker task for a freshly-opened sandbox frontier (no-op if live)."""
    key = (run.id, call_key)
    if has_inflight(*key):
        return
    _INFLIGHT[key] = asyncio.create_task(
        _run_sandbox_task(pool, run, call_key=call_key, command=command, timeout_s=timeout_s)
    )


async def _run_sandbox_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    command: str,
    timeout_s: float | None,
) -> None:
    """Run one command in the run's sandbox, write its ``sandbox_result`` signal, and
    wake the run (always-signals)."""
    try:
        result = await _execute(run, command=command, timeout_s=timeout_s)
        try:
            async with pool.acquire() as conn:
                await wf_queries.insert_run_signal(
                    conn, run_id=run.id, call_key=call_key, kind="sandbox_result", result=result
                )
            # batch: sandbox results are a high-frequency wake source, like tool
            # results and child completions — a burst coalesces into one re-drive.
            await defer_run_wake(run.id, batch=True)
        except Exception:
            # The command ran but persisting/waking failed (DB blip, pool
            # exhaustion). If the signal committed and only the wake failed, the
            # sweep's unharvested-signal clause re-wakes within a tick; with no
            # signal at all, the stale-call clause re-wakes at the horizon and the
            # harvest re-dispatches. Log so the stall is diagnosable.
            log.exception("run_sandbox.signal_failed", run_id=run.id, call_key=call_key)
    finally:
        # CancelledError (worker shutdown) propagates here with no signal written —
        # the periodic sweep re-wakes the run and the step re-dispatches.
        _INFLIGHT.pop((run.id, call_key), None)


async def _execute(run: WfRun, *, command: str, timeout_s: float | None) -> dict[str, Any]:
    """Provision (or reuse) the run's sandbox and run ``command``, returning the
    signal envelope: ``{"ok": bash_dict}`` on success, ``{"error": …}`` on infra
    failure.

    The provision and exec calls are wrapped separately so the error ``code``
    distinguishes a sandbox that couldn't be brought up (``provision_failed``) from
    a command that couldn't be dispatched into a live sandbox (``exec_failed``). A
    command that *ran* and hit its ``timeout_s`` returns a normal ``{"ok": …}``
    result carrying ``timed_out=True`` — only an infra-layer raise is an error.
    """
    settings = get_settings()
    registry = runtime.require_sandbox_registry()
    pool = runtime.require_pool()

    try:
        handle = await registry.get_or_provision_run(run.id, pool=pool)
    except SandboxBackendError as exc:
        log.warning("run_sandbox.provision_failed", run_id=run.id, error=str(exc))
        return {"error": {"capability": "sandbox", "code": "provision_failed", "message": str(exc)}}

    timeout_seconds = (
        int(timeout_s) if timeout_s is not None else settings.bash_default_timeout_seconds
    )
    try:
        result = await registry.exec(
            handle,
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=settings.bash_max_output_bytes,
            cwd="/workspace",
        )
    except SandboxBackendError as exc:
        log.warning("run_sandbox.exec_failed", run_id=run.id, error=str(exc))
        return {"error": {"capability": "sandbox", "code": "exec_failed", "message": str(exc)}}

    return {
        "ok": {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        }
    }


def teardown_run_sandbox(run_id: str) -> None:
    """Best-effort fire-and-forget release of a run's ephemeral sandbox at run
    termination (#988, Q8).

    Schedules ``registry.release_run(run_id)`` as a detached task (holding a strong
    ref so it isn't GC'd mid-destroy) and NEVER raises — it is called from the
    terminal commit path, where a teardown hiccup must not corrupt the run's
    completion. A missing registry (e.g. a test that wired only the DB) or any
    other error is logged and swallowed. If the release is dropped (worker crash),
    the idle reaper reclaims the orphaned run sandbox on its next tick.
    """
    try:
        registry = runtime.require_sandbox_registry()
    except RuntimeError:
        # No registry on this worker (e.g. unit/integration test harness): nothing
        # to tear down. The reaper / next provision converges if one exists later.
        return
    task = asyncio.create_task(
        registry.release_run(run_id), name=f"wf-run-sandbox-teardown:{run_id}"
    )
    _TEARDOWN_TASKS.add(task)

    def _done(t: asyncio.Task[None]) -> None:
        _TEARDOWN_TASKS.discard(t)
        if not t.cancelled() and (exc := t.exception()) is not None:
            log.warning("run_sandbox.teardown_failed", run_id=run_id, error=str(exc))

    task.add_done_callback(_done)
