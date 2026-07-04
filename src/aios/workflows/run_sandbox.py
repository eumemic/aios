"""Run-side ``bash`` execution — a workflow run runs a shell command in its own
ephemeral sandbox (#988).

The execution-class twin of :mod:`aios.workflows.run_tools`: ``bash`` rides the
SAME ``tool`` capability as the network/credential tools, but its handler needs a
provisioned container, so the step routes it here (by the tool's ``executes``
class) instead of to the worker tool path. The shape is otherwise identical — the
step journals ``call_started``, launches a fire-and-forget worker task, and parks
the run; the task provisions (or reuses) the run's scratch sandbox, runs the
command, writes a ``tool_result`` ``wf_run_signals`` row + wakes the run; the next
step's pre-replay harvest folds ``sig.result`` straight into the ``call_result``.
The run never holds its lock/slot while the command runs.

**Result is a value, not an envelope.** Because bash is a tool, ``tool()`` never
raises — every outcome is a value the script branches on. Success writes the BARE
bash dict (``{exit_code, stdout, stderr, timed_out, truncated}``); a malformed
argument, a gate rejection, or an infra failure (provision/exec) writes a flat
``{"error": str}``. A command that RAN and hit its own ``timeout_seconds`` is a success
carrying ``timed_out=True`` — not an error. There is no ``{"ok"}/{"error"}`` wrap
and no ``SandboxError``; the harvest folds the value unchanged (the whole point of
routing through the existing ``tool`` capability).

**Shared ``_INFLIGHT``.** Launch-gating uses :data:`run_tools._INFLIGHT` (keyed
``(run_id, call_key)``, class-agnostic) so the step's single ``has_inflight``
guard covers both worker and sandbox tools; only the launcher differs by class.

**At-least-once.** A hard worker crash mid-task leaves no signal; the periodic
sweep re-wakes the run and the step re-dispatches. Sound because the run sandbox
is ephemeral scratch — a re-executed command runs against a fresh (or reused)
scratch container, never corrupting durable state. An author who needs an
*external* effect to dedupe across a crash re-drive reads ``$AIOS_IDEMPOTENCY_KEY``
(exported into the command's env) and passes it to the external service.
"""

from __future__ import annotations

import asyncio
import math
import shlex
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.db.queries import workflows as wf_queries
from aios.harness import runtime
from aios.jobs.app import defer_run_wake
from aios.logging import get_logger
from aios.models.workflows import WfRun
from aios.sandbox.env_keys import (
    AIOS_IDEMPOTENCY_KEY_ENV_KEY,
    AIOS_RUN_ID_ENV_KEY,
)
from aios.workflows import run_tools
from aios.workflows.idempotency_key import idempotency_key

log = get_logger("aios.workflows.run_sandbox")

# Strong refs for the fire-and-forget teardown tasks (see ``teardown_run_sandbox``):
# asyncio only weak-refs tasks, so without this the release task could be GC'd
# before it finishes destroying the container.
_TEARDOWN_TASKS: set[asyncio.Task[None]] = set()


def launch_sandbox_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    tool_name: str,
    tool_input: Any,
) -> None:
    """Launch the worker task for a freshly-opened sandbox-tool frontier (no-op if
    already live).

    Registers the task in the SHARED :data:`run_tools._INFLIGHT` so the step's
    class-agnostic ``run_tools.has_inflight`` guard covers it.
    """
    key = (run.id, call_key)
    if run_tools.has_inflight(*key):
        return
    run_tools._INFLIGHT[key] = asyncio.create_task(
        _run_sandbox_task(pool, run, call_key=call_key, tool_name=tool_name, tool_input=tool_input)
    )


async def _run_sandbox_task(
    pool: asyncpg.Pool[Any],
    run: WfRun,
    *,
    call_key: str,
    tool_name: str,
    tool_input: Any,
) -> None:
    """Run one sandbox tool, write its ``tool_result`` signal, and wake the run
    (always-signals — mirrors ``run_tools._run_tool_task``)."""
    try:
        try:
            result = await _execute(
                run, call_key=call_key, tool_name=tool_name, tool_input=tool_input
            )
        except Exception as exc:
            # Backstop — _execute maps gate/validation/provision/exec to their own
            # error values, but its pre-phase setup (require_sandbox_registry /
            # get_settings) raises BEFORE either guarded block. An uncaught escape
            # would skip the signal write and, with the stale-call sweep horizon,
            # park the run forever.
            log.exception("run_sandbox.unexpected", run_id=run.id, call_key=call_key)
            result = {"error": f"sandbox task failed: {type(exc).__name__}: {exc}"}
        try:
            async with pool.acquire() as conn:
                await wf_queries.insert_run_signal(
                    conn, run_id=run.id, call_key=call_key, kind="tool_result", result=result
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
        run_tools._INFLIGHT.pop((run.id, call_key), None)


async def _execute(run: WfRun, *, call_key: str, tool_name: str, tool_input: Any) -> dict[str, Any]:
    """Provision (or reuse) the run's sandbox and run the bash command, returning the
    tool_result VALUE: the bare bash dict on success, ``{"error": str}`` on any
    recoverable failure (gate rejection, malformed argument, provision/exec error).

    Errors are values, never raises: gating reuses the SAME
    :func:`run_tools.gate_run_tool` the worker path uses (single source of the
    error strings); timeout validation is a recoverable value (a bad ``timeout_seconds``
    does NOT terminally error the run); and the provision/exec phases catch broadly
    (``build_spec_from_run``/``_resolve_image`` raise plain ``ValueError`` and
    asyncpg errors, not just ``SandboxBackendError``) so an infra failure becomes a
    branchable value the run can recover from.

    **Idempotency preamble.** The command handed to ``backend.exec`` is prefixed
    with ``export AIOS_RUN_ID=… AIOS_IDEMPOTENCY_KEY=…`` so an author can pass
    ``$AIOS_IDEMPOTENCY_KEY`` to an external service as an idempotency key (so the
    service dedupes an effect re-fired by a crash re-drive). The key is the sha256
    of ``run_id\\0call_key`` — opaque and fixed-width, so the structural punctuation
    of a raw ``call_key`` never leaks into the author's environment. The preamble is
    prepended ONLY to the execed string — the journaled ``call_started`` command
    stays the verbatim author command, so replay / ``call_key`` derivation are
    unaffected. A re-drive carries the SAME ``run.id`` + ``call_key`` → a
    byte-identical key; distinct calls differ.
    """
    if (err := run_tools.gate_run_tool(run, tool_name)) is not None:
        return err

    args = tool_input if isinstance(tool_input, dict) else {}
    command = args.get("command")
    if not isinstance(command, str) or not command:
        return {"error": "bash tool requires a non-empty 'command' string"}

    timeout_seconds = args.get("timeout_seconds")
    # Validate ``timeout_seconds`` as a recoverable value (bash is a tool — tool() never
    # raises). ``bool`` is excluded explicitly (it is an ``int`` subclass, so
    # ``True``/``False`` would otherwise pass the numeric check); NaN/inf are
    # rejected via ``math.isfinite``. Passing this gate means only ``None`` or a
    # finite positive number reaches ``int()`` below.
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return {
            "error": f"timeout_seconds must be None or a finite positive number, got {timeout_seconds!r}"
        }

    settings = get_settings()
    registry = runtime.require_sandbox_registry()

    try:
        handle = await registry.get_or_provision_run(run.id)
    except Exception as exc:
        log.warning("run_sandbox.provision_failed", run_id=run.id, error=str(exc))
        return {"error": f"sandbox provisioning failed: {exc}"}

    idem = idempotency_key(run.id, call_key)
    preamble = (
        f"export {AIOS_RUN_ID_ENV_KEY}={shlex.quote(run.id)} "
        f"{AIOS_IDEMPOTENCY_KEY_ENV_KEY}={shlex.quote(idem)}\n"
    )
    # Resolve the int container deadline the SAME way the bash tool does (#988):
    # floor a positive request to 1, then clamp to the bash ceiling. The floor is
    # load-bearing: the in-container ``timeout`` is GNU coreutils, which treats a
    # DURATION of 0 as "no limit", so a bare ``int(0.5) == 0`` would run UNBOUNDED.
    # ``max(1, int(...))`` keeps truncation (2.7 → 2) while guaranteeing a positive
    # request never disables the timeout; the ``min`` clamps to the same ceiling the
    # bash tool uses.
    ceiling = settings.bash_default_timeout_seconds
    resolved_timeout_seconds = (
        min(ceiling, max(1, int(timeout_seconds))) if timeout_seconds is not None else ceiling
    )
    try:
        result = await registry.exec(
            handle,
            preamble + command,
            timeout_seconds=resolved_timeout_seconds,
            max_output_bytes=settings.bash_max_output_bytes,
            cwd="/workspace",
        )
    except Exception as exc:
        log.warning("run_sandbox.exec_failed", run_id=run.id, error=str(exc))
        return {"error": f"sandbox exec failed: {exc}"}

    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }


def teardown_run_sandbox(run_id: str) -> None:
    """Best-effort fire-and-forget release of a run's ephemeral sandbox at run
    termination (#988).

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
