"""The bash tool — run a shell command inside the session's sandbox.

Phase 3's headline tool. Each call runs ``bash -c <command>`` inside the
session's sandbox container via :meth:`SandboxRegistry.exec`. The
container is lazily provisioned on the first bash call (or any tool
call) and reused for subsequent calls within the same turn.

Tool call schema:

.. code-block::

    {
        "name": "bash",
        "arguments": {
            "command": "curl -s https://news.ycombinator.com > /workspace/hn.html",
            "timeout_seconds": 60  // optional, capped at bash_default_timeout_seconds
        }
    }

The ``command`` parameter is a single shell string, interpreted by
``bash -c`` inside the sandbox. The agent can use pipes, redirects,
subshells, ``&&``, ``||``, etc. — anything bash supports.

Return shape (the dict returned by the handler, passed through to the
tool-role message's ``content`` by :mod:`aios.harness.tool_dispatch`):

.. code-block::

    {
        "exit_code": 0,
        "stdout": "...",
        "stderr": "...",
        "timed_out": false,
        "truncated": false
    }

A nonzero exit code is NOT an error at the handler level — it's normal
output; the model will see it and react. An ``is_error`` tool message is
only produced when the docker call itself fails (container dead,
daemon unreachable, host error). That path is handled by
:mod:`aios.harness.tool_dispatch`, not here.

One-shot semantics: no shell state persists between calls (see
``aios_v1_decisions.md``). ``cd``, ``export``, and shell functions are
gone after the command returns. The agent should chain with ``&&`` or
use absolute paths.
"""

from __future__ import annotations

from typing import Any

from aios.config import get_settings
from aios.errors import AiosError
from aios.harness import runtime
from aios.sandbox.spec import resolve_bash_timeout_ceiling
from aios.tools.bash_memory_reconcile import reconcile_memory_mounts, snapshot_memory_mounts
from aios.tools.registry import registry


class BashArgumentError(AiosError):
    """Raised when the bash tool is called with malformed arguments."""

    error_type = "bash_argument_error"
    status_code = 400


BASH_DESCRIPTION = (
    "Run a shell command inside the session's sandbox. The command is "
    "interpreted by `bash -c` inside a Linux container with curl, git, "
    "python3, ripgrep, and standard coreutils available. The sandbox "
    "filesystem persists across calls and idle periods: files you write and "
    "packages you install (pip, npm, apt) survive between calls and while the "
    "session is idle, subject to retention limits — if it is reset to reclaim "
    "space or after long inactivity you are told, and can reinstall as needed. "
    "The working directory is /workspace, which holds the session's working "
    "data and is never reclaimed. Shell state (cd, export, functions) does NOT "
    "persist between calls; chain commands with && or use absolute paths. The "
    "default timeout is 120 seconds unless this session's environment raises "
    "it. A nonzero exit code is reported in the result — not every nonzero "
    "exit is a failure."
)

BASH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": (
                "The shell command to run. Interpreted by `bash -c` inside the sandbox container."
            ),
        },
        "timeout_seconds": {
            "type": "integer",
            "description": (
                "Optional maximum runtime in seconds. Defaults to the "
                "server's configured default. Capped at the server's "
                "maximum. If the command exceeds this duration it is "
                "terminated (SIGKILL) inside the sandbox; any output "
                "produced up to that point is returned and `timed_out` "
                "is set true in the result."
            ),
            "minimum": 1,
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}


async def bash_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one bash command inside the session's sandbox container.

    Provisions a container lazily via the sandbox registry if this is
    the session's first tool call. Returns the structured result dict;
    any exception (docker daemon down, container crashed) propagates up
    to :mod:`aios.harness.tool_dispatch`, which turns it into an
    ``is_error=true`` tool message and evicts the (possibly dead)
    container from the registry.
    """
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        raise BashArgumentError("bash tool requires a non-empty 'command' string")

    settings = get_settings()
    # The ceiling is the environment's ``bash_timeout_seconds`` when the
    # session is bound to one that sets it, else the worker-global
    # ``bash_default_timeout_seconds`` (issue #725). Resolved per-call
    # (one cheap DB read, dwarfed by the container exec it gates) so an
    # environment-config change takes effect on the session's next bash
    # call without a worker restart.
    max_timeout = await resolve_bash_timeout_ceiling(session_id)
    raw_timeout = arguments.get("timeout_seconds", max_timeout)
    if not isinstance(raw_timeout, int) or raw_timeout <= 0:
        raise BashArgumentError("timeout_seconds must be a positive integer")
    timeout = min(raw_timeout, max_timeout)

    sandbox = runtime.require_sandbox_registry()
    handle = await sandbox.get_or_provision(session_id, pool=runtime.require_pool())

    before = snapshot_memory_mounts(session_id)
    reconcile_warnings: list[str] = []
    exec_raised = False

    try:
        result = await sandbox.exec(
            handle,
            command,
            timeout_seconds=timeout,
            max_output_bytes=settings.bash_max_output_bytes,
        )
    except Exception:
        exec_raised = True
        raise
    finally:
        # Reconcile unconditionally — even when exec raises (e.g. container death)
        # so that partial writes made before the crash are captured in the DB.
        # reconcile_memory_mounts returns [] immediately when there are no mounts,
        # so this is a cheap no-op in the common case.
        # When exec succeeded, reconcile errors propagate; when exec already raised,
        # we suppress them so the original exception is not shadowed.
        try:
            reconcile_warnings = await reconcile_memory_mounts(session_id, before)
        except Exception:
            if not exec_raised:
                raise  # fail hard when exec succeeded

    stderr = result.stderr
    if reconcile_warnings:
        suffix = "\n[memory-reconcile] " + "; ".join(reconcile_warnings)
        stderr = stderr + suffix

    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": stderr,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }


def _register() -> None:
    registry.register(
        name="bash",
        description=BASH_DESCRIPTION,
        parameters_schema=BASH_PARAMETERS_SCHEMA,
        handler=bash_handler,
        transport="agent_tool",
        executes="sandbox",
    )


_register()
