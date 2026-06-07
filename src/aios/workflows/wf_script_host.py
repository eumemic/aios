"""The out-of-process workflow script host (the child process).

Run as ``python -m aios.workflows.wf_script_host``. Reads one ``INIT`` frame from
stdin (the pinned source, the run input, and the memo of already-resolved
results), executes the author script, and drives it with a manual ``.send()``
pump: memoized capability results are fed straight back (fast-forward), and at
the first *unresolved* capability the host emits the frontier + a ``SUSPENDED``
frame and exits. The coroutine is never persisted — a fresh host rebuilds it from
the journal on the next wake.

ISOLATION (load-bearing): this module imports **only** stdlib +
``aios.workflows.determinism`` / ``._protocol`` (themselves stdlib-only). It must
never import ``aios.harness.*`` / ``aios.crypto*`` / ``aios.db*`` / ``aios.services.*``
/ ``aios.tools.*`` / ``aios.sandbox.spec`` — those hold the worker's master
``CryptoBox`` and all-accounts pool. Because the child is a fresh ``spawn``ed
interpreter that never imports them, even a full sandbox escape in author code
reaches only this credential-free process. ``SAFE_BUILTINS`` and the absence of
``__import__`` are determinism/footgun aids, **not** the security boundary.

Block 1: ``gate()`` is a real capability; ``agent()`` emits a frontier the parent
rejects (Block 2); ``parallel()``/``pipeline()`` raise (Block 2).
"""

from __future__ import annotations

import builtins
import contextlib
import os
import sys
import traceback
from typing import Any

from aios.workflows._protocol import (
    EMIT,
    INIT,
    RAISED,
    RETURNED,
    SUSPENDED,
    read_frame_sync,
    write_frame_sync,
)
from aios.workflows.determinism import CallKeyer


class WorkflowScriptError(Exception):
    """The author script has the wrong shape (no ``async def main(input)``)."""


# ─── the capability awaitables (author-facing API) ───────────────────────────


class _Yield:
    """The single value every workflow ``await`` yields up to the driver."""

    __slots__ = ("capability_id", "spec")

    def __init__(self, capability_id: str, spec: Any) -> None:
        self.capability_id = capability_id
        self.spec = spec


class _Capability:
    __slots__ = ("_capability_id", "_spec")

    def __init__(self, capability_id: str, spec: Any) -> None:
        self._capability_id = capability_id
        self._spec = spec

    def __await__(self) -> Any:
        result = yield _Yield(self._capability_id, self._spec)
        return result


def gate(spec: Any = None) -> _Capability:
    """Suspend until an external resume delivers a value for this gate."""
    return _Capability("gate", spec)


def agent(agent_id: str, input: Any = None, output_schema: Any = None) -> _Capability:
    """Invoke an agent: the parent spawns a child session and awaits its result."""
    return _Capability(
        "agent", {"agent_id": agent_id, "input": input, "output_schema": output_schema}
    )


def parallel(_thunks: Any) -> Any:
    raise NotImplementedError("parallel() lands in Block 2")


def pipeline(_items: Any, *_stages: Any) -> Any:
    raise NotImplementedError("pipeline() lands in Block 2")


def log(*args: Any) -> None:
    """Emit a diagnostic to stderr (never stdout — that's the frame stream).

    Re-fires on every replay; kept to stderr so it can't corrupt the protocol or
    the journal.
    """
    sys.stderr.write(" ".join(str(a) for a in args) + "\n")


# ─── restricted builtins (a footgun aid, NOT the security boundary) ──────────

_SAFE_BUILTIN_NAMES = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytes",
        "callable",
        "chr",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        # constants + exceptions (so authors can raise / except)
        "NotImplemented",
        "Ellipsis",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "Exception",
        "IndexError",
        "KeyError",
        "LookupError",
        "NotImplementedError",
        "RuntimeError",
        "StopIteration",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    }
)
SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)
}


def _build_coroutine(source: str, input_value: Any) -> Any:
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "gate": gate,
        "agent": agent,
        "parallel": parallel,
        "pipeline": pipeline,
        "log": log,
    }
    code = compile(source, "<workflow>", "exec")
    exec(code, namespace)
    entry = namespace.get("main")
    if entry is None or not callable(entry):
        raise WorkflowScriptError("workflow script must define `async def main(input)`")
    coro = entry(input_value)
    if not hasattr(coro, "send"):
        raise WorkflowScriptError("`main` must be an async function (`async def main(input)`)")
    return coro


def _exc_repr(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _emit_raised(emit: Any, exc: BaseException) -> None:
    emit({"type": RAISED, "repr": _exc_repr(exc), "traceback": traceback.format_exc()})


# ─── the manual .send() driver ───────────────────────────────────────────────


def _drive(coro: Any, memo: dict[str, Any], emit: Any) -> None:
    keyer = CallKeyer()
    to_send: Any = None
    while True:
        try:
            yielded = coro.send(to_send)
        except StopIteration as stop:
            emit({"type": RETURNED, "value": stop.value})
            return
        except BaseException as exc:
            _emit_raised(emit, exc)
            return
        if not isinstance(yielded, _Yield):
            emit(
                {
                    "type": RAISED,
                    "repr": f"workflow awaited a non-capability value: {type(yielded).__name__}",
                    "traceback": "",
                }
            )
            return
        try:
            call_key = keyer.next(yielded.capability_id, yielded.spec)
        except BaseException as exc:
            _emit_raised(emit, exc)
            return
        if call_key in memo:
            to_send = memo[call_key]  # FAST-FORWARD a journaled result
            continue
        # FRONTIER: emit it and suspend (suspension is the absence of a resume).
        # content_hash + ordinal are recoverable from call_key, so the frame
        # carries only what the parent can't derive.
        emit(
            {
                "type": EMIT,
                "capability_id": yielded.capability_id,
                "call_key": call_key,
                "spec": yielded.spec,
            }
        )
        emit({"type": SUSPENDED})
        return


def _apply_resource_limits() -> None:
    """Self-impose CPU + address-space ceilings from the parent's env, before any
    author code runs. Unix-only; a no-op if unset/unsupported (the parent's
    wall-clock SIGKILL deadline is the platform-independent backstop)."""
    try:
        import resource
    except ImportError:
        return
    cpu = os.environ.get("AIOS_WF_RLIMIT_CPU_S")
    if cpu:
        with contextlib.suppress(ValueError, OSError):
            secs = int(cpu)
            resource.setrlimit(resource.RLIMIT_CPU, (secs, secs))
    as_bytes = os.environ.get("AIOS_WF_RLIMIT_AS_BYTES")
    if as_bytes and as_bytes != "0":
        with contextlib.suppress(ValueError, OSError):
            n = int(as_bytes)
            resource.setrlimit(resource.RLIMIT_AS, (n, n))


def main() -> int:
    _apply_resource_limits()
    stdout = sys.stdout.buffer

    def emit(obj: dict[str, Any]) -> None:
        write_frame_sync(stdout, obj)

    try:
        init = read_frame_sync(sys.stdin.buffer)
    except (EOFError, ValueError):
        return 1
    if init is None or init.get("type") != INIT:
        return 1

    try:
        coro = _build_coroutine(init["source"], init.get("input"))
    except BaseException as exc:
        _emit_raised(emit, exc)
        return 0

    _drive(coro, init.get("memo") or {}, emit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
