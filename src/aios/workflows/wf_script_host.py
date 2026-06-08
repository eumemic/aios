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

``gate()`` and ``agent()`` are real capabilities (for ``agent()`` the parent spawns
a child session and harvests its completion marker). ``parallel()``/``pipeline()``
fan a run out across concurrent branches, scheduled cooperatively by the driver:
each branch keys its own capabilities (so every ``call_key`` is deterministic across
replays), and the whole live frontier is emitted at once when the run suspends.
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


class AgentError(Exception):
    """A spawned ``agent()`` failed to return a value.

    Raised at the ``await agent(...)`` site in the author script — the driver
    ``coro.throw``s it when the call's journaled outcome is an error. So an author
    can ``try/except AgentError`` and continue, or let it propagate to fail the run
    (the bubble). ``kind`` distinguishes the failure mode: ``None`` for an explicit
    ``error()`` from the child, ``"child_errored"`` for a model failure,
    ``"child_gone"`` if the child was archived/deleted before answering,
    ``"timeout"`` if the call outran its wall-clock budget without responding, and
    ``"no_return"`` for an idle-without-responding child (see
    :class:`AgentNoReturnError`).
    """

    def __init__(self, message: str, *, kind: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind


class AgentNoReturnError(AgentError):
    """An ``agent()`` whose child went idle without ever responding (the totality
    backstop). A subtype of :class:`AgentError`, so a blanket ``except AgentError``
    catches it too; catch it explicitly to treat "no response" differently."""


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


class _ParallelYield:
    """The value ``await parallel(...)`` yields up to the driver: the set of branch
    coroutines to run concurrently. The driver returns a results list (one per
    branch, in order; ``None`` for a branch that raised)."""

    __slots__ = ("branches",)

    def __init__(self, branches: list[Any]) -> None:
        self.branches = branches


class _ParallelAwait:
    __slots__ = ("_branches",)

    def __init__(self, branches: list[Any]) -> None:
        self._branches = branches

    def __await__(self) -> Any:
        results = yield _ParallelYield(self._branches)
        return results


def gate(spec: Any = None) -> _Capability:
    """Suspend until an external resume delivers a value for this gate."""
    return _Capability("gate", spec)


def agent(agent_id: str, input: Any, output_schema: Any = None) -> _Capability:
    """Invoke an agent: spawn a child session with ``input`` as its first user
    message and await its ``return``/``error`` result.

    ``input`` is **required and must not be None** — the child needs a real first
    message to act on (a child born with no user message would sit idle forever and
    poison the totality backstop). Pass a ``str`` (delivered verbatim) or any
    JSON-serialisable value (delivered as canonical JSON). The error surfaces as a
    normal exception in the author's script, so a bad call fails the run loudly.
    """
    if input is None:
        raise ValueError("agent() requires a non-None input (the child's first message)")
    return _Capability(
        "agent", {"agent_id": agent_id, "input": input, "output_schema": output_schema}
    )


async def _branch(thunk: Any) -> Any:
    """Run one parallel branch: call the thunk, await whatever it returns. Wrapping
    it in a coroutine lets the driver schedule a uniform set of branches and lets a
    raising thunk/await surface as this branch's exception (→ None in the barrier)."""
    return await thunk()


def parallel(thunks: Any) -> _ParallelAwait:
    """Run ``thunks`` concurrently and return a list of their results, one per thunk
    in order. Each thunk takes no args and returns an awaitable (e.g.
    ``lambda: agent('a', x)``). The barrier tolerates a *failed agent*: a branch
    whose ``agent()`` raises :class:`AgentError` (uncaught) yields ``None`` in its
    slot — so inspect the results for ``None``. Any *other* exception out of a branch
    (an author bug — ``KeyError``, ``TypeError``, …) is **not** absorbed: it fails the
    whole run loudly (fail-hard). The children fan out as a single frontier, so all
    run at once."""
    return _ParallelAwait([_branch(t) for t in thunks])


def _pipeline_thunk(item: Any, stages: tuple[Any, ...]) -> Any:
    async def run() -> Any:
        value = item
        for stage in stages:
            produced = stage(value)
            value = await produced if hasattr(produced, "__await__") else produced
        return value

    return run


def pipeline(items: Any, *stages: Any) -> _ParallelAwait:
    """Run each item through ``stages`` independently and concurrently — each item's
    chain advances on its own, with no barrier between stages. Returns a list of
    final values, one per item in order (``None`` for an item whose chain failed with
    an :class:`AgentError`; any other exception fails the whole run, per
    :func:`parallel`). Each stage is ``stage(value) -> awaitable | value``; a
    non-awaitable return is used as-is (a sync transform). Composition over
    :func:`parallel`."""
    return parallel([_pipeline_thunk(item, stages) for item in items])


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
        # The agent() failure type lives with its capability in the author API
        # (not SAFE_BUILTINS), so `try/except AgentError` resolves it as a global.
        "AgentError": AgentError,
        "AgentNoReturnError": AgentNoReturnError,
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


def _emit_error(emit: Any, repr_: str, tb: str = "", *, kind: str | None = None) -> None:
    """Emit the host's single error terminal (``RAISED``). The sole shape for every
    failure the driver reports — an escaped exception or a driver-detected invariant
    breach — so the parent always parses one frame layout. ``kind`` lets the host
    label a structural failure it detects itself (e.g. the fan-out cap) so the parent
    sees a specific ``error_kind``; an uncaught author exception carries none."""
    emit({"type": RAISED, "repr": repr_, "traceback": tb, "kind": kind})


def _emit_raised(emit: Any, exc: BaseException) -> None:
    _emit_error(emit, _exc_repr(exc), traceback.format_exc())


_AGENT_ERROR_DEFAULT_MESSAGES: dict[str | None, str] = {
    "child_errored": "the agent errored before responding to the request",
    "no_return": "the agent went idle without responding to the request",
    "child_gone": "the agent was archived or deleted before responding to the request",
    "timeout": "the agent did not respond within its wall-clock deadline",
}


def _agent_error_from(error_info: Any) -> AgentError:
    """Build the ``AgentError`` to throw at an ``await agent(...)`` from a memo
    outcome's ``error`` payload (``{kind?, message?}``). ``no_return`` maps to the
    :class:`AgentNoReturnError` subtype; everything else to the base class."""
    info = error_info if isinstance(error_info, dict) else {}
    kind = info.get("kind")
    message = info.get("message") or _AGENT_ERROR_DEFAULT_MESSAGES.get(kind, "the agent failed")
    if kind == "no_return":
        return AgentNoReturnError(message, kind=kind)
    return AgentError(message, kind=kind)


# ─── the manual .send() driver (a deterministic cooperative scheduler) ────────

# Width ceiling for a single ``parallel()``/``pipeline()`` fan-out: the run RAISES
# before spawning any of an over-wide fan-out's children. A host-side CONSTANT (not
# config) so it is identical across replays — the decision to RAISE is a pure
# function of the script. The lifetime total across all calls is bounded separately,
# parent-side, by ``Settings.workflow_max_agent_calls``.
MAX_PARALLEL_FANOUT = 1000


class _Branch:
    """One coroutine the scheduler is driving — the root ``main``, or a ``parallel``
    sub-branch. ``send``/``throw`` is its pending resume; ``blocked`` means it is
    waiting (on a frontier capability, or on its parallel children to finish);
    ``join``/``index`` say which ``parallel`` result slot it fills when it ends.

    ``keyer``/``path`` make ``call_key``s **branch-local**: each branch numbers its
    own capabilities (per-content-hash ordinals from its own ``CallKeyer``), and the
    ``path`` — a deterministic prefix derived from the parallel structure (root is
    ``""``, a sub-branch is ``"{parent_path}{kth_parallel}.{i}/"``) — disambiguates
    the same content hash across branches. Without this, ordinals would be assigned
    in *sweep* order, which is memo-dependent, so two same-hash branches of unequal
    depth would swap keys across replays. ``n_parallels`` counts the child-spawning
    (non-empty) parallels this branch has opened, to number their children's paths
    deterministically — an empty ``parallel([])`` resolves to ``[]`` inline and is
    not numbered (its emptiness is structural, so this stays replay-stable)."""

    __slots__ = (
        "blocked",
        "coro",
        "done",
        "index",
        "join",
        "keyer",
        "n_parallels",
        "path",
        "send",
        "throw",
    )

    def __init__(
        self,
        coro: Any,
        *,
        path: str,
        join: _Join | None = None,
        index: int = -1,
    ) -> None:
        self.coro = coro
        self.send: Any = None
        self.throw: BaseException | None = None
        self.blocked = False
        self.done = False
        self.join = join
        self.index = index
        self.path = path
        self.keyer = CallKeyer()
        self.n_parallels = 0


class _Join:
    """A ``parallel`` barrier: the parent branch is unblocked with ``results`` once
    all of its child branches have finished (each filling its slot; ``None`` on a
    raised child — the barrier never fails as a whole)."""

    __slots__ = ("parent", "remaining", "results")

    def __init__(self, parent: _Branch, n: int) -> None:
        self.parent = parent
        self.results: list[Any] = [None] * n
        self.remaining = n


def _drive(root_coro: Any, memo: dict[str, Any], emit: Any) -> None:
    """Drive one wake of the author script. A single coroutine is the common case;
    ``parallel``/``pipeline`` introduce concurrent branches, which this schedules
    cooperatively in a fixed (creation-order) sweep — so the capability emission
    order, and thus every ``call_key``, is deterministic across replays."""
    root = _Branch(root_coro, path="")  # root keys are unprefixed (back-compatible)
    branches: list[_Branch] = [root]
    frontier: dict[str, _Yield] = {}

    def _finish(branch: _Branch, result: Any) -> None:
        # Feed a finished sub-branch's result into its parallel join. Only ever called
        # for a non-root branch, which always carries a join (root is handled inline
        # by the caller and never reaches here). A raised branch passes result=None
        # (the barrier); a returned branch passes its value.
        branch.done = True
        join = branch.join
        assert join is not None  # non-root ⇒ join present, by construction
        join.results[branch.index] = result
        join.remaining -= 1
        if join.remaining == 0:
            join.parent.send = join.results
            join.parent.blocked = False

    while True:
        progressed = False
        for branch in list(branches):  # snapshot: parallel appends sub-branches mid-sweep
            if branch.done or branch.blocked:
                continue
            try:
                # A journaled error outcome is delivered as a throw AT the await, so
                # the author's try/except sees it; everything else is a resume value.
                if branch.throw is not None:
                    yielded = branch.coro.throw(branch.throw)
                else:
                    yielded = branch.coro.send(branch.send)
            except StopIteration as stop:
                progressed = True
                if branch is root:
                    emit({"type": RETURNED, "value": stop.value})
                    return
                _finish(branch, stop.value)
                continue
            except BaseException as exc:
                progressed = True
                # The barrier absorbs a *failed agent* — any AgentError reaching a
                # sub-branch — as a None slot; that is the point of parallel. Matching
                # by type (not by provenance) means an author who lets an AgentError
                # escape a branch, even one they raised directly, opts that branch into
                # the same None: AgentError IS the agent-failure signal. Any OTHER
                # exception (a KeyError in the branch's own glue, a root error) is an
                # author bug → fail the whole run loudly (fail-hard), never a silent None.
                if branch is root or not isinstance(exc, AgentError):
                    _emit_raised(emit, exc)
                    return
                _finish(branch, None)  # barrier: a failed agent → None slot
                continue
            branch.send = None
            branch.throw = None
            progressed = True

            if isinstance(yielded, _ParallelYield):
                if not yielded.branches:
                    branch.send = []  # empty parallel resumes immediately with []
                    continue
                if len(yielded.branches) > MAX_PARALLEL_FANOUT:
                    # Fail before spawning any child — a deterministic, structural
                    # (memo-independent) limit, so it RAISES identically on replay.
                    _emit_error(
                        emit,
                        f"parallel() fan-out of {len(yielded.branches)} exceeds the "
                        f"{MAX_PARALLEL_FANOUT} cap",
                        kind="too_wide_fanout",
                    )
                    return
                join = _Join(branch, len(yielded.branches))
                branch.blocked = True  # unblocks when its children all finish
                # Children's paths are deterministic — "{parent}{kth_parallel}.{i}/"
                # — so their call_keys are stable across replays regardless of the
                # sweep/emission order.
                for i, child in enumerate(yielded.branches):
                    child_path = f"{branch.path}{branch.n_parallels}.{i}/"
                    branches.append(_Branch(child, path=child_path, join=join, index=i))
                branch.n_parallels += 1
                continue

            if not isinstance(yielded, _Yield):
                # Awaiting a non-capability value is an author bug in ANY branch →
                # fail the run loudly (fail-hard), never a silent None slot.
                _emit_error(
                    emit, f"workflow awaited a non-capability value: {type(yielded).__name__}"
                )
                return

            # A capability (agent/gate). The call_key is branch-local: the branch's
            # own per-content-hash ordinal, prefixed by its deterministic path. A bad
            # input raises here (in the driver, not the coro) — a deterministic author
            # error, so fail the whole run loudly rather than None-ing the branch.
            try:
                call_key = branch.path + branch.keyer.next(yielded.capability_id, yielded.spec)
            except BaseException as exc:
                _emit_raised(emit, exc)
                return
            if call_key in memo:
                # FAST-FORWARD a journaled outcome: `{ok}` resumes with the value,
                # `{error}` throws AgentError at the await (catchable by the author).
                outcome = memo[call_key]
                if isinstance(outcome, dict) and "error" in outcome:
                    branch.throw = _agent_error_from(outcome["error"])
                else:
                    branch.send = outcome["ok"]
            else:
                branch.blocked = True  # awaits this frontier capability's resolution
                frontier[call_key] = yielded

        if not progressed:
            break

    # No branch can advance and root has not returned → suspend on the frontier (the
    # whole set of unresolved capabilities, fanned out at once for parallel). The
    # frontier is non-empty here by construction: a stuck branch is blocked on a join
    # or a frontier capability, and a join only stays blocked while some child is
    # itself unfinished — which recurses down to a frontier-blocked leaf — so at least
    # one capability is always open. The empty case is thus unreachable; assert it
    # loudly rather than emit a frontier-less SUSPENDED that would park the run forever
    # (a silent hang is the one failure mode fail-hard must never produce).
    if not frontier:
        _emit_error(emit, "workflow deadlocked with no frontier")
        return
    for call_key, yielded in frontier.items():
        emit(
            {
                "type": EMIT,
                "capability_id": yielded.capability_id,
                "call_key": call_key,
                "spec": yielded.spec,
            }
        )
    emit({"type": SUSPENDED})


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
