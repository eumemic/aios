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
reaches only this credential-free process. ``SAFE_BUILTINS`` and the curated
``__import__`` allowlist are determinism/footgun aids, **not** the security boundary.

``gate()`` and ``agent()`` are real capabilities (for ``agent()`` the parent spawns
a child session and harvests its completion marker). ``parallel()``/``pipeline()``
fan a run out across concurrent branches, scheduled cooperatively by the driver:
each branch keys its own capabilities (so every ``call_key`` is deterministic across
replays), and the whole live frontier is emitted at once when the run suspends.

The model-facing statement of this contract is
``aios.models.workflows.WORKFLOW_SCRIPT_CONTRACT``. The two are kept in sync
structurally: the injected author namespace lives in ``author_namespace`` and
``tests/unit/test_workflow_script_contract_drift.py`` asserts every public
capability name there appears in the contract prose.
"""

from __future__ import annotations

import builtins
import contextlib
import inspect
import linecache
import os
import sys
import traceback
import types
from typing import Any

from aios.workflows._protocol import (
    ANNOTATION,
    EMIT,
    INIT,
    RAISED,
    RETURNED,
    SUSPENDED,
    read_frame_sync,
    write_frame_sync,
)
from aios.workflows.determinism import (
    CallKeyer,
    canonical_schema_json,
    storable_text,
    validate_value,
)


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
    ``"timeout"`` if the call outran its wall-clock budget without responding,
    ``"no_return"`` for an idle-without-responding child (see
    :class:`AgentNoReturnError`), ``"agent_not_found"`` if the named agent does not
    exist, and ``"bad_agent_call"`` for a malformed call (a non-string ``agent_id``
    or an invalid ``output_schema``).
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

    __slots__ = ("annotations", "capability_id", "spec")

    def __init__(
        self, capability_id: str, spec: Any, annotations: dict[str, Any] | None = None
    ) -> None:
        self.capability_id = capability_id
        self.spec = spec
        self.annotations = annotations or {}


class _Capability:
    __slots__ = ("_annotations", "_capability_id", "_spec")

    def __init__(
        self, capability_id: str, spec: Any, annotations: dict[str, Any] | None = None
    ) -> None:
        self._capability_id = capability_id
        self._spec = spec
        self._annotations = annotations or {}

    def __await__(self) -> Any:
        result = yield _Yield(self._capability_id, self._spec, self._annotations)
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


def agent(
    input: Any,
    *,
    agent_id: str | None = None,
    output_schema: Any = None,
    model: str | None = None,
    label: str | None = None,
) -> _Capability:
    """Invoke an agent child and await its ``return``/``error`` result.

    ``agent_id`` omitted spawns the generic workflow subagent over the run's frozen
    surface. ``agent_id`` supplied spawns that named agent. ``input`` is required and
    must not be None; ``model`` is a per-call model override; ``label`` is an
    observability annotation and does not enter the call key.
    """
    if input is None:
        raise ValueError("agent() requires a non-None input (the child's first message)")
    # Carry output_schema as a canonical JSON *string* so a schema's numeric literals
    # (minimum/multipleOf/…) are preserved verbatim — canonical_json normalises
    # integer-valued floats (1.0 → 1), which would alter schema constraints like
    # "minimum": 1.0. A schema is metadata, not workflow data. The worker
    # reconstructs the dict with json.loads. None stays None (no schema demanded).
    annotations: dict[str, Any] = {}
    if label is not None:
        annotations["label"] = label
    return _Capability(
        "agent",
        {
            "agent_id": agent_id,
            "input": input,
            "output_schema": None
            if output_schema is None
            else canonical_schema_json(output_schema),
            "model": model,
        },
        annotations,
    )


def budget() -> _Capability:
    """Read this run's shared direct-child spend budget, or None when unset."""
    return _Capability("budget", None)


def tool(name: str, input: Any) -> _Capability:
    """Invoke one of the workflow's declared tools and await its result.

    The network/credential tools (``http_request``, ``web_search``, ``web_fetch``)
    run in the worker against the run's bound vaults and declared surface.
    ``'bash'`` — when declared in the workflow's tool surface — runs a shell command
    in the run's own ephemeral sandbox (``cwd="/workspace"``, scratch space that
    lives only for the run); its ``input`` is ``{"command": str, "timeout_seconds":
    float|None}`` and its result is the bash dict the script branches on:
    ``{"exit_code", "stdout", "stderr", "timed_out", "truncated"}``. A nonzero exit
    (or a command that hit its own ``timeout_seconds``, surfacing ``timed_out=True``) is a
    VALUE, not a raise.

    ``input`` is the tool's arguments (a JSON-serialisable dict). The result is the
    tool's own return value — a plain dict the script branches on (e.g. ``{"status":
    200, …}`` or ``{"error": "…"}``); a tool error resolves as a value, it does
    **not** raise. The script subprocess only emits the request; the tool runs on
    the worker (or, for ``'bash'``, against the run's provisioned sandbox).

    **bash re-run tolerance (at-least-once).** The run sandbox is ephemeral scratch,
    and a hard worker crash mid-command re-drives the call — so a ``'bash'`` command
    may run more than once. Write filesystem-side commands to be re-run-tolerant
    (e.g. ``rm -rf <dir>; git clone …``); the scratch container absorbs a re-run
    with no durable damage. An irreversible *external* effect (an HTTP POST, a
    payment) WILL re-fire on a crash re-drive — pass ``$AIOS_IDEMPOTENCY_KEY``
    (exported into the command's environment, alongside ``$AIOS_RUN_ID``) to the
    external service as an idempotency key so it dedupes the re-fired effect. The key
    is stable across re-drives of the same call and distinct across calls.
    """
    return _Capability("tool", {"tool_name": name, "input": input})


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


def _stage_arity(stage: Any) -> int:
    """How many of ``(prev, item, index)`` to hand ``stage`` — inspected once at
    construction. A ``*args`` stage opts into all three; a plain stage gets its count
    of positional parameters, clamped to ``[1, 3]`` (so a 0-param stage still receives
    ``prev``, a >3 stage gets the three available). A callable whose signature can't be
    inspected (e.g. a C builtin) falls back to the legacy 1-arg (prev-only) call."""
    try:
        params = inspect.signature(stage).parameters.values()
    except (ValueError, TypeError):
        return 1  # non-inspectable callable (e.g. C builtin) → legacy 1-arg (prev only)
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return 3  # *args → all three (prev, item, index)
    n = sum(
        1
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return min(max(n, 1), 3)  # clamp: 0-param → 1, >3 → 3


def _pipeline_thunk(
    item: Any, index: int, stages: tuple[Any, ...], arities: tuple[int, ...]
) -> Any:
    async def run() -> Any:
        value = item
        for stage, arity in zip(stages, arities, strict=True):
            args = (value, item, index)[:arity]
            produced = stage(*args)
            value = await produced if hasattr(produced, "__await__") else produced
        return value

    return run


def pipeline(items: Any, *stages: Any) -> _ParallelAwait:
    """Run each item through ``stages`` independently and concurrently — each item's
    chain advances on its own, with no barrier between stages. Returns a list of
    final values, one per item in order (``None`` for an item whose chain failed with
    an :class:`AgentError`; any other exception fails the whole run, per
    :func:`parallel`).

    Each stage may declare 1, 2, or 3 of ``(prev, item, index)``: ``prev`` is the
    previous stage's output (the item itself for the first stage), ``item`` is the
    original element, and ``index`` is its position in ``items``. A stage's arity is
    inspected once at construction. A ``*args`` stage receives all three; a callable
    whose signature can't be inspected receives only ``prev`` (the legacy 1-arg
    shape); every stage must accept at least ``prev``. A stage returns
    ``awaitable | value`` — a non-awaitable return is used as-is (a sync transform).
    Composition over :func:`parallel`."""
    arities = tuple(_stage_arity(s) for s in stages)
    return parallel(
        [_pipeline_thunk(item, index, stages, arities) for index, item in enumerate(items)]
    )


# Annotations (log/phase) produced during the branch step currently being driven.
# A synchronous log()/phase() cannot reach the driver's per-branch keyer directly,
# so it buffers here; the driver drains this right after each branch step, keying
# every entry through THAT branch's keyer+path (see _flush_annotations). The host is
# single-threaded and drives exactly one branch at a time, so a module list is the
# whole of the ambient context.
_pending: list[tuple[str, str]] = []


def log(*args: Any) -> None:
    """Record a progress line on the run's journal (a durable ``annotation`` event,
    surfaced in the run stream). Space-joined like ``print``; re-runs on every replay
    but the journal keeps it emit-once. ``storable_text`` neutralizes the NUL/unpaired-
    surrogate bytes that arbitrary logged output may carry, so the annotation's key
    derivation can't reject it — a stray control byte never fails the run. (A value so
    huge its frame exceeds the protocol's MAX_FRAME_BYTES is the one exception, and the
    same pre-existing cap that bounds every frame — not specific to annotations.)"""
    _pending.append(("log", storable_text(" ".join(str(a) for a in args))))


def phase(title: Any) -> None:
    """Record a phase marker on the run's journal (a durable ``annotation`` event) —
    a lightweight progress label, not a step boundary. Emit-once across replays."""
    _pending.append(("phase", storable_text(str(title))))


def _flush_annotations(branch: _Branch, emit: Any) -> None:
    """Emit the annotations buffered during ``branch``'s just-finished step, keying
    each through the BRANCH's own keyer+path — the identical derivation a capability
    call uses. That makes every annotation's call_key branch-local and replay-stable:
    ``parallel()`` branches never collide, and a replay re-emits the identical key,
    which the journal's ``UNIQUE (run_id, call_key, type)`` folds to one row (the
    ``type`` discriminator keeps an annotation distinct from a same-keyed capability).
    Drained right after the producing step so attribution to the branch is exact.

    Total by construction: ``storable_text`` already made every text jsonb-storable, so
    ``keyer.next`` (which validates against that same domain) cannot raise here — a
    raise would propagate out of the driver's inner ``finally`` and wrongly error the
    run (or mask a clean return)."""
    if not _pending:
        return
    buffered = _pending[:]
    _pending.clear()
    for kind, text in buffered:
        payload = {"kind": kind, "text": text}
        call_key = branch.path + branch.keyer.next(ANNOTATION, payload)
        emit({"type": ANNOTATION, "call_key": call_key, "payload": payload})


# ─── restricted builtins + curated imports (footgun aids, NOT the boundary) ──

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
        "classmethod",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
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
        "object",
        "oct",
        "ord",
        "pow",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
        # constants + exceptions (so authors can raise / except)
        "NotImplemented",
        "Ellipsis",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BaseException",
        "Exception",
        "ImportError",
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

# The curated import surface: modules whose direct API is deterministic, I/O-free
# parse/transform glue. An entry admits itself and its submodules
# (``collections.abc``). First-order claim only — module *attributes* stay
# reachable regardless (``statistics.random`` is one hop away), and the
# fail-closed replay check, not this list, is the determinism enforcement layer.
# Like SAFE_BUILTINS, a footgun aid, NOT the security boundary. ``__future__`` is
# deliberately rejected: under PEP 563 every annotation becomes a string and
# dataclasses' ClassVar/InitVar detection silently misclassifies fields (see
# ``dont_inherit`` in ``_build_coroutine``).
_IMPORTABLE_MODULES = frozenset(
    {
        "base64",
        "collections",
        "dataclasses",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "re",
        "statistics",
        "string",
        "textwrap",
        "urllib.parse",
    }
)


def _safe_import(
    name: str,
    globals: Any = None,
    locals: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    """The ``__import__`` behind a script's ``import`` statements: allowlist the
    dotted name (an entry or a submodule of one), then the real import machinery.
    The error message carries the full allowlist — it IS the author-facing
    documentation. ``from urllib import parse`` is rejected (the name ``__import__``
    receives is ``urllib``, which is not listed); the author retries as
    ``import urllib.parse`` / ``from urllib.parse import …``."""
    allowed = level == 0 and any(name == m or name.startswith(m + ".") for m in _IMPORTABLE_MODULES)
    if not allowed:
        raise ImportError(
            f"cannot import {name!r}: workflow scripts may only import "
            f"{', '.join(sorted(_IMPORTABLE_MODULES))}"
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


# The two dunders an author never calls by name: ``import x`` desugars to
# ``__import__`` (the curated shim) and ``class X:`` to ``__build_class__`` (the
# real one — classes are fully enabled; the allowlist includes ``dataclasses``).
SAFE_BUILTINS["__import__"] = _safe_import
SAFE_BUILTINS["__build_class__"] = builtins.__build_class__


def author_namespace() -> dict[str, Any]:
    """The names injected into a workflow-author script's global namespace.

    This is the single source of truth for the author-facing capability set. The
    model-facing prose statement of it is ``WORKFLOW_SCRIPT_CONTRACT``
    (``aios.models.workflows``); the two are kept structurally in agreement by
    ``tests/unit/test_workflow_script_contract_drift.py``, which introspects this
    dict and asserts every public capability name appears in the contract prose.
    """
    return {
        "__builtins__": SAFE_BUILTINS,
        "gate": gate,
        "agent": agent,
        "tool": tool,
        "budget": budget,
        "parallel": parallel,
        "pipeline": pipeline,
        "log": log,
        "phase": phase,
        # The agent() failure type lives with its capability in the author API
        # (not SAFE_BUILTINS), so `try/except AgentError` resolves it as a global.
        "AgentError": AgentError,
        "AgentNoReturnError": AgentNoReturnError,
    }


def _build_coroutine(source: str, input_value: Any) -> Any:
    # The script runs as a REAL module registered in sys.modules: a class body's
    # first op is ``__module__ = __name__`` (so the name must resolve), and
    # consumers like dataclasses resolve string annotations against
    # ``sys.modules[cls.__module__].__dict__`` — pointing that at the script's own
    # namespace keeps such lookups truthful (an explicit ``"InitVar[int]"``
    # annotation finds the *script's* import, not a bystander module's globals).
    module = types.ModuleType("<workflow>")
    sys.modules[module.__name__] = module
    namespace: dict[str, Any] = module.__dict__
    namespace.update(author_namespace())
    # dont_inherit: without it the script inherits THIS module's ``from __future__
    # import annotations`` (PEP 563), stringifying every annotation — which silently
    # breaks dataclasses' ClassVar/InitVar detection. Compiled clean, annotations
    # are live objects, evaluated eagerly against the script's own namespace.
    code = compile(source, "<workflow>", "exec", dont_inherit=True)
    linecache.cache["<workflow>"] = (len(source), None, source.splitlines(True), "<workflow>")
    exec(code, namespace)
    entry = namespace.get("main")
    if entry is None or not callable(entry):
        raise WorkflowScriptError("workflow script must define `async def main(input)`")
    coro = entry(input_value)
    if not hasattr(coro, "send"):
        raise WorkflowScriptError("workflow script must define `async def main(input)`")
    return coro


def _exc_repr(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _author_traceback(exc: BaseException) -> str:
    """Return an author-facing traceback containing only workflow source frames.

    The subprocess host stack is implementation detail (and may include sandbox
    internals). The script itself is author-visible by definition, so keep its
    filename/line/function/code entries plus the exception type/message footer.
    """
    frames = [
        frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == "<workflow>"
    ]
    if not frames:
        return ""
    lines = ["Traceback (most recent call last):\n"]
    lines.extend(traceback.format_list(frames))
    lines.extend(traceback.format_exception_only(type(exc), exc))
    return "".join(lines)


def _emit_error(emit: Any, repr_: str, tb: str = "", *, kind: str | None = None) -> None:
    """Emit the host's single error terminal (``RAISED``). The sole shape for every
    failure the driver reports — an escaped exception or a driver-detected invariant
    breach — so the parent always parses one frame layout. ``kind`` lets the host
    label a structural failure it detects itself (e.g. the fan-out cap) so the parent
    sees a specific ``error_kind``; an uncaught author exception carries none."""
    emit({"type": RAISED, "repr": repr_, "traceback": tb, "kind": kind})


def _emit_raised(emit: Any, exc: BaseException) -> None:
    _emit_error(emit, _exc_repr(exc), _author_traceback(exc))


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
                try:
                    # A journaled error outcome is delivered as a throw AT the await, so
                    # the author's try/except sees it; everything else is a resume value.
                    if branch.throw is not None:
                        yielded = branch.coro.throw(branch.throw)
                    else:
                        yielded = branch.coro.send(branch.send)
                finally:
                    # Drain log()/phase() buffered during THIS step, keyed by THIS
                    # branch — before the outcome is handled, so annotation frames
                    # precede the step's EMIT/terminal in the stream, and the live
                    # exception (if the send raised) is still bound for the outer
                    # handlers' traceback capture. _flush_annotations is total, so it
                    # can never replace that exception.
                    _flush_annotations(branch, emit)
            except StopIteration as stop:
                progressed = True
                if branch is root:
                    # The return value must live in the same JSON value domain as
                    # capability inputs — validate it at the source (path-precise
                    # author error) rather than letting ``json.dumps`` decide: a
                    # dataclass instance would die as an opaque host crash, and a
                    # NaN would sail through ``allow_nan`` only to detonate at the
                    # parent's jsonb cast. ANY exception out of the walk is
                    # author-caused (RecursionError on a cyclic value, a raising
                    # ``.items()`` on a dict subclass), so all are author errors.
                    try:
                        validate_value(stop.value, path="return")
                    except Exception as exc:
                        _emit_raised(emit, exc)
                        return
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
                "annotations": yielded.annotations,
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
