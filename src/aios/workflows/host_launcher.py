"""Parent side of the out-of-process workflow script host.

Spawns ``python -m aios.workflows.wf_script_host`` as a fresh interpreter
(``spawn`` semantics — the worker's loaded ``aios.harness.runtime`` with the
master ``CryptoBox`` + pool is never inherited), hands it the pinned source +
input + memo over a pipe, and reads back the frontier + terminal frames.

Containment is layered: the child self-imposes ``RLIMIT_CPU``/``RLIMIT_AS`` from
env (best-effort, Unix), and the parent enforces a **wall-clock SIGKILL
deadline** that is platform-independent and fires even on a tight CPU loop —
because the parent only ever ``await``s a pipe read, never synchronous author
code. A child that crashes / OOMs / is killed surfaces as a ``raised`` outcome;
it can never wedge the worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

from aios.workflows._protocol import (
    ANNOTATION,
    EMIT,
    INIT,
    RAISED,
    RETURNED,
    SUSPENDED,
    decode_length,
    encode_frame,
)

DEFAULT_DEADLINE_SECONDS = 30.0
# Per-wake budgets SCALE with the INIT frame (#780): a wake that replays a bigger
# memo is entitled to proportionally more wall/CPU time, so a run can never time
# out merely because it has accumulated results. 30s/MiB is ~3 orders of magnitude
# above measured parse+replay cost — the headroom is for the author's own glue
# (sorting/joining N results), which is what actually grows with the memo. The cap
# bounds how long one wake can hold a worker (a run cancel is only consumed at the
# next step, and a deploy drains in-flight steps): a capped budget is still
# ~15,000x the measured replay cost of a frame-cap-sized INIT.
DEADLINE_SECONDS_PER_INIT_MIB = 30.0
MAX_SCALED_SECONDS = 600.0


def _scaled_seconds(base: float, init_len: int) -> float:
    """The effective per-wake budget: ``base`` plus the INIT-size allowance, capped."""
    return min(
        base + (init_len / (1024 * 1024)) * DEADLINE_SECONDS_PER_INIT_MIB, MAX_SCALED_SECONDS
    )


# 4 GiB virtual-address ceiling: bounds a runaway allocation (e.g. ``[0]*10**10``
# ≈ 80 GB) that the wall-clock deadline does NOT bound — the deadline caps
# duration, not peak memory — while leaving ample headroom for a coordination
# script. A ``setrlimit`` failure (platform-dependent) degrades to the deadline;
# pass ``None`` to opt out.
DEFAULT_ADDRESS_SPACE_BYTES: int | None = 4 * 1024**3

HostOutcomeKind = Literal["suspended", "returned", "raised"]
HostErrorKind = Literal[
    "author_exception",
    "too_wide_fanout",  # a single parallel()/pipeline() exceeded MAX_PARALLEL_FANOUT
    "script_host_crash",
    "script_host_timeout",
    "script_host_spawn_failed",
]

# SECURITY (load-bearing): the child IS the isolation boundary — author code is
# assumed able to escape the restricted builtins and read the child's entire
# environment (e.g. ``gate.__globals__['os'].environ``). So the child must inherit
# NO secret-bearing variable: we pass a deny-by-default allowlist of launch /
# locale essentials only — never ``AIOS_VAULT_KEY`` / ``AIOS_DB_URL`` / any
# ``*_API_KEY`` / token. This is what makes "credential-free subprocess" (§3.4)
# actually true; without it a malicious or buggy script reaches every tenant's
# secrets via the inherited env. Extend this set, never widen to ``os.environ``.
_CHILD_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "__PYVENV_LAUNCHER__",  # macOS venv re-exec
        "SYSTEMROOT",  # Windows interpreter launch
        "PATHEXT",
    }
)


@dataclass(frozen=True)
class EmittedCapability:
    capability_id: str
    # "sha:<hex>#<ordinal>", optionally prefixed by a parallel branch path
    # ("0.0/sha:<hex>#<ordinal>"). Treated as an opaque, deterministic key.
    call_key: str
    spec: Any
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmittedAnnotation:
    """One ``log()``/``phase()`` progress line the child emitted this wake. ``call_key``
    is the branch-local annotation key; ``payload`` is ``{"kind": "log"|"phase", "text"}``.
    The step journals these as ``annotation`` events; the memo UNIQUE makes a replay's
    re-emission a no-op (emit-once)."""

    call_key: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class HostOutcome:
    """The result of driving one wake of the author script in the subprocess."""

    kind: HostOutcomeKind
    emitted: list[EmittedCapability] = field(default_factory=list)
    annotations: list[EmittedAnnotation] = field(default_factory=list)
    value: Any = None
    error_repr: str | None = None
    error_traceback: str | None = None
    error_kind: HostErrorKind | None = None
    # The child's real stderr — crash diagnostics only (a host-crash traceback, an
    # rlimit message). log()/phase() no longer route through here; they are journaled
    # annotation frames on stdout. The step deliberately does NOT consume this field:
    # capturing-at-the-boundary-then-dropping is the "explicitly dropped" disposition the
    # design calls for — it is never plumbed into the run journal (author-visible output
    # is the annotations now), and is here for a host-crash post-mortem path to read.
    stderr: str = ""


def _parse_frames(buf: bytes) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    i = 0
    while i + 4 <= len(buf):
        n = decode_length(buf[i : i + 4])
        if i + 4 + n > len(buf):
            break  # truncated tail (crash mid-write) — ignore it
        frames.append(json.loads(buf[i + 4 : i + 4 + n]))
        i += 4 + n
    return frames


def _outcome_from_frames(
    frames: list[dict[str, Any]], stderr: str, returncode: int | None
) -> HostOutcome:
    emitted = [
        EmittedCapability(
            capability_id=f["capability_id"],
            call_key=f["call_key"],
            spec=f.get("spec"),
            annotations=f.get("annotations") or {},
        )
        for f in frames
        if f.get("type") == EMIT
    ]
    # Progress annotations (log()/phase()), in stream/execution order. Carried on every
    # outcome — including a no-terminal crash below — so a script that logged before
    # dying is still journaled. (A timeout/spawn failure returns earlier with none: its
    # frames are discarded, which is exactly the zero-growth bound for a runaway loop.)
    annotations = [
        EmittedAnnotation(call_key=f["call_key"], payload=f["payload"])
        for f in frames
        if f.get("type") == ANNOTATION
    ]
    terminal = next(
        (f for f in reversed(frames) if f.get("type") in (SUSPENDED, RETURNED, RAISED)), None
    )
    if terminal is None:
        return HostOutcome(
            kind="raised",
            emitted=emitted,
            annotations=annotations,
            error_kind="script_host_crash",
            error_repr=f"script host exited (rc={returncode}) without a terminal frame",
            stderr=stderr,
        )
    kind = terminal["type"]
    if kind == SUSPENDED:
        return HostOutcome(
            kind="suspended", emitted=emitted, annotations=annotations, stderr=stderr
        )
    if kind == RETURNED:
        return HostOutcome(
            kind="returned",
            emitted=emitted,
            annotations=annotations,
            value=terminal.get("value"),
            stderr=stderr,
        )
    return HostOutcome(
        kind="raised",
        emitted=emitted,
        annotations=annotations,
        error_repr=terminal.get("repr"),
        error_traceback=terminal.get("traceback"),
        # The host stamps a specific kind on structural failures it detects itself
        # (e.g. the fan-out cap); an uncaught author exception carries none → generic.
        error_kind=terminal.get("kind") or "author_exception",
        stderr=stderr,
    )


async def run_script_host(
    *,
    source: str,
    input: Any,
    memo: dict[str, Any],
    address_space_bytes: int | None = DEFAULT_ADDRESS_SPACE_BYTES,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> HostOutcome:
    """Drive one wake of ``source`` in a fresh, credential-free subprocess.

    ``deadline_seconds`` is the BASE budget; the effective budget scales with the
    INIT frame size (see :func:`_scaled_seconds`). The CPU rlimit is DERIVED —
    one second above the wall deadline — so a single-threaded CPU-bound child is
    killed by the parent's deadline (a clean ``script_host_timeout``), never by
    the rlimit SIGKILL (an opaque host crash). RLIMIT_CPU sums across threads, so
    a child that escapes the restricted builtins into GIL-releasing threads can
    still hit the rlimit first — containment holds either way (both outcomes are
    terminal raised); only the error kind differs.
    """
    init_bytes = encode_frame({"type": INIT, "source": source, "input": input, "memo": memo})
    deadline = _scaled_seconds(deadline_seconds, len(init_bytes))
    cpu = int(deadline) + 1
    # Deny-by-default env: only non-secret launch/locale essentials cross the
    # spawn (see ``_CHILD_ENV_ALLOWLIST``), plus the two rlimit knobs the child
    # self-applies. The child must stay credential-free.
    env = {k: os.environ[k] for k in _CHILD_ENV_ALLOWLIST if k in os.environ}
    # DETERMINISM (load-bearing): pin the hash seed so str-hash-dependent orderings
    # are identical across wakes. Every wake is a fresh interpreter; under an
    # inherited/random seed, ``list({"a", "b"})`` reorders per wake, and if such an
    # ordering feeds a capability spec the call_key desyncs and replay-with-memo
    # breaks. (``canonical_json`` rejects sets directly, but a list *built from* a
    # set is indistinguishable from any other list.)
    env["PYTHONHASHSEED"] = "0"
    env["AIOS_WF_RLIMIT_CPU_S"] = str(cpu)
    env["AIOS_WF_RLIMIT_AS_BYTES"] = str(address_space_bytes or 0)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "aios.workflows.wf_script_host",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except OSError as exc:
        return HostOutcome(
            kind="raised", error_kind="script_host_spawn_failed", error_repr=str(exc)
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=init_bytes), timeout=deadline
        )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        # Free the parent-side pipe FDs now; awaiting communicate()/wait() again
        # can re-hang on a D-state child (cf. sandbox/_subprocess.py).
        with contextlib.suppress(Exception):
            proc._transport.close()  # type: ignore[attr-defined]
        return HostOutcome(
            kind="raised",
            error_kind="script_host_timeout",
            # The EFFECTIVE (scaled) deadline — this string becomes the run's
            # terminal output, so it must state the number actually enforced.
            error_repr=f"script host exceeded the {deadline:.1f}s wall-clock deadline",
        )
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            proc._transport.close()  # type: ignore[attr-defined]
        raise

    stderr = stderr_bytes.decode("utf-8", "replace")
    # A corrupt / oversized stream (``decode_length`` ValueError, ``json.loads``
    # JSONDecodeError — the latter is a ValueError subclass) must error the run as
    # a host crash, never escape the step uncaught and wedge a non-terminal run
    # into a sweep crashloop.
    try:
        frames = _parse_frames(stdout_bytes)
    except ValueError as exc:
        return HostOutcome(
            kind="raised",
            error_kind="script_host_crash",
            error_repr=f"unparseable script host output: {exc}",
            stderr=stderr,
        )
    return _outcome_from_frames(frames, stderr, proc.returncode)
