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
    EMIT,
    INIT,
    RAISED,
    RETURNED,
    SUSPENDED,
    decode_length,
    encode_frame,
)

DEFAULT_CPU_SECONDS = 30
DEFAULT_DEADLINE_SECONDS = 30.0
DEFAULT_ADDRESS_SPACE_BYTES: int | None = None  # no AS cap by default; the deadline backstops

HostOutcomeKind = Literal["suspended", "returned", "raised"]


@dataclass(frozen=True)
class EmittedCapability:
    capability_id: str
    call_key: str  # "sha:<hex>#<ordinal>" — content_hash + ordinal are derivable from this
    spec: Any


@dataclass(frozen=True)
class HostOutcome:
    """The result of driving one wake of the author script in the subprocess."""

    kind: HostOutcomeKind
    emitted: list[EmittedCapability] = field(default_factory=list)
    value: Any = None
    error_repr: str | None = None
    error_kind: str | None = (
        None  # author_exception | script_host_crash | script_host_timeout | ...
    )
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
        )
        for f in frames
        if f.get("type") == EMIT
    ]
    terminal = next(
        (f for f in reversed(frames) if f.get("type") in (SUSPENDED, RETURNED, RAISED)), None
    )
    if terminal is None:
        return HostOutcome(
            kind="raised",
            emitted=emitted,
            error_kind="script_host_crash",
            error_repr=f"script host exited (rc={returncode}) without a terminal frame",
            stderr=stderr,
        )
    kind = terminal["type"]
    if kind == SUSPENDED:
        return HostOutcome(kind="suspended", emitted=emitted, stderr=stderr)
    if kind == RETURNED:
        return HostOutcome(
            kind="returned", emitted=emitted, value=terminal.get("value"), stderr=stderr
        )
    return HostOutcome(
        kind="raised",
        emitted=emitted,
        error_repr=terminal.get("repr"),
        error_kind="author_exception",
        stderr=stderr,
    )


async def run_script_host(
    *,
    source: str,
    input: Any,
    memo: dict[str, Any],
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    address_space_bytes: int | None = DEFAULT_ADDRESS_SPACE_BYTES,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> HostOutcome:
    """Drive one wake of ``source`` in a fresh, credential-free subprocess."""
    init_bytes = encode_frame({"type": INIT, "source": source, "input": input, "memo": memo})
    env = {
        **os.environ,
        "AIOS_WF_RLIMIT_CPU_S": str(cpu_seconds),
        "AIOS_WF_RLIMIT_AS_BYTES": str(address_space_bytes or 0),
    }
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
            proc.communicate(input=init_bytes), timeout=deadline_seconds
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
            error_repr=f"script host exceeded the {deadline_seconds}s wall-clock deadline",
        )
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            proc._transport.close()  # type: ignore[attr-defined]
        raise

    stderr = stderr_bytes.decode("utf-8", "replace")
    return _outcome_from_frames(_parse_frames(stdout_bytes), stderr, proc.returncode)
