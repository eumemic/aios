"""Length-prefixed JSON frame protocol between the parent step and the
out-of-process script host.

Wire format: a 4-byte big-endian length prefix followed by that many bytes of
UTF-8 JSON. Stdlib-only — imported by the credential-free child, so it must never
pull in anything from ``aios.harness``/``aios.db``/``aios.crypto``.

Message flow (the child emits zero or more ANNOTATION + frontier-capability frames,
then a terminal; a single-coroutine run emits at most one EMIT, a
``parallel``/``pipeline`` fan-out emits one EMIT per open branch capability before
the terminal; ANNOTATION frames — ``log()``/``phase()`` progress — interleave freely,
fire-and-forget, in execution order):

    parent → child:  INIT {source, input, memo}
    child  → parent: ANNOTATION {call_key, payload} *  (zero or more, interleaved)
                     EMIT {capability_id, call_key, spec} *  (zero or more)
                     then exactly one of SUSPENDED | RETURNED {value} | RAISED {repr, traceback}
"""

from __future__ import annotations

import json
import struct
from typing import IO, Any

_LEN = struct.Struct(">I")
MAX_FRAME_BYTES = 64 * 1024 * 1024  # guard against a corrupt/oversized length prefix

# message "type" tags
INIT = "init"
EMIT = "emit"
ANNOTATION = "annotation"
SUSPENDED = "suspended"
RETURNED = "returned"
RAISED = "raised"


def encode_frame(obj: dict[str, Any]) -> bytes:
    body = json.dumps(obj).encode("utf-8")
    return _LEN.pack(len(body)) + body


def decode_length(prefix: bytes) -> int:
    n = int(_LEN.unpack(prefix)[0])
    if n > MAX_FRAME_BYTES:
        raise ValueError(f"workflow host frame too large: {n} bytes")
    return n


def write_frame_sync(stream: IO[bytes], obj: dict[str, Any]) -> None:
    """Blocking write of one frame (the child's stdout side)."""
    stream.write(encode_frame(obj))
    stream.flush()


def read_frame_sync(stream: IO[bytes]) -> dict[str, Any] | None:
    """Blocking read of one frame, or ``None`` at clean EOF (the child's stdin)."""
    prefix = stream.read(4)
    if not prefix:
        return None
    if len(prefix) < 4:
        raise EOFError("truncated workflow host frame prefix")
    n = decode_length(prefix)
    body = stream.read(n)
    if len(body) < n:
        raise EOFError("truncated workflow host frame body")
    result: dict[str, Any] = json.loads(body)
    return result
