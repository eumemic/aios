"""Per-call idempotency token for workflow-run tool calls.

A workflow run executes tools **at-least-once**: a hard worker crash between a
tool's execution and its journaled signal re-drives the call on resume. For a
*non-idempotent* external effect (an HTTP POST that charges, sends, or
publishes) that re-fire is a duplicate side effect with no guard. The fix is a
deterministic per-call token an author threads to the external service as an
idempotency key, so the upstream dedupes the re-fired effect.

**Derivation (CEO-binding).** The token is per-CALL — ``sha256(run_id ‖ call_key)``,
NUL-separated. Per-call (not per-run) is load-bearing for correctness: a per-run
token would make two *distinct* mutating calls in one run share one key, so the
upstream would silently dedup the second. Per-call gets every case right:

- crash re-drive of the SAME call → same ``run_id`` + ``call_key`` → byte-identical
  token (dedupes correctly);
- two distinct calls in one run → distinct ``call_key`` → distinct tokens;
- a new/cloned run → new ``run_id`` → re-fires (a deliberate new effect);
- an edited call → new ``call_key`` → re-fires.

The NUL separator (not ``:``) makes ``(run_id, call_key)`` an unambiguous pair: no
``run_id``/``call_key`` boundary ambiguity can collide two distinct pairs onto one
token, since a ULID ``run_id`` never contains a NUL.

This is the SINGLE source of truth for the token, shared by both deliveries: the
sandbox/``bash`` path exports it as ``$AIOS_IDEMPOTENCY_KEY`` (env preamble,
:mod:`aios.workflows.run_sandbox`) and the worker/``http_request`` path
substitutes it for the author's sentinel header (:mod:`aios.workflows.run_tools`).
"""

from __future__ import annotations

import hashlib

# The author-facing opt-in sentinel for the worker ``http_request`` delivery: a
# script writes ``headers={"Idempotency-Key": AIOS_IDEMPOTENCY_KEY_SENTINEL}`` and
# the worker substitutes the real per-call token. Mirrors the bash path's
# ``$AIOS_IDEMPOTENCY_KEY`` env var — same opt-in ("pass it OR knowingly accept
# at-least-once"), one ergonomic across both deliveries.
AIOS_IDEMPOTENCY_KEY_SENTINEL = "$AIOS_IDEMPOTENCY_KEY"


def idempotency_key(run_id: str, call_key: str) -> str:
    """The per-call idempotency token: ``sha256(run_id ‖ call_key)``, NUL-separated.

    Opaque and fixed-width — the structural punctuation of a raw ``call_key`` never
    leaks to the external service. Stable across crash re-drives of the same call,
    distinct across distinct calls. See the module docstring for the full contract.
    """
    return hashlib.sha256(f"{run_id}\0{call_key}".encode()).hexdigest()
