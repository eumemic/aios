"""Deterministic child-session id for a workflow ``agent()`` spawn.

The child id is a pure function of ``(run_id, call_key)`` so a replayed spawn
reproduces the *same* id — the basis of idempotent ``create_child_session`` +
harvest-on-reattach (a crash between the row insert and ``call_started`` is
recovered by recomputing the same id, finding the existing row, and harvesting
instead of double-spawning).

The seed folds the **full** ``call_key`` string (``"sha:<hex>#<ordinal>"``), NOT
the content hash — so N identical siblings (``parallel([agent("ping")]*3)`` →
keys ``…#0/#1/#2``) get THREE distinct child ids. Folding the content hash alone
would collapse them to one child (the rev-1 regression).

Parent-side only: the step computes this at spawn and journals it into
``call_started``; harvest reads it back from the journal and never recomputes.
Kept out of :mod:`aios.workflows.determinism` so that module stays stdlib-only
for the credential-free host.
"""

from __future__ import annotations

import hashlib

from aios.ids import SESSION, make_id


def child_session_id(run_id: str, call_key: str) -> str:
    """``sess_<ulid>`` deterministically derived from ``(run_id, call_key)``."""
    seed = hashlib.sha256(f"{run_id}:{call_key}".encode()).digest()
    return make_id(SESSION, body=seed[:16])
