"""Deterministic sub-run id for a workflow ``invoke_workflow()`` spawn (#1129).

The dual of :mod:`aios.workflows.child_id` (which derives a child *session* id for
``agent()``): a sub-run id is a pure function of ``(run_id, call_key)`` so a
replayed spawn reproduces the *same* id — the basis of idempotent ``create_run``
re-attach + harvest-on-reattach (a crash between the run insert and the
``call_started`` journal write is recovered by recomputing the same id, finding
the existing row, and harvesting instead of double-spawning).

The seed folds the **full** ``call_key`` string, NOT the content hash — so N
identical sibling ``invoke_workflow`` calls (``parallel([invoke_workflow(...)]*3)``
→ keys ``…#0/#1/#2``) get THREE distinct sub-run ids. Folding the content hash
alone would collapse them to one sub-run (the rev-1 regression documented in
``child_id.py``). A bare call_key is ``"sha:<hex>#<ordinal>"``; a key reached
inside a ``parallel`` branch additionally carries that branch's deterministic
path prefix — either way the **whole** string is folded opaquely, so distinct
branches/ordinals → distinct ids.

Parent-side only: the step computes this at spawn and journals it into
``call_started``; harvest reads it back from the journal and never recomputes.
"""

from __future__ import annotations

import hashlib

from aios.ids import WORKFLOW_RUN, make_id


def child_run_id(run_id: str, call_key: str) -> str:
    """``wfr_<ulid>`` deterministically derived from ``(run_id, call_key)``."""
    seed = hashlib.sha256(f"run:{run_id}:{call_key}".encode()).digest()
    return make_id(WORKFLOW_RUN, body=seed[:16])
