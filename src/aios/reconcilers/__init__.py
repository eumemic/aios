"""Reconcilers: loops that re-check an *asserted* state against the world.

A reconciler exists because an assertion nobody re-checks is not state — it is a
sticker. Each module here reads a declared/asserted projection (a GitHub label, a
committed manifest) and the authoritative substrate it claims to describe, joins
them, and REPORTS the disagreement.

``work_state`` is the pure join/classify core for work-state reconciliation
(labels vs. aios runs); ``work_state_cli`` is its I/O shell.
"""
