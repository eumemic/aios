"""Shared maker-marker comment-idempotency helper (aios#1292) — the CLASS fix.

Comment-POST is **not idempotent**: GitHub does not honor an ``Idempotency-Key`` on
issue-comment create, so two POSTs of the "same" comment produce two comments. Two
distinct hazards drive that duplication, and BOTH the triage pipeline and the dev
pipeline POST chairman-facing markered comments, so this is a constellation-wide
recurrence-class (eumemic-company#24, fractal-retro recurrence discipline). The class is
closed HERE, in one shared helper that both pipelines inject into their workflow source,
rather than patched per-call-site in each pipeline (which would let the fix drift).

The two hazards:

1. **At-least-once replay.** The aios crash contract (``run_tools.py``) is at-least-once:
   a worker can run the comment POST and then crash *before journaling the result*, so
   replay re-drives the POST → a duplicate comment. The guard: before POSTing, scan the
   already-fetched comment thread for the comment's stable ``## <marker>`` heading and
   SKIP if present. This is the maker-marker pattern — a posted comment is its own
   "already done" marker, observed on the next read.

2. **Error-path ordering (triage's needs-decision specifically).** When a comment is
   paired with an additive idempotent label, the LABEL must be applied first and the
   comment posted only after the label returns 2xx — so the label guards the comment (a
   labeled issue drops out of the untriaged working set, so a re-run never re-classifies
   and re-comments). That ordering is the *caller's* concern; this helper only owns the
   maker-marker skip, which closes the replay hazard for EVERY markered comment.

WHY A SOURCE STRING, NOT AN IMPORTED FUNCTION: each workflow is rendered to a single
self-contained *script source* that the script-host ``exec``s in a curated namespace
(stdlib ``re``/``json`` only — no ``aios`` import at runtime). So a genuinely shared
helper has to be shared at *authoring* time: this module exports the helper SOURCE TEXT,
authored once here, and both ``build_*_script`` functions splice it into their ``_BODY``.
The text references ``gh`` and ``_ipath`` — names both pipeline bodies already define with
identical semantics — so it composes without per-pipeline edits.

DETERMINISM: the helper does only pure list/dict/string work plus the same ``gh`` /
``_ipath`` calls the surrounding body already makes; it imports nothing and reads no
clock. A skip returns a synthetic ``{"status": 200, "skipped": True}`` value (never a
real tool() await), so the skip path emits no capability and replay stays stable.
"""

from __future__ import annotations

# The shared helper source, authored once. Spliced verbatim into each pipeline's _BODY.
# References ``gh`` and ``_ipath`` from the surrounding body (both pipelines define them
# with identical signatures: ``await gh(method, path, body=None)`` and
# ``_ipath(repo, suffix)``).
COMMENT_IDEMPOTENCY_HELPERS = r'''
# ─── shared comment-idempotency helper (aios#1292 — close the comment-POST class) ──
# Comment-POST is NOT idempotent (GitHub ignores Idempotency-Key on comment create) and
# the aios crash contract is at-least-once, so a replay can re-drive a comment POST and
# duplicate it. The maker-marker guard: a posted comment carries a stable ``## <marker>``
# heading; before POSTing, scan the already-fetched thread for that heading and skip if
# present. The posted comment is its own "already done" marker on the next read.

def _comment_thread_has_marker(comments, marker):
    """True if any already-fetched comment's body contains ``marker``. ``comments`` is the
    array from ``GET /issues/{n}/comments`` (or None/[] when unread → no marker → POST
    proceeds, the safe default that preserves the first-post). Match is a substring test on
    the stable heading, mirroring the maker-marker pattern."""
    needle = (marker or "").strip()
    if not needle or not isinstance(comments, list):
        return False
    for c in comments:
        if isinstance(c, dict):
            text = c.get("body")
            if isinstance(text, str) and needle in text:
                return True
    return False


async def post_comment_once(repo, number, marker, body, existing_comments):
    """POST one issue comment UNLESS the already-fetched thread already carries ``marker``
    (the maker-marker dedup). Returns the gh() response, or a synthetic skipped sentinel
    (``{"status": 200, "skipped": True}``) when the marker is already present — so the
    caller can treat a skip as a benign success and never emits a duplicate-POST capability
    on replay. ``body`` MUST contain ``marker`` (the heading IS the marker) so the posted
    comment is self-identifying on the next read."""
    if _comment_thread_has_marker(existing_comments, marker):
        log("comment skip (marker already present): #%s %r" % (number, marker))
        return {"status": 200, "skipped": True}
    return await gh("POST", _ipath(repo, "/issues/%d/comments" % number), {"body": body})


def _comment_posted_ok(resp):
    """True if a post_comment_once() result is a success — a real 2xx POST OR a skip."""
    if not isinstance(resp, dict):
        return False
    if resp.get("skipped"):
        return True
    st = resp.get("status")
    return isinstance(st, int) and 200 <= st <= 299
'''
