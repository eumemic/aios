"""Shared GitHub-body parsing helper (aios#1294) — the CLASS fix for silent
truncate-then-degrade-to-empty.

``http_request`` caps a response body at a configurable char limit and now sets
``"truncated": True`` whenever it cuts one (the tool no longer truncates
silently). But the *caller* contract matters just as much: BOTH the triage
pipeline and the dev pipeline shared an identical ``_json_body`` /
``gh_paginated`` pair that, on a truncated body or a JSON-parse failure of a 2xx
response, returned ``None`` / broke the page loop and degraded to ``[]`` — a
**silent ``scanned:0`` no-op** that blocked the triage loop on aios-sized repos
(a single ``per_page=100`` issue page is ~591 KB, far over the old 100 KB cap).

This module closes that recurrence-class in ONE place: a loud-failing
``_json_body`` and ``gh_paginated`` authored once here and spliced into each
pipeline's workflow ``_BODY`` (the same source-string sharing pattern as
``comment_idempotency.COMMENT_IDEMPOTENCY_HELPERS``). The rule, applied
identically by both pipelines:

  * A ``truncated`` 2xx body → RAISE (the body is structurally incomplete; a
    JSON parse would either crash mid-stream or — worse — succeed on a coincidentally
    balanced prefix and silently drop the tail).
  * A 2xx body that fails to JSON-parse → RAISE (a 200 with unparseable JSON is a
    real defect, not an empty result; never swallow it into ``None``/``[]``).
  * A non-2xx response is still a VALUE the caller branches on (auth/404/422 are
    the caller's concern, mirroring the existing ``gh`` contract) — only a 2xx
    body that we cannot faithfully parse is fatal.

WHY A SOURCE STRING, NOT AN IMPORTED FUNCTION: each workflow is rendered to a
single self-contained *script source* the script-host ``exec``s in a curated
namespace (stdlib ``re``/``json`` only — no ``aios`` import at runtime). A
genuinely shared helper must therefore be shared at *authoring* time: this module
exports the helper SOURCE TEXT, authored once, and both ``build_*_script``
functions splice it into their ``_BODY``. The text references ``gh``, ``_status``,
``_headers``, ``_link_next_page``, ``_with_query`` and ``log`` — names both
pipeline bodies already define with identical semantics — so it composes without
per-pipeline edits.

DETERMINISM: the helper does only pure string/list/dict work plus the same ``gh``
calls the surrounding body already makes; it imports nothing and reads no clock.
The raise is a deterministic function of the (replayed) tool result, so replay
stays stable — a run that failed loud once fails loud identically on replay.
"""

from __future__ import annotations

# The shared helper source, authored once. Spliced verbatim into each pipeline's
# _BODY. References ``gh`` / ``_status`` / ``_headers`` / ``_link_next_page`` /
# ``_with_query`` / ``log`` from the surrounding body (both pipelines define them
# with identical semantics).
GH_BODY_HELPERS = r'''
# ─── shared GitHub-body parsing (aios#1294 — fail loud, never degrade-to-empty) ──
# http_request caps a response body and flags ``truncated: True`` when it cuts one.
# A truncated 2xx body is structurally incomplete and a 2xx body that won't
# JSON-parse is a real defect — NEITHER may degrade silently to None/[] (that was
# the silent ``scanned:0`` no-op that blocked the triage loop). Both raise here so
# the failure is loud; a non-2xx response stays a value the caller branches on.

class GitHubBodyError(RuntimeError):
    """A 2xx GitHub response whose body could not be faithfully parsed —
    truncated by the response cap, or 2xx-but-unparseable JSON. Raised instead of
    returning None/[] so a structurally-incomplete read never degrades to a silent
    empty result (aios#1294)."""


def _resp_truncated(resp):
    """True when http_request signalled it cut this body (``truncated: True``).
    The flag is present ONLY when the body was truncated, so its mere presence is
    the signal."""
    return isinstance(resp, dict) and bool(resp.get("truncated"))


def _json_body(resp):
    """Parse a GitHub response body to JSON, failing LOUD on a faithless read.

    - Not a dict / no body string → None (an empty or error-envelope response the
      caller already branches on; a non-2xx is the caller's concern).
    - A ``truncated`` body → raise GitHubBodyError (the body lost its tail; a parse
      would crash mid-stream or silently succeed on a balanced prefix).
    - A 2xx body that fails to JSON-parse → raise GitHubBodyError (a 200 with
      unparseable JSON is a defect, never an empty result).
    - A non-2xx body that fails to parse → None (e.g. a 404/422 with a non-JSON or
      partial body is the caller's branch, not a fatal read).
    """
    if not isinstance(resp, dict):
        return None
    if _resp_truncated(resp):
        raise GitHubBodyError(
            "GitHub response body was TRUNCATED by the response cap (status %s) — "
            "refusing to parse a structurally-incomplete body. Raise "
            "AIOS_HTTP_RESPONSE_MAX_CHARS or page the endpoint more finely; never "
            "degrade a truncated read to empty." % (_status(resp),)
        )
    raw = resp.get("body")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        st = _status(resp)
        if isinstance(st, int) and 200 <= st <= 299:
            raise GitHubBodyError(
                "GitHub returned a 2xx (%s) whose body failed to JSON-parse: %s — "
                "a 2xx with unparseable JSON is a defect, not an empty result; "
                "refusing to degrade to None." % (st, exc)
            ) from exc
        return None


async def gh_paginated(path, per_page=100, max_pages=50):
    """GET every page of a GitHub list endpoint, following ``Link: rel="next"``.

    Concatenates the per-page JSON arrays. A non-2xx page stops pagination and
    returns what was gathered so far (auth/404 is the caller's branch). But a 2xx
    page that is TRUNCATED or fails to JSON-parse RAISES (via ``_json_body``) — it
    NEVER degrades to a partial/empty list, which is the aios#1294 silent
    ``scanned:0`` no-op. A 2xx page that parses to a non-list is a genuine shape
    change and stops pagination (returns what was gathered). The page number is
    rebuilt onto OUR path, so no host-bearing next URL crosses the route gate; each
    request is a distinct ``tool()`` await (per-content-hash call_key), so replay
    stays stable. Requires the route to allow a query string (``allow_query``)."""
    items = []
    page = 1
    for _ in range(max_pages):
        resp = await gh("GET", _with_query(path, per_page=per_page, page=page))
        if _status(resp) != 200:
            break
        chunk = _json_body(resp)  # raises loud on truncated / unparseable 2xx
        if not isinstance(chunk, list):
            break
        items.extend(chunk)
        nxt = _link_next_page(_headers(resp).get("link"))
        if nxt is None:
            break
        page = nxt
    return items
'''
