"""Shared GitHub-body parsing helper (aios#1294, aios#1323) — the CLASS fix for
silent truncate-then-degrade-to-empty AND silent list under-count.

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

aios#1323 — NO-SILENT-DEGRADE LIST-READ INVARIANT (generalises #1294 from response
BODIES to list PAGINATION): a LIST read is only honest if it can prove it saw the
WHOLE list. The shipped ``gh_paginated`` followed ``Link: rel="next"`` but, when it
hit its ``max_pages`` ceiling with a ``rel="next"`` still dangling, RETURNED THE
PARTIAL LIST AS IF COMPLETE — the same look-green-while-doing-less shape #1294
polices, one layer up. A reconciler that reads a paginated check-runs list, gets
page 1, and concludes "all green" is silently under-counting the substrate it
judges. The fix here:

  * ``gh_paginated`` now RAISES ``GitHubListIncomplete`` if it exhausts ``max_pages``
    while GitHub still advertises an unfetched ``rel="next"`` page — it NEVER returns
    a partial set silently treated as complete. (A truncated/unparseable 2xx page
    still raises ``GitHubBodyError`` as before.)
  * ``graphql_list_complete`` asserts ``len(nodes) == totalCount`` for a GraphQL
    connection and raises ``GitHubListIncomplete`` on a mismatch — the GraphQL twin
    of the REST ``rel="next"`` walk.

``GitHubListIncomplete`` is the first-class ``cannot-determine`` verdict: a LIST
read that cannot prove completeness is fatal-loud, NEVER coerced to an empty / ok /
green partial result.

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


class GitHubListIncomplete(RuntimeError):
    """A LIST read that could NOT prove it fetched the whole list — REST
    pagination that still advertised an unfetched ``Link: rel="next"`` page when
    the page ceiling was hit, or a GraphQL connection where ``len(nodes)`` did not
    equal ``totalCount``. This is the first-class ``cannot-determine`` verdict of
    aios#1323: a list read that cannot prove completeness is raised LOUD, NEVER
    returned as a partial set silently treated as complete (the dominant
    look-green-while-doing-less shape — a paginated check-runs read that sees page
    1 and concludes "all green")."""


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
    stays stable. Requires the route to allow a query string (``allow_query``).

    aios#1323 — the no-silent-degrade list-read invariant: if the page ceiling
    (``max_pages``) is reached while GitHub STILL advertises an unfetched
    ``rel="next"`` page, we RAISE ``GitHubListIncomplete`` rather than return the
    partial list as if it were the whole list. Silently truncating a list read at
    the ceiling is exactly the look-green-while-doing-less shape (a check-runs read
    that returns page 1 and is treated as "all green"); the cap exists to bound work,
    not to license a silent under-count. Raise the ceiling for genuinely huge lists.
    """
    items = []
    page = 1
    for _ in range(max_pages):
        resp = await gh("GET", _with_query(path, per_page=per_page, page=page))
        if _status(resp) != 200:
            return items  # non-2xx (auth/404) is the caller's branch
        chunk = _json_body(resp)  # raises loud on truncated / unparseable 2xx
        if not isinstance(chunk, list):
            return items  # genuine shape change — stop, return what we have
        items.extend(chunk)
        nxt = _link_next_page(_headers(resp).get("link"))
        if nxt is None:
            return items  # last page reached cleanly — the list is COMPLETE
        page = nxt
    # Loop fell through the ceiling with a rel="next" still dangling: we CANNOT
    # prove we saw the whole list, so this read is cannot-determine, never a
    # silently-truncated "complete" list (aios#1323).
    raise GitHubListIncomplete(
        "GitHub list read for %r exhausted the %d-page ceiling while a "
        "rel=\"next\" page was still unfetched — refusing to treat a partial "
        "(%d-item) read as the complete list. Raise max_pages or paginate more "
        "finely; never silently under-count a list read." % (path, max_pages, len(items))
    )


def graphql_list_complete(connection, *, where=""):
    """Assert a GraphQL connection was fetched in FULL and return its ``nodes``.

    ``connection`` is a parsed GraphQL connection object — a dict carrying both a
    ``totalCount`` (the server's authoritative count) and a ``nodes`` list (what we
    actually fetched). The GraphQL twin of the REST ``rel="next"`` walk: a single
    GraphQL page returns at most ``first:N`` nodes, so a connection with more than
    ``N`` items comes back with ``len(nodes) < totalCount`` and a ``pageInfo`` cursor
    we did not follow. Returning those ``nodes`` as if complete is the same silent
    under-count #1294 polices for bodies.

    Returns ``nodes`` iff ``len(nodes) == totalCount``. On ANY mismatch — fewer
    nodes than the count (unfetched pages) OR a malformed connection missing either
    field — raise ``GitHubListIncomplete`` (cannot-determine), NEVER a partial list.
    ``where`` is an optional label for the error message (e.g. ``"check runs for
    <sha>"``). Pure: deterministic over its input, no I/O."""
    label = (" for %s" % where) if where else ""
    if not isinstance(connection, dict):
        raise GitHubListIncomplete(
            "GraphQL connection%s is not an object (%r) — cannot prove the list is "
            "complete; refusing to degrade to empty (aios#1323)." % (label, type(connection).__name__)
        )
    total = connection.get("totalCount")
    nodes = connection.get("nodes")
    if not isinstance(total, int) or not isinstance(nodes, list):
        raise GitHubListIncomplete(
            "GraphQL connection%s is missing totalCount/nodes (totalCount=%r, "
            "nodes=%r) — cannot prove the list is complete; refusing to degrade to "
            "empty (aios#1323)." % (label, total, type(nodes).__name__)
        )
    if len(nodes) != total:
        raise GitHubListIncomplete(
            "GraphQL list%s under-counted: fetched %d node(s) but totalCount is %d "
            "— there are unfetched pages, so this read is cannot-determine, NOT a "
            "complete list. Paginate the connection (follow pageInfo.endCursor) "
            "before treating it as whole (aios#1323)." % (label, len(nodes), total)
        )
    return nodes
'''
