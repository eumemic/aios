"""Unit tests for the shared comment-idempotency helper (aios#1292).

The helper closes the comment-POST non-idempotency CLASS shared by the triage and dev
pipelines: GitHub ignores ``Idempotency-Key`` on comment create, and the aios crash
contract is at-least-once, so a replay (or a label-fail re-classify loop) can re-drive a
comment POST and duplicate it. The maker-marker guard scans the already-fetched thread for
the comment's stable ``## <marker>`` heading and skips the POST if present.

The helper source is authored once in ``comment_idempotency.COMMENT_IDEMPOTENCY_HELPERS``
and spliced into BOTH pipeline scripts. We exec it in a namespace with a recording ``gh``
stub (the helper references ``gh`` / ``_ipath`` from the surrounding body) and exercise the
pure logic directly — no LLM, no real tool, no time.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aios.workflows.comment_idempotency import COMMENT_IDEMPOTENCY_HELPERS


def _namespace() -> dict[str, Any]:
    """Exec the helper source with the two names it references from a pipeline body:
    a recording async ``gh`` and ``_ipath`` / ``log``."""
    posts: list[dict[str, Any]] = []

    async def gh(method: str, path: str, body: Any = None) -> dict[str, Any]:
        posts.append({"method": method, "path": path, "body": body})
        return {"status": 201, "body": "{}"}

    def _ipath(repo: str, suffix: str) -> str:
        return f"/repos/{repo}{suffix}"

    ns: dict[str, Any] = {"gh": gh, "_ipath": _ipath, "log": lambda *a, **k: None, "posts": posts}
    exec(compile(COMMENT_IDEMPOTENCY_HELPERS, "comment_idempotency", "exec"), ns)
    return ns


MARKER = "## Triage: needs a decision"
BODY = MARKER + "\n\nWhy: needs capital sign-off"


# ─── _comment_thread_has_marker ──────────────────────────────────────────────


def test_marker_detected_in_thread() -> None:
    ns = _namespace()
    has = ns["_comment_thread_has_marker"]
    comments = [{"body": "unrelated chatter"}, {"body": BODY}]
    assert has(comments, MARKER) is True


def test_marker_absent_from_thread() -> None:
    ns = _namespace()
    has = ns["_comment_thread_has_marker"]
    comments = [{"body": "unrelated"}, {"body": "more chatter"}]
    assert has(comments, MARKER) is False


def test_unread_thread_reports_no_marker() -> None:
    # None / [] / non-list → no marker → the first POST proceeds (safe default).
    ns = _namespace()
    has = ns["_comment_thread_has_marker"]
    assert has(None, MARKER) is False
    assert has([], MARKER) is False
    assert has("boom", MARKER) is False


def test_non_dict_and_empty_comments_are_ignored() -> None:
    ns = _namespace()
    has = ns["_comment_thread_has_marker"]
    assert has(["a string", {"body": None}, {"nope": 1}], MARKER) is False


# ─── post_comment_once: the maker-marker dedup ───────────────────────────────


def test_first_post_goes_through() -> None:
    ns = _namespace()
    post = ns["post_comment_once"]
    resp = asyncio.run(post("o/r", 7, MARKER, BODY, existing_comments=[]))
    assert resp["status"] == 201
    assert not resp.get("skipped")
    assert ns["posts"] == [
        {"method": "POST", "path": "/repos/o/r/issues/7/comments", "body": {"body": BODY}}
    ]


def test_replay_with_marker_present_skips_the_post() -> None:
    # The replay/duplicate hazard: the marker is already in the thread → skip, no POST.
    ns = _namespace()
    post = ns["post_comment_once"]
    resp = asyncio.run(post("o/r", 7, MARKER, BODY, existing_comments=[{"body": BODY}]))
    assert resp.get("skipped") is True
    assert resp["status"] == 200
    assert ns["posts"] == []  # NO duplicate POST


def test_skip_and_real_post_both_count_as_posted_ok() -> None:
    ns = _namespace()
    ok = ns["_comment_posted_ok"]
    assert ok({"status": 200, "skipped": True}) is True
    assert ok({"status": 201}) is True
    assert ok({"status": 200}) is True
    assert ok({"status": 403}) is False
    assert ok({"error": "boom"}) is False
    assert ok(None) is False
