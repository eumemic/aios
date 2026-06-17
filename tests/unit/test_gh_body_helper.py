"""Unit tests for the shared GitHub-body parsing helper (aios#1294).

The helper closes the silent-truncate-then-degrade-to-empty CLASS shared by the triage and
dev pipelines: ``http_request`` now flags ``truncated: True`` when it cuts a body, and the
shared ``_json_body`` / ``gh_paginated`` must FAIL LOUD on a truncated or unparseable-2xx
body instead of returning ``None`` / breaking the page loop and degrading to ``[]`` (the
silent ``scanned:0`` no-op that blocked the triage loop).

The helper source is authored once in ``gh_body.GH_BODY_HELPERS`` and spliced into BOTH
pipeline scripts. We exec it in a namespace with the names it references from the
surrounding body (``gh`` / ``_status`` / ``_headers`` / ``_link_next_page`` /
``_with_query`` / ``log``) and exercise the pure logic directly — no LLM, no real tool, no
time.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pytest

from aios.workflows.gh_body import GH_BODY_HELPERS


def _namespace(pages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Exec the helper source with the body-provided names it references. ``pages`` is a
    queue of canned http_request responses returned by the recording ``gh`` stub, one per
    GET (so ``gh_paginated`` can be driven page by page)."""
    queue = list(pages or [])
    calls: list[str] = []

    async def gh(method: str, path: str, body: Any = None) -> dict[str, Any]:
        calls.append(path)
        if queue:
            return queue.pop(0)
        return {"status": 404, "body": ""}

    def _status(resp: Any) -> Any:
        return resp.get("status") if isinstance(resp, dict) else None

    def _headers(resp: Any) -> dict[str, Any]:
        if not isinstance(resp, dict):
            return {}
        h = resp.get("headers")
        if not isinstance(h, dict):
            return {}
        return {str(k).lower(): v for k, v in h.items()}

    def _link_next_page(link_header: Any) -> int | None:
        if not isinstance(link_header, str) or not link_header:
            return None
        for part in link_header.split(","):
            seg = part.strip()
            if 'rel="next"' not in seg:
                continue
            m = re.search(r"[?&]page=(\d+)", seg)
            if m:
                return int(m.group(1))
        return None

    def _with_query(path: str, **params: Any) -> str:
        qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
        return path + ("?" if qs else "") + qs

    ns: dict[str, Any] = {
        "json": json,
        "gh": gh,
        "_status": _status,
        "_headers": _headers,
        "_link_next_page": _link_next_page,
        "_with_query": _with_query,
        "log": lambda *a, **k: None,
        "calls": calls,
    }
    exec(compile(GH_BODY_HELPERS, "gh_body", "exec"), ns)
    return ns


# ─── _json_body ──────────────────────────────────────────────────────────────


def test_json_body_parses_a_clean_2xx() -> None:
    ns = _namespace()
    out = ns["_json_body"]({"status": 200, "body": '[{"number": 1}]'})
    assert out == [{"number": 1}]


def test_json_body_none_for_non_dict_or_empty_body() -> None:
    ns = _namespace()
    assert ns["_json_body"]("boom") is None
    assert ns["_json_body"]({"status": 200}) is None
    assert ns["_json_body"]({"status": 200, "body": ""}) is None


def test_json_body_raises_on_truncated_2xx() -> None:
    # A truncated body lost its tail — refuse to parse it; never degrade to empty.
    ns = _namespace()
    err = ns["GitHubBodyError"]
    resp = {"status": 200, "body": "[0,0,0", "truncated": True}
    with pytest.raises(err):
        ns["_json_body"](resp)


def test_json_body_raises_on_truncated_even_if_prefix_parses() -> None:
    # The dangerous case: a truncated body whose prefix happens to be valid JSON. The
    # ``truncated`` flag must win — parsing it would silently drop the tail.
    ns = _namespace()
    err = ns["GitHubBodyError"]
    resp = {"status": 200, "body": "[1,2,3]", "truncated": True}
    with pytest.raises(err):
        ns["_json_body"](resp)


def test_json_body_raises_on_unparseable_2xx() -> None:
    # A 200 with unparseable JSON is a defect, not an empty result.
    ns = _namespace()
    err = ns["GitHubBodyError"]
    with pytest.raises(err):
        ns["_json_body"]({"status": 200, "body": "not json at all"})


def test_json_body_non_2xx_unparseable_stays_a_value() -> None:
    # A 404/422 with a non-JSON body is the caller's branch, not a fatal read.
    ns = _namespace()
    assert ns["_json_body"]({"status": 404, "body": "Not Found"}) is None
    assert ns["_json_body"]({"status": 422, "body": "<<garbage>>"}) is None


# ─── gh_paginated ────────────────────────────────────────────────────────────


def test_gh_paginated_concatenates_pages() -> None:
    pages = [
        {"status": 200, "body": "[1,2]", "headers": {"Link": '<...?page=2>; rel="next"'}},
        {"status": 200, "body": "[3,4]"},
    ]
    ns = _namespace(pages)
    items = asyncio.run(ns["gh_paginated"]("/repos/o/r/issues"))
    assert items == [1, 2, 3, 4]


def test_gh_paginated_raises_on_truncated_page_never_returns_partial() -> None:
    # The exact aios#1294 failure: a truncated 2xx list page must RAISE, not degrade to a
    # partial / empty list and report ``scanned:0``.
    pages = [
        {"status": 200, "body": "[1,2]", "headers": {"Link": '<...?page=2>; rel="next"'}},
        {"status": 200, "body": "[3,4", "truncated": True},
    ]
    ns = _namespace(pages)
    err = ns["GitHubBodyError"]
    with pytest.raises(err):
        asyncio.run(ns["gh_paginated"]("/repos/o/r/issues"))


def test_gh_paginated_raises_on_unparseable_2xx_page() -> None:
    pages = [{"status": 200, "body": "definitely not json"}]
    ns = _namespace(pages)
    err = ns["GitHubBodyError"]
    with pytest.raises(err):
        asyncio.run(ns["gh_paginated"]("/repos/o/r/issues"))


def test_gh_paginated_non_2xx_page_stops_and_returns_gathered() -> None:
    # A non-2xx page (auth/404) is the caller's branch — stop and return what we have.
    pages = [
        {"status": 200, "body": "[1,2]", "headers": {"Link": '<...?page=2>; rel="next"'}},
        {"status": 403, "body": "Forbidden"},
    ]
    ns = _namespace(pages)
    items = asyncio.run(ns["gh_paginated"]("/repos/o/r/issues"))
    assert items == [1, 2]
