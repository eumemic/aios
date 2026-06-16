"""Unit tests for the post-merge source-issue cleanup (#1188).

A merged dev-pipeline PR used to leave its source issue **OPEN** with stale
``autodev:in-progress`` + ``dispatched`` claim labels: the pipeline's PR bodies carry no
``Closes #N`` linkage, so GitHub never auto-closes the issue on merge, and the post-merge
node only stripped ``autodev:in-progress``. To a dispatch-gate sweep / rank-6 staleness
reaper the completed work then read as in-flight (#1156, #1176 both merged-but-open).

The fix adds ``_close_source_issue(repo, issue_number)`` to the workflow *script body*: the
terminal post-merge cleanup that idempotently strips BOTH claim labels and closes the issue
(``state:closed`` + ``state_reason:completed``), logging each gh() response (rank-6
label-write visibility). ``_unlabel`` now also logs its DELETE outcome.

These helpers live inside the workflow script source (``_BODY``), so they are not importable
as module attributes. We build the production script and ``exec`` it in a fresh namespace,
inject a fake async ``gh`` (recording every call) and a ``log`` sink, then drive the
coroutine to completion. No LLM, no real I/O, no time.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aios.workflows.dev_pipeline import build_dev_pipeline_script


def _ns(gh: Any) -> dict[str, Any]:
    """A fresh exec namespace for the script body with a fake ``gh`` and ``log`` injected.

    ``gh``/``log`` are runtime-provided names in the deployed workflow (not defined in the
    body); injecting them into the exec namespace lets the coroutine resolve them as globals.
    """
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
    )
    namespace: dict[str, Any] = {}
    exec(compile(src, "dev_pipeline_script", "exec"), namespace)
    logs: list[str] = []
    namespace["gh"] = gh
    namespace["log"] = lambda *a: logs.append(" ".join(str(x) for x in a))
    namespace["_LOGS"] = logs
    return namespace


class _FakeGH:
    """A recording fake for the workflow's ``gh(method, path, body=None)`` coroutine.

    ``responses`` maps a ``(method, path)`` to the dict response to return; an unmapped call
    returns a 200 with empty body (the GitHub default for a successful PATCH/DELETE here).
    Every call is appended to ``calls`` so a test can assert exactly what was issued.
    """

    def __init__(self, responses: dict[tuple[str, str], dict[str, Any]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str, Any]] = []

    async def __call__(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        self.calls.append((method, path, body))
        return self.responses.get((method, path), {"status": 200, "body": ""})


REPO = "eumemic/aios"
ISSUE = 1156
IN_PROGRESS_PATH = "/repos/eumemic/aios/issues/1156/labels/autodev%3Ain-progress"
DISPATCHED_PATH = "/repos/eumemic/aios/issues/1156/labels/dispatched"
ISSUE_PATH = "/repos/eumemic/aios/issues/1156"


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ─── acceptance: a merged PR closes the issue and strips BOTH claim labels ─────


def test_close_source_issue_strips_both_labels_and_closes() -> None:
    gh = _FakeGH()
    ns = _ns(gh)
    _run(ns["_close_source_issue"](REPO, ISSUE))

    methods_paths = [(m, p) for (m, p, _b) in gh.calls]
    # Both in-flight claim labels are DELETEd...
    assert ("DELETE", IN_PROGRESS_PATH) in methods_paths
    assert ("DELETE", DISPATCHED_PATH) in methods_paths
    # ...and the issue is PATCHed closed with the completed reason.
    patch = [(m, p, b) for (m, p, b) in gh.calls if m == "PATCH"]
    assert len(patch) == 1
    assert patch[0][1] == ISSUE_PATH
    assert patch[0][2] == {"state": "closed", "state_reason": "completed"}


def test_close_source_issue_removes_dispatched_label() -> None:
    # Regression: the old node stripped only autodev:in-progress, leaving `dispatched`.
    gh = _FakeGH()
    ns = _ns(gh)
    _run(ns["_close_source_issue"](REPO, ISSUE))
    assert ("DELETE", DISPATCHED_PATH) in [(m, p) for (m, p, _b) in gh.calls]


# ─── idempotence: a re-drive over an already-cleaned issue is a no-op ──────────


def test_idempotent_when_labels_absent_and_issue_already_closed() -> None:
    # A re-launched run finds the labels already gone (404) and the issue already closed
    # (PATCH returns 200 — re-closing is a GitHub no-op). It must NOT raise.
    gh = _FakeGH(
        {
            ("DELETE", IN_PROGRESS_PATH): {"status": 404, "body": ""},
            ("DELETE", DISPATCHED_PATH): {"status": 404, "body": ""},
            ("PATCH", ISSUE_PATH): {"status": 200, "body": '{"state": "closed"}'},
        }
    )
    ns = _ns(gh)
    # No exception == idempotent no-op.
    resp = _run(ns["_close_source_issue"](REPO, ISSUE))
    assert resp == {"status": 200, "body": '{"state": "closed"}'}


def test_label_already_absent_404_is_not_a_failure_log() -> None:
    # A 404 unlabel (label absent) is the idempotent no-op a re-drive expects: logged as
    # ok, NOT as a FAILED strip.
    gh = _FakeGH(
        {
            ("DELETE", IN_PROGRESS_PATH): {"status": 404, "body": ""},
            ("DELETE", DISPATCHED_PATH): {"status": 404, "body": ""},
        }
    )
    ns = _ns(gh)
    _run(ns["_close_source_issue"](REPO, ISSUE))
    logs = "\n".join(ns["_LOGS"])
    assert "FAILED" not in logs


# ─── visibility: a failed strip / close is logged, not silently swallowed ──────


def test_failed_label_strip_is_logged() -> None:
    # A non-benign DELETE status (e.g. 403/500-after-retries) must be surfaced in the log.
    gh = _FakeGH({("DELETE", DISPATCHED_PATH): {"status": 403, "body": "forbidden"}})
    ns = _ns(gh)
    _run(ns["_close_source_issue"](REPO, ISSUE))
    logs = "\n".join(ns["_LOGS"])
    assert "FAILED" in logs
    assert "dispatched" in logs


def test_failed_close_is_logged() -> None:
    # A non-200 PATCH (close rejected) must be visible, never silently swallowed.
    gh = _FakeGH({("PATCH", ISSUE_PATH): {"status": 422, "body": "unprocessable"}})
    ns = _ns(gh)
    _run(ns["_close_source_issue"](REPO, ISSUE))
    logs = "\n".join(ns["_LOGS"])
    assert "close source issue" in logs
    assert "FAILED" in logs


def test_unlabel_returns_and_logs_response() -> None:
    # _unlabel now returns the gh() response and logs the outcome (rank-6 visibility).
    gh = _FakeGH({("DELETE", DISPATCHED_PATH): {"status": 200, "body": ""}})
    ns = _ns(gh)
    resp = _run(ns["_unlabel"](REPO, ISSUE, "dispatched"))
    assert resp == {"status": 200, "body": ""}
    assert any("unlabel" in line for line in ns["_LOGS"])
