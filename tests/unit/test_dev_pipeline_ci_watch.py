"""Unit tests for the dev-pipeline DETERMINISTIC CI watch (#1316).

The CI-watch role used to be an ``agent()`` (Opus) that polled GitHub until CI was terminal
and returned ``CI_SCHEMA`` ``{status: green|red|no_ci, detail}``. Waiting for a CI run to reach
a terminal state requires ZERO judgment (``architecture/intelligence-vs-computation.md``):
burning model tokens to call the Checks API in a loop is wasteful, and an LLM interpreting
check-status free text is the completion-marker-from-free-text failure class. #1316 replaces
the watch ``agent()`` at BOTH call sites — the A9 PR-CI loop and the post-merge master-CI read
— with a deterministic poll of the GitHub Checks + combined-status REST API via the
``http_request`` tool. ``fix_ci`` STAYS an ``agent()`` (repairing a red build is judgment).

These tests prove the acceptance criteria WITHOUT a live model:

* ``_ci_poll_verdict`` (pure) maps the Checks + combined-status payloads to green / red / no_ci,
  with a None sentinel for "CI present but still running" (keep polling).
* ``watch_ci`` (the A9 node) polls to a terminal verdict, returns red fail-SAFE on a build
  that never goes terminal, and NEVER consults an agent/model.
* ``watch_ci_advisory`` (the post-merge node) distinguishes a real verdict from INDETERMINATE
  (both surfaces unreadable, or never-terminal) — the #1176 advisory contract preserved.
* The rendered production script makes NO ``agent()`` call carrying ``task == "watch_ci"`` at
  either call site (the watch is fully de-intelligenced); ``fix_ci`` is still an agent task.

Like the risk-floor tests, the watch helpers live inside the workflow *script source*
(``_BODY``), so they are not importable as module attributes: we build the production script,
``exec`` it in a fresh namespace, and exercise the functions directly. ``watch_ci`` /
``watch_ci_advisory`` / ``_read_ci`` are async and call the module-level ``gh`` helper, which
we monkeypatch in the exec namespace to feed canned GitHub responses — no tool, no network, no
model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from aios.workflows.dev_pipeline import build_dev_pipeline_script


def _ns() -> dict[str, Any]:
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
        max_ci_iters=3,
    )
    namespace: dict[str, Any] = {}
    exec(compile(src, "dev_pipeline_script", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def ns() -> dict[str, Any]:
    return _ns()


@pytest.fixture(scope="module")
def ci_verdict(ns: dict[str, Any]) -> Callable[[Any, Any], Any]:
    return ns["_ci_poll_verdict"]  # type: ignore[no-any-return]


# A GitHub Checks-API response shape (GET /commits/{sha}/check-runs).
def _checks(*runs: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": 200,
        "headers": {},
        "body": _json({"total_count": len(runs), "check_runs": list(runs)}),
    }


# A combined-status response shape (GET /commits/{sha}/status).
def _combined(state: str, total: int) -> dict[str, Any]:
    return {"status": 200, "headers": {}, "body": _json({"state": state, "total_count": total})}


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj)


def _err(status: int = 500) -> dict[str, Any]:
    return {"status": status, "headers": {}, "body": "boom"}


# ─── the pure verdict reducer: green / red / no_ci / still-running ─────────────


def test_no_ci_when_neither_surface_reports(ci_verdict: Callable[[Any, Any], Any]) -> None:
    # Empty check-runs AND an empty combined status => no_ci (CI_SCHEMA status).
    out = ci_verdict([], {"total_count": 0, "state": "pending"})
    assert out["status"] == "no_ci"


def test_green_when_all_checks_completed_success(ci_verdict: Callable[[Any, Any], Any]) -> None:
    out = ci_verdict([{"status": "completed", "conclusion": "success"}], {"total_count": 0})
    assert out["status"] == "green"


def test_red_when_a_check_failed(ci_verdict: Callable[[Any, Any], Any]) -> None:
    out = ci_verdict([{"status": "completed", "conclusion": "failure"}], {"total_count": 0})
    assert out["status"] == "red"


@pytest.mark.parametrize(
    "conclusion", ["failure", "timed_out", "cancelled", "action_required", "startup_failure"]
)
def test_failing_conclusions_are_red(
    ci_verdict: Callable[[Any, Any], Any], conclusion: str
) -> None:
    out = ci_verdict([{"status": "completed", "conclusion": conclusion}], {"total_count": 0})
    assert out["status"] == "red"


@pytest.mark.parametrize("conclusion", ["success", "neutral", "skipped", "stale", None])
def test_non_failing_completed_conclusions_are_green(
    ci_verdict: Callable[[Any, Any], Any], conclusion: str | None
) -> None:
    out = ci_verdict([{"status": "completed", "conclusion": conclusion}], {"total_count": 0})
    assert out["status"] == "green"


def test_still_running_returns_none_sentinel(ci_verdict: Callable[[Any, Any], Any]) -> None:
    # A queued / in-progress check is NOT terminal — the verdict is the None "keep polling"
    # sentinel, NOT a CI_SCHEMA status. (None is never returned to the caller as a verdict.)
    assert ci_verdict([{"status": "in_progress", "conclusion": None}], {"total_count": 0}) is None
    assert ci_verdict([{"status": "queued", "conclusion": None}], {"total_count": 0}) is None


def test_red_is_eager_even_with_a_running_check(ci_verdict: Callable[[Any, Any], Any]) -> None:
    # A doomed build fails fast: a terminal failure is red even while other checks run.
    out = ci_verdict(
        [{"status": "completed", "conclusion": "failure"}, {"status": "in_progress"}],
        {"total_count": 0},
    )
    assert out["status"] == "red"


def test_combined_status_only_green(ci_verdict: Callable[[Any, Any], Any]) -> None:
    # External CI via the legacy commit-status protocol (no check-runs) is honoured.
    out = ci_verdict([], {"total_count": 1, "state": "success"})
    assert out["status"] == "green"


def test_combined_status_only_red(ci_verdict: Callable[[Any, Any], Any]) -> None:
    out = ci_verdict(None, {"total_count": 2, "state": "failure"})
    assert out["status"] == "red"


def test_combined_status_pending_is_still_running(
    ci_verdict: Callable[[Any, Any], Any],
) -> None:
    assert ci_verdict(None, {"total_count": 1, "state": "pending"}) is None


def test_mixed_surface_pending_blocks_green(ci_verdict: Callable[[Any, Any], Any]) -> None:
    # Checks all green but a legacy status still pending => keep polling (not green yet).
    out = ci_verdict(
        [{"status": "completed", "conclusion": "success"}],
        {"total_count": 1, "state": "pending"},
    )
    assert out is None


def test_garbage_payloads_fold_to_no_ci(ci_verdict: Callable[[Any, Any], Any]) -> None:
    # A non-list check_runs and non-dict combined contribute nothing -> no_ci, never a crash.
    out = ci_verdict("not-a-list", "not-a-dict")
    assert out["status"] == "no_ci"


# ─── the A9 PR-CI watch node: polls to terminal, deterministic, fail-safe ──────


def _patch_gh(ns: dict[str, Any], responses: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Patch the namespace ``gh`` to pop canned responses in order; record (method, path)."""
    calls: list[tuple[str, str]] = []
    queue = list(responses)

    async def fake_gh(method: str, path: str, body: Any = None) -> dict[str, Any]:
        calls.append((method, path))
        return queue.pop(0) if queue else {"status": 200, "headers": {}, "body": _json({})}

    ns["gh"] = fake_gh
    return calls


def test_watch_ci_returns_green(ns: dict[str, Any]) -> None:
    # Both surface reads (check-runs, then status) on the first poll yield a green verdict.
    _patch_gh(
        ns, [_checks({"status": "completed", "conclusion": "success"}), _combined("success", 0)]
    )
    out = asyncio.run(ns["watch_ci"]("o/r", "a" * 40, 3))
    assert out["status"] == "green"


def test_watch_ci_returns_red(ns: dict[str, Any]) -> None:
    _patch_gh(
        ns, [_checks({"status": "completed", "conclusion": "failure"}), _combined("success", 0)]
    )
    out = asyncio.run(ns["watch_ci"]("o/r", "a" * 40, 3))
    assert out["status"] == "red"


def test_watch_ci_returns_no_ci(ns: dict[str, Any]) -> None:
    _patch_gh(ns, [_checks(), _combined("pending", 0)])
    out = asyncio.run(ns["watch_ci"]("o/r", "a" * 40, 3))
    assert out["status"] == "no_ci"


def test_watch_ci_polls_until_terminal(ns: dict[str, Any]) -> None:
    # First poll: still running (in_progress + pending). Second poll: green. The node must
    # keep polling and return the terminal verdict — not give up after one read.
    _patch_gh(
        ns,
        [
            _checks({"status": "in_progress"}),  # poll 1: check-runs
            _combined("pending", 0),  # poll 1: status
            _checks({"status": "completed", "conclusion": "success"}),  # poll 2: check-runs
            _combined("success", 0),  # poll 2: status
        ],
    )
    out = asyncio.run(ns["watch_ci"]("o/r", "a" * 40, 3))
    assert out["status"] == "green"


def test_watch_ci_never_terminal_is_red_failsafe(ns: dict[str, Any]) -> None:
    # A build that never reaches a terminal state within max_iters must NOT be waved through
    # as green — it returns red so the A9 loop escalates to the verify gate (fail-SAFE).
    never = [_checks({"status": "in_progress"}), _combined("pending", 0)] * 5
    _patch_gh(ns, never)
    out = asyncio.run(ns["watch_ci"]("o/r", "a" * 40, 3))
    assert out["status"] == "red"
    assert "did not reach a terminal state" in out["detail"]


def test_watch_ci_calls_the_checks_and_status_endpoints(ns: dict[str, Any]) -> None:
    # The watch reads the Checks API + combined-status endpoints deterministically — NO agent.
    calls = _patch_gh(
        ns, [_checks({"status": "completed", "conclusion": "success"}), _combined("success", 0)]
    )
    asyncio.run(ns["watch_ci"]("o/r", "a" * 40, 3))
    paths = [p for _, p in calls]
    assert any("/commits/" in p and "/check-runs" in p for p in paths)
    assert any("/commits/" in p and "/status" in p for p in paths)
    assert all(m == "GET" for m, _ in calls)  # read-only


# ─── the post-merge master-CI advisory node: distinguishes indeterminate ───────


def test_watch_ci_advisory_green(ns: dict[str, Any]) -> None:
    _patch_gh(
        ns, [_checks({"status": "completed", "conclusion": "success"}), _combined("success", 0)]
    )
    status, _detail = asyncio.run(ns["watch_ci_advisory"]("o/r", "a" * 40, 3))
    assert status == "green"


def test_watch_ci_advisory_red(ns: dict[str, Any]) -> None:
    _patch_gh(
        ns, [_checks({"status": "completed", "conclusion": "failure"}), _combined("success", 0)]
    )
    status, _detail = asyncio.run(ns["watch_ci_advisory"]("o/r", "a" * 40, 3))
    assert status == "red"


def test_watch_ci_advisory_no_ci(ns: dict[str, Any]) -> None:
    _patch_gh(ns, [_checks(), _combined("pending", 0)])
    status, _detail = asyncio.run(ns["watch_ci_advisory"]("o/r", "a" * 40, 3))
    assert status == "no_ci"


def test_watch_ci_advisory_unreadable_is_indeterminate(ns: dict[str, Any]) -> None:
    # Both CI surfaces non-2xx on every poll => INDETERMINATE (status None), DISTINCT from a
    # 'no_ci' verdict and NEVER coerced to the most-blocking 'red' (#1176 advisory contract).
    _patch_gh(ns, [_err(500), _err(500)] * 5)
    status, detail = asyncio.run(ns["watch_ci_advisory"]("o/r", "a" * 40, 3))
    assert status is None
    assert "could not read either CI surface" in detail


def test_watch_ci_advisory_never_terminal_is_indeterminate(ns: dict[str, Any]) -> None:
    # CI readable but never terminal within max_iters => INDETERMINATE (NOT red): the merge is
    # a committed fact, so the post-merge watch never manufactures a blocking red verdict.
    _patch_gh(ns, [_checks({"status": "in_progress"}), _combined("pending", 0)] * 5)
    status, detail = asyncio.run(ns["watch_ci_advisory"]("o/r", "a" * 40, 3))
    assert status is None
    assert "did not reach a terminal state" in detail


def test_watch_ci_advisory_one_surface_down_still_verdicts(ns: dict[str, Any]) -> None:
    # check-runs down (500) but combined-status green => still a real green verdict (a single
    # transient surface blip cannot manufacture a false indeterminate when the other surface
    # gives a clean terminal answer).
    _patch_gh(ns, [_err(500), _combined("success", 1)])
    status, _detail = asyncio.run(ns["watch_ci_advisory"]("o/r", "a" * 40, 3))
    assert status == "green"


# ─── acceptance: the rendered script makes NO watch_ci agent call; fix_ci stays ─


def test_rendered_script_has_no_watch_ci_agent_call() -> None:
    # The whole point of #1316: neither call site invokes an agent for the watch. The old
    # sites passed {"task": "watch_ci", ...} to agent(); that must be gone entirely.
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
    )
    assert '"task": "watch_ci"' not in src
    assert "'task': 'watch_ci'" not in src
    # The deterministic nodes ARE present and called.
    assert "async def watch_ci(" in src
    assert "async def watch_ci_advisory(" in src
    assert "await watch_ci(" in src
    assert "await watch_ci_advisory(" in src


def test_rendered_script_keeps_fix_ci_as_agent() -> None:
    # fix_ci is genuine judgment and STAYS an agent() task — only the wait/read is de-intelligenced.
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
    )
    assert '"task": "fix_ci"' in src
    assert "FIX_AGENT_ID" in src
