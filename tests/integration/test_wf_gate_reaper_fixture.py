"""Standing reaper / gate dead-man fixture (aios#1386).

Drives the production ``build_reaper_script`` directly against the real script host (no
DB, no live model) with simulated ``list_runs`` / ``http_request`` resolutions, walking
the deterministic config → run-index → in-progress → gates → dirty-PRs → summary machine.
Because the host re-runs from the start each wake (replaying the whole growing memo),
these tests also assert the load-bearing durable property: **replay is deterministic**
(every call_key is stable across every replay).

It proves the #1386 acceptance — a parked-and-forgotten run, an in-progress-with-no-run
issue, or a DIRTY-no-run PR is detected within one sweep and re-driven or escalated, never
silently abandoned:

  * class 1 — ``autodev:in-progress`` issue with NO live run → re-drive recommended
    (below cap) / escalate-with-owner (at cap); an issue WITH a live run is left alone.
  * class 2 — a suspended run (open gate) parked > N hours → escalated; the reaper NEVER
    auto-resolves a gate (structural: no resume_gate/gate in its surface). A fresh gate
    (within N hours) is left alone — and staleness is measured against the FROZEN
    ``trigger.fired_at``, never a wall clock (replay-stable).
  * class 3 — an open DIRTY/CONFLICTING PR with no live run → re-drive into rebase /
    escalate; a clean PR and a ``dirty``-but-live PR are left alone.
  * bounded re-drive — the terminal-run tally caps re-drive at K, then escalates.
  * the no-silent-degrade trip — a truncated list_runs page / non-2xx GitHub read yields
    cannot-determine, NEVER a silent under-count read as "nothing abandoned".
  * the structured per-sweep summary (found + acted).
  * the cron trigger envelope unwrap (config under ``input``; clock from ``trigger``).

Mirrors the host-subprocess style of ``test_wf_telemetry_observer_fixture.py``; needs no
Postgres.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aios.workflows.gate_reaper import (
    DEFAULT_GATE_STALE_HOURS,
    DEFAULT_MAX_REDRIVE_ATTEMPTS,
    REQUIRED_TOOLS,
    build_reaper_fixture_script,
    build_reaper_workflow_create,
)
from aios.workflows.host_launcher import run_script_host

pytestmark = pytest.mark.integration

REPO = "o/r"
WF = "wf_dev_pipeline"
FIRED_AT = "2026-06-18T12:00:00+00:00"  # the frozen cron "now"


def _run_row(
    run_id: str,
    *,
    status: str,
    issue_number: int | None,
    updated_at: str = "2026-06-18T11:00:00+00:00",
    repo: str = REPO,
    wrap_envelope: bool = False,
) -> dict[str, Any]:
    """A list_runs row, real-shaped: a WfRun dump minus the script/surface blobs. ``input``
    carries the dev-pipeline's ``{repo, issue_number}`` (optionally inside a trigger
    envelope, the run_completion / cron re-launch shape)."""
    inp: Any = None
    if issue_number is not None:
        inp = {"repo": repo, "issue_number": issue_number}
        if wrap_envelope:
            inp = {"trigger": {"source": "cron", "fired_at": FIRED_AT}, "input": inp}
    return {
        "id": run_id,
        "workflow_id": WF,
        "status": status,
        "input": inp,
        "output": None,
        "updated_at": updated_at,
        "created_at": "2026-06-18T09:00:00+00:00",
    }


def _gh_list(items: list[Any]) -> dict[str, Any]:
    return {"status": 200, "headers": {}, "body": json.dumps(items)}


def _issue(number: int) -> dict[str, Any]:
    return {"number": number, "state": "open"}


def _pr_row(number: int) -> dict[str, Any]:
    return {"number": number, "state": "open"}


class ReaperScenario:
    """Deterministic responder for the reaper, keyed on the call so replay is stable.

    ``run_pages`` maps a live/terminal run status -> the list_runs rows to return.
    ``in_progress`` / ``open_pulls`` are the GitHub issue/PR lists. ``pr_states`` maps a
    PR number -> its ``mergeable_state`` (the per-PR full read). Comment threads are empty
    (so post_comment_once always posts). Records every mutating POST for assertions."""

    def __init__(
        self,
        *,
        run_pages: dict[str, list[dict[str, Any]]] | None = None,
        in_progress: list[dict[str, Any]] | None = None,
        open_pulls: list[dict[str, Any]] | None = None,
        pr_states: dict[int, str] | None = None,
        list_runs_error: bool = False,
    ) -> None:
        self.run_pages = run_pages or {}
        self.in_progress = in_progress or []
        self.open_pulls = open_pulls or []
        self.pr_states = pr_states or {}
        self.list_runs_error = list_runs_error
        self.comments_posted: list[tuple[str, str]] = []  # (path, body)
        self.labels_added: list[tuple[str, list[str]]] = []  # (path, labels)
        self.list_runs_calls: list[dict[str, Any]] = []
        self.http_calls: list[tuple[str, str]] = []  # (method, path)

    # ── list_runs ──
    def _list_runs(self, args: dict[str, Any]) -> dict[str, Any]:
        self.list_runs_calls.append(args)
        if self.list_runs_error:
            return {"error": "boom"}
        return {"runs": list(self.run_pages.get(args["status"], []))}

    # ── http_request ──
    def _http(self, args: dict[str, Any]) -> dict[str, Any]:
        method = args["method"]
        path = args["path"]
        self.http_calls.append((method, path))
        base = path.split("?", 1)[0]

        if method == "POST" and base.endswith("/comments"):
            body = json.loads(args.get("body") or "{}")
            self.comments_posted.append((base, body.get("body", "")))
            return {"status": 201, "body": "{}"}
        if method == "POST" and base.endswith("/labels"):
            body = json.loads(args.get("body") or "{}")
            self.labels_added.append((base, body.get("labels", [])))
            return {"status": 200, "body": "[]"}
        if method == "GET" and base.endswith("/comments"):
            return _gh_list([])  # empty thread -> post proceeds
        if method == "GET" and "/issues" in base and not base.endswith("/comments"):
            return _gh_list(self.in_progress)
        if method == "GET" and base.endswith("/pulls"):
            return _gh_list(self.open_pulls)
        if method == "GET" and "/pulls/" in base:
            num = int(base.rsplit("/", 1)[1])
            return {
                "status": 200,
                "body": json.dumps(
                    {"number": num, "mergeable_state": self.pr_states.get(num, "clean")}
                ),
            }
        raise AssertionError(f"unexpected http_request {method} {path}")

    def outcome(self, cap: Any) -> dict[str, Any]:
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool":
            name = spec["tool_name"]
            if name == "list_runs":
                return {"ok": self._list_runs(spec["input"])}
            if name == "http_request":
                return {"ok": self._http(spec["input"])}
            raise AssertionError(f"reaper called an unexpected tool: {name}")
        raise AssertionError(f"reaper must not emit capability {cid} (no agent/gate); {spec!r}")


async def _drive(
    scenario: ReaperScenario,
    *,
    input: dict[str, Any],
    max_steps: int = 200,
    **build_kwargs: Any,
) -> Any:
    """Drive the production reaper to a terminal outcome; assert replay-determinism."""
    src = build_reaper_fixture_script(**build_kwargs)
    memo: dict[str, Any] = {}
    keys: list[str] = []
    for _ in range(max_steps):
        out = await run_script_host(source=src, input=input, memo=memo)
        if out.kind == "returned":
            assert len(keys) == len(set(keys)), "replay produced a duplicate call_key"
            return out.value
        assert out.kind == "suspended", (out.kind, out.error_repr, out.error_traceback)
        assert len(out.emitted) == 1, [(e.capability_id, e.spec) for e in out.emitted]
        cap = out.emitted[0]
        keys.append(cap.call_key)
        memo[cap.call_key] = scenario.outcome(cap)
    raise AssertionError(f"reaper did not terminate within {max_steps} steps")


def _cron_input(**cfg: Any) -> dict[str, Any]:
    """The cron WorkflowAction envelope the reaper receives (config under ``input``, the
    frozen clock under ``trigger.fired_at``)."""
    return {
        "trigger": {"id": "trg", "name": "reaper", "source": "cron", "fired_at": FIRED_AT},
        "input": {"repo": REPO, "dev_pipeline_workflow_id": WF, **cfg},
    }


# ─── class 1: in-progress issue with no live run ──────────────────────────────


async def test_in_progress_issue_with_no_run_is_redriven() -> None:
    scn = ReaperScenario(
        run_pages={"pending": [], "running": [], "suspended": [], "errored": [], "cancelled": []},
        in_progress=[_issue(7)],
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "abandonment-found"
    f = [x for x in value["found"] if x["class"] == "in-progress-no-run"]
    assert len(f) == 1 and f[0]["target"] == 7
    assert f[0]["action"] == "redrive-recommended"
    # it escalated: one comment + one escalation label on the issue
    assert any(p.endswith("/issues/7/comments") for p, _ in scn.comments_posted)
    assert any(p.endswith("/issues/7/labels") for p, _ in scn.labels_added)


async def test_in_progress_issue_with_live_run_is_left_alone() -> None:
    scn = ReaperScenario(
        run_pages={
            "pending": [],
            "running": [_run_row("r1", status="running", issue_number=7)],
            "suspended": [],
            "errored": [],
            "cancelled": [],
        },
        in_progress=[_issue(7)],
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "ok"
    assert value["found"] == []
    assert scn.comments_posted == []


async def test_in_progress_correlates_through_trigger_envelope_input() -> None:
    """A re-launched run's issue key lives at input.trigger.input — the reaper must dig
    through the envelope to see it as a live run for issue 7."""
    scn = ReaperScenario(
        run_pages={
            "pending": [],
            "running": [_run_row("r1", status="running", issue_number=7, wrap_envelope=True)],
            "suspended": [],
            "errored": [],
            "cancelled": [],
        },
        in_progress=[_issue(7)],
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "ok"


# ─── bounded re-drive: cap at K terminal runs, then escalate ──────────────────


async def test_redrive_caps_and_escalates_to_owner() -> None:
    """K=2 spent (errored/cancelled) runs for the issue → at the cap, the reaper stops
    recommending re-drive and escalates-with-owner."""
    spent = [
        _run_row("e1", status="errored", issue_number=7),
        _run_row("c1", status="cancelled", issue_number=7),
    ]
    scn = ReaperScenario(
        run_pages={
            "pending": [],
            "running": [],
            "suspended": [],
            "errored": [spent[0]],
            "cancelled": [spent[1]],
        },
        in_progress=[_issue(7)],
    )
    value = await _drive(scn, input=_cron_input(max_redrive_attempts=2))
    f = next(x for x in value["found"] if x["class"] == "in-progress-no-run")
    assert f["action"] == "escalated"
    assert "needs a human" in f["detail"]


# ─── class 2: stale gate ──────────────────────────────────────────────────────


async def test_stale_gate_is_escalated_not_resolved() -> None:
    """A suspended run parked 11h (> 6h default) → escalated. The reaper has no
    resume_gate in its surface, so it CANNOT auto-resolve — escalation is the only act."""
    scn = ReaperScenario(
        run_pages={
            "pending": [],
            "running": [],
            "suspended": [
                _run_row(
                    "g1", status="suspended", issue_number=9, updated_at="2026-06-18T01:00:00+00:00"
                )
            ],
            "errored": [],
            "cancelled": [],
        },
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "abandonment-found"
    f = [x for x in value["found"] if x["class"] == "stale-gate"]
    assert len(f) == 1 and f[0]["target"] == 9
    assert f[0]["action"] == "escalated"
    assert any(p.endswith("/issues/9/comments") for p, _ in scn.comments_posted)


async def test_fresh_gate_within_threshold_is_left_alone() -> None:
    """A suspended run parked only 1h (<= 6h) is healthy — not escalated."""
    scn = ReaperScenario(
        run_pages={
            "pending": [],
            "running": [],
            "suspended": [
                _run_row(
                    "g1", status="suspended", issue_number=9, updated_at="2026-06-18T11:00:00+00:00"
                )
            ],
            "errored": [],
            "cancelled": [],
        },
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "ok"
    assert value["found"] == []
    assert scn.comments_posted == []


async def test_gate_staleness_uses_frozen_fired_at_not_wall_clock() -> None:
    """Staleness is fired_at - updated_at. With NO frozen clock reachable the reaper
    refuses to read a parked gate as fresh — it surfaces cannot-determine."""
    scn = ReaperScenario(
        run_pages={
            "pending": [],
            "running": [],
            "suspended": [
                _run_row(
                    "g1", status="suspended", issue_number=9, updated_at="2020-01-01T00:00:00+00:00"
                )
            ],
            "errored": [],
            "cancelled": [],
        },
    )
    # bare config (no trigger envelope, no `now`) -> no frozen clock
    value = await _drive(scn, input={"repo": REPO, "dev_pipeline_workflow_id": WF})
    assert value["verdict"] == "cannot-determine"


# ─── class 3: dirty PR with no live run ───────────────────────────────────────


async def test_dirty_pr_with_no_run_is_redriven() -> None:
    scn = ReaperScenario(
        run_pages={"pending": [], "running": [], "suspended": [], "errored": [], "cancelled": []},
        open_pulls=[_pr_row(42)],
        pr_states={42: "dirty"},
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "abandonment-found"
    f = [x for x in value["found"] if x["class"] == "dirty-pr-no-run"]
    assert len(f) == 1 and f[0]["target"] == 42
    assert f[0]["action"] == "redrive-recommended"
    assert "rebase" in f[0]["detail"]


async def test_clean_pr_is_left_alone() -> None:
    scn = ReaperScenario(
        run_pages={"pending": [], "running": [], "suspended": [], "errored": [], "cancelled": []},
        open_pulls=[_pr_row(42)],
        pr_states={42: "clean"},
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "ok"
    assert value["found"] == []


async def test_dirty_pr_with_live_run_is_left_alone() -> None:
    scn = ReaperScenario(
        run_pages={
            "pending": [],
            "running": [_run_row("r1", status="running", issue_number=42)],
            "suspended": [],
            "errored": [],
            "cancelled": [],
        },
        open_pulls=[_pr_row(42)],
        pr_states={42: "dirty"},
    )
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "ok"
    assert value["found"] == []


# ─── no-silent-degrade: a degraded read is cannot-determine, never a false ok ─


async def test_list_runs_error_is_cannot_determine() -> None:
    scn = ReaperScenario(list_runs_error=True, in_progress=[_issue(7)])
    value = await _drive(scn, input=_cron_input())
    assert value["verdict"] == "cannot-determine"
    # it never reached the GitHub mutation path
    assert scn.comments_posted == []


async def test_truncated_run_page_is_cannot_determine() -> None:
    """A full list_runs page (len == requested limit) means runs may exist unseen — the
    reaper cannot prove the liveness index, so cannot-determine, NEVER a silent
    under-count that reads a live run as dead and re-drives a healthy issue."""
    scn = ReaperScenario(
        run_pages={
            "pending": [
                _run_row("a", status="pending", issue_number=1),
                _run_row("b", status="pending", issue_number=2),
            ],
            "running": [],
            "suspended": [],
            "errored": [],
            "cancelled": [],
        },
        in_progress=[_issue(7)],
    )
    value = await _drive(scn, input=_cron_input(limit=2))
    assert value["verdict"] == "cannot-determine"
    assert "truncated" in value["reason"]


# ─── the structured summary + a clean sweep ───────────────────────────────────


async def test_clean_sweep_is_ok_with_structured_summary() -> None:
    scn = ReaperScenario(
        run_pages={"pending": [], "running": [], "suspended": [], "errored": [], "cancelled": []},
    )
    value = await _drive(scn, input=_cron_input())
    assert set(value.keys()) >= {"verdict", "scanned", "found", "acted", "bands_version"}
    assert value["verdict"] == "ok"
    assert value["found"] == []
    assert value["acted"] == []


async def test_missing_config_is_cannot_determine() -> None:
    scn = ReaperScenario()
    value = await _drive(
        scn, input={"trigger": {"source": "cron", "fired_at": FIRED_AT}, "input": {}}
    )
    assert value["verdict"] == "cannot-determine"
    assert "missing" in value["reason"]


# ─── surface: structural gate-resolution prohibition ──────────────────────────


def test_reaper_surface_cannot_resolve_gates() -> None:
    """The reaper escalates approval gates; it NEVER auto-resolves them. That rule is
    STRUCTURAL: resume_gate / gate are absent from its tool surface, and the GitHub route
    grants only GET/POST (no merge/close/unlabel)."""
    types = {t.type for t in REQUIRED_TOOLS}
    assert types == {"list_runs", "http_request"}
    assert "resume_gate" not in types
    assert "gate" not in types
    wc = build_reaper_workflow_create()
    assert {t.type for t in wc.tools} == {"list_runs", "http_request"}
    assert len(wc.http_servers) == 1
    route = wc.http_servers[0].routes[0]
    assert set(route.methods) == {"GET", "POST"}
    assert "DELETE" not in route.methods and "PUT" not in route.methods


def test_reaper_script_validates_against_its_surface() -> None:
    """The #1285 create-time validator: the script compiles, defines async def
    main(input), and REQUIRED_TOOLS is a superset of the script's literal tool() names."""
    from aios.workflows.gate_reaper import build_reaper_script
    from aios.workflows.script_validation import validate_workflow_script

    validate_workflow_script(script=build_reaper_script(), tools=list(REQUIRED_TOOLS))


def test_defaults_exposed() -> None:
    assert DEFAULT_GATE_STALE_HOURS > 0
    assert DEFAULT_MAX_REDRIVE_ATTEMPTS > 0
