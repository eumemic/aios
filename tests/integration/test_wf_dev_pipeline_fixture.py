"""Dev-pipeline reference workflow fixture (autodev#19 / aios#987).

Drives the production ``build_dev_pipeline_script`` directly against the real script
host (no DB) with simulated agent/tool/gate resolutions, walking the state machine
clutch by clutch. Because the pipeline is sequential (one capability per wake) and the
driver re-runs from the start each wake — replaying the whole growing memo — these
tests also assert the load-bearing durable property: **replay is deterministic** (every
call_key is stable across every replay).

Mirrors the host-subprocess style of ``test_wf_host.py``; needs no Postgres or sandbox.

NOTE (a known limit, by design): ``run_script_host`` feeds *memoized* capability results,
so these tests exercise the state machine but NOT the live ``http_request`` route gate —
e.g. a query string in ``path`` is rejected by the real tool, which is why the script
lists PRs with a clean path and filters in-script. The deploy-time live smoke covers the
real tool surface (the deep_research "live demo" analog).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aios.workflows.dev_pipeline import build_dev_pipeline_script
from aios.workflows.host_launcher import run_script_host

pytestmark = pytest.mark.integration

REPO = "o/r"
ISSUE = 5
BRANCH = "issue-5"
# A concrete, >50-word spec body that clears the scripted spec gate.
LONG_BODY = (
    "Add a durable retry wrapper around the outbound webhook dispatcher so that transient "
    "five hundred level responses are retried with exponential backoff up to five attempts "
    "on a jittered schedule, with a dead letter table for exhausted deliveries. The "
    "implementation must add a new module, wire it into the dispatcher call site, cover the "
    "backoff math with deterministic unit tests, and document the dead letter replay "
    "procedure for operators in the runbook so the on call engineer has a clear recovery path."
)


def _issue_json(body: str = LONG_BODY) -> str:
    return json.dumps(
        {"number": ISSUE, "title": "Add webhook retry", "body": body, "state": "open"}
    )


def _pr_json(*, sha: str = "sha_initial", merged: bool = False) -> str:
    return json.dumps(
        {
            "number": 42,
            "node_id": "PR_node42",
            "html_url": "https://github.com/o/r/pull/42",
            "head": {"sha": sha, "ref": BRANCH},
            "merged": merged,
        }
    )


# ─── the capability responder (a deterministic function of the capability) ────


class Scenario:
    """A deterministic responder + a record of what the script asked for. Keying every
    decision on the capability spec (and, for the bounded loops, the per-iteration
    ``label``) keeps the responder a pure function of the call_key — so replay is stable.
    """

    def __init__(
        self,
        *,
        body: str = LONG_BODY,
        existing_pr: bool = False,
        implement_escalated: bool = False,
        review_results: list[dict[str, Any]] | None = None,
        ci_results: list[dict[str, Any]] | None = None,
        risk_result: dict[str, Any] | None = None,
        merge_guard_exit: int = 0,
        merge_status: int = 200,
        pr_merged_on_confirm: bool = False,
        gate_results: dict[str, Any] | None = None,
        master_ci: str = "green",
        master_ci_error: bool = False,
    ) -> None:
        self.body = body
        self.existing_pr = existing_pr
        self.implement_escalated = implement_escalated
        self.review_results = review_results or [
            {"verdict": "pass", "issues": [], "artifact_posted": True}
        ]
        self.ci_results = ci_results or [{"status": "green", "detail": ""}]
        self.risk_result = (
            risk_result if risk_result is not None else {"tier": 1, "summary": "safe"}
        )
        self.merge_guard_exit = merge_guard_exit
        self.merge_status = merge_status
        self.pr_merged_on_confirm = pr_merged_on_confirm
        self.gate_results = gate_results or {}
        self.master_ci = master_ci
        self.master_ci_error = master_ci_error
        self.tasks: list[str] = []
        self.gates: list[dict[str, Any]] = []
        self.http: list[tuple[str, str]] = []

    def _http(self, args: dict[str, Any]) -> dict[str, Any]:
        path, method = args["path"], args["method"]
        self.http.append((method, path))
        assert "?" not in path, f"query string in path is rejected by http_request: {path!r}"
        is_issue_get = (
            method == "GET"
            and "/issues/5" in path
            and "/comments" not in path
            and "/labels" not in path
        )
        if is_issue_get:
            return {"status": 200, "body": _issue_json(self.body)}
        if method == "GET" and path == "/repos/o/r/pulls":  # list open PRs (clean path)
            return {"status": 200, "body": "[" + _pr_json() + "]" if self.existing_pr else "[]"}
        if method == "POST" and path == "/repos/o/r/pulls":
            return {"status": 201, "body": _pr_json()}
        if method == "GET" and path == "/repos/o/r/pulls/42":  # merge-confirm read
            return {"status": 200, "body": _pr_json(merged=self.pr_merged_on_confirm)}
        if method == "PUT" and path.endswith("/merge"):
            return {"status": self.merge_status, "body": "{}"}
        return {"status": 200, "body": "{}"}  # comments / labels / graphql

    def outcome(self, cap: Any) -> dict[str, Any]:
        """Return the full memo outcome ({"ok": value} or {"error": {...}}) for a capability."""
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool":
            if spec["tool_name"] == "http_request":
                return {"ok": self._http(spec["input"])}
            if spec["tool_name"] == "bash":  # merge-ref guard
                return {
                    "ok": {
                        "exit_code": self.merge_guard_exit,
                        "stdout": "MERGE_GUARD_OK"
                        if self.merge_guard_exit == 0
                        else "sentinel failed",
                        "stderr": "",
                        "timed_out": False,
                        "truncated": False,
                    }
                }
        if cid == "agent":
            task = spec["input"].get("task")
            label = cap.annotations.get("label", "")
            self.tasks.append(label or task)
            if task == "implement":
                return {
                    "ok": {
                        "branch": BRANCH,
                        "pr_title": "Add webhook retry",
                        "pr_body": "Closes #5",
                        "escalated": self.implement_escalated and label == "implement",
                        "escalation_reason": "interface shape unsettled",
                    }
                }
            if task == "review":
                idx = int(label.rsplit("-", 1)[1]) if "-" in label else 0
                return {"ok": self.review_results[min(idx, len(self.review_results) - 1)]}
            if task == "watch_ci":
                if spec["input"].get("ref"):  # post-merge master watch
                    if self.master_ci_error:
                        return {"error": {"kind": "child_errored"}}
                    return {"ok": {"status": self.master_ci, "detail": ""}}
                idx = int(label.rsplit("-", 1)[1]) if label.startswith("ci-") else 0
                return {"ok": self.ci_results[min(idx, len(self.ci_results) - 1)]}
            if task == "risk":
                return {"ok": self.risk_result}
            if task in ("fix", "fix_ci"):
                return {"ok": {"head_sha": f"sha_{label}", "pushed": True}}
        if cid == "gate":
            self.gates.append(spec)
            result: dict[str, Any] = self.gate_results.get(spec.get("kind"), {})
            return {"ok": result}
        raise AssertionError(f"unhandled capability {cid} spec={spec!r}")


async def _drive(
    scenario: Scenario,
    *,
    input: dict[str, Any] | None = None,
    max_steps: int = 60,
    **build_kwargs: Any,
) -> tuple[Any, list[str], list[str]]:
    """Drive the production script to a terminal outcome, returning
    (return_value, phases, ordered_call_keys). Asserts replay-determinism (unique keys)."""
    src = build_dev_pipeline_script(**build_kwargs)
    inp = input or {"repo": REPO, "issue_number": ISSUE, "kind": "issue"}
    memo: dict[str, Any] = {}
    keys: list[str] = []
    phases: list[str] = []
    for _ in range(max_steps):
        out = await run_script_host(source=src, input=inp, memo=memo)
        phases = [a.payload["text"] for a in out.annotations if a.payload["kind"] == "phase"]
        if out.kind == "returned":
            assert len(keys) == len(set(keys)), "replay produced a duplicate call_key"
            return out.value, phases, keys
        assert out.kind == "suspended", (out.kind, out.error_repr, out.error_traceback)
        assert len(out.emitted) == 1, [(e.capability_id, e.spec) for e in out.emitted]
        cap = out.emitted[0]
        keys.append(cap.call_key)
        memo[cap.call_key] = scenario.outcome(cap)
    raise AssertionError(f"workflow did not terminate within {max_steps} steps")


# ─── happy path ───────────────────────────────────────────────────────────────


async def test_happy_path_merges_and_completes() -> None:
    scn = Scenario(risk_result={"tier": 1, "summary": "additive"})
    value, phases, _keys = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value == {
        "state": "done",
        "pr_url": "https://github.com/o/r/pull/42",
        "pr_number": 42,
        "merged": True,
        "risk_tier": 1,
        "escalations": [],
    }
    assert phases == [
        "ingest",
        "spec-gate",
        "implement",
        "open-pr",
        "verify",
        "risk",
        "merge-guard",
        "mark-ready",
        "merge",
        "post-merge",
    ]
    assert scn.gates == []  # tier-1 auto-merges; no gate opened
    assert scn.tasks == ["implement", "review-0", "ci-0", "risk", "master-ci"]
    # never used a query string (the production-path bug the review caught)
    assert all("?" not in path for _, path in scn.http)


async def test_adopts_existing_open_pr_instead_of_creating() -> None:
    scn = Scenario(existing_pr=True)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert ("POST", "/repos/o/r/pulls") not in scn.http  # adopted, did not create


# ─── scripted spec gate short-circuits before any spend ───────────────────────


async def test_spec_gate_short_circuits_without_spawning_agents() -> None:
    scn = Scenario(body="too short")
    value, phases, _ = await _drive(scn)
    assert value["state"] == "spec_failed"
    assert "too short" in value["reason"]
    assert phases == ["ingest", "spec-gate"]
    assert scn.tasks == []  # no implement agent — failed before any spend
    assert ("POST", "/repos/o/r/issues/5/labels") in scn.http


async def test_spec_gate_blocks_unresolved_marker() -> None:
    body = LONG_BODY + "\n\nApproach: TBD — still an open question we must settle first."
    scn = Scenario(body=body)
    value, _, _ = await _drive(scn)
    assert value["state"] == "spec_failed"
    assert "unresolved marker" in value["reason"]


# ─── escalations park at a gate (the durable-workflow upgrade of awaiting_triage) ─


async def test_design_escalation_parks_then_resumes_and_completes() -> None:
    scn = Scenario(
        implement_escalated=True,
        gate_results={"design": {"resolved": True, "resolution": "use a typed enum"}},
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert "design" in value["escalations"]
    assert any(g["kind"] == "design" for g in scn.gates)
    assert "implement-resumed" in scn.tasks  # resumed -> re-implemented


async def test_design_escalation_unresolved_stays_escalated() -> None:
    scn = Scenario(implement_escalated=True, gate_results={"design": {"resolved": False}})
    value, _, _ = await _drive(scn)
    assert value["state"] == "escalated"
    assert value["reason"] == "design"


async def test_merge_guard_refusal_parks_at_gate() -> None:
    scn = Scenario(merge_guard_exit=73, gate_results={"merge_guard": {"proceed": False}})
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "escalated"
    assert value["reason"] == "merge_guard_refused"
    assert any(g["kind"] == "merge_guard" for g in scn.gates)


async def test_merge_guard_refusal_proceed_merges() -> None:
    scn = Scenario(merge_guard_exit=73, gate_results={"merge_guard": {"proceed": True}})
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["merged"] is True


async def test_high_risk_parks_for_merge_approval() -> None:
    scn = Scenario(
        risk_result={"tier": 4, "summary": "touches the scheduler"},
        gate_results={"merge_approval": {"approve": True}},
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2, auto_merge_max_tier=2)
    assert value["state"] == "done"
    assert value["risk_tier"] == 4
    assert any(g["kind"] == "merge_approval" for g in scn.gates)


async def test_high_risk_hold_does_not_merge() -> None:
    scn = Scenario(
        risk_result={"tier": 4, "summary": "risky"},
        gate_results={"merge_approval": {"approve": False}},
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2, auto_merge_max_tier=2)
    assert value["state"] == "held"
    assert value["reason"] == "merge_approval"


# ─── robustness the adversarial review surfaced ───────────────────────────────


async def test_malformed_risk_return_defaults_to_tier3_and_does_not_crash() -> None:
    # A risk agent that returns a value missing 'tier' must NOT discard a verified PR; it
    # falls back to the conservative tier-3 (which then parks for merge approval).
    scn = Scenario(
        risk_result={"summary": "no tier field"},
        gate_results={"merge_approval": {"approve": True}},
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["risk_tier"] == 3
    assert any(g["kind"] == "merge_approval" for g in scn.gates)


async def test_merge_405_but_already_merged_is_treated_as_done() -> None:
    # A crash-re-driven merge PUT against an already-merged PR returns 405; the confirm GET
    # shows merged=true, so the run completes instead of mis-reporting merge_failed.
    scn = Scenario(merge_status=405, pr_merged_on_confirm=True)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["merged"] is True
    assert ("GET", "/repos/o/r/pulls/42") in scn.http  # the confirm read happened


async def test_merge_failure_not_confirmed_reports_merge_failed() -> None:
    scn = Scenario(merge_status=405, pr_merged_on_confirm=False)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "merge_failed"


async def test_post_merge_master_ci_agent_error_parks_master_red_not_lost_merge() -> None:
    # If the post-merge master-CI agent ERRORS, the merge (a committed fact) must not be
    # discarded: the run parks at master_red and still reports merged on resume.
    scn = Scenario(master_ci_error=True)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["merged"] is True
    assert "master_red" in value["escalations"]
    assert any(g["kind"] == "master_red" for g in scn.gates)


async def test_red_master_ci_parks_at_gate() -> None:
    scn = Scenario(master_ci="red")
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"  # merge happened; parked then resumed
    assert "master_red" in value["escalations"]


# ─── bounded self-repair loops ────────────────────────────────────────────────


async def test_ci_red_then_fix_then_green_completes() -> None:
    scn = Scenario(
        ci_results=[{"status": "red", "detail": "test_x failed"}, {"status": "green", "detail": ""}]
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=3)
    assert value["state"] == "done"
    assert "ci-fix-0" in scn.tasks
    assert "ci-1" in scn.tasks


async def test_review_fail_then_fix_then_pass_completes() -> None:
    scn = Scenario(
        review_results=[
            {"verdict": "fail", "issues": ["unguarded null"], "artifact_posted": True},
            {"verdict": "pass", "issues": [], "artifact_posted": True},
        ]
    )
    value, _, _ = await _drive(scn, max_review_iters=3, max_ci_iters=2)
    assert value["state"] == "done"
    assert "fix-0" in scn.tasks
    assert "review-1" in scn.tasks


async def test_review_loop_does_final_re_review_of_last_fix() -> None:
    # max_review_iters=2 => up to 2 fixes, then a FINAL re-review (review-2) of the last fix.
    # Here review-0 fail, review-1 fail, fix, review-2 PASS -> done (autodev's for/else shape).
    scn = Scenario(
        review_results=[
            {"verdict": "fail", "issues": ["a"], "artifact_posted": True},
            {"verdict": "fail", "issues": ["b"], "artifact_posted": True},
            {"verdict": "pass", "issues": [], "artifact_posted": True},
        ]
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert "review-2" in scn.tasks  # the final re-review ran
    assert "fix-1" in scn.tasks


async def test_missing_review_artifact_parks_at_gate() -> None:
    scn = Scenario(
        review_results=[{"verdict": "pass", "issues": [], "artifact_posted": False}],
        gate_results={"verify": {"resolved": False}},
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "escalated"
    assert value["reason"] == "verify_no_artifact"


async def test_review_exhaustion_parks_at_gate() -> None:
    scn = Scenario(
        review_results=[{"verdict": "fail", "issues": ["x"], "artifact_posted": True}],
        gate_results={"verify": {"resolved": False}},
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "escalated"
    assert value["reason"] == "verify_exhausted"


# ─── trigger envelope ─────────────────────────────────────────────────────────


async def test_accepts_trigger_envelope_input() -> None:
    scn = Scenario()
    value, _, _ = await _drive(
        scn,
        input={
            "trigger": {"kind": "cron"},
            "input": {"repo": REPO, "issue_number": ISSUE, "kind": "issue"},
        },
        max_review_iters=2,
        max_ci_iters=2,
    )
    assert value["state"] == "done"
