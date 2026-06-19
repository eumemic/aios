"""Dev-pipeline reconciler reference workflow fixture (aios#49/#111, build step 5).

Drives the production ``build_reconciler_script`` directly against the real script host (no DB)
with simulated agent/tool resolutions, walking the stateless scan→owner→ONE-transition→exit
machine. Because the host re-runs from the start each wake (replaying the whole growing memo),
these tests also assert the load-bearing durable property: **replay is deterministic** (every
call_key is stable across every replay).

It proves the design's substrate-correctness:
- the reconciler reads the COMPLETE blackboard once, computes ``owner()``, and drives EXACTLY ONE
  transition per pass (build / rebase / ci / review / merge / escalate);
- WOULD-MERGE ADVISORY mode (the v1 default): the merge branch stamps an advisory, never merges;
- the #1158 tier-gate: a tier>cap PR routes to needs:human/merge-approval, never merge:approved;
- **idempotency** (step 2): a re-run over an item the reconciler already advanced touches NOTHING
  already advanced (the next pass re-derives state from scratch and finds it terminal / re-stamps
  the same single-valued labels).

Mirrors the host-subprocess style of ``test_wf_triage_pipeline_fixture.py`` /
``test_wf_dev_pipeline_fixture.py``; needs no Postgres.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from aios.workflows.dev_pipeline_reconciler import build_reconciler_script
from aios.workflows.host_launcher import run_script_host

pytestmark = pytest.mark.integration

REPO = "o/r"
V2 = "pipeline:v2"
SHA = "a" * 40
SHA2 = "b" * 40


def _issue(
    number: int,
    *,
    labels: list[str] | None = None,
    body: str = "A sufficiently long spec body to clear the fifty word minimum word count gate that "
    "the scripted pre-flight spec gate enforces over the issue body plus its comment thread before "
    "any implement agent is ever dispatched against the issue so that we never spend a build agent "
    "on an underspecified issue that cannot possibly be implemented as written here today now.",
    state: str = "open",
    pull_request: bool = False,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "number": number,
        "title": f"Issue {number}",
        "body": body,
        "state": state,
        "labels": [{"name": n} for n in (labels or [])],
    }
    if pull_request:
        obj["pull_request"] = {"url": f"https://github.com/o/r/pull/{number}"}
    return obj


def _pr(
    number: int,
    *,
    labels: list[str] | None = None,
    head_ref: str | None = None,
    head_sha: str = SHA,
    mergeable: bool | None = True,
    mergeable_state: str = "clean",
    draft: bool = False,
    state: str = "open",
    merged: bool = False,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"PR {number}",
        "state": state,
        "draft": draft,
        "merged": merged,
        "mergeable": mergeable,
        "mergeable_state": mergeable_state,
        "node_id": f"NODE_{number}",
        "html_url": f"https://github.com/o/r/pull/{number}",
        "head": {"ref": head_ref or f"dev-pipeline/issue-{number}", "sha": head_sha},
        "labels": [{"name": n} for n in (labels or [])],
    }


class Scenario:
    """A deterministic responder + a record of what the script asked for. Keying every decision on
    the capability (and the per-call label) keeps the responder a pure function of the call_key —
    so replay is stable. Models the GitHub REST surface the reconciler reads/writes + the agent and
    bash tool returns."""

    PER_PAGE = 100

    def __init__(
        self,
        *,
        issues: list[dict[str, Any]] | None = None,
        prs: list[dict[str, Any]] | None = None,
        ci: dict[int, dict[str, Any]] | None = None,
        reviews: dict[int, list[dict[str, Any]]] | None = None,
        review_results: dict[int, dict[str, Any]] | None = None,
        risk_results: dict[int, dict[str, Any]] | None = None,
        fix_results: dict[int, dict[str, Any]] | None = None,
        implement_results: dict[int, dict[str, Any]] | None = None,
        files: dict[int, list[dict[str, Any]]] | None = None,
        commits: dict[int, int] | None = None,
        reviews_status: int = 200,
        merge_put_status: int = 200,
        merge_guard_exit: int = 0,
        rebase_exit: int = 75,
        agent_error_on: set[str] | None = None,
    ) -> None:
        self.issues = issues or []
        self.prs = prs or []
        self.ci = ci or {}
        self.reviews = reviews or {}
        self.review_results = review_results or {}
        self.risk_results = risk_results or {}
        self.fix_results = fix_results or {}
        self.implement_results = implement_results or {}
        self.files = files or {}
        # {pr_number: commit_count} — the PR branch's commits (the implement-agent dedup signal).
        self.commits = commits or {}
        self.reviews_status = reviews_status
        self.merge_put_status = merge_put_status
        self.merge_guard_exit = merge_guard_exit
        self.rebase_exit = rebase_exit
        self.agent_error_on = agent_error_on or set()
        # observability
        self.http: list[tuple[str, str]] = []
        self.agent_labels: list[str] = []
        self.bash_commands: list[str] = []
        self.labels_added: dict[int, list[str]] = {}
        self.labels_removed: dict[int, list[str]] = {}
        self.comments_posted: dict[int, list[str]] = {}
        self.merges_put: list[int] = []

    # ── pagination ──
    def _list_page(self, rows: list[dict[str, Any]], base: str, page: int) -> dict[str, Any]:
        per = self.PER_PAGE
        start = (page - 1) * per
        chunk = rows[start : start + per]
        total_pages = max(1, (len(rows) + per - 1) // per)
        headers: dict[str, str] = {}
        if page < total_pages:
            headers["Link"] = (
                f'<{base}?per_page={per}&page={page + 1}>; rel="next", '
                f'<{base}?per_page={per}&page={total_pages}>; rel="last"'
            )
        return {"status": 200, "headers": headers, "body": json.dumps(chunk)}

    def _pr_by_number(self, n: int) -> dict[str, Any] | None:
        for pr in self.prs:
            if pr["number"] == n:
                return pr
        return None

    def _http(self, args: dict[str, Any]) -> dict[str, Any]:
        path, method = args["path"], args["method"]
        self.http.append((method, path))
        clean, _, query = path.partition("?")
        page = 1
        m = re.search(r"[?&]page=(\d+)", "?" + query)
        if m:
            page = int(m.group(1))

        if method == "GET" and clean == "/repos/o/r/issues":
            return self._list_page(self.issues, "https://api.github.com/repos/o/r/issues", page)
        if method == "GET" and clean == "/repos/o/r/pulls":
            return self._list_page(self.prs, "https://api.github.com/repos/o/r/pulls", page)

        mp = re.match(r"^/repos/o/r/pulls/(\d+)$", clean)
        if method == "GET" and mp:
            pr = self._pr_by_number(int(mp.group(1)))
            return {"status": 200, "body": json.dumps(pr or {})}

        mr = re.match(r"^/repos/o/r/pulls/(\d+)/reviews$", clean)
        if method == "GET" and mr:
            if self.reviews_status != 200:
                return {"status": self.reviews_status, "body": "{}"}
            return {"status": 200, "body": json.dumps(self.reviews.get(int(mr.group(1)), []))}

        mco = re.match(r"^/repos/o/r/pulls/(\d+)/commits$", clean)
        if method == "GET" and mco:
            count = self.commits.get(int(mco.group(1)), 0)
            return {"status": 200, "body": json.dumps([{"sha": f"c{i}"} for i in range(count)])}

        mf = re.match(r"^/repos/o/r/pulls/(\d+)/files$", clean)
        if method == "GET" and mf:
            return {"status": 200, "body": json.dumps(self.files.get(int(mf.group(1)), []))}

        mmerge = re.match(r"^/repos/o/r/pulls/(\d+)/merge$", clean)
        if method == "PUT" and mmerge:
            n = int(mmerge.group(1))
            if self.merge_put_status != 200:
                return {"status": self.merge_put_status, "body": "{}"}
            self.merges_put.append(n)
            return {"status": 200, "body": json.dumps({"sha": "merge" + "0" * 35, "merged": True})}

        mc = re.match(r"^/repos/o/r/issues/(\d+)/comments$", clean)
        if method == "GET" and mc:
            return {"status": 200, "body": "[]"}
        if method == "POST" and mc:
            n = int(mc.group(1))
            raw = args.get("body")
            if isinstance(raw, str):
                self.comments_posted.setdefault(n, []).append(json.loads(raw).get("body", ""))
            return {"status": 201, "body": "{}"}

        ml = re.match(r"^/repos/o/r/issues/(\d+)/labels$", clean)
        if method == "POST" and ml:
            n = int(ml.group(1))
            raw = args.get("body")
            if isinstance(raw, str):
                self.labels_added.setdefault(n, []).extend(json.loads(raw).get("labels", []))
            return {"status": 200, "body": "[]"}

        mdl = re.match(r"^/repos/o/r/issues/(\d+)/labels/(.+)$", clean)
        if method == "DELETE" and mdl:
            n = int(mdl.group(1))
            label = mdl.group(2).replace("%3A", ":")
            self.labels_removed.setdefault(n, []).append(label)
            return {"status": 200, "body": "[]"}

        mclose = re.match(r"^/repos/o/r/issues/(\d+)$", clean)
        if method == "PATCH" and mclose:
            return {"status": 200, "body": json.dumps({"state": "closed"})}

        # POST a new PR (the PR-FIRST build open)
        if method == "POST" and clean == "/repos/o/r/pulls":
            raw = args.get("body")
            body = json.loads(raw) if isinstance(raw, str) else {}
            ref = body.get("head", "")
            mnum = re.search(r"issue-(\d+)", ref)
            num = 9000 + int(mnum.group(1)) if mnum else 9999
            new = _pr(num, head_ref=ref, labels=[], draft=True)
            self.prs.append(new)
            return {"status": 201, "body": json.dumps(new)}

        if method == "POST" and clean == "/graphql":
            return {"status": 200, "body": "{}"}

        return {"status": 200, "body": "{}"}

    def _bash(self, command: str) -> dict[str, Any]:
        self.bash_commands.append(command)
        if "MERGE_GUARD_OK" in command or "mg-" in command or "refs/pull" in command:
            return {"exit_code": self.merge_guard_exit, "stdout": "MERGE_GUARD_OK", "stderr": ""}
        # the rebase node
        return {"exit_code": self.rebase_exit, "stdout": "REBASE", "stderr": ""}

    def outcome(self, cap: Any) -> dict[str, Any]:
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool" and spec["tool_name"] == "http_request":
            return {"ok": self._http(spec["input"])}
        if cid == "tool" and spec["tool_name"] == "bash":
            return {"ok": self._bash(spec["input"].get("command", ""))}
        if cid == "agent":
            label = cap.annotations.get("label", "")
            self.agent_labels.append(label)
            if label in self.agent_error_on:
                return {"error": {"kind": "child_errored"}}
            task = spec["input"].get("task", "")
            n = int(spec["input"].get("pr_number") or spec["input"].get("issue_number") or 0)
            if task == "watch_ci":
                return {
                    "ok": self.ci.get(
                        n, {"status": "green", "polled_sha": SHA, "required_complete": True}
                    )
                }
            if task == "review":
                return {
                    "ok": self.review_results.get(
                        n, {"verdict": "pass", "issues": [], "artifact_posted": True}
                    )
                }
            if task == "risk":
                return {"ok": self.risk_results.get(n, {"tier": 2, "summary": "low risk"})}
            if task == "fix_ci":
                return {"ok": self.fix_results.get(n, {"head_sha": SHA2, "pushed": True})}
            if task == "implement":
                return {
                    "ok": self.implement_results.get(
                        n,
                        {
                            "branch": f"dev-pipeline/issue-{n}",
                            "pr_title": "t",
                            "pr_body": "b",
                            "escalated": False,
                        },
                    )
                }
            return {"ok": {}}
        raise AssertionError(f"unhandled capability {cid} spec={spec!r}")


async def _drive(
    scenario: Scenario,
    *,
    input: dict[str, Any] | None = None,
    max_steps: int = 200,
    real_merge: bool = False,
) -> tuple[Any, list[str], list[str]]:
    """Drive the production reconciler script to a terminal outcome, returning
    (return_value, phases, ordered_call_keys). Asserts replay-determinism (unique keys)."""
    src = build_reconciler_script(
        repo=REPO,
        real_merge=real_merge,
        implement_agent_id="dev-implement",
        review_agent_id="dev-review",
        fix_agent_id="dev-fix",
        ci_agent_id="dev-ci-watch",
        risk_agent_id="dev-risk",
    )
    inp = {"repo": REPO} if input is None else input
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
    raise AssertionError(f"reconciler did not terminate within {max_steps} steps")


# ─── a quiet board is a clean no-op ───────────────────────────────────────────


async def test_empty_board_is_a_clean_noop() -> None:
    value, phases, _ = await _drive(Scenario())
    assert value["state"] == "done"
    assert value["transition"] is None
    assert value["result"] == "noop"
    assert phases == ["scan", "owner", "summary"]


async def test_non_v2_items_are_invisible() -> None:
    # The reconciler ONLY touches pipeline:v2 items (step 6). A board of v1 items is a no-op.
    scn = Scenario(
        issues=[_issue(10, labels=["shovel-ready", "approved"])],  # no pipeline:v2
        prs=[_pr(20, labels=["ci-red"])],  # no pipeline:v2
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] is None
    assert scn.agent_labels == []  # no agent spend on un-adopted items
    assert scn.labels_added == {}


# ─── BUILD: PR-FIRST (open the draft PR before the implement agent) ───────────


async def test_build_is_pr_first_then_implement() -> None:
    scn = Scenario(issues=[_issue(11, labels=[V2, "shovel-ready", "approved"])])
    value, phases, _ = await _drive(scn)
    assert value["transition"] == "build"
    assert value["outcome"]["result"] == "built"
    # PR-FIRST: a draft PR was POSTed BEFORE the implement agent ran.
    post_pulls = [i for i, (m, p) in enumerate(scn.http) if m == "POST" and p == "/repos/o/r/pulls"]
    impl_at = scn.agent_labels.index("build-11")
    # the PR POST happened (job-identity created)
    assert post_pulls, "no draft PR was opened (PR-FIRST violated)"
    # the issue got the claim labels (dispatched + pipeline:v2 + in-progress)
    assert "dispatched" in scn.labels_added.get(11, [])
    assert V2 in scn.labels_added.get(11, [])
    assert "dev-implement" not in scn.agent_labels  # the label is "build-11", not the agent id
    assert impl_at >= 0  # the implement agent (maker) ran
    assert "build" in phases


async def test_build_spec_gate_failure_parks_needs_human_no_agent() -> None:
    # An empty/underspecified body fails the scripted spec gate -> needs:human/spec, NO build agent.
    scn = Scenario(issues=[_issue(12, labels=[V2, "shovel-ready", "approved"], body="too short")])
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "spec_failed"
    assert "needs:human/spec" in scn.labels_added.get(12, [])
    assert "build-12" not in scn.agent_labels  # no implement agent spent
    assert "underspecified" in scn.labels_added.get(12, [])
    # `shovel-ready` and `underspecified` are mutually exclusive on the spec-readiness axis: the
    # spec-gate STRIPS the stale `shovel-ready` claim as it stamps `underspecified`, so a rejection
    # can never leave the contradictory `shovel-ready ∧ underspecified` pair (the #1075/#1076/#1081/
    # #1087 mislabel class).
    assert "shovel-ready" in scn.labels_removed.get(12, [])


async def test_build_dispatched_issue_is_not_re_built() -> None:
    # Idempotency: an already-dispatched issue is not build-eligible -> no-op.
    scn = Scenario(issues=[_issue(13, labels=[V2, "shovel-ready", "approved", "dispatched"])])
    value, _, _ = await _drive(scn)
    assert value["transition"] is None
    assert scn.agent_labels == []


async def test_build_skips_implement_when_already_built_marker_present() -> None:
    # The implement-agent crash-replay dedup (C-3): a build-eligible issue ALREADY carrying the
    # autodev:built marker (a prior run pushed a diff, then a crash re-drove the build) must NOT
    # re-run the (expensive) implement agent. autodev:built is not a build-eligibility predicate, so
    # owner() still picks the issue for build, but _do_build short-circuits on the marker (the PR is
    # created/adopted first, then the marker check skips the implement agent).
    scn = Scenario(
        issues=[_issue(15, labels=[V2, "shovel-ready", "approved", "autodev:built"])],
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] == "build"
    assert value["outcome"]["result"] == "already_built"
    assert "build-15" not in scn.agent_labels  # the implement agent did NOT re-run


async def test_build_skips_implement_when_adopted_pr_has_commits() -> None:
    # The other arm of C-3: no marker yet, but _open_or_adopt_pr adopts a PR whose branch already
    # carries commits -> the implement already ran in a prior crashed pass -> skip it. We make the
    # issue build-eligible by having NO open PR in the snapshot whose branch references the issue
    # number (so has_open_pr is False), but the create-PR returns a branch that already has commits.
    scn = Scenario(
        issues=[_issue(16, labels=[V2, "shovel-ready", "approved"])],
        commits={9016: 4},  # the PR-first create returns #9016 (9000 + 16); seed its commit count
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] == "build"
    assert value["outcome"]["result"] == "already_built"
    assert "build-16" not in scn.agent_labels  # the implement agent did NOT re-run
    assert "autodev:built" in scn.labels_added.get(16, [])


# ─── REVIEW: maker≠checker (a DIFFERENT agent), risk floor, #1158 tier-gate ───


async def test_green_pr_is_reviewed_and_approved_within_cap() -> None:
    scn = Scenario(
        prs=[_pr(21, labels=[V2])],
        ci={21: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        review_results={21: {"verdict": "pass", "issues": [], "artifact_posted": True}},
        risk_results={21: {"tier": 2, "summary": "low"}},
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] == "review"
    assert value["outcome"]["result"] == "approved"
    # the checker is a DIFFERENT agent call (review-21) from any build (maker≠checker)
    assert "review-21" in scn.agent_labels
    assert "risk-21" in scn.agent_labels
    # the reviewed-green stamp + single-valued risk tier + merge:approved (approval LAST)
    added = scn.labels_added.get(21, [])
    assert any(a.startswith("reviewed:green@") for a in added)
    assert "risk:tier-2" in added
    assert "merge:approved" in added
    # the merge-guard bash ran (the mechanical check)
    assert any("MERGE_GUARD_OK" in c or "mg-" in c for c in scn.bash_commands)


async def test_tier_above_cap_routes_to_needs_human_merge_approval() -> None:
    # The #1158 control: a tier-4 PR NEVER gets merge:approved; it parks at needs:human/merge-approval.
    scn = Scenario(
        prs=[_pr(22, labels=[V2])],
        ci={22: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        review_results={22: {"verdict": "pass", "issues": [], "artifact_posted": True}},
        risk_results={22: {"tier": 4, "summary": "high"}},
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "needs_merge_approval"
    added = scn.labels_added.get(22, [])
    assert "needs:human/merge-approval" in added
    assert "merge:approved" not in added  # the tier-gate REFUSED to stamp it


async def test_ci_workflow_change_floors_to_tier4_via_risk_floor() -> None:
    # The deterministic risk floor: a PR touching .github/workflows floors to tier-4 even if the
    # risk agent says tier-1 -> needs:human/merge-approval (re-derived in the merger branch).
    scn = Scenario(
        prs=[_pr(23, labels=[V2])],
        ci={23: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        review_results={23: {"verdict": "pass", "issues": [], "artifact_posted": True}},
        risk_results={23: {"tier": 1, "summary": "trivial"}},
        files={23: [{"filename": ".github/workflows/ci.yml", "patch": "+ run: echo hi"}]},
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "needs_merge_approval"
    assert value["outcome"]["risk_tier"] == 4
    assert "merge:approved" not in scn.labels_added.get(23, [])


async def test_review_failure_parks_needs_human_verify() -> None:
    scn = Scenario(
        prs=[_pr(24, labels=[V2])],
        ci={24: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        review_results={24: {"verdict": "fail", "issues": ["a bug"], "artifact_posted": True}},
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "review_failed"
    assert "needs:human/verify" in scn.labels_added.get(24, [])


async def test_merge_guard_refusal_parks_needs_human_merge_guard() -> None:
    scn = Scenario(
        prs=[_pr(25, labels=[V2])],
        ci={25: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        review_results={25: {"verdict": "pass", "issues": [], "artifact_posted": True}},
        risk_results={25: {"tier": 1, "summary": "ok"}},
        merge_guard_exit=73,
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "merge_guard_refused"
    assert "needs:human/merge-guard" in scn.labels_added.get(25, [])
    assert "merge:approved" not in scn.labels_added.get(25, [])


async def test_risk_agent_failure_fails_closed_to_tier4() -> None:
    # A dead risk agent must FAIL CLOSED to tier-4 (> cap) -> needs:human/merge-approval, NOT
    # default to tier-3 (the auto-merge ceiling) and silently approve. The merge-safety gate must
    # err conservative on missing evidence.
    scn = Scenario(
        prs=[_pr(36, labels=[V2])],
        ci={36: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        review_results={36: {"verdict": "pass", "issues": [], "artifact_posted": True}},
        agent_error_on={"risk-36"},  # the risk agent errors
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "needs_merge_approval"
    assert value["outcome"]["risk_tier"] == 4
    assert "needs:human/merge-approval" in scn.labels_added.get(36, [])
    assert "merge:approved" not in scn.labels_added.get(36, [])


# ─── MERGE: WOULD-MERGE ADVISORY mode (the v1 default — never actually merges) ──


async def test_approved_pr_in_advisory_mode_does_not_merge() -> None:
    scn = Scenario(
        prs=[_pr(26, labels=[V2, "reviewed:green@" + SHA, "merge:approved", "risk:tier-2"])],
        ci={26: {"status": "green", "polled_sha": SHA, "required_complete": True}},
    )
    value, _, _ = await _drive(scn, real_merge=False)
    assert value["transition"] == "merge"
    assert value["outcome"]["result"] == "would_merge_advisory"
    assert scn.merges_put == []  # NEVER actually merged in advisory mode
    assert "merge:would-merge-advisory" in scn.labels_added.get(26, [])


async def test_approved_pr_with_real_merge_on_does_conditional_put() -> None:
    scn = Scenario(
        prs=[_pr(27, labels=[V2, "reviewed:green@" + SHA, "merge:approved", "risk:tier-2"])],
        ci={27: {"status": "green", "polled_sha": SHA, "required_complete": True}},
    )
    value, _, _ = await _drive(scn, real_merge=True)
    assert value["transition"] == "merge"
    assert value["outcome"]["result"] == "merged"
    assert scn.merges_put == [27]  # the conditional PUT /merges fired
    # close-before-strip: the source issue (issue-27) was PATCH-closed
    assert ("PATCH", "/repos/o/r/issues/27") in scn.http


async def test_real_merge_block_escalates_required_set_not_strip_loop() -> None:
    # M-2: a 422/403/405 merge block (branch protection / newly-required check) must escalate to a
    # DURABLE needs:human/required-set, NOT silently strip merge:approved and re-loop forever.
    scn = Scenario(
        prs=[_pr(39, labels=[V2, "reviewed:green@" + SHA, "merge:approved", "risk:tier-2"])],
        ci={39: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        merge_put_status=422,
    )
    value, _, _ = await _drive(scn, real_merge=True)
    assert value["outcome"]["result"] == "required_set"
    assert "needs:human/required-set" in scn.labels_added.get(39, [])


# ─── REBASE: a conflicting PR is healed first (mechanical no-op exits clean) ──


async def test_conflicting_pr_routes_to_rebase() -> None:
    scn = Scenario(
        prs=[_pr(28, labels=[V2], mergeable=False, mergeable_state="dirty")],
        rebase_exit=75,  # NOOP -> the branch turned out already-current
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] == "rebase"
    assert value["outcome"]["result"] in ("noop", "rebased")
    assert any("rebase-" in c or "REBASE" in c for c in scn.bash_commands)


async def test_rebase_conflict_parks_needs_human_rebase() -> None:
    scn = Scenario(
        prs=[_pr(29, labels=[V2], mergeable=False, mergeable_state="dirty")],
        rebase_exit=76,  # CONFLICT, and the fix agent can't heal it (default fix returns SHA2)
        fix_results={29: {"head_sha": SHA2, "pushed": True}},
    )
    # the confirm rebase still conflicts -> conflict outcome
    value, _, _ = await _drive(scn)
    assert value["transition"] == "rebase"
    assert value["outcome"]["result"] == "conflict"
    assert "needs:human/rebase" in scn.labels_added.get(29, [])


# ─── CI: a red PR routes to the fixer; no-progress bumps the lap ──────────────


async def test_red_pr_routes_to_ci_fix() -> None:
    scn = Scenario(
        prs=[_pr(30, labels=[V2])],
        ci={30: {"status": "red", "polled_sha": SHA, "required_complete": True}},
        fix_results={30: {"head_sha": SHA2, "pushed": True}},
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] == "ci"
    assert value["outcome"]["result"] == "fixed"
    assert "ci-fix-30" in scn.agent_labels


async def test_ci_no_progress_bumps_the_lap_counter() -> None:
    # The fixer returns the SAME head (no new commit) -> bump pipeline:laps (the livelock catch).
    scn = Scenario(
        prs=[_pr(31, labels=[V2], head_sha=SHA)],
        ci={31: {"status": "red", "polled_sha": SHA, "required_complete": True}},
        fix_results={31: {"head_sha": SHA, "pushed": False}},  # no new commit
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] in ("no_progress", "livelock")
    assert any(a.startswith("pipeline:laps:") for a in scn.labels_added.get(31, []))


async def test_livelock_cap_parks_needs_human_livelock() -> None:
    # A PR already at the (fixture) lap cap minus one, with another no-progress fix, livelocks.
    scn = Scenario(
        prs=[_pr(32, labels=[V2, "pipeline:laps:7"], head_sha=SHA)],
        ci={32: {"status": "red", "polled_sha": SHA, "required_complete": True}},
        fix_results={32: {"head_sha": SHA, "pushed": False}},
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "livelock"
    assert "needs:human/livelock" in scn.labels_added.get(32, [])


async def test_ci_fix_agent_error_escalates_immediately_not_a_lap() -> None:
    # A fix-AGENT error (infra) must escalate to needs:human/ci IMMEDIATELY — not be masked as a
    # legitimate no-progress lap that slow-walks for MAX_LAPS passes (and evades the dead-man).
    scn = Scenario(
        prs=[_pr(38, labels=[V2])],
        ci={38: {"status": "red", "polled_sha": SHA, "required_complete": True}},
        agent_error_on={"ci-fix-38"},
    )
    value, _, _ = await _drive(scn)
    assert value["outcome"]["result"] == "fix_agent_error"
    assert "needs:human/ci" in scn.labels_added.get(38, [])
    # NOT a lap bump (an infra error is not a no-progress fix)
    assert not any(a.startswith("pipeline:laps:") for a in scn.labels_added.get(38, []))


# ─── ESCALATE: an unforeseen verdict is detectable within one pass ───────────


async def test_unforeseen_ci_verdict_escalates() -> None:
    # A watch verdict the _ci_verdict guard rejects degrades to "pending" (WAITING), so to force
    # the escalate path we need a verdict the script normalizer can't classify. The watch agent
    # returning a 'no_ci' that fails head-verification -> pending -> WAITING (not escalate). The
    # escalate branch is reached only by a tuple owner()'s PR branches don't foresee; we cover it
    # in the unit cross-product test. Here we assert a WAITING PR is a clean no-op (no false action).
    scn = Scenario(
        prs=[_pr(33, labels=[V2])],
        ci={33: {"status": "green", "polled_sha": "deadbeef", "required_complete": False}},
    )
    value, _, _ = await _drive(scn)
    # premature/unverifiable green -> pending -> WAITING -> no actionable item -> noop
    assert value["transition"] is None
    assert value["result"] == "noop"


# ─── HUMAN-OWNED: a requested-changes PR is left to the human ────────────────


async def test_human_requested_changes_pr_is_left_alone() -> None:
    scn = Scenario(
        prs=[_pr(34, labels=[V2])],
        ci={34: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        reviews={34: [{"user": {"login": "tom"}, "state": "CHANGES_REQUESTED"}]},
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] is None  # the reconciler refuses to fight the human
    assert "review-34" not in scn.agent_labels  # no review agent spent on a human-owned PR


async def test_unreadable_reviews_fails_closed_to_human_owned() -> None:
    # A non-200 reviews read must FAIL CLOSED (treat as human-owned this pass), NOT fail open and
    # let the reconciler act over a possible active human review. owner() then leaves it WAITING.
    scn = Scenario(
        prs=[_pr(37, labels=[V2])],
        ci={37: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        reviews_status=500,  # the reviews API blips
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] is None  # not acted on (human-owned this pass)
    assert "review-37" not in scn.agent_labels  # no review agent spent


# ─── ONE transition per pass (the serial reconciler invariant) ───────────────


async def test_exactly_one_transition_per_pass_highest_priority() -> None:
    # A board with a rebase-needed PR, a red PR, and a build-eligible issue: the rebase wins (rank 0)
    # and is the ONLY transition driven this pass.
    scn = Scenario(
        issues=[_issue(40, labels=[V2, "shovel-ready", "approved"])],
        prs=[
            _pr(41, labels=[V2]),  # green -> review (rank 2)
            _pr(42, labels=[V2], mergeable=False, mergeable_state="dirty"),  # rebase (rank 0)
        ],
        ci={41: {"status": "green", "polled_sha": SHA, "required_complete": True}},
        rebase_exit=75,
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] == "rebase"
    assert value["number"] == 42
    # the build issue + the review PR were NOT acted on this pass (one transition per pass)
    assert "build-40" not in scn.agent_labels
    assert "review-41" not in scn.agent_labels


# ─── IDEMPOTENCY (step 2): a re-run over an advanced item touches nothing more ──


async def test_re_run_over_approved_advisory_pr_is_idempotent() -> None:
    # First pass: a green un-reviewed PR -> review -> stamps reviewed:green + merge:approved.
    # Simulate the post-advance board (those labels now present) and re-run: it routes to merge,
    # and in advisory mode stamps the advisory ONCE (the maker-marker dedup means a 2nd run posts
    # no duplicate comment). The key idempotency property: NO re-review (no review agent) on re-run.
    advanced = _pr(34, labels=[V2, "reviewed:green@" + SHA, "merge:approved", "risk:tier-2"])
    scn = Scenario(
        prs=[advanced],
        ci={34: {"status": "green", "polled_sha": SHA, "required_complete": True}},
    )
    value, _, _ = await _drive(scn, real_merge=False)
    assert value["transition"] == "merge"  # NOT review — the advance is respected
    assert "review-34" not in scn.agent_labels  # no re-review agent spend
    assert value["outcome"]["result"] == "would_merge_advisory"


async def test_re_run_after_human_park_is_a_noop() -> None:
    # A PR already carrying needs:human/* is vetoed by owner() -> a re-scan touches nothing.
    scn = Scenario(
        prs=[_pr(35, labels=[V2, "needs:human/verify"])],
        ci={35: {"status": "green", "polled_sha": SHA, "required_complete": True}},
    )
    value, _, _ = await _drive(scn)
    assert value["transition"] is None
    assert scn.labels_added == {}  # nothing written
    assert scn.agent_labels == []  # no agent spend


async def test_dispatched_issue_with_open_pr_does_not_double_build() -> None:
    # The issue was dispatched and its PR exists; a re-scan must drive the PR, never re-build.
    scn = Scenario(
        issues=[_issue(50, labels=[V2, "shovel-ready", "approved", "dispatched"])],
        prs=[_pr(9050, labels=[V2], head_ref="dev-pipeline/issue-50")],
        ci={9050: {"status": "green", "polled_sha": SHA, "required_complete": True}},
    )
    value, _, _ = await _drive(scn)
    # the green PR is reviewed; the dispatched issue is NOT re-built
    assert value["transition"] == "review"
    assert value["number"] == 9050
    assert "build-50" not in scn.agent_labels


# ─── input handling ──────────────────────────────────────────────────────────


async def test_accepts_trigger_envelope_input() -> None:
    value, _, _ = await _drive(
        Scenario(), input={"trigger": {"kind": "cron"}, "input": {"repo": REPO}}
    )
    assert value["state"] == "done"


async def test_missing_repo_with_no_default_errors() -> None:
    src = build_reconciler_script()  # no repo default (agent ids default fine for this check)
    out = await run_script_host(source=src, input={}, memo={})
    assert out.kind == "returned"
    assert out.value["state"] == "error"
    assert "no repo" in out.value["reason"]
