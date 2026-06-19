"""Dev-pipeline POST-MERGE checker reference workflow fixture (task #76; aios#49/#111).

Drives the production ``build_post_merge_checker_script`` directly against the real script host
(no DB, no live model) with simulated ``http_request`` / ``list_runs`` resolutions, walking the
stateless config → scan → provenance-witness → master-ci → verify → summary machine. Because the
host re-runs from the start each wake (replaying the whole growing memo), these tests also assert
the load-bearing durable property: **replay is deterministic** (every call_key is stable).

It proves task #76's acceptance — the post-merge twin of ``owner()``'s pre-merge gate verifies,
from durable off-the-run GitHub state, that every recently-merged ``pipeline:v2`` PR:

  * (a) closed its source issue — an OPEN source issue for a merged PR → ``needs:human/issue-open``;
  * (b) did not leave master red — a RED master post-merge → ``needs:human/master-red`` (#1188);
  * (c) merged THROUGH the maker≠checker gate — a ``pipeline:v2`` PR merged WITHOUT the
    ``reviewed:green@<sha>`` + in-cap ``risk:tier-N`` stamp → ``needs:human/bad-provenance``.

Plus: a CLEAN merge passes and is stamped ``post-merge:checked@<sha>`` (verified ONCE — a re-sweep
skips it, idempotent); a violation is NEVER silently passed; a truncated/degraded read fails LOUD
cannot-determine, never a silent "no regressions"; and the structured per-sweep summary (verdict +
found + checked) is the primary output.

Mirrors the host-subprocess style of ``test_wf_gate_reaper_fixture.py`` /
``test_wf_dev_pipeline_reconciler_fixture.py``; needs no Postgres.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from aios.models.agents import HttpServerSpec
from aios.workflows.dev_pipeline_post_merge import (
    REQUIRED_TOOLS,
    build_post_merge_checker_fixture_script,
    build_post_merge_checker_workflow_create,
)
from aios.workflows.host_launcher import run_script_host

pytestmark = pytest.mark.integration

REPO = "o/r"
WF = "wf_dev_pipeline"
V2 = "pipeline:v2"
SHA = "a" * 40  # a merged PR's merge_commit_sha
MASTER_SHA = "f" * 40  # the resolved BASE_BRANCH head


def _merged_pr(
    number: int,
    *,
    labels: list[str] | None = None,
    head_ref: str | None = None,
    merge_commit_sha: str = SHA,
    merged: bool = True,
) -> dict[str, Any]:
    """A CLOSED+MERGED PR row, as the closed-PR list returns it (carries merged_at +
    merge_commit_sha + the head ref the source-issue number is derived from)."""
    return {
        "number": number,
        "title": f"PR {number}",
        "state": "closed",
        "merged_at": "2026-06-18T12:00:00+00:00" if merged else None,
        "merge_commit_sha": merge_commit_sha,
        "head": {"ref": head_ref or f"dev-pipeline/issue-{number}", "sha": SHA},
        "labels": [{"name": n} for n in (labels or [])],
    }


def _gate_labels(reviewed_sha: str = SHA, tier: int = 2) -> list[str]:
    """The reconciler's durable gate provenance: pipeline:v2 + reviewed:green@<sha> + risk:tier-N."""
    return [V2, f"reviewed:green@{reviewed_sha}", f"risk:tier-{tier}"]


class Scenario:
    """A deterministic responder + a record of what the checker asked for. Models the GitHub REST
    surface the checker reads/writes (closed-PR list, the source issue's state, master's CI
    surfaces, comment threads, the escalation POSTs) + the list_runs provenance witness. Keyed on
    the capability so replay is a pure function of the call_key."""

    def __init__(
        self,
        *,
        merged_prs: list[dict[str, Any]] | None = None,
        issue_states: dict[int, str] | None = None,
        master_check_runs: list[dict[str, Any]] | None = None,
        master_combined: dict[str, Any] | None = None,
        completed_run_issues: list[int] | None = None,
        prs_list_status: int = 200,
        issue_read_status: int = 200,
        list_runs_error: bool = False,
        existing_comments: dict[int, list[str]] | None = None,
    ) -> None:
        self.merged_prs = merged_prs or []
        # issue_number -> "open"|"closed" (default closed: the happy path)
        self.issue_states = issue_states or {}
        # master CI surfaces (default: a single passing check-run => green master)
        self.master_check_runs = (
            master_check_runs
            if master_check_runs is not None
            else [{"status": "completed", "conclusion": "success"}]
        )
        self.master_combined = (
            master_combined
            if master_combined is not None
            else {"total_count": 0, "state": "success"}
        )
        self.completed_run_issues = completed_run_issues or []
        self.prs_list_status = prs_list_status
        self.issue_read_status = issue_read_status
        self.list_runs_error = list_runs_error
        self.existing_comments = existing_comments or {}
        # observability
        self.http: list[tuple[str, str]] = []
        self.labels_added: dict[int, list[str]] = {}
        self.comments_posted: dict[int, list[str]] = {}
        self.list_runs_calls: list[dict[str, Any]] = []

    def _http(self, args: dict[str, Any]) -> dict[str, Any]:
        method, path = args["method"], args["path"]
        self.http.append((method, path))
        # BEHAVIORAL maker≠checker-across-the-boundary guard: the checker READS + ESCALATES only.
        # It must NEVER emit a mutating verb — no PUT (merge), no PATCH (close an issue), no DELETE
        # (unlabel). The deploy surface (GET·POST-only) rejects these in prod, but assert it here
        # too so a future _BODY edit that regresses into _unlabel/merge/close fails LOUD in CI
        # rather than slipping past the catch-all into the surface-rejection failure mode.
        assert method in ("GET", "POST"), (
            f"post-merge checker emitted a MUTATING verb {method} {path} — it must only read + "
            "escalate (it can never mutate the merge it judges)"
        )
        clean = path.split("?", 1)[0]

        # the closed/merged PR list
        if method == "GET" and clean == "/repos/o/r/pulls":
            if self.prs_list_status != 200:
                return {"status": self.prs_list_status, "body": "{}"}
            return {"status": 200, "headers": {}, "body": json.dumps(self.merged_prs)}

        # the source issue's state (check a)
        mi = re.match(r"^/repos/o/r/issues/(\d+)$", clean)
        if method == "GET" and mi:
            if self.issue_read_status != 200:
                return {"status": self.issue_read_status, "body": "{}"}
            n = int(mi.group(1))
            state = self.issue_states.get(n, "closed")
            return {"status": 200, "body": json.dumps({"number": n, "state": state})}

        # master's CI surfaces (check b) — resolve BASE_BRANCH, then check-runs + combined status
        if method == "GET" and re.match(r"^/repos/o/r/commits/[^/]+$", clean):
            # _resolve_sha1: GET /commits/{ref} -> canonical SHA-1
            return {"status": 200, "body": json.dumps({"sha": MASTER_SHA})}
        if method == "GET" and clean.endswith("/check-runs"):
            return {"status": 200, "body": json.dumps({"check_runs": self.master_check_runs})}
        if method == "GET" and re.match(r"^/repos/o/r/commits/[^/]+/status$", clean):
            return {"status": 200, "body": json.dumps(self.master_combined)}

        # comment thread reads (post_markered_comment fetches first)
        mc = re.match(r"^/repos/o/r/issues/(\d+)/comments$", clean)
        if method == "GET" and mc:
            n = int(mc.group(1))
            return {
                "status": 200,
                "body": json.dumps([{"body": b} for b in self.existing_comments.get(n, [])]),
            }
        if method == "POST" and mc:
            n = int(mc.group(1))
            raw = args.get("body")
            if isinstance(raw, str):
                self.comments_posted.setdefault(n, []).append(json.loads(raw).get("body", ""))
            return {"status": 201, "body": "{}"}

        # label POSTs (the escalation + the checked marker)
        ml = re.match(r"^/repos/o/r/issues/(\d+)/labels$", clean)
        if method == "POST" and ml:
            n = int(ml.group(1))
            raw = args.get("body")
            if isinstance(raw, str):
                self.labels_added.setdefault(n, []).extend(json.loads(raw).get("labels", []))
            return {"status": 200, "body": "[]"}

        return {"status": 200, "body": "{}"}

    def _list_runs(self, args: dict[str, Any]) -> dict[str, Any]:
        self.list_runs_calls.append(args)
        if self.list_runs_error:
            return {"error": "boom"}
        runs = [
            {"input": {"repo": REPO, "issue_number": n}, "status": "completed"}
            for n in self.completed_run_issues
        ]
        return {"runs": runs}

    def outcome(self, cap: Any) -> dict[str, Any]:
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool":
            name = spec["tool_name"]
            if name == "http_request":
                return {"ok": self._http(spec["input"])}
            if name == "list_runs":
                return {"ok": self._list_runs(spec["input"])}
            raise AssertionError(f"checker called an unexpected tool: {name}")
        raise AssertionError(f"checker must not emit capability {cid} (no agent/gate); {spec!r}")


async def _drive(
    scenario: Scenario,
    *,
    input: dict[str, Any] | None = None,
    max_steps: int = 200,
    **build_kwargs: Any,
) -> Any:
    """Drive the production post-merge checker to a terminal outcome; assert replay-determinism."""
    src = build_post_merge_checker_fixture_script(repo=REPO, **build_kwargs)
    inp = {"repo": REPO} if input is None else input
    memo: dict[str, Any] = {}
    keys: list[str] = []
    for _ in range(max_steps):
        out = await run_script_host(source=src, input=inp, memo=memo)
        if out.kind == "returned":
            assert len(keys) == len(set(keys)), "replay produced a duplicate call_key"
            return out.value
        assert out.kind == "suspended", (out.kind, out.error_repr, out.error_traceback)
        assert len(out.emitted) == 1, [(e.capability_id, e.spec) for e in out.emitted]
        cap = out.emitted[0]
        keys.append(cap.call_key)
        memo[cap.call_key] = scenario.outcome(cap)
    raise AssertionError(f"checker did not terminate within {max_steps} steps")


# ─── a board with no merged pipeline PRs is a clean ok no-op ───────────────────


async def test_no_merged_v2_prs_is_a_clean_ok() -> None:
    value = await _drive(Scenario())
    assert value["verdict"] == "ok"
    assert value["scanned"] == 0
    assert value["found"] == []


async def test_non_v2_merges_are_invisible() -> None:
    # Only pipeline:v2 merges are in scope — a merged PR without the routing label is ignored.
    scn = Scenario(merged_prs=[_merged_pr(50, labels=["some-other-label"])])
    value = await _drive(scn)
    assert value["scanned"] == 0
    assert value["found"] == []
    assert scn.labels_added == {}  # nothing escalated


async def test_closed_but_unmerged_pr_is_ignored() -> None:
    # A closed-but-NOT-merged PR merged nothing → nothing to verify.
    scn = Scenario(merged_prs=[_merged_pr(51, labels=_gate_labels(), merged=False)])
    value = await _drive(scn)
    assert value["scanned"] == 0


# ─── (1) THE CLEAN MERGE PASSES and is checked exactly once ───────────────────


async def test_clean_merge_passes_and_is_marked_checked() -> None:
    # A merged v2 PR: source issue closed, master green, full gate provenance → ok, no escalation,
    # and stamped post-merge:checked@<merge_sha> (verified once).
    scn = Scenario(
        merged_prs=[_merged_pr(60, labels=_gate_labels())],
        issue_states={60: "closed"},
        completed_run_issues=[60],
    )
    value = await _drive(scn)
    assert value["verdict"] == "ok"
    assert value["scanned"] == 1
    assert value["found"] == []
    assert 60 in value["checked"]
    assert f"post-merge:checked@{SHA}" in scn.labels_added.get(60, [])
    # NO needs:human/* escalation on a clean merge
    assert not any(lbl.startswith("needs:human/") for lbl in scn.labels_added.get(60, []))


async def test_already_checked_merge_is_skipped_idempotent() -> None:
    # A merged PR already carrying post-merge:checked@<sha> for THIS merge sha is skipped — a
    # re-sweep is a no-op (verified once; idempotent).
    scn = Scenario(
        merged_prs=[_merged_pr(61, labels=[*_gate_labels(), f"post-merge:checked@{SHA}"])],
        issue_states={61: "closed"},
    )
    value = await _drive(scn)
    assert value["verdict"] == "ok"
    assert value["checked"] == []  # not re-checked
    assert scn.labels_added == {}  # nothing re-stamped
    # no source-issue read spent on an already-checked merge
    assert ("GET", "/repos/o/r/issues/61") not in scn.http


# ─── (2) ISSUE STAYED OPEN → needs:human/issue-open ───────────────────────────


async def test_merged_pr_with_open_source_issue_escalates() -> None:
    scn = Scenario(
        merged_prs=[_merged_pr(62, labels=_gate_labels())],
        issue_states={62: "open"},  # the reconciler's best-effort close did not stick
        completed_run_issues=[62],
    )
    value = await _drive(scn)
    assert value["verdict"] == "regression-found"
    assert any(f["class"] == "issue-open" for f in value["found"])
    assert "needs:human/issue-open" in scn.labels_added.get(62, [])
    assert "autodev:post-merge-regression" in scn.labels_added.get(62, [])
    # a markered escalation comment was posted (never silent)
    assert any("source issue stayed OPEN" in c for c in scn.comments_posted.get(62, []))
    # a violated merge is NOT marked checked (it stays pull-able for the seat)
    assert f"post-merge:checked@{SHA}" not in scn.labels_added.get(62, [])
    assert 62 not in value["checked"]


# ─── (3) MASTER WENT RED → needs:human/master-red (#1188) ─────────────────────


async def test_merge_that_leaves_master_red_escalates() -> None:
    scn = Scenario(
        merged_prs=[_merged_pr(63, labels=_gate_labels())],
        issue_states={63: "closed"},
        master_check_runs=[{"status": "completed", "conclusion": "failure"}],  # master is RED
        master_combined={"total_count": 0, "state": "success"},
        completed_run_issues=[63],
    )
    value = await _drive(scn)
    assert value["verdict"] == "regression-found"
    assert value["master_verdict"] == "red"
    assert any(f["class"] == "master-red" for f in value["found"])
    assert "needs:human/master-red" in scn.labels_added.get(63, [])
    assert any("master is RED" in c for c in scn.comments_posted.get(63, []))
    # NOT marked checked while master is red
    assert 63 not in value["checked"]


async def test_indeterminate_master_ci_is_cannot_determine_not_ok() -> None:
    # Master CI still running (a non-terminal check-run) — the sweep cannot declare the merge
    # clean. Verdict is cannot-determine, NOT a silent ok, and the merge is NOT marked checked.
    scn = Scenario(
        merged_prs=[_merged_pr(64, labels=_gate_labels())],
        issue_states={64: "closed"},
        master_check_runs=[{"status": "in_progress", "conclusion": None}],  # not terminal
        master_combined={"total_count": 0, "state": "pending"},
    )
    value = await _drive(scn)
    assert value["verdict"] == "cannot-determine"
    assert value["master_verdict"] == "indeterminate"
    assert value["found"] == []  # no false regression
    assert 64 not in value["checked"]  # not declared clean while master CI unsettled


# ─── (4) BAD PROVENANCE → needs:human/bad-provenance ──────────────────────────


async def test_merged_v2_pr_without_review_stamp_is_bad_provenance() -> None:
    # A pipeline:v2 PR MERGED carrying NO reviewed:green@<sha> stamp — it bypassed the gate's
    # maker≠checker review the auto-merge authority is scoped to.
    scn = Scenario(
        merged_prs=[_merged_pr(65, labels=[V2, "risk:tier-2"])],  # no reviewed:green@
        issue_states={65: "closed"},
    )
    value = await _drive(scn)
    assert value["verdict"] == "regression-found"
    assert any(f["class"] == "bad-provenance" for f in value["found"])
    assert "needs:human/bad-provenance" in scn.labels_added.get(65, [])
    assert any("without the maker" in c or "bypassed" in c for c in scn.comments_posted.get(65, []))


async def test_merged_v2_pr_with_tier_above_cap_is_bad_provenance() -> None:
    # A pipeline:v2 PR merged carrying risk:tier-4 (ABOVE the auto-merge cap of 3) — the #1158
    # tier-gate should have routed it to a human, so a merge AT tier-4 is illegitimate provenance.
    scn = Scenario(
        merged_prs=[_merged_pr(66, labels=[V2, f"reviewed:green@{SHA}", "risk:tier-4"])],
        issue_states={66: "closed"},
    )
    value = await _drive(scn)
    assert value["verdict"] == "regression-found"
    assert any(f["class"] == "bad-provenance" for f in value["found"])
    assert "needs:human/bad-provenance" in scn.labels_added.get(66, [])


async def test_clean_provenance_with_run_witness_passes() -> None:
    # Full gate labels + a completed dev-pipeline run for the issue in the journal → clean.
    scn = Scenario(
        merged_prs=[_merged_pr(67, labels=_gate_labels(tier=3))],  # tier-3 == cap, in-cap
        issue_states={67: "closed"},
        completed_run_issues=[67],
    )
    value = await _drive(scn)
    assert value["verdict"] == "ok"
    assert value["found"] == []
    assert 67 in value["checked"]


async def test_require_run_provenance_flags_missing_run() -> None:
    # With REQUIRE_RUN_PROVENANCE on: gate labels present but NO completed run drove the issue →
    # bad-provenance (the strict posture; the gate labels may have been applied out-of-band).
    scn = Scenario(
        merged_prs=[_merged_pr(68, labels=_gate_labels())],
        issue_states={68: "closed"},
        completed_run_issues=[],  # no driving run in the journal
    )
    value = await _drive(scn, require_run_provenance=True, dev_pipeline_workflow_id=WF)
    assert value["verdict"] == "regression-found"
    assert any(f["class"] == "bad-provenance" for f in value["found"])


async def test_degraded_run_journal_never_manufactures_a_violation() -> None:
    # The run-journal read degrades (list_runs error) — even with REQUIRE_RUN_PROVENANCE on, a
    # degraded witness must NOT manufacture a false bad-provenance; the durable gate labels stand.
    scn = Scenario(
        merged_prs=[_merged_pr(69, labels=_gate_labels())],
        issue_states={69: "closed"},
        list_runs_error=True,
    )
    value = await _drive(scn, require_run_provenance=True, dev_pipeline_workflow_id=WF)
    assert value["verdict"] == "ok"
    assert value["witness_ok"] is False
    assert value["found"] == []


# ─── MULTIPLE violations on one merge are all surfaced ────────────────────────


async def test_multiple_violations_on_one_merge_all_escalate() -> None:
    # Open issue AND bad provenance on the same merged PR → both classes found + both labels.
    scn = Scenario(
        merged_prs=[_merged_pr(70, labels=[V2])],  # no review stamp, no tier
        issue_states={70: "open"},
    )
    value = await _drive(scn)
    assert value["verdict"] == "regression-found"
    classes = {f["class"] for f in value["found"]}
    assert "issue-open" in classes
    assert "bad-provenance" in classes
    added = scn.labels_added.get(70, [])
    assert "needs:human/issue-open" in added
    assert "needs:human/bad-provenance" in added


# ─── FAIL-LOUD: a degraded merged-PR read is cannot-determine, never silent ok ─


async def test_degraded_merged_pr_read_is_cannot_determine() -> None:
    scn = Scenario(prs_list_status=500)  # the closed-PR list read blips
    value = await _drive(scn)
    assert value["verdict"] == "cannot-determine"
    assert value["scanned"] == 0
    assert "reason" in value


async def test_unreadable_source_issue_is_indeterminate_not_a_false_violation() -> None:
    # A non-2xx issue read must NOT read as 'issue open' (a false violation) — it is indeterminate
    # this sweep (re-checked next), and the merge is not declared clean either while a check is
    # unresolved (provenance still passes, master green, but the issue check abstained).
    scn = Scenario(
        merged_prs=[_merged_pr(71, labels=_gate_labels())],
        issue_read_status=404,
    )
    value = await _drive(scn)
    # the issue check abstained (no issue-open finding manufactured)
    assert not any(f["class"] == "issue-open" for f in value["found"])
    assert "needs:human/issue-open" not in scn.labels_added.get(71, [])


# ─── input handling ────────────────────────────────────────────────────────────


async def test_accepts_cron_trigger_envelope_input() -> None:
    value = await _drive(Scenario(), input={"trigger": {"source": "cron"}, "input": {"repo": REPO}})
    assert value["verdict"] == "ok"


async def test_missing_repo_with_no_default_is_cannot_determine() -> None:
    src = build_post_merge_checker_fixture_script()  # no repo default
    out = await run_script_host(source=src, input={}, memo={})
    assert out.kind == "returned"
    assert out.value["verdict"] == "cannot-determine"


async def test_string_scan_limit_is_coerced_not_a_crash() -> None:
    # A cron config can deliver scan_limit as a JSON STRING. It must coerce to an int (used in a
    # `len(runs) >= limit` comparison on the run-journal witness path) — a string would TypeError
    # mid-sweep. Pair a string limit WITH a wired workflow_id (the witness path that does the
    # comparison) so the regression would actually fire.
    scn = Scenario(
        merged_prs=[_merged_pr(80, labels=_gate_labels())],
        issue_states={80: "closed"},
        completed_run_issues=[80],
    )
    value = await _drive(
        scn,
        input={"repo": REPO, "scan_limit": "50", "dev_pipeline_workflow_id": WF},
    )
    assert value["verdict"] == "ok"
    assert value["scanned"] == 1


async def test_nonnumeric_scan_limit_falls_back_to_default() -> None:
    # A garbage scan_limit must fall back to the default, never crash the sweep.
    value = await _drive(Scenario(), input={"repo": REPO, "scan_limit": "not-a-number"})
    assert value["verdict"] == "ok"


# ─── deploy surface sanity (mirrors the unit assertions through the create) ───


def test_workflow_create_surface_is_read_and_escalate_only() -> None:
    wc = build_post_merge_checker_workflow_create(name="x")
    assert {t.type for t in REQUIRED_TOOLS} == {"http_request", "list_runs"}
    assert len(wc.http_servers) == 1
    server = wc.http_servers[0]
    assert isinstance(server, HttpServerSpec)
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert set(repos[0].methods or []) == {"GET", "POST"}
