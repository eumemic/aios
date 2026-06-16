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
import re
from typing import Any

import pytest

from aios.workflows.dev_pipeline import build_dev_pipeline_script
from aios.workflows.host_launcher import run_script_host

pytestmark = pytest.mark.integration

REPO = "o/r"
ISSUE = 5
BRANCH = "issue-5"
# The merge PUT returns the merge commit's GitHub-canonical SHA-1 (40-char hex). The
# post-merge master-CI watch threads THIS into the watch instead of re-resolving the branch
# name (issue #1178) — a SHA-256 clone would re-resolve `master` to a 64-char id GitHub's
# REST API rejects, hard-erroring the watch on a green master.
MERGE_SHA = "f823360f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"
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
        comments: list[str] | None = None,
        existing_pr: bool = False,
        implement_escalated: bool = False,
        implement_error: bool = False,
        review_results: list[dict[str, Any]] | None = None,
        review_error_on: str | None = None,
        ci_results: list[dict[str, Any]] | None = None,
        ci_error_on: str | None = None,
        risk_result: dict[str, Any] | None = None,
        merge_guard_exit: int = 0,
        merge_status: int = 200,
        pr_merged_on_confirm: bool = False,
        gate_results: dict[str, Any] | None = None,
        master_ci: str = "green",
        master_ci_error: bool = False,
        master_ci_results: list[dict[str, Any] | str] | None = None,
        merge_returns_sha: bool = True,
        commit_sha: str | None = MERGE_SHA,
        transient_5xx: dict[tuple[str, str], int] | None = None,
    ) -> None:
        self.body = body
        # The comment-thread the GET /issues/{n}/comments responder returns (list of body
        # strings → GitHub comment objects). None => no comments (empty array).
        self.comments = comments or []
        self.existing_pr = existing_pr
        self.implement_escalated = implement_escalated
        self.implement_error = implement_error
        self.review_results = review_results or [
            {"verdict": "pass", "issues": [], "artifact_posted": True}
        ]
        self.review_error_on = review_error_on  # an agent label that raises AgentError
        self.ci_results = ci_results or [{"status": "green", "detail": ""}]
        self.ci_error_on = ci_error_on  # an agent label that raises AgentError
        self.risk_result = (
            risk_result if risk_result is not None else {"tier": 1, "summary": "safe"}
        )
        self.merge_guard_exit = merge_guard_exit
        self.merge_status = merge_status
        self.pr_merged_on_confirm = pr_merged_on_confirm
        self.gate_results = gate_results or {}
        self.master_ci = master_ci
        self.master_ci_error = master_ci_error
        # Per-attempt post-merge master-CI verdicts, indexed by the watch's per-retry label
        # ("master-ci-{i}"). Each item is either a verdict dict ({"status": ..., "detail": ...})
        # or the string "error" (the watch agent raises AgentError that attempt). When set, this
        # overrides master_ci/master_ci_error. The last item is reused for any later attempt.
        self.master_ci_results = master_ci_results
        # issue #1178: whether the merge PUT body carries the merge commit's SHA-1. When False,
        # the workflow falls back to resolving BASE_BRANCH via GET /repos/{repo}/commits/master.
        self.merge_returns_sha = merge_returns_sha
        # The SHA-1 the GET /commits/master canonicalisation returns. Set to None to model a ref
        # that can't be resolved to a SHA-1 (the watch must then NOT be dispatched at all).
        self.commit_sha = commit_sha
        # {(method, path): N} — return a transient 5xx for the FIRST N attempts of that call,
        # then the real response. Counts are mutated as attempts arrive (replay-stable because
        # each retry is a distinct call_key, so a replay re-asks each attempt exactly once).
        self.transient_5xx = dict(transient_5xx or {})
        self.tasks: list[str] = []
        self.agent_inputs: list[dict[str, Any]] = []
        self.gates: list[dict[str, Any]] = []
        self.http: list[tuple[str, str]] = []
        self.labels_added: list[str] = []  # every label name POSTed to /labels (item 3)
        self.labels_removed: list[str] = []  # every label name DELETEd from /labels
        self.followups: list[dict[str, Any]] = []  # advisory follow-up issues created (#1176)

    # GitHub's per-page cap for the comments endpoint in this fixture. The script asks for
    # per_page=100; we paginate ``self.comments`` at this size and emit Link: rel="next"
    # until the final page so a >COMMENTS_PER_PAGE thread exercises the pagination walk.
    COMMENTS_PER_PAGE = 100

    def _comments_page(self, page: int) -> dict[str, Any]:
        """One page of the comment thread + a GitHub-shaped Link header for rel=next.

        Mirrors GitHub's pagination: page 1 of N carries Link: rel="next" (and rel="last");
        the last page carries no rel="next". The script follows the parsed ``page`` number,
        rebuilding the path itself — so the Link URL host here is illustrative only.
        """
        per = self.COMMENTS_PER_PAGE
        start = (page - 1) * per
        chunk = self.comments[start : start + per]
        global_ids = list(range(start, start + len(chunk)))
        body = json.dumps([{"id": i, "body": c} for i, c in zip(global_ids, chunk, strict=True)])
        total_pages = max(1, (len(self.comments) + per - 1) // per)
        headers: dict[str, str] = {}
        if page < total_pages:
            base = "https://api.github.com/repos/o/r/issues/5/comments"
            headers["Link"] = (
                f'<{base}?per_page={per}&page={page + 1}>; rel="next", '
                f'<{base}?per_page={per}&page={total_pages}>; rel="last"'
            )
        return {"status": 200, "headers": headers, "body": body}

    def _http(self, args: dict[str, Any]) -> dict[str, Any]:
        path, method = args["path"], args["method"]
        remaining = self.transient_5xx.get((method, path), 0)
        if remaining > 0:
            self.transient_5xx[(method, path)] = remaining - 1
            self.http.append((method, path))
            return {"status": 503, "body": ""}  # transient -> gh_retry re-issues
        self.http.append((method, path))
        # The comment-thread read is the one route that carries a query string (per_page/page);
        # the GitHub /repos/** route opts into allow_query (#1156). Split it off, serve the page.
        clean, _, query = path.partition("?")
        if method == "GET" and clean == "/repos/o/r/issues/5/comments":  # the item-1 thread read
            page = 1
            m = re.search(r"[?&]page=(\d+)", "?" + query)
            if m:
                page = int(m.group(1))
            return self._comments_page(page)
        assert "?" not in path, f"query string in path is rejected by http_request: {path!r}"
        if method == "POST" and path.endswith("/labels"):  # add label(s) — capture names
            raw = args.get("body")
            if isinstance(raw, str):
                self.labels_added.extend(json.loads(raw).get("labels", []))
            return {"status": 200, "body": "[]"}
        if method == "DELETE" and "/labels/" in path:  # remove one label — capture decoded name
            self.labels_removed.append(path.rsplit("/labels/", 1)[1].replace("%3A", ":"))
            return {"status": 200, "body": "[]"}
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
        if method == "POST" and path == "/repos/o/r/issues":  # advisory follow-up issue (#1176)
            raw = args.get("body")
            self.followups.append(json.loads(raw) if isinstance(raw, str) else {})
            return {"status": 201, "body": json.dumps({"number": 99})}
        if method == "GET" and path == "/repos/o/r/pulls/42":  # merge-confirm read
            body = json.loads(_pr_json(merged=self.pr_merged_on_confirm))
            body["merge_commit_sha"] = self.commit_sha  # confirm-read fallback for master sha
            return {"status": 200, "body": json.dumps(body)}
        if method == "GET" and path == "/repos/o/r/commits/master":  # ref->SHA-1 canonicalise
            # GitHub canonicalises any ref to its SHA-1 commit id. The watch must use THIS,
            # never a 64-char SHA-256 a local clone might produce (issue #1178).
            if self.commit_sha is None:  # ref can't be resolved to a SHA-1
                return {"status": 422, "body": "{}"}
            return {"status": 200, "body": json.dumps({"sha": self.commit_sha})}
        if method == "PUT" and path.endswith("/merge"):
            # The merge PUT returns the merge commit's SHA-1 (issue #1178).
            if self.merge_status == 200 and self.merge_returns_sha:
                body = json.dumps({"sha": MERGE_SHA, "merged": True})
            else:
                body = "{}"
            return {"status": self.merge_status, "body": body}
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
            self.agent_inputs.append(spec["input"])
            if task == "implement":
                if self.implement_error and label == "implement":
                    return {"error": {"kind": "agent_not_found"}}
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
                if self.review_error_on == label:
                    return {"error": {"kind": "child_errored"}}
                idx = int(label.rsplit("-", 1)[1]) if "-" in label else 0
                return {"ok": self.review_results[min(idx, len(self.review_results) - 1)]}
            if task == "watch_ci":
                if spec["input"].get("ref"):  # post-merge master watch (advisory, retried)
                    if self.master_ci_results is not None:
                        idx = int(label.rsplit("-", 1)[1]) if "-" in label else 0
                        item = self.master_ci_results[min(idx, len(self.master_ci_results) - 1)]
                        if item == "error":
                            return {"error": {"kind": "child_errored"}}
                        return {"ok": item}
                    if self.master_ci_error:
                        return {"error": {"kind": "child_errored"}}
                    return {"ok": {"status": self.master_ci, "detail": ""}}
                if self.ci_error_on == label:
                    return {"error": {"kind": "child_errored"}}
                idx = int(label.rsplit("-", 1)[1]) if label.startswith("ci-") else 0
                return {"ok": self.ci_results[min(idx, len(self.ci_results) - 1)]}
            if task == "risk":
                return {"ok": self.risk_result}
            if task in ("fix", "fix_ci"):
                if self.review_error_on == label or self.ci_error_on == label:
                    return {"error": {"kind": "child_errored"}}
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
        "master_ci": "green",
    }
    # a green master-CI watch raises no advisory and files no follow-up issue
    assert "advisories" not in value
    assert scn.followups == []
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
    assert scn.tasks == ["implement", "review-0", "ci-0", "risk", "master-ci-0"]
    # never used a query string EXCEPT on the allow_query comments-pagination path
    # (#1156); every other GitHub call stays clean-path (the production-path bug review caught)
    assert all(
        "?" not in path
        for _, path in scn.http
        if not path.startswith("/repos/o/r/issues/5/comments")
    )
    # item 1: the comment thread is read at ingest (paginated -> ?per_page/?page carried)
    assert any(m == "GET" and p.startswith("/repos/o/r/issues/5/comments") for m, p in scn.http)
    # item 3: in-progress claimed at ingest, released on success, never marked failed
    assert "autodev:in-progress" in scn.labels_added
    assert "autodev:in-progress" in scn.labels_removed
    assert "autodev:failed" not in scn.labels_added


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
    # item 3: a spec failure is a terminal NON-gate failure -> labelled autodev:failed
    assert "autodev:failed" in scn.labels_added
    assert "autodev:in-progress" in scn.labels_removed


async def test_spec_gate_blocks_unresolved_marker() -> None:
    body = LONG_BODY + "\n\nApproach: TBD — still an open question we must settle first."
    scn = Scenario(body=body)
    value, _, _ = await _drive(scn)
    assert value["state"] == "spec_failed"
    assert "unresolved marker" in value["reason"]


async def test_spec_gate_marker_resolved_in_comment_does_not_bounce() -> None:
    # The body carries a marker; a LATER comment quotes that exact line AND signals a
    # resolution -> the gate must NOT bounce, and the run proceeds to implement.
    marker_line = "Approach: TBD — still an open question we must settle first."
    body = LONG_BODY + "\n\n" + marker_line
    scn = Scenario(
        body=body,
        comments=["Resolved: " + marker_line + " We will use a typed enum with a fallback arm."],
    )
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"  # marker suppressed -> full pipeline ran
    assert "implement" in scn.tasks


async def test_spec_gate_comment_without_resolution_signal_still_bounces() -> None:
    # A comment that merely QUOTES the marker line but carries NO resolution signal must
    # NOT suppress it (no regression of the marker-trap protection).
    marker_line = "Approach: TBD — still an open question we must settle first."
    body = LONG_BODY + "\n\n" + marker_line
    scn = Scenario(body=body, comments=["I also wonder: " + marker_line])
    value, _, _ = await _drive(scn)
    assert value["state"] == "spec_failed"
    assert "unresolved marker" in value["reason"]


async def test_spec_gate_comment_resolution_word_without_quote_still_bounces() -> None:
    # A resolution word with NO quote of the offending line must not unlock the gate.
    marker_line = "Approach: TBD — still an open question we must settle first."
    body = LONG_BODY + "\n\n" + marker_line
    scn = Scenario(body=body, comments=["This is all resolved now, trust me."])
    value, _, _ = await _drive(scn)
    assert value["state"] == "spec_failed"
    assert "unresolved marker" in value["reason"]


async def test_thin_body_satisfied_by_comment_thread_proceeds() -> None:
    # A body too short on its own clears the word-count once the comment thread (a design
    # pass) is threaded in — autodev's "comments count toward the spec" behaviour.
    scn = Scenario(body="Add a retry wrapper.", comments=[LONG_BODY])
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"


async def test_comments_threaded_into_every_agent_node() -> None:
    # The comment array reaches implement, review, and fix agents (item 1).
    scn = Scenario(
        comments=["design note: prefer a typed enum"],
        review_results=[
            {"verdict": "fail", "issues": ["x"], "artifact_posted": True},
            {"verdict": "pass", "issues": [], "artifact_posted": True},
        ],
    )
    value, _, _ = await _drive(scn, max_review_iters=3, max_ci_iters=2)
    assert value["state"] == "done"
    threaded = "design note: prefer a typed enum"
    implement_in = [i for i in scn.agent_inputs if i.get("task") == "implement"]
    review_in = [i for i in scn.agent_inputs if i.get("task") == "review"]
    fix_in = [i for i in scn.agent_inputs if i.get("task") == "fix"]
    assert implement_in and all(threaded in i.get("comments", []) for i in implement_in)
    assert review_in and all(threaded in i.get("comments", []) for i in review_in)
    assert fix_in and all(threaded in i.get("comments", []) for i in fix_in)


async def test_comment_thread_past_first_page_fully_ingested() -> None:
    # #1156: a heavily-discussed issue whose AUTHORITATIVE spec resolution lands as the LAST
    # comment, well past GitHub's first page. The pre-fix single GET returned only page 1 (the
    # first COMMENTS_PER_PAGE comments) and silently dropped the rest, so the resolving comment
    # never reached the spec gate or the coder. gh_paginated must follow Link: rel="next" and
    # ingest the WHOLE thread.
    n = Scenario.COMMENTS_PER_PAGE * 2 + 7  # spans three pages (2 full + a partial last)
    last = "design resolved: the interface is a typed Verdict enum — shovel-ready."
    thread = [f"chatter comment number {i}" for i in range(n - 1)] + [last]
    scn = Scenario(comments=thread)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"

    # The final comment (page 3) is threaded into the implement agent — not dropped.
    implement_in = [i for i in scn.agent_inputs if i.get("task") == "implement"]
    assert implement_in
    assert all(last in i.get("comments", []) for i in implement_in), (
        "the last comment (past the first page) must be ingested and threaded to the coder"
    )
    # The full thread (every page) reached the agent, not just page 1.
    assert all(len(i.get("comments", [])) == n for i in implement_in)

    # Pagination walked every page: page 1, 2, and the final page 3 were each fetched.
    comment_pages = sorted(
        int(re.search(r"[?&]page=(\d+)", p).group(1))  # type: ignore[union-attr]
        for m, p in scn.http
        if m == "GET" and p.startswith("/repos/o/r/issues/5/comments?")
    )
    assert comment_pages == [1, 2, 3], comment_pages


async def test_comment_thread_single_page_does_not_over_fetch() -> None:
    # A short thread (one page) must stop after page 1 — no rel="next" => no page-2 request.
    scn = Scenario(comments=["only design note: prefer a typed enum"])
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    pages = [p for m, p in scn.http if m == "GET" and p.startswith("/repos/o/r/issues/5/comments?")]
    assert len(pages) == 1, f"single page must not over-fetch; fetched {pages!r}"
    assert "page=1" in pages[0]


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


async def test_post_merge_master_ci_agent_error_returns_done_without_parking() -> None:
    # #1176: a persistently-erroring post-merge master-CI watch must NOT park the completed
    # run at a (false) human gate. The merge is a committed fact, so the run returns done; the
    # erroring watch degrades to a DISTINCT, non-blocking "indeterminate" advisory (NOT
    # master_red) recorded in result["advisories"], leaving the blocking escalations clean.
    scn = Scenario(master_ci_error=True)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["merged"] is True
    # agent flakiness is never coerced to the most-blocking outcome: no gate, no blocking escalation
    assert scn.gates == []
    assert value["escalations"] == []
    assert "master_red" not in value["escalations"]
    # the distinct "could not determine master state" signal, kept apart from "master is red"
    assert value["master_ci"] == "indeterminate"
    assert value["advisories"] == ["master_ci_indeterminate"]
    # the watch was RETRIED a bounded number of times before being declared indeterminate
    assert scn.tasks.count("master-ci-0") == 1
    assert "master-ci-1" in scn.tasks
    assert "master-ci-2" in scn.tasks
    assert "master-ci-3" not in scn.tasks  # bounded at MAX_MASTER_CI_ITERS=3
    # a best-effort follow-up issue is filed for the indeterminate state (non-blocking alert)
    assert len(scn.followups) == 1
    assert "INDETERMINATE" in scn.followups[0]["title"]


async def test_red_master_ci_files_advisory_and_returns_done_without_parking() -> None:
    # #1176: a genuinely-RED master is a SEPARATE non-blocking signal, not a gate-park. The run
    # returns done (the merge happened), files a follow-up issue, and records the distinct
    # "master_red" advisory — never re-suspending the completed run.
    scn = Scenario(master_ci="red")
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert scn.gates == []  # not a gate-park
    assert value["escalations"] == []  # not coerced into the blocking escalations list
    assert value["master_ci"] == "red"
    assert value["advisories"] == ["master_red"]
    assert len(scn.followups) == 1
    assert "RED" in scn.followups[0]["title"]


async def test_red_and_indeterminate_are_distinct_advisory_reasons() -> None:
    # Acceptance: "master is red" and "could not determine master state" are DISTINCT reasons.
    red = Scenario(master_ci="red")
    red_value, _, _ = await _drive(red, max_review_iters=2, max_ci_iters=2)
    indet = Scenario(master_ci_error=True)
    indet_value, _, _ = await _drive(indet, max_review_iters=2, max_ci_iters=2)
    assert red_value["advisories"] == ["master_red"]
    assert indet_value["advisories"] == ["master_ci_indeterminate"]
    assert red_value["advisories"] != indet_value["advisories"]
    assert red_value["master_ci"] != indet_value["master_ci"]


async def test_post_merge_watch_error_then_success_recovers_to_green() -> None:
    # #1176: a TRANSIENTLY-flaky watch (errors once, then returns green) is retried within the
    # bound and recovers — no advisory, no follow-up issue, run done.
    scn = Scenario(master_ci_results=["error", {"status": "green", "detail": ""}])
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["master_ci"] == "green"
    assert "advisories" not in value
    assert scn.followups == []
    assert "master-ci-0" in scn.tasks and "master-ci-1" in scn.tasks
    assert "master-ci-2" not in scn.tasks  # stopped retrying once a verdict arrived


def _sha1_re():
    import re as _re
    return _re.compile(r"^[0-9a-f]{40}$")


async def test_post_merge_watch_is_handed_the_merge_sha1_not_the_branch_name() -> None:
    # issue #1178: the post-merge master-CI watch must be handed the merge commit's
    # GitHub-canonical SHA-1 (the merge PUT already returned it) — NOT a bare branch name a
    # SHA-256 clone would re-resolve to a 64-char id GitHub's REST API rejects. On a green
    # master this yields a real green verdict with no error and no advisory.
    scn = Scenario()  # merge PUT returns MERGE_SHA, master_ci defaults to green
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["master_ci"] == "green"
    assert "advisories" not in value  # green master raises no advisory
    assert scn.followups == []
    # the watch input carries a 40-char SHA-1 head_sha, not just a branch name
    watch_in = [i for i in scn.agent_inputs
                if i.get("task") == "watch_ci" and i.get("ref")]
    assert watch_in, "post-merge master-CI watch was never dispatched"
    for w in watch_in:
        assert w["head_sha"] == MERGE_SHA
        assert _sha1_re().match(w["head_sha"]), "watch head_sha must be a SHA-1"


async def test_post_merge_watch_never_passes_a_sha256_to_a_github_endpoint() -> None:
    # Regression guard (issue #1178): the watch must NEVER be handed a 64-char SHA-256 object
    # id. Even when the merge PUT carries no sha (confirm-only re-drive), the workflow
    # canonicalises the branch via GitHub and threads the resulting SHA-1 — and any head_sha
    # it ever passes is provably 40-char SHA-1, never 64-char.
    scn = Scenario(merge_returns_sha=False)  # force the GET /commits/master fallback
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["master_ci"] == "green"
    watch_in = [i for i in scn.agent_inputs
                if i.get("task") == "watch_ci" and i.get("ref")]
    assert watch_in, "post-merge master-CI watch was never dispatched"
    for w in watch_in:
        sha = w["head_sha"]
        assert len(sha) != 64, "a 64-char SHA-256 id must never reach the watch"
        assert _sha1_re().match(sha), "watch head_sha must be a SHA-1"
    # the fallback actually went to GitHub's canonicalising endpoint
    assert ("GET", "/repos/o/r/commits/master") in scn.http


async def test_post_merge_watch_unresolvable_sha_degrades_to_indeterminate() -> None:
    # issue #1178 regression guard: if BASE_BRANCH cannot be resolved to a SHA-1 at all, the
    # workflow must NOT fall back to dispatching the watch with a branch name (the very path
    # that produced the SHA-256 error). It degrades to the non-blocking indeterminate advisory
    # and the watch agent is never invoked.
    scn = Scenario(merge_returns_sha=False, commit_sha=None)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["merged"] is True
    assert value["escalations"] == []  # never coerced to a blocking outcome
    assert value["master_ci"] == "indeterminate"
    assert value["advisories"] == ["master_ci_indeterminate"]
    # the watch agent was NOT dispatched with an unresolved ref
    assert not any(t.startswith("master-ci-") for t in scn.tasks)
    assert len(scn.followups) == 1
    assert "INDETERMINATE" in scn.followups[0]["title"]


async def test_post_merge_advisory_never_marks_failed_and_releases_in_progress() -> None:
    # Telemetry must distinguish an advisory post-merge signal from genuinely-blocked work: an
    # indeterminate/red watch must not label the issue failed, and the in-progress claim is
    # released on merge (the issue is done).
    scn = Scenario(master_ci="red")
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert "autodev:failed" not in scn.labels_added
    assert "autodev:in-progress" in scn.labels_removed


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


# ─── transient-5xx retry (item 2) ─────────────────────────────────────────────


async def test_transient_5xx_then_success_completes() -> None:
    # The first two attempts of the ingest GET and the merge PUT return 503; gh's bounded
    # retry re-issues and the third attempt succeeds -> the run completes normally.
    scn = Scenario(
        transient_5xx={
            ("GET", "/repos/o/r/issues/5"): 2,
            ("PUT", "/repos/o/r/pulls/42/merge"): 2,
        }
    )
    value, _, keys = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"
    assert value["merged"] is True
    # the ingest GET was attempted 3 times (2 transient + 1 success), and so was the merge
    assert sum(1 for m, p in scn.http if (m, p) == ("GET", "/repos/o/r/issues/5")) == 3
    assert sum(1 for m, p in scn.http if (m, p) == ("PUT", "/repos/o/r/pulls/42/merge")) == 3
    # replay stayed stable across all the extra attempts (no duplicate call_key)
    assert len(keys) == len(set(keys))


async def test_persistent_5xx_surfaces_failure_not_zombie() -> None:
    # All three ingest attempts 503 -> gh returns the last 503; the run reports the read
    # failure AND labels autodev:failed (item 3) rather than hanging silently.
    scn = Scenario(transient_5xx={("GET", "/repos/o/r/issues/5"): 99})
    value, _, _ = await _drive(scn)
    assert value["state"] == "error"
    assert "autodev:failed" in scn.labels_added
    # exactly 3 attempts (the bounded retry cap), then it gives up
    assert sum(1 for m, p in scn.http if (m, p) == ("GET", "/repos/o/r/issues/5")) == 3


# ─── failure surfacing (item 3) ───────────────────────────────────────────────


async def test_merge_failed_is_labelled_autodev_failed() -> None:
    scn = Scenario(merge_status=405, pr_merged_on_confirm=False)
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "merge_failed"
    assert "autodev:failed" in scn.labels_added
    assert "autodev:in-progress" in scn.labels_removed


async def test_gate_park_is_not_labelled_failed() -> None:
    # An escalation parks at a gate; a deliberate "stay escalated" resume is NOT a silent
    # failure, so the run must NOT raise autodev:failed (the contract: a gate park ≠ fail).
    scn = Scenario(implement_escalated=True, gate_results={"design": {"resolved": False}})
    value, _, _ = await _drive(scn)
    assert value["state"] == "escalated"
    assert "autodev:failed" not in scn.labels_added


# ─── AgentError guards on every judgment node (item 5) ─────────────────────────


async def test_implement_agent_error_escalates_to_design_gate() -> None:
    scn = Scenario(implement_error=True, gate_results={"design": {"resolved": False}})
    value, _, _ = await _drive(scn)
    assert value["state"] == "escalated"
    assert value["reason"] == "design"
    assert any(g["kind"] == "design" for g in scn.gates)  # parked, did not crash the run


async def test_review_agent_error_escalates_to_verify_gate() -> None:
    scn = Scenario(review_error_on="review-0", gate_results={"verify": {"resolved": False}})
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "escalated"
    assert value["reason"] == "verify_agent_error"
    assert any(g["kind"] == "verify" for g in scn.gates)


async def test_ci_watch_agent_error_escalates_to_verify_gate() -> None:
    scn = Scenario(ci_error_on="ci-0", gate_results={"verify": {"resolved": False}})
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "escalated"
    assert value["reason"] == "ci_agent_error"
    assert any(g["kind"] == "verify" for g in scn.gates)


async def test_agent_error_resume_resolved_continues() -> None:
    # A reviewer error that the human resumes as resolved lets the run proceed to done.
    scn = Scenario(review_error_on="review-0", gate_results={"verify": {"resolved": True}})
    value, _, _ = await _drive(scn, max_review_iters=2, max_ci_iters=2)
    assert value["state"] == "done"


# ─── auto-recovery retry budget (item 4) ──────────────────────────────────────


async def test_retry_count_under_budget_runs_normally() -> None:
    scn = Scenario()
    value, _, _ = await _drive(
        scn,
        input={"repo": REPO, "issue_number": ISSUE, "kind": "issue", "retry_count": 1},
        max_review_iters=2,
        max_ci_iters=2,
    )
    assert value["state"] == "done"


async def test_retry_count_at_budget_dead_letters() -> None:
    # A re-launched run that has already burned MAX_RUN_RETRIES (2) terminates at the
    # dead-letter state instead of looping, and is labelled autodev:failed.
    scn = Scenario()
    value, phases, _ = await _drive(
        scn,
        input={"repo": REPO, "issue_number": ISSUE, "kind": "issue", "retry_count": 2},
    )
    assert value["state"] == "dead_letter"
    assert value["retry_count"] == 2
    assert phases == []  # dead-lettered before even the ingest phase
    assert scn.tasks == []  # no spend
    assert "autodev:failed" in scn.labels_added
    # item 3 (never a silent zombie): a prior attempt left autodev:in-progress; the
    # dead-letter terminal must DROP it (replace with autodev:failed), exactly like _fail
    # does on every other terminal failure — otherwise the issue is stuck both in-progress
    # AND failed, perpetually claimed and never re-dispatchable.
    assert "autodev:in-progress" in scn.labels_removed


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
