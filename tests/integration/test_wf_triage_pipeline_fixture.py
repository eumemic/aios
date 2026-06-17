"""Triage-pipeline reference workflow fixture (aios#1226).

Drives the production ``build_triage_pipeline_script`` directly against the real script
host (no DB) with simulated agent/tool resolutions, walking the stateless scan→triage→
summary machine. Because the host re-runs from the start each wake (replaying the whole
growing memo), these tests also assert the load-bearing durable property: **replay is
deterministic** (every call_key is stable across every replay).

It proves the #1226 acceptance:
- correct classification + labeling on a sample set (modelled on #1221-#1225),
- the structured run summary (per-class counts + the explicit needs-decision list),
- **idempotency**: a re-run over the now-labeled set touches NOTHING (no agents, no labels).

Mirrors the host-subprocess style of ``test_wf_dev_pipeline_fixture.py``; needs no Postgres.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from aios.workflows.host_launcher import run_script_host
from aios.workflows.triage_pipeline import (
    TRIAGE_STATE_LABELS,
    build_triage_pipeline_script,
)

pytestmark = pytest.mark.integration

REPO = "o/r"


def _issue(number: int, *, labels: list[str] | None = None, body: str = "a spec",
           pull_request: bool = False, state: str = "open") -> dict[str, Any]:
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


# The concrete target set today (#1226): the #777 parity issues #1221-#1225, currently
# labeled only `enhancement,workflows`.
PARITY_LABELS = ["enhancement", "workflows"]


class Scenario:
    """A deterministic responder + a record of what the script asked for. Keying every
    decision on the capability (and the per-issue agent label) keeps the responder a pure
    function of the call_key — so replay is stable."""

    ISSUES_PER_PAGE = 100

    def __init__(
        self,
        *,
        issues: list[dict[str, Any]],
        classifications: dict[int, dict[str, Any]] | None = None,
        comments: dict[int, list[str]] | None = None,
        agent_error_on: set[int] | None = None,
        label_post_status: int = 200,
    ) -> None:
        self.issues = issues
        # {issue_number: {"classification": ..., "reason": ...}} the classifier returns.
        self.classifications = classifications or {}
        self.comments = comments or {}
        self.agent_error_on = agent_error_on or set()
        self.label_post_status = label_post_status
        # observability
        self.http: list[tuple[str, str]] = []
        self.tasks: list[str] = []
        self.agent_inputs: list[dict[str, Any]] = []
        self.labels_added: dict[int, list[str]] = {}     # issue -> labels POSTed
        self.comments_posted: dict[int, list[str]] = {}  # issue -> comment bodies POSTed

    # ── pagination of the issue list ──
    def _issues_page(self, page: int) -> dict[str, Any]:
        per = self.ISSUES_PER_PAGE
        start = (page - 1) * per
        chunk = self.issues[start : start + per]
        total_pages = max(1, (len(self.issues) + per - 1) // per)
        headers: dict[str, str] = {}
        if page < total_pages:
            base = "https://api.github.com/repos/o/r/issues"
            headers["Link"] = (
                f'<{base}?per_page={per}&page={page + 1}>; rel="next", '
                f'<{base}?per_page={per}&page={total_pages}>; rel="last"'
            )
        return {"status": 200, "headers": headers, "body": json.dumps(chunk)}

    def _comments_page(self, number: int, page: int) -> dict[str, Any]:
        bodies = self.comments.get(number, [])
        # single page for the fixture's small threads
        return {"status": 200, "headers": {},
                "body": json.dumps([{"id": i, "body": b} for i, b in enumerate(bodies)])}

    def _http(self, args: dict[str, Any]) -> dict[str, Any]:
        path, method = args["path"], args["method"]
        self.http.append((method, path))
        clean, _, query = path.partition("?")

        if method == "GET" and clean == "/repos/o/r/issues":
            page = 1
            m = re.search(r"[?&]page=(\d+)", "?" + query)
            if m:
                page = int(m.group(1))
            return self._issues_page(page)

        m = re.match(r"^/repos/o/r/issues/(\d+)/comments$", clean)
        if method == "GET" and m:
            page = 1
            pm = re.search(r"[?&]page=(\d+)", "?" + query)
            if pm:
                page = int(pm.group(1))
            return self._comments_page(int(m.group(1)), page)

        if method == "POST" and m:  # post a comment (needs-decision)
            number = int(m.group(1))
            raw = args.get("body")
            if isinstance(raw, str):
                self.comments_posted.setdefault(number, []).append(json.loads(raw).get("body", ""))
            return {"status": 201, "body": "{}"}

        ml = re.match(r"^/repos/o/r/issues/(\d+)/labels$", clean)
        if method == "POST" and ml:  # apply label(s)
            number = int(ml.group(1))
            raw = args.get("body")
            if isinstance(raw, str):
                self.labels_added.setdefault(number, []).extend(json.loads(raw).get("labels", []))
            return {"status": self.label_post_status, "body": "[]"}

        return {"status": 200, "body": "{}"}

    def outcome(self, cap: Any) -> dict[str, Any]:
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool" and spec["tool_name"] == "http_request":
            return {"ok": self._http(spec["input"])}
        if cid == "agent":
            label = cap.annotations.get("label", "")
            self.tasks.append(label)
            self.agent_inputs.append(spec["input"])
            number = int(spec["input"].get("issue_number", 0))
            if number in self.agent_error_on:
                return {"error": {"kind": "child_errored"}}
            return {"ok": self.classifications.get(
                number, {"classification": "needs-design", "reason": ""})}
        raise AssertionError(f"unhandled capability {cid} spec={spec!r}")


async def _drive(
    scenario: Scenario,
    *,
    input: dict[str, Any] | None = None,
    max_steps: int = 120,
    **build_kwargs: Any,
) -> tuple[Any, list[str], list[str]]:
    """Drive the production script to a terminal outcome, returning
    (return_value, phases, ordered_call_keys). Asserts replay-determinism (unique keys)."""
    src = build_triage_pipeline_script(**build_kwargs)
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
    raise AssertionError(f"workflow did not terminate within {max_steps} steps")


# ─── the #1221-#1225 sample set: classify + label ─────────────────────────────


def _parity_issues() -> list[dict[str, Any]]:
    return [_issue(n, labels=list(PARITY_LABELS)) for n in range(1221, 1226)]


async def test_classifies_and_labels_the_sample_set() -> None:
    issues = _parity_issues()
    classifications = {
        1221: {"classification": "shovel-ready", "reason": "scope clear"},
        1222: {"classification": "shovel-ready", "reason": "scope clear"},
        1223: {"classification": "needs-design", "reason": "architecture unresolved"},
        1224: {"classification": "needs-decision", "reason": "needs capital sign-off"},
        1225: {"classification": "needs-decision", "reason": "settled-vs-forks: A or B"},
    }
    scn = Scenario(issues=issues, classifications=classifications)
    value, phases, _keys = await _drive(scn)

    assert value["state"] == "done"
    assert value["repo"] == REPO
    assert value["scanned"] == 5
    assert value["untriaged"] == 5
    assert value["classified"] == 5
    assert value["counts"] == {"shovel-ready": 2, "needs-design": 1, "needs-decision": 2}
    assert phases == ["scan", "triage", "summary"]

    # the right label was applied to each issue
    assert scn.labels_added[1221] == ["shovel-ready"]
    assert scn.labels_added[1222] == ["shovel-ready"]
    assert scn.labels_added[1223] == ["needs-design"]
    assert scn.labels_added[1224] == ["needs-decision"]
    assert scn.labels_added[1225] == ["needs-decision"]

    # triage NEVER applies `approved` (the two-axis invariant)
    assert all("approved" not in labs for labs in scn.labels_added.values())

    # one ephemeral agent per untriaged issue, in sorted order
    assert scn.tasks == [f"triage-{n}" for n in range(1221, 1226)]


async def test_needs_decision_posts_one_comment_with_the_reason() -> None:
    issues = [_issue(1224, labels=list(PARITY_LABELS))]
    scn = Scenario(
        issues=issues,
        classifications={1224: {"classification": "needs-decision",
                                "reason": "requires external partner commitment"}},
    )
    value, _, _ = await _drive(scn)
    assert value["counts"]["needs-decision"] == 1
    # exactly one comment, and it states the why
    assert len(scn.comments_posted[1224]) == 1
    posted = scn.comments_posted[1224][0]
    assert "needs a decision" in posted.lower()
    assert "requires external partner commitment" in posted
    # the explicit needs-decision list in the summary carries issue + reason
    assert value["needs_decision"] == [
        {"issue": 1224, "reason": "requires external partner commitment"}
    ]


async def test_shovel_ready_and_needs_design_post_no_comment() -> None:
    issues = [_issue(1221, labels=list(PARITY_LABELS)),
              _issue(1223, labels=list(PARITY_LABELS))]
    scn = Scenario(
        issues=issues,
        classifications={
            1221: {"classification": "shovel-ready", "reason": "x"},
            1223: {"classification": "needs-design", "reason": "y"},
        },
    )
    await _drive(scn)
    assert scn.comments_posted == {}  # only needs-decision posts a comment


# ─── idempotency: a re-run over the labeled set is a no-op ─────────────────────


async def test_re_run_over_labeled_set_is_a_noop() -> None:
    # After the first run, the issues now carry their triage state-label. A re-run must
    # touch NOTHING: no classifier agent, no label POST.
    already = [
        _issue(1221, labels=["enhancement", "shovel-ready"]),
        _issue(1223, labels=["enhancement", "needs-design"]),
        _issue(1224, labels=["enhancement", "needs-decision"]),
    ]
    scn = Scenario(issues=already)
    value, phases, _ = await _drive(scn)
    assert value["state"] == "done"
    assert value["scanned"] == 3
    assert value["untriaged"] == 0
    assert value["classified"] == 0
    assert value["counts"] == {"shovel-ready": 0, "needs-design": 0, "needs-decision": 0}
    assert scn.tasks == []  # no agent spend
    assert scn.labels_added == {}  # no label written
    assert scn.comments_posted == {}
    assert phases == ["scan", "triage", "summary"]


async def test_each_state_label_marks_an_issue_already_triaged() -> None:
    # Any ONE of the five triage state-labels means the issue is already triaged.
    issues = [_issue(1000 + i, labels=["enhancement", lab])
              for i, lab in enumerate(TRIAGE_STATE_LABELS)]
    scn = Scenario(issues=issues)
    value, _, _ = await _drive(scn)
    assert value["untriaged"] == 0
    assert scn.tasks == []


async def test_mixed_untriaged_and_triaged_only_touches_untriaged() -> None:
    issues = [
        _issue(1221, labels=["enhancement"]),                 # untriaged
        _issue(1222, labels=["enhancement", "shovel-ready"]),  # already triaged
        _issue(1223, labels=[]),                               # untriaged (no labels at all)
        _issue(1224, labels=["needs-decision"]),               # already triaged
    ]
    scn = Scenario(
        issues=issues,
        classifications={
            1221: {"classification": "shovel-ready", "reason": "r"},
            1223: {"classification": "needs-design", "reason": "r"},
        },
    )
    value, _, _ = await _drive(scn)
    assert value["untriaged"] == 2
    assert value["classified"] == 2
    assert scn.tasks == ["triage-1221", "triage-1223"]
    assert set(scn.labels_added) == {1221, 1223}


# ─── empty / no-op scans ──────────────────────────────────────────────────────


async def test_empty_repo_is_a_clean_noop() -> None:
    scn = Scenario(issues=[])
    value, phases, _ = await _drive(scn)
    assert value["state"] == "done"
    assert value["scanned"] == 0
    assert value["untriaged"] == 0
    assert value["counts"] == {"shovel-ready": 0, "needs-design": 0, "needs-decision": 0}
    assert scn.tasks == []
    assert phases == ["scan", "triage", "summary"]


async def test_pull_requests_are_skipped() -> None:
    # The issues endpoint returns PRs too; they must never be triaged.
    issues = [
        _issue(1221, labels=["enhancement"]),
        _issue(1222, labels=["enhancement"], pull_request=True),  # a PR — skip
    ]
    scn = Scenario(
        issues=issues,
        classifications={1221: {"classification": "shovel-ready", "reason": "r"}},
    )
    value, _, _ = await _drive(scn)
    assert value["scanned"] == 1  # the PR is not counted as a scanned issue
    assert value["untriaged"] == 1
    assert scn.tasks == ["triage-1221"]


# ─── full comment-thread read (a later comment can supersede the body) ─────────


async def test_comment_thread_is_threaded_into_the_classifier() -> None:
    issues = [_issue(1221, labels=["enhancement"])]
    scn = Scenario(
        issues=issues,
        classifications={1221: {"classification": "shovel-ready", "reason": "r"}},
        comments={1221: ["design resolved: use a typed enum — shovel-ready"]},
    )
    await _drive(scn)
    triage_in = [i for i in scn.agent_inputs if i.get("task") == "triage"]
    assert triage_in
    assert "design resolved: use a typed enum — shovel-ready" in triage_in[0]["comments"]


# ─── robustness: a flaky/garbage classification doesn't abort the sweep ────────


async def test_agent_error_on_one_issue_records_and_continues() -> None:
    issues = [_issue(1221, labels=["enhancement"]), _issue(1222, labels=["enhancement"])]
    scn = Scenario(
        issues=issues,
        classifications={1222: {"classification": "shovel-ready", "reason": "r"}},
        agent_error_on={1221},
    )
    value, _, _ = await _drive(scn)
    assert value["state"] == "done"  # the sweep completed despite one failure
    assert value["classified"] == 1  # only 1222 was classified
    assert value["counts"]["shovel-ready"] == 1
    assert len(value["errors"]) == 1
    assert value["errors"][0]["issue"] == 1221
    assert "agent error" in value["errors"][0]["error"]
    # 1221 got no label (it errored); 1222 was labeled
    assert 1221 not in scn.labels_added
    assert scn.labels_added[1222] == ["shovel-ready"]


async def test_label_post_failure_is_recorded_per_issue() -> None:
    issues = [_issue(1221, labels=["enhancement"])]
    scn = Scenario(
        issues=issues,
        classifications={1221: {"classification": "shovel-ready", "reason": "r"}},
        label_post_status=403,
    )
    value, _, _ = await _drive(scn)
    assert value["state"] == "done"
    assert value["classified"] == 0  # the label failed, so it's not counted classified
    assert len(value["errors"]) == 1
    assert value["errors"][0]["issue"] == 1221
    assert "label apply failed" in value["errors"][0]["error"]


# ─── input handling ───────────────────────────────────────────────────────────


async def test_accepts_trigger_envelope_input() -> None:
    scn = Scenario(issues=[])
    value, _, _ = await _drive(
        scn, input={"trigger": {"kind": "cron"}, "input": {"repo": REPO}}
    )
    assert value["state"] == "done"


async def test_default_repo_used_when_input_omits_one() -> None:
    scn = Scenario(issues=[_issue(1221, labels=["enhancement"])])
    scn_class = scn.classifications
    scn_class[1221] = {"classification": "shovel-ready", "reason": "r"}
    value, _, _ = await _drive(scn, input={}, repo=REPO)
    assert value["state"] == "done"
    assert value["repo"] == REPO
    assert value["classified"] == 1


async def test_missing_repo_with_no_default_errors() -> None:
    scn = Scenario(issues=[])
    value, _, _ = await _drive(scn, input={})
    assert value["state"] == "error"
    assert "no repo" in value["reason"]


# ─── max-issues-per-run bound (stateless re-run picks up the remainder) ────────


async def test_max_issues_per_run_bounds_one_sweep() -> None:
    issues = [_issue(1221 + i, labels=["enhancement"]) for i in range(5)]
    scn = Scenario(
        issues=issues,
        classifications={n: {"classification": "shovel-ready", "reason": "r"}
                         for n in range(1221, 1226)},
    )
    value, _, _ = await _drive(scn, max_issues_per_run=2)
    # only the first 2 (sorted) are classified this run; a re-run handles the rest
    assert value["untriaged"] == 5
    assert value["classified"] == 2
    assert scn.tasks == ["triage-1221", "triage-1222"]
