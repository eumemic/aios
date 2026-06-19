"""The dev-pipeline reconciler's ``owner()`` — completeness + idempotency property-tests + the
deploy-surface + the safety invariants (aios#49/#111, build steps 1, 1b, 2, 4).

``owner()`` is the load-bearing TOTAL function: over a normalized blackboard tuple (an issue/PR's
labels + CI status + mergeable + draft + approval + human-review + closed), it returns AT MOST ONE
(item, transition) in a fixed first-match-wins priority order, with a final ``else``. The whole
correctness guarantee of the reconciler is that this function is (a) returns ≤1 action and (b) is
total over the enumerated cross-product. These tests turn that guarantee into an enforced CI gate.

``owner()`` is PURE + SYNCHRONOUS — it reads no clock, performs no I/O, emits no capability — so we
exec the SAME source string that ships into the production script (``owner_source()``) and call it
directly across the entire enumerated cross-product. Exercising the exact text that ships (one
source of truth) means the property-test can never pass on a different function than the one running.
"""

from __future__ import annotations

import ast
import itertools
from typing import Any

from aios.models.agents import HttpServerSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.dev_pipeline_reconciler import (
    AUTO_MERGE_MAX_TIER,
    LABEL_MERGE_APPROVED,
    LABEL_REVIEWED_PREFIX,
    NEEDS_HUMAN_MERGE_APPROVAL,
    REQUIRED_HTTP_SERVERS,
    REQUIRED_TOOLS,
    build_reconciler_fixture_script,
    build_reconciler_script,
    build_reconciler_workflow_create,
    owner_source,
)

# ─── exec the SHIPPED owner() source into a namespace (one source of truth) ──

# We exec OUR OWN authored source (owner_source()) in-process with no I/O — the SAME text that
# ships into the production script — so the property-tests exercise exactly the function that runs.
_OWNER_NS: dict[str, Any] = {}
exec(owner_source(), _OWNER_NS)
owner = _OWNER_NS["owner"]
select_actionable = _OWNER_NS["select_actionable"]

_SHA = "a" * 40
_SHA2 = "b" * 40


# ─── normalized-item builders (the blackboard owner() reads) ─────────────────


def _issue(
    *,
    number: int = 1,
    shovel_ready: bool = False,
    approved: bool = False,
    dispatched: bool = False,
    has_open_pr: bool = False,
    closed: bool = False,
    labels: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    return {
        "kind": "issue",
        "number": number,
        "labels": labels,
        "closed": closed,
        "shovel_ready": shovel_ready,
        "approved": approved,
        "dispatched": dispatched,
        "has_open_pr": has_open_pr,
    }


def _pr(
    *,
    number: int = 1,
    labels: frozenset[str] = frozenset(),
    closed: bool = False,
    draft: bool = False,
    head_sha: str = _SHA,
    needs_rebase: bool = False,
    ci_verdict: str = "green",
    human_blocked: bool = False,
    risk_tier: int | None = None,
) -> dict[str, Any]:
    return {
        "kind": "pr",
        "number": number,
        "labels": labels,
        "closed": closed,
        "draft": draft,
        "head_sha": head_sha,
        "needs_rebase": needs_rebase,
        "ci_verdict": ci_verdict,
        "human_blocked": human_blocked,
        "risk_tier": risk_tier,
    }


def _transition(item: dict[str, Any]) -> str | None:
    """The transition owner() picks, or None (terminal)."""
    decision = owner(item)
    if decision is None:
        return None
    _it, transition = decision
    return str(transition)


# ════════════════════════════════════════════════════════════════════════════
# STEP 1b — COMPLETENESS: owner() returns <=1 action AND is TOTAL over the
#           enumerated (ci x mergeable x draft x approval x human-review x closed)
#           cross-product. This is the whole correctness guarantee.
# ════════════════════════════════════════════════════════════════════════════

# The enumerated axes. ``mergeable`` is modelled by ``needs_rebase`` (the conflicting/behind state);
# ``ci`` by the verdict vocabulary owner() reads; ``approval`` by presence of merge:approved;
# ``human-review`` by ``human_blocked``; plus draft + closed + reviewed-for-head.
_CI_VERDICTS = ["green", "red", "pending", "no_ci", "unknown", "weird-unforeseen-status"]
_BOOLS = [False, True]
_TIERS = [None, 1, 2, 3, 4]
_VALID_TRANSITIONS = {"build", "rebase", "ci", "review", "merge", "escalate", None}


def _pr_cross_product() -> list[dict[str, Any]]:
    """Every PR tuple in the enumerated cross-product (ci x needs_rebase x draft x approval x
    human-review x closed x reviewed-for-head x tier)."""
    out = []
    for ci, rebase, draft, approved, human, closed, reviewed, tier in itertools.product(
        _CI_VERDICTS, _BOOLS, _BOOLS, _BOOLS, _BOOLS, _BOOLS, _BOOLS, _TIERS
    ):
        labels = set()
        if approved:
            labels.add(LABEL_MERGE_APPROVED)
        if reviewed:
            labels.add(LABEL_REVIEWED_PREFIX + _SHA)
        out.append(
            _pr(
                number=1,
                labels=frozenset(labels),
                closed=closed,
                draft=draft,
                head_sha=_SHA,
                needs_rebase=rebase,
                ci_verdict=ci,
                human_blocked=human,
                risk_tier=tier,
            )
        )
    return out


def _issue_cross_product() -> list[dict[str, Any]]:
    out = []
    for sr, ap, disp, has_pr, closed in itertools.product(_BOOLS, _BOOLS, _BOOLS, _BOOLS, _BOOLS):
        out.append(
            _issue(shovel_ready=sr, approved=ap, dispatched=disp, has_open_pr=has_pr, closed=closed)
        )
    return out


def test_owner_is_total_over_the_pr_cross_product() -> None:
    # TOTAL: every enumerated PR tuple maps to a VALID transition (or None) — never an unhandled
    # state. The final else→escalate makes a tuple no branch foresaw detectable, not a silent gap.
    for item in _pr_cross_product():
        decision = owner(item)
        assert decision is None or decision[1] in _VALID_TRANSITIONS, (item, decision)


def test_owner_is_total_over_the_issue_cross_product() -> None:
    for item in _issue_cross_product():
        decision = owner(item)
        assert decision is None or decision[1] in _VALID_TRANSITIONS, (item, decision)


def test_owner_returns_at_most_one_action() -> None:
    # ≤1 ACTION: owner() returns either None or exactly ONE (item, transition) — never a list /
    # multiple actions. First-match-wins over a fixed priority order is disjoint by construction.
    for item in _pr_cross_product() + _issue_cross_product():
        decision = owner(item)
        assert decision is None or (isinstance(decision, tuple) and len(decision) == 2), decision


def test_owner_never_returns_an_unforeseen_verdict_as_a_silent_noop() -> None:
    # An unforeseen ci-status string (not in the verdict vocabulary) must ESCALATE (the relocated
    # gap is detectable), not silently fall through to None. A mergeable, non-human, open PR with a
    # garbage verdict is the canonical "a tuple the authors didn't foresee" case.
    item = _pr(ci_verdict="weird-unforeseen-status", needs_rebase=False, human_blocked=False)
    assert _transition(item) == "escalate"


def test_unforeseen_kind_escalates() -> None:
    weird = {"kind": "something-new", "number": 1, "labels": frozenset(), "closed": False}
    assert _transition(weird) == "escalate"


# ─── the priority order itself (first-match-wins disjointness) ───────────────


def test_needs_human_is_the_hard_veto_first() -> None:
    # ANY needs:human/* label makes owner() return None FIRST — even over an otherwise-actionable
    # state (a red PR that would otherwise route to ci). A terminal label can't freeze a loop.
    red_human = _pr(ci_verdict="red", labels=frozenset({"needs:human/ci"}))
    assert owner(red_human) is None
    build_human = _issue(
        shovel_ready=True, approved=True, labels=frozenset({"shovel-ready", "needs:human/spec"})
    )
    assert owner(build_human) is None


def test_closed_is_terminal() -> None:
    assert owner(_pr(closed=True, ci_verdict="red")) is None
    assert owner(_issue(closed=True, shovel_ready=True, approved=True)) is None


def test_rebase_precedes_ci() -> None:
    # A conflicting PR can't run CI, so rebase is checked FIRST among PR states — even when the
    # (stale) verdict reads red.
    assert _transition(_pr(needs_rebase=True, ci_verdict="red")) == "rebase"


def test_human_blocked_pr_is_terminal() -> None:
    # The reconciler refuses to fight a human (requested-changes / CODEOWNERS), even on a red PR.
    assert owner(_pr(human_blocked=True, ci_verdict="red")) is None


def test_pending_ci_waits() -> None:
    assert owner(_pr(ci_verdict="pending")) is None
    assert owner(_pr(ci_verdict="unknown")) is None


def test_green_unreviewed_routes_to_review() -> None:
    assert _transition(_pr(ci_verdict="green", labels=frozenset())) == "review"
    assert _transition(_pr(ci_verdict="no_ci", labels=frozenset())) == "review"


def test_reviewed_for_stale_sha_is_re_reviewed_on_new_head() -> None:
    # reviewed:green stamped for an OLD sha; the head advanced -> not reviewed-for-head -> review.
    item = _pr(head_sha=_SHA2, labels=frozenset({LABEL_REVIEWED_PREFIX + _SHA}))
    assert _transition(item) == "review"


def test_reviewed_for_head_without_approval_re_drives_review() -> None:
    # reviewed-for-head but no merge:approved and no needs:human/* (a crash between label writes) ->
    # re-pick review to complete it (idempotent re-stamp), never a silent stall.
    item = _pr(head_sha=_SHA, labels=frozenset({LABEL_REVIEWED_PREFIX + _SHA}))
    assert _transition(item) == "review"


# ════════════════════════════════════════════════════════════════════════════
# #1158 TIER-REFUSE — owner() never routes a tier>cap PR to the merge effector.
# ════════════════════════════════════════════════════════════════════════════


def test_owner_refuses_merge_for_tier_above_cap() -> None:
    # The merge branch is the EFFECTOR. owner() must STRUCTURALLY exclude tier>AUTO_MERGE_MAX_TIER
    # from it — even a (wrongly) merge:approved-labeled tier-4 PR is NOT routed to merge; it
    # escalates defensively. The review branch is what refuses to STAMP merge:approved for tier>cap;
    # owner() is the second, structural line of the same #1158 control.
    approved_labels = frozenset({LABEL_MERGE_APPROVED, LABEL_REVIEWED_PREFIX + _SHA})
    for tier in range(AUTO_MERGE_MAX_TIER + 1, 6):
        item = _pr(labels=approved_labels, risk_tier=tier, head_sha=_SHA)
        assert _transition(item) != "merge", f"tier {tier} must NOT route to merge"
        assert _transition(item) == "escalate"


def test_owner_routes_merge_for_explicit_in_cap_tier() -> None:
    approved_labels = frozenset({LABEL_MERGE_APPROVED, LABEL_REVIEWED_PREFIX + _SHA})
    for tier in (1, 2, AUTO_MERGE_MAX_TIER):
        item = _pr(labels=approved_labels, risk_tier=tier, head_sha=_SHA)
        assert _transition(item) == "merge", f"tier {tier} (explicit, in cap) must route to merge"


def test_owner_refuses_merge_for_approved_pr_with_no_tier() -> None:
    # Defense in depth: a merge:approved PR carrying NO risk:tier-N label (tier is None) means the
    # review branch never completed its tier stamp — owner() must NOT treat 'no tier' as 'in cap'
    # and merge. The effector merges only on POSITIVE evidence of an in-cap tier; it escalates here.
    approved_labels = frozenset({LABEL_MERGE_APPROVED, LABEL_REVIEWED_PREFIX + _SHA})
    item = _pr(labels=approved_labels, risk_tier=None, head_sha=_SHA)
    assert _transition(item) != "merge"
    assert _transition(item) == "escalate"


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — IDEMPOTENCY: re-evaluating an item owner() already advanced is a NO-OP
#          at the decision layer. owner() is a pure function of the item, so two
#          calls on the SAME tuple are identical; and an item whose advance landed
#          a terminal/claim label is NOT re-picked.
# ════════════════════════════════════════════════════════════════════════════


def test_owner_is_a_pure_function_of_the_item() -> None:
    # Determinism: the same tuple yields the same decision on every call (no hidden state).
    for item in _pr_cross_product() + _issue_cross_product():
        assert owner(item) == owner(item)


def test_advanced_build_issue_is_not_re_picked() -> None:
    # After a build advances the issue (it stamps `dispatched`), a re-scan must NOT re-build it.
    before = _issue(shovel_ready=True, approved=True, dispatched=False, has_open_pr=False)
    assert _transition(before) == "build"
    after = _issue(shovel_ready=True, approved=True, dispatched=True, has_open_pr=False)
    assert _transition(after) is None  # dispatched -> no longer build-eligible
    after_pr = _issue(shovel_ready=True, approved=True, dispatched=False, has_open_pr=True)
    assert _transition(after_pr) is None  # a PR already exists -> no re-build


def test_advanced_review_pr_is_not_re_reviewed_at_same_head() -> None:
    # After review advances the PR to merge:approved (tier<=cap), a re-scan routes to MERGE, never
    # back to review — the reviewed:green@<sha> + merge:approved labels guard re-review.
    advanced = _pr(
        labels=frozenset({LABEL_REVIEWED_PREFIX + _SHA, LABEL_MERGE_APPROVED}),
        head_sha=_SHA,
        risk_tier=2,
    )
    assert _transition(advanced) == "merge"


def test_advanced_review_pr_to_human_approval_is_terminal() -> None:
    # After review routes a tier>cap PR to needs:human/merge-approval, a re-scan vetoes it (None).
    advanced = _pr(
        labels=frozenset({LABEL_REVIEWED_PREFIX + _SHA, NEEDS_HUMAN_MERGE_APPROVAL}),
        head_sha=_SHA,
        risk_tier=4,
    )
    assert owner(advanced) is None


def test_escalated_item_is_not_re_escalated() -> None:
    # After escalate stamps needs:human/stuck, a re-scan vetoes it (the needs:human/* hard veto).
    advanced = _pr(ci_verdict="weird", labels=frozenset({"needs:human/stuck"}))
    assert owner(advanced) is None


# ════════════════════════════════════════════════════════════════════════════
# select_actionable — exactly ONE transition per pass, priority-then-number order.
# ════════════════════════════════════════════════════════════════════════════


def test_select_actionable_picks_one_highest_priority_item() -> None:
    # rebase < ci < review < merge < build < escalate; within a rank, lowest number wins.
    items = [
        _issue(number=2, shovel_ready=True, approved=True),  # build (rank 4)
        _pr(number=5, ci_verdict="red"),  # ci (rank 1)
        _pr(number=3, needs_rebase=True),  # rebase (rank 0) — wins
    ]
    decision = select_actionable(items)
    assert decision is not None
    item, transition = decision
    assert transition == "rebase"
    assert item["number"] == 3


def test_select_actionable_lowest_number_tiebreak_within_rank() -> None:
    items = [
        _pr(number=9, ci_verdict="red"),
        _pr(number=4, ci_verdict="red"),
        _pr(number=7, ci_verdict="red"),
    ]
    decision = select_actionable(items)
    assert decision is not None
    item, transition = decision
    assert transition == "ci"
    assert item["number"] == 4  # lowest number wins


def test_select_actionable_returns_none_when_nothing_actionable() -> None:
    items = [
        _pr(number=1, ci_verdict="pending"),
        _pr(number=2, human_blocked=True),
        _issue(number=3, closed=True),
        _pr(number=4, labels=frozenset({"needs:human/ci"}), ci_verdict="red"),
    ]
    assert select_actionable(items) is None


def test_select_actionable_empty_is_none() -> None:
    assert select_actionable([]) is None


# ════════════════════════════════════════════════════════════════════════════
# SAFETY — gate() is BANNED in the assembled script; no datetime/time import.
# ════════════════════════════════════════════════════════════════════════════


def _gate_call_count(src: str) -> int:
    """Count actual ``gate(...)`` CALL nodes (not prose mentions of "gate()" in docstrings).

    Catches BOTH the bare-name form ``gate(...)`` AND the attribute form ``x.gate(...)`` — so a
    ``from aios... import gate as g`` alias OR a ``module.gate()`` getattr escape can't smuggle a
    suspend past the ban. The author namespace injects ``gate`` as a bare global (no module to
    attribute off), but we match the attribute form too so the guard is robust to a future splice
    that imports it differently. (We also assert no aliasing import statement below.)"""
    tree = ast.parse(src)
    count = 0
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        func = n.func
        if (isinstance(func, ast.Name) and func.id == "gate") or (
            isinstance(func, ast.Attribute) and func.attr == "gate"
        ):
            count += 1
    return count


def test_assembled_script_has_zero_gate_calls() -> None:
    # gate() durably SUSPENDS the whole run (the orphan bug relocated, not fixed). The reconciler
    # BANS it in the loop — escalation is a durable needs:human/* label, the queryable mailbox.
    for src in (
        build_reconciler_script(repo="o/r"),
        build_reconciler_fixture_script(
            implement_agent_id="i",
            review_agent_id="r",
            fix_agent_id="f",
            ci_agent_id="c",
            risk_agent_id="k",
            repo="o/r",
        ),
    ):
        assert _gate_call_count(src) == 0
        # Alias-proofing: no ``import gate``/``gate as``/``= gate`` rebinding can sneak a call past
        # the AST matcher above (the author namespace exposes gate as a bare global, so any such
        # binding would be a deliberate smuggle). The script does no aliasing of capabilities.
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert all(a.name != "gate" for a in n.names), "gate must never be imported/aliased"
            # a `g = gate` style rebinding of the bare global
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name):
                assert n.value.id != "gate", "gate must never be rebound to an alias"


def test_assembled_script_excludes_datetime_import() -> None:
    # The restricted import allowlist: no datetime/time (a timestamp must be passed via input).
    src = build_reconciler_script(repo="o/r")
    assert "import datetime" not in src
    assert "import time" not in src


def test_assembled_script_parses() -> None:
    ast.parse(build_reconciler_script(repo="o/r"))


def test_real_merge_defaults_off_advisory_mode() -> None:
    # WOULD-MERGE ADVISORY mode (chairman decision 2): real-merge is a single flag, default OFF.
    src = build_reconciler_script(repo="o/r")
    assert "REAL_MERGE = False" in src
    # and the advisory label is present (the seat clears it)
    assert "merge:would-merge-advisory" in src


def test_real_merge_flag_flips_on() -> None:
    src = build_reconciler_script(repo="o/r", real_merge=True)
    assert "REAL_MERGE = True" in src


# ════════════════════════════════════════════════════════════════════════════
# DEPLOY SURFACE — REQUIRED_TOOLS / REQUIRED_HTTP_SERVERS / WorkflowCreate.
# ════════════════════════════════════════════════════════════════════════════


def test_required_tools_is_the_full_dev_union() -> None:
    # The reconciler clones/edits (via the implement/fix child agents) AND reaches GitHub — the
    # same union as the monolith.
    got = {t.type for t in REQUIRED_TOOLS}
    assert got == {"bash", "read", "write", "edit", "glob", "grep", "http_request"}


def test_required_tools_have_no_duplicates() -> None:
    types = [t.type for t in REQUIRED_TOOLS]
    assert len(types) == len(set(types))


def test_single_github_server_with_full_rest_verbs() -> None:
    assert len(REQUIRED_HTTP_SERVERS) == 1
    server = REQUIRED_HTTP_SERVERS[0]
    assert server.name == "github"
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert len(repos) == 1
    # DELETE (unlabel), PATCH (#1208 close), PUT (merge) are all load-bearing.
    assert set(repos[0].methods or []) == {"GET", "POST", "PUT", "DELETE", "PATCH"}
    assert repos[0].allow_query is True


def test_graphql_route_present_for_mark_ready() -> None:
    server = REQUIRED_HTTP_SERVERS[0]
    assert {r.path_pattern for r in server.routes} == {"/repos/**", "/graphql"}


def test_workflow_create_is_valid_and_carries_surface() -> None:
    wc = build_reconciler_workflow_create(name="dev-pipeline-reconciler")
    assert isinstance(wc, WorkflowCreate)
    assert wc.name == "dev-pipeline-reconciler"
    assert wc.script == build_reconciler_script()
    assert {t.type for t in wc.tools} == {t.type for t in REQUIRED_TOOLS}
    assert len(wc.http_servers) == 1
    server = wc.http_servers[0]
    assert isinstance(server, HttpServerSpec)
    assert server.name == "github"


def test_workflow_create_forwards_real_merge_kwarg() -> None:
    wc = build_reconciler_workflow_create(name="r", real_merge=True)
    assert "REAL_MERGE = True" in wc.script


def test_workflow_create_github_server_name_threads_into_script() -> None:
    wc = build_reconciler_workflow_create(name="r", github_server="gh")
    assert any(isinstance(s, HttpServerSpec) and s.name == "gh" for s in wc.http_servers)
    assert "'gh'" in wc.script or '"gh"' in wc.script
