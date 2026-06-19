"""The dev-pipeline RECONCILER — one stateless serial state-reconciler (aios#49/#111).

The dev_pipeline monolith dissolves into a SINGLE serial state-reconciler: one stateless aios
cron workflow that owns every PR/issue transition off the GitHub blackboard. This is the
build of the design-of-record (``eumemic-company/architecture/dev-pipeline-reconciler-design.md``),
steps 1-5 of its build plan.

WHY (the verdict, verified at the source): the monolith parks each job in an ephemeral run via
``gate()`` (a durable ``SUSPENDED`` frame), and recovering an errored run is unbuildable today
(no ``run_completion`` triggers, a ``WorkflowAction`` fires a static template so it can't carry
per-issue context, an errored run's ``output`` is ``null``). So every stuck PR is a true ORPHAN.
Moving job-identity OUT of the ephemeral run and INTO the durable GitHub blackboard (the PR + its
labels), then replacing the one suspend-at-gate driver with a stateless pull-loop, dissolves the
entire orphan class. ``triage_pipeline.py`` already ships exactly this stateless-rescan-cron shape
in prod — the proven precedent, not a new pattern.

THE DESIGN (each pass):
1. Read the COMPLETE blackboard once — one snapshot (open issues + open PRs + their
   labels/CI/mergeable/draft), one GitHub-token read reused for the whole pass.
2. Compute ``owner()`` — a single deterministic TOTAL function over the blackboard tuple, fixed
   priority order, first-match-wins (disjoint by construction) with an ``else`` (total by
   construction). It returns AT MOST ONE (item, transition).
3. Pick the ONE highest-priority actionable item (lowest PR/issue number as tiebreak).
4. Drive EXACTLY ONE transition via the right SEQUENTIAL ``agent()`` call (``dev-implement`` for
   build [PR-FIRST: open the draft PR before the long implement agent], ``dev-review``+``dev-risk``
   for review, ``dev-fix`` for ci, ``dev-resolve`` (the fix agent) for rebase, merge-guard+PUT for
   merge), then EXIT. The next cron pass re-evaluates from scratch.

WHAT HOLDS (substrate-verified):
- ``gate()`` is BANNED in the loop. One ``gate()`` durably suspends the WHOLE run (the orphan bug
  relocated, not fixed). Escalation relocates to durable terminal ``needs:human/*`` labels read off
  the board — the queryable mailbox that replaces unrecoverable suspended runs.
- maker≠checker preserved STRUCTURALLY as DIFFERENT AGENT CALLS within the reconciler
  (``dev-implement`` on a build item ≠ ``dev-review`` on a review item). Different DECIDERS, which
  sequential calls satisfy; never required different concurrent processes.
- #1158 foreclosed by ``owner()`` REFUSING to emit ``merge:approved`` for tier>3 (it emits
  ``needs:human/merge-approval`` instead), AND the merge branch re-deriving the deterministic
  ``_risk_floor`` itself + treating ANY ``needs:human/*`` as a hard veto.
- WOULD-MERGE ADVISORY mode (chairman decision 2): in v1 the merge branch does NOT actually merge
  — it stamps an advisory label/comment the seat clears, until the post-merge checker (task #76)
  exists. Real-merge is a single flag (``real_merge``), default OFF.

RECOVERY IS INTRINSIC: the run is ephemeral, the job lives in durable state (PR + labels), the next
cron pass re-attaches by re-deriving state from scratch. No dead-man-for-coordination is needed.

SELF-RACE: a single serial actor has no concurrent claimant → no lease, no run_id, no clock, no
steal-check. The only residual is the reconciler racing ITSELF if a cron fire overlaps a still-
running prior pass. Because every transition is idempotent (the next pass re-derives from scratch),
an overlapping fire is at worst a harmless re-evaluation, not a double-act.

The exported builders return workflow *source code* (the ``dev_pipeline.py``/``triage_pipeline.py``
pattern): ``build_reconciler_script`` is the production workload authored into the runtime via
``aios workflows create``; ``build_reconciler_fixture_script`` is the CI variant with a tight scan
cap, driven by ``tests/integration/test_wf_dev_pipeline_reconciler_fixture.py`` against the host
with simulated agent/tool returns.

DETERMINISM: the script imports only ``re``/``json`` (the curated allowlist — NO ``datetime``/
``time``), keeps all capability I/O in the value domain, scans the blackboard in a fixed sorted
order, and emits capabilities in a deterministic order — so replay-with-memo is stable.
"""

from __future__ import annotations

from typing import Any

from aios.models.agents import HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.comment_idempotency import COMMENT_IDEMPOTENCY_HELPERS

# Reused VERBATIM from the monolith so the reconciler and dev_pipeline can never drift in the spec
# gate, the rebase exit codes, the fix hints, or what they require of the shared judgment agents.
# The schemas (IMPLEMENT/REVIEW/FIX/CI/RISK_SCHEMA) are imported inside ``_render_constants`` (they
# are only rendered into the script header, never referenced at this module's top level).
from aios.workflows.dev_pipeline import (
    FIX_CI_LINT_HINT,
    FIX_REBASE_HINT,
    MIN_BUG_BODY_WORDS,
    MIN_SPEC_BODY_WORDS,
    REBASE_EXIT_CONFLICT,
    REBASE_EXIT_DONE,
    REBASE_EXIT_ERROR,
    REBASE_EXIT_NOOP,
    RESOLUTION_SIGNALS,
    SPEC_BLOCKERS,
)
from aios.workflows.dev_pipeline_lib import DEV_PIPELINE_LIB
from aios.workflows.gh_body import GH_BODY_HELPERS

# ─── default judgment-node agent ids (override per deployment) ───────────────
# maker≠checker is preserved by these being DIFFERENT agent calls in the reconciler: the build
# transition uses ``dev-implement`` and the review transition uses ``dev-review`` — different
# deciders, which is what the chairman's maker≠checker requirement means.
DEFAULT_IMPLEMENT_AGENT_ID = "dev-implement"
DEFAULT_REVIEW_AGENT_ID = "dev-review"
DEFAULT_FIX_AGENT_ID = "dev-fix"
DEFAULT_CI_AGENT_ID = "dev-ci-watch"
DEFAULT_RISK_AGENT_ID = "dev-risk"

# ─── the routing label (step 6) ─────────────────────────────────────────────
# The reconciler ONLY touches PRs/issues carrying ``pipeline:v2``; the monolith no-op-skips them.
# Mechanically the two can't both act on the same item (reversible). A v2 build stamps it; the
# one-time adoption sweep (step 6, deferred to deploy) stamps existing open PRs.
LABEL_PIPELINE_V2 = "pipeline:v2"

# ─── the dispatch / claim labels (shared with the monolith + conveyor) ───────
# The dispatch gate is ``shovel-ready ∧ approved ∧ ¬dispatched`` (the two-axis issue-state model).
# ``dispatched`` is the in-flight claim the reconciler stamps when it opens the PR for a build
# (PR-FIRST), so a re-scan never re-builds an in-flight issue.
LABEL_SHOVEL_READY = "shovel-ready"
LABEL_APPROVED = "approved"
LABEL_DISPATCHED = "dispatched"
# Stamped on the issue once the implement agent has produced a diff (the design's row-1 ``→
# autodev:built`` marker). It is the at-least-once dedup for the implement agent itself: a build
# re-driven by a crash mid-implement re-reads this marker (and the PR branch's commit count) and
# SKIPS the long, expensive implement agent rather than re-building / re-pushing a second time.
LABEL_BUILT = "autodev:built"

# ─── the review-claim labels (the reviewed-green stamp + risk tier) ──────────
# ``reviewed:green@<sha>`` records that the review transition has run for a specific head sha; the
# review branch is re-picked ONLY while it is absent (or stamped for a stale sha), so a re-scan
# never re-reviews an already-reviewed-at-this-head PR. ``risk:tier-N`` is single-valued (any prior
# tier label is deleted before the new one is stamped). ``merge:approved`` is the green light the
# merge branch acts on; the review branch stamps it LAST (escalation-first, approval-last).
LABEL_REVIEWED_PREFIX = "reviewed:green@"
LABEL_RISK_PREFIX = "risk:tier-"
LABEL_MERGE_APPROVED = "merge:approved"

# ─── the would-merge advisory label (WOULD-MERGE ADVISORY mode, chairman dec. 2) ──
# In v1 (``real_merge`` OFF) the merge branch does NOT merge — it stamps this advisory the seat
# clears, until the post-merge checker (task #76) exists. maker≠checker is structural separation but
# NOT detection-independence (reviewer and builder are the same correlated LLM family); the only
# check that reaches PAST the merge boundary (the #1188 post-merge-regression class) is #76. So the
# reconciler does not auto-merge until #76 — it advises, and the seat is the uncorrelated checker.
LABEL_WOULD_MERGE = "merge:would-merge-advisory"

# ─── the terminal needs:human/* mailbox (the relocated escalation) ───────────
# ``gate()`` is BANNED in the loop. Every escalation becomes a DURABLE terminal label read off the
# board — the queryable mailbox that replaces an unrecoverable suspended run. ``owner()`` returns
# None for ANY ``needs:human/*`` (a hard veto, terminal, never re-picked) so a human-gated item
# can't freeze a loop (unlike a suspended ``gate()``).
NEEDS_HUMAN_PREFIX = "needs:human/"
NEEDS_HUMAN_SPEC = "needs:human/spec"
NEEDS_HUMAN_BUILD = "needs:human/build"
NEEDS_HUMAN_CI = "needs:human/ci"
NEEDS_HUMAN_REBASE = "needs:human/rebase"
NEEDS_HUMAN_VERIFY = "needs:human/verify"
NEEDS_HUMAN_MERGE_GUARD = "needs:human/merge-guard"
NEEDS_HUMAN_MERGE_APPROVAL = "needs:human/merge-approval"
NEEDS_HUMAN_REQUIRED_SET = "needs:human/required-set"
NEEDS_HUMAN_LIVELOCK = "needs:human/livelock"
NEEDS_HUMAN_STUCK = "needs:human/stuck"

# ─── the per-PR livelock lap counter (step 4) ────────────────────────────────
# A durable ``pipeline:laps:<n>`` label incremented each time the reconciler re-drives a PR that has
# CYCLED (rebase↔ci↔review↔merge-strip). Independent of per-stage retry caps (each lap is a fresh
# first-attempt on a NEW sha, so per-stage caps never fire). At CAP → ``needs:human/livelock`` —
# closes the master-moves livelock the dead-man can't catch (the item "progresses" every interval).
LABEL_LAPS_PREFIX = "pipeline:laps:"
MAX_LAPS = 8

# ─── failure-surfacing labels (shared with the monolith) ─────────────────────
LABEL_IN_PROGRESS = "autodev:in-progress"

# Stable heading markers on the chairman-facing comments the reconciler posts. Each heading is BOTH
# the comment's first line AND the maker-marker the replay guard scans for (aios#1292): a posted
# comment is its own "already done" marker on the next read, so an at-least-once replay never
# duplicates it.
MARKER_SPEC_NOT_READY = "## Spec not ready for implementation"
MARKER_RISK_ASSESSMENT = "## Risk Assessment"
MARKER_MERGE_GUARD = "## Merge guard refused"
MARKER_REBASE_CONFLICT = "## Rebase conflict — branch could not be healed"
MARKER_WOULD_MERGE = "## WOULD-MERGE (advisory — auto-merge held until the post-merge checker)"
MARKER_STUCK = "## Reconciler: unforeseen state (needs:human/stuck)"

# The #1158 control: the merge branch (and ``owner()``) refuse ``merge:approved`` for any tier
# strictly above this — those route to ``needs:human/merge-approval``. tier≤AUTO_MERGE_MAX_TIER
# clears mechanically. (The deterministic ``_risk_floor`` floors CI-workflow/missing-files PRs to
# tier-4, structurally above this, so they always park for human review.)
AUTO_MERGE_MAX_TIER = 3


def _py(name: str, value: Any) -> str:
    """One ``NAME = <repr>`` constant line for the prepended header (mirrors dev_pipeline)."""
    return f"{name} = {value!r}"


def _render_constants(
    *,
    implement_agent_id: str,
    review_agent_id: str,
    fix_agent_id: str,
    ci_agent_id: str,
    risk_agent_id: str,
    github_server: str,
    base_branch: str,
    merge_sentinels: list[str],
    max_ci_iters: int,
    max_review_iters: int,
    max_rebase_attempts: int,
    auto_merge_max_tier: int,
    max_laps: int,
    max_items_per_pass: int,
    merge_method: str,
    real_merge: bool,
    default_model: str | None,
) -> str:
    # Re-import the verbatim schemas/spec-gate constants from dev_pipeline at render time (they
    # were ``del``'d at module top so they don't read as locals; importing here keeps ONE source).
    from aios.workflows.dev_pipeline import (
        CI_SCHEMA,
        FIX_SCHEMA,
        IMPLEMENT_SCHEMA,
        REVIEW_SCHEMA,
        RISK_SCHEMA,
    )

    lines = [
        _py("IMPLEMENT_AGENT_ID", implement_agent_id),
        _py("REVIEW_AGENT_ID", review_agent_id),
        _py("FIX_AGENT_ID", fix_agent_id),
        _py("CI_AGENT_ID", ci_agent_id),
        _py("RISK_AGENT_ID", risk_agent_id),
        _py("GITHUB_SERVER", github_server),
        _py("BASE_BRANCH", base_branch),
        _py("MERGE_SENTINELS", list(merge_sentinels)),
        _py("MAX_CI_ITERS", max_ci_iters),
        _py("MAX_REVIEW_ITERS", max_review_iters),
        _py("MAX_REBASE_ATTEMPTS", max_rebase_attempts),
        _py("AUTO_MERGE_MAX_TIER", auto_merge_max_tier),
        _py("MAX_LAPS", max_laps),
        _py("MAX_ITEMS_PER_PASS", max_items_per_pass),
        _py("MERGE_METHOD", merge_method),
        _py("REAL_MERGE", real_merge),
        _py("DEFAULT_MODEL", default_model),
        # spec-gate constants (reused verbatim — the lib's spec_ok references them)
        _py("MIN_SPEC_BODY_WORDS", MIN_SPEC_BODY_WORDS),
        _py("MIN_BUG_BODY_WORDS", MIN_BUG_BODY_WORDS),
        _py("SPEC_BLOCKERS", list(SPEC_BLOCKERS)),
        _py("RESOLUTION_SIGNALS", list(RESOLUTION_SIGNALS)),
        # rebase exit codes (the lib's _rebase_command / _sync_branch reference them)
        _py("REBASE_EXIT_DONE", REBASE_EXIT_DONE),
        _py("REBASE_EXIT_NOOP", REBASE_EXIT_NOOP),
        _py("REBASE_EXIT_CONFLICT", REBASE_EXIT_CONFLICT),
        _py("REBASE_EXIT_ERROR", REBASE_EXIT_ERROR),
        _py("FIX_CI_LINT_HINT", FIX_CI_LINT_HINT),
        _py("FIX_REBASE_HINT", FIX_REBASE_HINT),
        # the reconciler's own routing / claim / escalation / advisory labels + markers
        _py("LABEL_PIPELINE_V2", LABEL_PIPELINE_V2),
        _py("LABEL_SHOVEL_READY", LABEL_SHOVEL_READY),
        _py("LABEL_APPROVED", LABEL_APPROVED),
        _py("LABEL_DISPATCHED", LABEL_DISPATCHED),
        _py("LABEL_BUILT", LABEL_BUILT),
        _py("LABEL_IN_PROGRESS", LABEL_IN_PROGRESS),
        _py("LABEL_REVIEWED_PREFIX", LABEL_REVIEWED_PREFIX),
        _py("LABEL_RISK_PREFIX", LABEL_RISK_PREFIX),
        _py("LABEL_MERGE_APPROVED", LABEL_MERGE_APPROVED),
        _py("LABEL_WOULD_MERGE", LABEL_WOULD_MERGE),
        _py("LABEL_LAPS_PREFIX", LABEL_LAPS_PREFIX),
        _py("NEEDS_HUMAN_PREFIX", NEEDS_HUMAN_PREFIX),
        _py("NEEDS_HUMAN_SPEC", NEEDS_HUMAN_SPEC),
        _py("NEEDS_HUMAN_BUILD", NEEDS_HUMAN_BUILD),
        _py("NEEDS_HUMAN_CI", NEEDS_HUMAN_CI),
        _py("NEEDS_HUMAN_REBASE", NEEDS_HUMAN_REBASE),
        _py("NEEDS_HUMAN_VERIFY", NEEDS_HUMAN_VERIFY),
        _py("NEEDS_HUMAN_MERGE_GUARD", NEEDS_HUMAN_MERGE_GUARD),
        _py("NEEDS_HUMAN_MERGE_APPROVAL", NEEDS_HUMAN_MERGE_APPROVAL),
        _py("NEEDS_HUMAN_REQUIRED_SET", NEEDS_HUMAN_REQUIRED_SET),
        _py("NEEDS_HUMAN_LIVELOCK", NEEDS_HUMAN_LIVELOCK),
        _py("NEEDS_HUMAN_STUCK", NEEDS_HUMAN_STUCK),
        _py("MARKER_SPEC_NOT_READY", MARKER_SPEC_NOT_READY),
        _py("MARKER_RISK_ASSESSMENT", MARKER_RISK_ASSESSMENT),
        _py("MARKER_MERGE_GUARD", MARKER_MERGE_GUARD),
        _py("MARKER_REBASE_CONFLICT", MARKER_REBASE_CONFLICT),
        _py("MARKER_WOULD_MERGE", MARKER_WOULD_MERGE),
        _py("MARKER_STUCK", MARKER_STUCK),
        _py("IMPLEMENT_SCHEMA", IMPLEMENT_SCHEMA),
        _py("REVIEW_SCHEMA", REVIEW_SCHEMA),
        _py("FIX_SCHEMA", FIX_SCHEMA),
        _py("CI_SCHEMA", CI_SCHEMA),
        _py("RISK_SCHEMA", RISK_SCHEMA),
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# owner() — THE LOAD-BEARING TOTAL FUNCTION (step 1)
# ════════════════════════════════════════════════════════════════════════════
#
# Authored ONCE as a shared source string (the DEV_PIPELINE_LIB pattern) so the SAME text is
# spliced into the reconciler script AND exec'd directly by the completeness/idempotency property-
# tests. ``owner()`` is PURE and SYNCHRONOUS — it operates only on a normalized blackboard tuple
# already read by the caller, reads no clock, performs no I/O, and emits no capability — so the
# property-test can call it across the entire enumerated cross-product without driving the host.
#
# CONTRACT (the whole correctness guarantee):
#   * Returns AT MOST ONE (item, transition) — first-match-wins over a FIXED priority order makes
#     the branches disjoint by construction, so ≤1 action ever fires per item.
#   * TOTAL over the enumerated (ci-status x mergeable_state x draft x approval x human-review x
#     closed) cross-product — the final ``else -> escalate`` row makes a tuple no branch foresaw
#     DETECTABLE (it escalates to needs:human/stuck), never a silent gap. This is the structural
#     guarantee the gate()-suspending monolith lacked.
#
# A "blackboard item" is the normalized dict the caller builds from one open issue OR open PR:
#   {kind: "issue"|"pr", number: int, labels: frozenset[str], closed: bool,
#    # issue-only: approved, shovel_ready, dispatched, has_open_pr, spec_blocked
#    # pr-only: draft, ci_verdict ("green"|"red"|"pending"|"no_ci"|"unknown"), needs_rebase,
#    #          human_blocked (requested-changes / CODEOWNERS / human-commits-after-claim),
#    #          reviewed_for_head, risk_tier, head_sha }
# The transition the owner returns is a string the driver dispatches: "build" | "rebase" | "ci" |
# "review" | "merge" | "escalate" | None (terminal — no action).
RECONCILER_OWNER = r'''
# ─── label predicates (the durable blackboard the reconciler reads) ──────────

def _has_needs_human(labels):
    """True if ANY ``needs:human/*`` label is present — a HARD VETO. ``owner()`` returns None
    for such an item: terminal, never re-picked, can't freeze a loop (unlike a suspended
    gate()). The seat clears it by removing the label."""
    return any(isinstance(n, str) and n.startswith(NEEDS_HUMAN_PREFIX) for n in labels)


def _reviewed_label_for(labels, head_sha):
    """The ``reviewed:green@<sha>`` label present for THIS head sha, or None. The review branch
    is re-picked ONLY while no such label exists, so a re-scan never re-reviews an already-
    reviewed-at-this-head PR (idempotency). A reviewed-for-a-STALE-sha PR (head advanced) is
    NOT reviewed-for-head → it is re-picked for review on the new sha."""
    if not (isinstance(head_sha, str) and head_sha):
        return None
    want = LABEL_REVIEWED_PREFIX + head_sha
    for n in labels:
        if n == want:
            return n
    return None


# ─── owner(): the single total deterministic function ────────────────────────

def owner(item):
    """The load-bearing total function. Over a normalized blackboard ``item`` (an open issue OR an
    open PR), return AT MOST ONE ``(item, transition)`` in a FIXED priority order (first-match-wins
    = disjoint), with a final ``else`` (= total). The driver then dispatches the one transition.

    Priority order (the design-of-record's 13-state map, collapsed to disjoint branches):

      0. TERMINAL / needs:human/*  → None  (hard veto; never re-picked) — the #1158 control and
         every escalation mailbox live here; a terminal label can't freeze a loop.
      1. ISSUE build-eligible (shovel-ready ∧ approved ∧ ¬dispatched ∧ no-open-PR ∧ ¬needs:human)
         → "build"  (PR-FIRST: the driver opens the draft PR before the long implement agent).
         A non-build-eligible issue (needs-design/needs-decision/blocked/already-dispatched) is
         terminal here (None) — deliberately not dispatch-eligible by THIS reconciler.
      2. PR human-owned (requested-changes / CODEOWNERS / human commits after claim) → None
         (the reconciler refuses to fight the human; pipeline:v2 is a routing hint, NOT a firewall).
      3. PR needs rebase (conflicting/behind — checked FIRST among PR states; a conflicting PR
         can't even run CI) → "rebase".
      4. PR ci-red (mergeable ∧ verdict red) → "ci".
      5. PR ci pending / no verdict yet / unknown mergeable → None (WAITING; re-evaluated next
         pass; the dead-man owns liveness past the horizon — out of owner()'s scope).
      6. PR ci-green ∧ not-yet-reviewed-for-this-head → "review" (the checker call — a DIFFERENT
         agent from build; maker≠checker).
      7. PR reviewed-green-for-head ∧ ¬merge:approved → "review" re-drive ONLY if the review
         stamp is missing; if reviewed-for-head AND tier>AUTO_MERGE_MAX_TIER the review branch
         already routed it to needs:human/merge-approval (caught at branch 0). A reviewed PR that
         still lacks merge:approved AND lacks needs:human/* means the review transition has not
         completed its label writes yet → re-pick "review" (idempotent: re-stamps the same labels).
      8. PR merge:approved (∧ reviewed-for-head ∧ tier≤AUTO_MERGE_MAX_TIER) → "merge"  (the effector
         branch; never reviews; #1158 tier-gate already enforced — but re-checked here defensively).
      9. else → "escalate"  (the relocated gap: a tuple no branch foresaw is DETECTABLE within one
         pass, not a silent freeze).
    """
    labels = item.get("labels") or frozenset()

    # 0. TERMINAL / needs:human/* — the hard veto. FIRST so a human-gated item is NEVER re-picked.
    if item.get("closed"):
        return None
    if _has_needs_human(labels):
        return None

    kind = item.get("kind")

    if kind == "issue":
        # 1. build-eligible issue → PR-FIRST build. The dispatch gate is the two-axis model:
        # shovel-ready ∧ approved ∧ ¬dispatched, plus no PR already open for it.
        if (item.get("shovel_ready") and item.get("approved")
                and not item.get("dispatched")
                and not item.get("has_open_pr")):
            return (item, "build")
        # Any other issue (needs-design/needs-decision/blocked/already-dispatched/has-open-PR) is
        # terminal for THIS reconciler — not dispatch-eligible. (An already-dispatched issue's work
        # lives on its PR, which the PR branches below own.)
        return None

    if kind == "pr":
        # 2. human-owned PR — the reconciler refuses to act (does not fight the human).
        if item.get("human_blocked"):
            return None

        # 3. needs rebase FIRST among PR states — a conflicting/behind PR can't run CI.
        if item.get("needs_rebase"):
            return (item, "rebase")

        verdict = item.get("ci_verdict")
        head_sha = item.get("head_sha")

        # 4. ci-red (mergeable, not conflicting) → fix.
        if verdict == "red":
            return (item, "ci")

        # 5. ci pending / no verdict / unknown mergeable → WAITING (no actor; re-evaluated next
        # pass). null/unknown is "not yet," never "clean."
        if verdict in ("pending", "unknown"):
            return None

        # verdict is now "green" or "no_ci" (a trusted-able verdict).
        if verdict in ("green", "no_ci"):
            reviewed = _reviewed_label_for(labels, head_sha)
            approved = LABEL_MERGE_APPROVED in labels
            # 8. merge:approved → merge (effector). Re-check the tier gate defensively: a
            # merge:approved that somehow carries tier>cap (it never should — the review branch
            # refuses to stamp it) is NOT merged; it falls through to escalate.
            if approved:
                tier = item.get("risk_tier")
                if not isinstance(tier, int) or tier <= AUTO_MERGE_MAX_TIER:
                    return (item, "merge")
                return (item, "escalate")
            # 6/7. green ∧ not-yet-(or stale-)reviewed-for-head, OR reviewed-for-head but the
            # approval label-writes have not completed → (re-)review. Idempotent: a re-driven
            # review re-stamps the SAME reviewed:green@<sha> + risk:tier-N + (advisory|approval),
            # all additive/single-valued, so a re-scan over an already-reviewed item that simply
            # hasn't reached merge yet is a harmless no-op re-stamp — NOT double work, because
            # once merge:approved lands the merge branch (8) owns it, and once tier>cap lands the
            # needs:human/merge-approval label owns it (caught at branch 0).
            if reviewed is None:
                return (item, "review")
            # reviewed-for-head, no merge:approved, no needs:human/* (caught at 0): the review
            # transition stamped the reviewed label but its terminal approval label-write did not
            # land (a crash between writes). Re-pick review to complete it (idempotent).
            return (item, "review")

        # 9. else (an unforeseen verdict string) → escalate (the relocated gap; detectable).
        return (item, "escalate")

    # 9. else (an unforeseen kind) → escalate.
    return (item, "escalate")


def select_actionable(items):
    """The ONE highest-priority actionable item, lowest number as tiebreak. Scans every blackboard
    item through ``owner()``, keeps those with a non-None transition, and returns the FIRST in a
    deterministic priority-then-number order — so the reconciler drives EXACTLY ONE per pass.

    Priority rank (lower = sooner): rebase < ci < review < merge < build < escalate. Rebase first
    (a conflicting PR blocks everything downstream), then the in-flight PR ladder, then new builds,
    then the catch-all escalate. Within a rank, lowest number wins (stable, deterministic)."""
    rank = {"rebase": 0, "ci": 1, "review": 2, "merge": 3, "build": 4, "escalate": 5}
    actionable = []
    for it in items:
        decision = owner(it)
        if decision is None:
            continue
        _item, transition = decision
        actionable.append((rank.get(transition, 99), int(it.get("number", 0)), it, transition))
    if not actionable:
        return None
    actionable.sort(key=lambda t: (t[0], t[1]))
    _r, _n, it, transition = actionable[0]
    return (it, transition)
'''


# The static script body — references the prepended constants AND the spliced ``owner()`` /
# ``select_actionable`` (RECONCILER_OWNER) + the shared lib helpers (DEV_PIPELINE_LIB). Pure stdlib
# (re/json), value-domain I/O, bounded loops, no datetime/time: replay-stable by construction.
_BODY = r'''
import json
import re

# NOTE: the proven scripted/agentic helpers (spec gate, gh plumbing, SHA checks, merge-guard +
# rebase, risk floor, label/close plumbing, _ci_verdict/_shas_equal, _watch_ci/_sync_branch) live
# in ``dev_pipeline_lib.DEV_PIPELINE_LIB`` and are spliced in below by ``build_reconciler_script``.
# The total function ``owner()`` + ``select_actionable()`` live in ``RECONCILER_OWNER`` and are
# spliced in too. Module-level defs resolve names at CALL time, so this state machine can call them
# freely even though their ``def``s are appended after this block.


# ─── blackboard normalization (one snapshot → normalized items owner() reads) ──

def _label_names(obj):
    """The set of label names on an issue/PR. GitHub returns labels as objects with a ``name``
    field (or, defensively, bare strings)."""
    names = set()
    if not isinstance(obj, dict):
        return names
    for lab in obj.get("labels") or []:
        if isinstance(lab, dict):
            n = lab.get("name")
            if isinstance(n, str):
                names.add(n)
        elif isinstance(lab, str):
            names.add(lab)
    return names


def _is_pull_request(issue):
    """True if this issues-endpoint row is actually a PR (the issues endpoint returns both)."""
    return isinstance(issue, dict) and "pull_request" in issue


def _classify_ci_verdict(ci):
    """Map a ``_ci_verdict`` outcome ('pass'|'red'|'retry') + raw status onto the owner()'s
    verdict vocabulary. 'pass' over a green → 'green'; 'pass' over a no_ci → 'no_ci'; 'red' →
    'red'; 'retry' (unverifiable / premature) → 'pending' (WAITING, re-evaluated next pass)."""
    raw = ci.get("status") if isinstance(ci, dict) else None
    if raw == "red":
        return "red"
    if raw == "no_ci":
        return "no_ci"
    if raw == "green":
        return "green"
    return "pending"


async def _read_pr_blackboard(repo, pr):
    """Normalize ONE open PR into the blackboard item ``owner()`` reads. Reads the live mergeable
    fields (already on the listed PR object where present; we trust the list payload's
    ``mergeable_state`` and re-read ``GET /pulls/{n}`` only for the fields the list omits), the CI
    verdict (script-verified via the shared ``_ci_verdict``), the review stamp, the risk tier, and
    the human-ownership signals. PURE-ish: only GitHub reads, no mutation."""
    number = int(pr.get("number", 0))
    labels = frozenset(_label_names(pr))
    head_sha = _pr_head_sha(pr)
    # TERMINAL short-circuit: a PR carrying any ``needs:human/*`` (or closed) is vetoed by owner()
    # FIRST, so it is never actionable. Skip the expensive normalization (the CI watch agent + the
    # mergeable/reviews reads) entirely — spending a watch_ci agent on a human-gated PR every pass
    # is pure waste, and it re-arms the floodgates this design exists to avoid. Return the minimal
    # terminal item; owner() returns None for it regardless of the unread fields.
    if any(isinstance(n, str) and n.startswith(NEEDS_HUMAN_PREFIX) for n in labels) \
            or pr.get("state") == "closed":
        return {"kind": "pr", "number": number, "labels": labels,
                "closed": pr.get("state") == "closed", "draft": bool(pr.get("draft")),
                "head_sha": head_sha, "needs_rebase": False, "ci_verdict": "unknown",
                "human_blocked": False, "risk_tier": _risk_tier_from_labels(labels),
                "branch": (pr.get("head") or {}).get("ref", ""),
                "node_id": pr.get("node_id", ""), "html_url": pr.get("html_url", "")}
    # Re-read the single PR for the authoritative mergeable/draft/state (the list payload can omit
    # ``mergeable``). A read failure degrades to the list payload's fields (fail-safe for SAFETY: an
    # unknown mergeable is WAITING, never "clean"). LIVENESS note: a PERSISTENTLY-failing single-PR
    # read pins the PR in WAITING/unknown every pass with no escalation here — that slow-stall is
    # caught ONLY by the Step-8 dead-man (task #104), which MUST treat 'ci_verdict==unknown for N
    # consecutive passes' as flaggable. Until the dead-man is live this degrade is the one uncovered
    # slow-orphan path; it is intentional (re-reading every pass self-heals a transient).
    full = _json_body(await gh("GET", _ipath(repo, "/pulls/%d" % number)))
    pr_full = full if isinstance(full, dict) else pr
    head_sha = _pr_head_sha(pr_full) or head_sha
    draft = bool(pr_full.get("draft"))
    closed = pr_full.get("state") == "closed"
    needs_rebase = _needs_rebase(pr_full)
    # CI verdict: dispatch the watch agent for THIS head, then script-verify it (uncorrelated).
    # A conflicting PR (needs_rebase) skips the CI read — rebase owns it first (and a conflicting
    # PR can't compute refs/pull/N/merge anyway), so we don't spend a watch agent on it.
    verdict = "unknown"
    if not needs_rebase and not draft:
        verdict = await _read_ci_verdict(repo, number, head_sha)
    # Human ownership: a requested-changes / CODEOWNERS review, or human commits after the claim.
    human_blocked = await _pr_human_blocked(repo, number, labels)
    # Risk tier from the single-valued risk:tier-N label, if the review branch already stamped it.
    risk_tier = _risk_tier_from_labels(labels)
    return {
        "kind": "pr", "number": number, "labels": labels, "closed": closed,
        "draft": draft, "head_sha": head_sha, "needs_rebase": needs_rebase,
        "ci_verdict": verdict, "human_blocked": human_blocked, "risk_tier": risk_tier,
        "branch": (pr_full.get("head") or {}).get("ref", ""),
        "node_id": pr_full.get("node_id", ""),
        "html_url": pr_full.get("html_url", ""),
    }


async def _read_ci_verdict(repo, pr_number, head_sha):
    """One watch_ci agent call for the PR head, script-verified via the shared ``_ci_verdict``
    (the false-green guard #1392: a green/no_ci is trusted only when the polled commit IS the live
    head and the required-check set concluded). Returns the owner() verdict vocabulary."""
    try:
        ci = await agent(
            {"task": "watch_ci", "repo": repo, "pr_number": pr_number, "head_sha": head_sha},
            agent_id=CI_AGENT_ID, output_schema=CI_SCHEMA, model=_MODEL,
            label="scan-ci-%d" % pr_number)
    except AgentError as exc:
        log("scan-ci agent error on PR #%d -> pending:" % pr_number, exc)
        return "pending"
    trusted = await _ci_verdict(repo, pr_number, ci, head_sha)
    if trusted == "pass":
        return _classify_ci_verdict(ci)  # "green" or "no_ci"
    if trusted == "red":
        return "red"
    return "pending"  # "retry" -> WAITING


async def _pr_human_blocked(repo, pr_number, labels):
    """True if a HUMAN owns this PR: an active requested-changes review. The reconciler refuses to
    act on a human-owned PR (it does not fight the human; ``pipeline:v2`` is a routing hint, NOT a
    human firewall).

    FAILS CLOSED on an unreadable reviews response. ``human_blocked`` is the ONLY thing standing
    between the reconciler and a PR a human is mid-review on — so a non-200 / error-envelope /
    non-list response must NOT read as 'no block' (the fail-OPEN hole that, with REAL_MERGE on,
    would let a transient reviews-API blip merge OVER an active requested-changes review). On an
    unreadable read we return True (treat as human-owned THIS pass → owner() leaves it WAITING); the
    next pass re-reads and clears it if the read succeeds with no block — a one-pass defer, never a
    permanent freeze (the read is re-derived every pass)."""
    resp = await gh("GET", _ipath(repo, "/pulls/%d/reviews" % pr_number))
    if _status(resp) != 200:
        log("human-blocked: reviews read for PR #%d returned %r -> fail closed (treat as blocked)"
            % (pr_number, _status(resp)))
        return True
    reviews = _json_body(resp)
    if not isinstance(reviews, list):
        log("human-blocked: reviews body for PR #%d not a list -> fail closed (treat as blocked)"
            % pr_number)
        return True
    # GitHub returns the full review history; the LATEST review per author is authoritative. A
    # CHANGES_REQUESTED that a later APPROVED/DISMISSED superseded is not active.
    latest_by_author = {}
    for r in reviews:
        if not isinstance(r, dict):
            continue
        author = (r.get("user") or {}).get("login", "")
        state = r.get("state", "")
        if state in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
            latest_by_author[author] = state
    return any(state == "CHANGES_REQUESTED" for state in latest_by_author.values())


def _risk_tier_from_labels(labels):
    """The single-valued risk tier from a ``risk:tier-N`` label, or None if unstamped. The review
    branch keeps it single-valued (deletes any prior tier label before stamping the new one)."""
    for n in labels:
        if isinstance(n, str) and n.startswith(LABEL_RISK_PREFIX):
            try:
                return int(n[len(LABEL_RISK_PREFIX):])
            except (ValueError, TypeError):
                continue
    return None


def _build_eligible_issue(repo, issue, open_prs):
    """Normalize ONE open issue into the blackboard item ``owner()`` reads. The build gate is the
    two-axis model: ``shovel-ready ∧ approved ∧ ¬dispatched`` AND no PR already open for it (the PR,
    if any, is the live job; the issue is then terminal for the issue-branch)."""
    number = int(issue.get("number", 0))
    labels = frozenset(_label_names(issue))
    closed = issue.get("state") == "closed"
    # ``has_open_pr`` is conservative: any open PR whose branch references this issue number, OR
    # whose head branch is the conventional ``issue-<n>`` / ``*-<n>`` shape. We can't always know
    # the linkage, so a dispatched issue (its claim label) is the primary "already in flight" guard;
    # has_open_pr is the backstop for a PR opened without the dispatched stamp landing.
    has_open_pr = _issue_has_open_pr(number, open_prs)
    return {
        "kind": "issue", "number": number, "labels": labels, "closed": closed,
        "shovel_ready": LABEL_SHOVEL_READY in labels,
        "approved": LABEL_APPROVED in labels,
        "dispatched": LABEL_DISPATCHED in labels,
        "has_open_pr": has_open_pr,
        "title": issue.get("title", ""), "body": issue.get("body", ""),
    }


def _issue_has_open_pr(issue_number, open_prs):
    """Whether an open PR already exists for this issue (its branch name carries the issue number).
    Conservative substring/suffix match on the head ref — a dispatched issue is the primary guard;
    this is the backstop for a PR opened before the ``dispatched`` stamp landed."""
    needle = str(issue_number)
    for pr in open_prs:
        ref = ((pr.get("head") or {}).get("ref") or "") if isinstance(pr, dict) else ""
        # match issue-<n>, <n> as a trailing token, or /<n> boundary — bounded by word edges
        if re.search(r"(^|[^0-9])%s($|[^0-9])" % re.escape(needle), ref):
            return True
    return False


# ─── the lap counter (step 4) ─────────────────────────────────────────────────

def _current_laps(labels):
    """The current ``pipeline:laps:<n>`` count from the labels (0 if unstamped, the MAX if several
    slipped in). A durable per-PR counter incremented each time the reconciler re-drives a PR that
    has CYCLED — independent of per-stage retry caps (each lap is a fresh first-attempt on a new
    sha, so per-stage caps never fire)."""
    best = 0
    for n in labels:
        if isinstance(n, str) and n.startswith(LABEL_LAPS_PREFIX):
            try:
                best = max(best, int(n[len(LABEL_LAPS_PREFIX):]))
            except (ValueError, TypeError):
                continue
    return best


async def _bump_lap(repo, pr_number, labels):
    """Increment the durable ``pipeline:laps:<n>`` label (delete the old, add the new). At CAP →
    stamp ``needs:human/livelock`` (terminal; owner() then vetoes the PR). Returns True if the PR
    was livelocked (the caller must NOT then drive a transition on it). Idempotent: re-deleting an
    absent old label is a 404 no-op, re-adding the same new label is additive."""
    current = _current_laps(labels)
    new = current + 1
    if new >= MAX_LAPS:
        log("PR #%d hit livelock cap (%d laps) -> needs:human/livelock" % (pr_number, new))
        await _label(repo, pr_number, NEEDS_HUMAN_LIVELOCK)
        return True
    if current > 0:
        await _unlabel(repo, pr_number, "%s%d" % (LABEL_LAPS_PREFIX, current))
    await _label(repo, pr_number, "%s%d" % (LABEL_LAPS_PREFIX, new))
    return False


# ─── the transitions (one per pass; each idempotent under at-least-once replay) ──

async def _do_build(repo, item):
    """BUILD transition — PR-FIRST: open the draft PR (job-identity) BEFORE the long implement
    agent, so there is NO ``dispatched``-with-no-PR window ever. Stamp ``dispatched`` + ``pipeline:
    v2`` + ``autodev:in-progress`` (the claim), run the scripted spec gate, then the implement agent,
    then push to the PR branch. Idempotent: a re-drive that finds the PR already open ADOPTS it
    (PR-first dedup) and re-runs the implement agent against the same branch (the agent's own clone
    is fresh each time)."""
    number = item["number"]
    body = item.get("body", "")
    title = item.get("title", "")
    # The conventional branch for this issue — the implement agent pushes here, and the PR-first
    # open targets it. (The implement agent may return a different branch; we adopt whatever it
    # reports, but the PR-first open uses this deterministic name so the dedup is stable.)
    branch = "dev-pipeline/issue-%d" % number

    # Scripted pre-flight spec gate (over body + comments). A failure stamps needs:human/spec and
    # exits — NO build agent is spent on an underspecified issue.
    comments = await gh_paginated(_ipath(repo, "/issues/%d/comments" % number))
    if not isinstance(comments, list):
        comments = []
    comment_bodies = _comment_texts(comments)
    issue_obj = {"body": body, "title": title}
    ok, reason = spec_ok(issue_obj, "issue", comments)
    if not ok:
        log("build: spec gate failed on #%d:" % number, reason)
        await _label(repo, number, "underspecified")
        await post_comment_once(repo, number, MARKER_SPEC_NOT_READY,
                                MARKER_SPEC_NOT_READY + "\n\n" + reason, comments)
        await _label(repo, number, NEEDS_HUMAN_SPEC)
        return {"transition": "build", "number": number, "result": "spec_failed",
                "reason": reason}

    # PR-FIRST: claim the issue (dispatched + v2 + in-progress) and open the draft PR BEFORE the
    # long implement agent. The claim labels keep a re-scan from re-building the in-flight issue.
    await _label(repo, number, LABEL_DISPATCHED)
    await _label(repo, number, LABEL_PIPELINE_V2)
    await _label(repo, number, LABEL_IN_PROGRESS)
    pr = await _open_or_adopt_pr(repo, branch, number, title)
    if pr is None:
        log("build: could not open/adopt a draft PR for #%d -> needs:human/build" % number)
        await _label(repo, number, NEEDS_HUMAN_BUILD)
        return {"transition": "build", "number": number, "result": "no_pr"}
    pr_number = int(pr["number"])
    # Stamp pipeline:v2 on the PR too so the reconciler owns it on the next pass.
    await _label(repo, pr_number, LABEL_PIPELINE_V2)

    # IMPLEMENT-AGENT DEDUP (the at-least-once boundary). The implement agent is the single most
    # expensive op in the system; a crash AFTER it pushed commits but BEFORE its step result was
    # journaled would replay _do_build and re-run it (a double-build / re-push). agent() exposes no
    # idempotency key, so we dedup on DURABLE evidence instead: if the issue already carries
    # `autodev:built` OR the PR branch already has commits beyond base, the implement already ran —
    # SKIP it and let the normal CI/review ladder take the PR forward on the next pass.
    if LABEL_BUILT in (item.get("labels") or frozenset()) or await _pr_has_commits(repo, pr_number):
        log("build: #%d already built (marker/commits present) -> skipping implement re-run" % number)
        await _label(repo, number, LABEL_BUILT)
        return {"transition": "build", "number": number, "pr_number": pr_number,
                "result": "already_built"}

    # The implement agent (maker). An error stamps needs:human/build (the relocated escalation —
    # NOT a gate). A no-return AgentNoReturnError is also an AgentError (caught here).
    try:
        impl = await agent(
            {"task": "implement", "repo": repo, "issue_number": number, "kind": "issue",
             "title": title, "body": body, "comments": comment_bodies, "branch": branch,
             "pr_number": pr_number},
            agent_id=IMPLEMENT_AGENT_ID, output_schema=IMPLEMENT_SCHEMA, model=_MODEL,
            label="build-%d" % number)
    except AgentError as exc:
        log("build: implement agent error on #%d -> needs:human/build:" % number, exc)
        await _label(repo, number, NEEDS_HUMAN_BUILD)
        return {"transition": "build", "number": number, "result": "implement_error",
                "reason": str(exc)}
    if impl.get("escalated"):
        log("build: implement escalated on #%d -> needs:human/build" % number)
        await _label(repo, number, NEEDS_HUMAN_BUILD)
        return {"transition": "build", "number": number, "result": "escalated",
                "reason": impl.get("escalation_reason", "")}
    # Stamp the built marker LAST — the durable dedup signal for any subsequent crash-replay.
    await _label(repo, number, LABEL_BUILT)
    return {"transition": "build", "number": number, "pr_number": pr_number, "result": "built"}


async def _pr_has_commits(repo, pr_number):
    """Whether the PR branch already carries at least one commit (the implement agent already
    pushed). A read failure is conservative: returns False (so a flaky read NEVER skips a build that
    has not happened — the re-run is the safe direction; the worst case is a second implement on a
    truly-empty branch, which the marker check above already guards in the common path). Uses the
    PR commits endpoint (paginated; one page is enough — we only need 'any')."""
    commits = await gh_paginated(_ipath(repo, "/pulls/%d/commits" % pr_number))
    return isinstance(commits, list) and len(commits) > 0


async def _open_or_adopt_pr(repo, branch, issue_number, title):
    """PR-FIRST open with at-least-once dedup: list open PRs, adopt one for ``branch`` if present
    (the re-drive case), else create a DRAFT PR. A create 422 (already exists) re-lists and adopts.
    Returns the PR dict or None. The body is a minimal placeholder the implement agent fills in via
    its own push; PR-FIRST means the durable job-identity exists before the long agent runs.

    The list read goes through ``gh_paginated`` (NOT a single-page ``gh``): on a repo carrying >30
    open PRs the existing draft for ``branch`` can sit on page 2, and a single-page read would miss
    it → re-create-422 → re-miss → a FALSE ``needs:human/build`` on an issue that already has a
    healthy PR (and a split-brain with the paginated snapshot in main()). ``gh_paginated`` also fails
    LOUD on a truncated read (#1294/#1323), so a partial list can never silently masquerade as
    'no PR exists'."""
    pr = _find_open_pr(await gh_paginated(_ipath(repo, "/pulls")), branch)
    if pr is not None:
        return pr
    created = await gh("POST", _ipath(repo, "/pulls"),
                       {"title": title or ("dev-pipeline: issue #%d" % issue_number),
                        "body": "Automated dev-pipeline build for issue #%d. "
                                "The implement agent fills in the diff." % issue_number,
                        "head": branch, "base": BASE_BRANCH, "draft": True})
    if _status(created) in (200, 201):
        return _json_body(created)
    if _status(created) == 422:  # already exists (re-drive / race) -> adopt it (paginated)
        return _find_open_pr(await gh_paginated(_ipath(repo, "/pulls")), branch)
    return None


async def _do_rebase(repo, item):
    """REBASE transition — heal a branch master moved under. ``_sync_branch`` does the mechanical
    rebase (idempotent no-op-on-equal-sha) + bounded fix-agent. A real conflict after the budget
    stamps ``needs:human/rebase`` (the relocated escalation). A clean rebase exits — the next pass
    re-reads the (now-mergeable) PR and drives CI on the new head."""
    pr_number = item["number"]
    branch = item.get("branch", "")
    comments = _comment_texts(
        await gh_paginated(_ipath(repo, "/issues/%d/comments" % pr_number)) or [])
    sync = await _sync_branch(repo, pr_number, branch, comments, _MODEL)
    if sync["outcome"] in ("conflict", "error"):
        detail = sync.get("detail", "")
        await post_markered_comment(repo, pr_number, MARKER_REBASE_CONFLICT,
                                    "%s\n\n```\n%s\n```" % (MARKER_REBASE_CONFLICT, detail))
        await _label(repo, pr_number, NEEDS_HUMAN_REBASE)
        return {"transition": "rebase", "number": pr_number, "result": "conflict"}
    # noop or rebased: exit; the next pass re-reads mergeable + drives CI on the (new) head.
    return {"transition": "rebase", "number": pr_number, "result": sync["outcome"]}


async def _do_ci(repo, item):
    """CI transition — the tree is genuinely red (script-verified). Dispatch the bounded fixer
    (``dev-fix``), which pushes ONE new head sha; re-pending CI IS the exit (the next pass re-reads
    the verdict on the new head).

    Three failure dispositions, kept DISTINCT (an infra failure must not be masked as a legitimate
    no-op that the lap counter slow-walks for MAX_LAPS passes):
      - fix-AGENT error (crash / no-return / schema break) → ``needs:human/ci`` IMMEDIATELY. An
        agent host being down is not a CI-fix the lap counter should retry 8x; it is an infra
        escalation that must surface fast (and a hard-down agent would otherwise 'progress' every
        pass via the lap bump and so evade the dead-man horizon too).
      - fix ran but pushed NO new commit (same head) → bump the lap (the legitimate no-progress
        case; a fresh sha each real lap, so the per-stage cap never fires — the lap counter is the
        livelock catch).
      - fix pushed a new head → exit; the next pass re-reads CI on the new head."""
    pr_number = item["number"]
    head_sha = item.get("head_sha", "")
    comments = _comment_texts(
        await gh_paginated(_ipath(repo, "/issues/%d/comments" % pr_number)) or [])
    before = head_sha
    try:
        fix = await agent(
            {"task": "fix_ci", "repo": repo, "pr_number": pr_number, "detail": "CI is red",
             "comments": comments, "lint_hint": FIX_CI_LINT_HINT},
            agent_id=FIX_AGENT_ID, output_schema=FIX_SCHEMA, model=_MODEL,
            label="ci-fix-%d" % pr_number)
        new_head = fix.get("head_sha", "")
    except AgentError as exc:
        # An AGENT error is an infra failure, NOT a CI-fix that produced no commit. Escalate fast.
        log("ci: fix agent error on PR #%d -> needs:human/ci:" % pr_number, exc)
        await _label(repo, pr_number, NEEDS_HUMAN_CI)
        return {"transition": "ci", "number": pr_number, "result": "fix_agent_error",
                "reason": str(exc)}
    if not new_head or (before and new_head == before):
        # No new commit -> the fixer ran but made no progress on this head. Bump the lap.
        livelocked = await _bump_lap(repo, pr_number, item.get("labels") or frozenset())
        return {"transition": "ci", "number": pr_number,
                "result": "livelock" if livelocked else "no_progress"}
    return {"transition": "ci", "number": pr_number, "result": "fixed", "head_sha": new_head}


async def _do_review(repo, item):
    """REVIEW transition (the CHECKER call — a DIFFERENT agent from build; maker≠checker). Runs
    ``dev-review`` (artifact-required) → ``dev-risk`` → RE-DERIVE the deterministic ``_risk_floor``
    itself (the #1158 control) → ``_merge_guard_command`` (a check, mutates labels only). Stamps
    (escalation-FIRST, approval-LAST):
      - risk:tier-N (single-valued: delete any prior tier label first),
      - reviewed:green@<head_sha>,
      - THEN exactly one terminal: merge:approved (pass ∧ guard-ok ∧ tier≤cap)
        | needs:human/merge-approval (tier>cap — the #1158 control)
        | needs:human/verify (review failed) | needs:human/merge-guard (guard refused).
    Idempotent: re-stamping the same single-valued labels on a re-drive is a no-op."""
    pr_number = item["number"]
    head_sha = item.get("head_sha", "")
    comments = _comment_texts(
        await gh_paginated(_ipath(repo, "/issues/%d/comments" % pr_number)) or [])

    # dev-review (the checker). An error / no-artifact / failing verdict → needs:human/verify.
    try:
        review = await agent(
            {"task": "review", "repo": repo, "pr_number": pr_number, "head_sha": head_sha,
             "comments": comments},
            agent_id=REVIEW_AGENT_ID, output_schema=REVIEW_SCHEMA, model=_MODEL,
            label="review-%d" % pr_number)
    except AgentError as exc:
        log("review: agent error on PR #%d -> needs:human/verify:" % pr_number, exc)
        await _label(repo, pr_number, NEEDS_HUMAN_VERIFY)
        return {"transition": "review", "number": pr_number, "result": "verify_error"}
    if not review.get("artifact_posted"):
        await _label(repo, pr_number, NEEDS_HUMAN_VERIFY)
        return {"transition": "review", "number": pr_number, "result": "no_artifact"}
    if review.get("verdict") != "pass" and review.get("issues"):
        # The review found unresolved issues. The reconciler does NOT loop review→fix→review in a
        # single pass (that is the monolith's job); it parks for a human (the relocated escalation).
        # A subsequent fix-and-re-push advances the head sha, which clears reviewed-for-head and
        # re-picks review on the new sha — but a STILL-failing review with no head advance is a
        # human-park, not an in-loop spin.
        await _label(repo, pr_number, NEEDS_HUMAN_VERIFY)
        return {"transition": "review", "number": pr_number, "result": "review_failed",
                "issues": review.get("issues", [])}

    # dev-risk (best-effort) → RE-DERIVE the deterministic _risk_floor ourselves (#1158). The merge
    # gate must FAIL CLOSED on missing risk evidence, so a flaky/failed/malformed risk node defaults
    # to tier-4 (ABOVE AUTO_MERGE_MAX_TIER=3) → the PR parks at needs:human/merge-approval rather
    # than silently auto-merging at the cap. tier-3 is NOT conservative when the cap is 3 — it is the
    # maximally-permissive auto-merge tier; defaulting there would let a dead risk agent wave PRs
    # through. The floor can only RAISE the tier further (CI-workflow PRs etc.).
    tier = 4
    summary = "risk assessment unavailable (failing closed to tier-4 — human merge-approval required)"
    try:
        risk = await agent(
            {"task": "risk", "repo": repo, "pr_number": pr_number},
            agent_id=RISK_AGENT_ID, output_schema=RISK_SCHEMA, model=_MODEL,
            label="risk-%d" % pr_number)
        tier = int(risk["tier"])
        summary = str(risk["summary"])
    except (AgentError, KeyError, ValueError, TypeError) as exc:
        log("review: risk node failed -> failing CLOSED to tier-4 (needs human merge-approval):", exc)
    try:
        files = _json_body(await gh("GET", _ipath(repo, "/pulls/%d/files" % pr_number)))
    except Exception as exc:  # noqa: BLE001 — a files-fetch failure must fail CLOSED, not open
        log("review: risk-floor files-fetch failed (failing closed to tier-4):", exc)
        files = None
    tier, floored_files = _risk_floor(tier, files)
    if floored_files:
        summary = ("%s\n\n_Risk floored to tier %d (privileged surface / files-fetch failure — "
                   "#1185/#1187)._" % (summary, tier))

    # The merge-ref guard (fail-closed; the one mechanical check that reaches the broken-on-merge
    # case). A non-zero exit is a guard refusal → needs:human/merge-guard.
    guard = await tool("bash", {"command": _merge_guard_command(repo, pr_number),
                                "timeout_seconds": 600})
    guard_ok = isinstance(guard, dict) and guard.get("exit_code") == 0

    # Stamp risk:tier-N single-valued (delete any prior tier label first), then reviewed:green@<sha>.
    await _restamp_risk_tier(repo, pr_number, item.get("labels") or frozenset(), tier)
    await post_markered_comment(repo, pr_number, MARKER_RISK_ASSESSMENT,
                                "%s\n\n**Tier %d**\n\n%s" % (MARKER_RISK_ASSESSMENT, tier, summary))
    if head_sha:
        await _label(repo, pr_number, LABEL_REVIEWED_PREFIX + head_sha)

    # Terminal label, escalation-FIRST / approval-LAST. ANY needs:human/* is a hard veto downstream.
    if not guard_ok:
        detail = "%s\n%s\nexit=%r" % (
            (guard.get("stdout", "") if isinstance(guard, dict) else "")[-1200:],
            (guard.get("stderr", "") if isinstance(guard, dict) else "")[-600:],
            guard.get("exit_code") if isinstance(guard, dict) else None)
        await post_markered_comment(repo, pr_number, MARKER_MERGE_GUARD,
                                    "%s\n\n```\n%s\n```" % (MARKER_MERGE_GUARD, detail))
        await _label(repo, pr_number, NEEDS_HUMAN_MERGE_GUARD)
        return {"transition": "review", "number": pr_number, "result": "merge_guard_refused",
                "risk_tier": tier}
    # The #1158 tier-gate: REFUSE to emit merge:approved for tier>cap; route to human approval.
    if tier > AUTO_MERGE_MAX_TIER:
        await _label(repo, pr_number, NEEDS_HUMAN_MERGE_APPROVAL)
        return {"transition": "review", "number": pr_number, "result": "needs_merge_approval",
                "risk_tier": tier}
    # pass ∧ guard-ok ∧ tier≤cap → the green light (approval LAST).
    await _label(repo, pr_number, LABEL_MERGE_APPROVED)
    return {"transition": "review", "number": pr_number, "result": "approved", "risk_tier": tier}


async def _restamp_risk_tier(repo, pr_number, labels, tier):
    """Stamp ``risk:tier-N`` single-valued: delete any DIFFERENT prior risk:tier-* label, then add
    the new one. Idempotent — re-stamping the same tier deletes nothing and re-adds the same label."""
    want = "%s%d" % (LABEL_RISK_PREFIX, tier)
    for n in labels:
        if isinstance(n, str) and n.startswith(LABEL_RISK_PREFIX) and n != want:
            await _unlabel(repo, pr_number, n)
    await _label(repo, pr_number, want)


async def _do_merge(repo, item):
    """MERGE transition (the EFFECTOR branch; never reviews). In WOULD-MERGE ADVISORY mode
    (``REAL_MERGE`` OFF — the v1 default per chairman decision 2) it does NOT merge: it stamps an
    advisory label + comment the seat clears, until the post-merge checker (task #76) exists. With
    ``REAL_MERGE`` ON it re-checks the gate defensively (re-derive nothing new — the review branch
    already enforced #1158; here we re-read the live head + required-set + mergeable), then does a
    CONDITIONAL ``PUT /merges`` with the expected-head sha (GitHub 409 on mismatch — closes the
    read→PUT TOCTOU), then ``_close_source_issue`` (close-before-strip). A sha-mismatch / master-
    moved → STRIP merge:approved (re-evaluate from scratch); a newly-required check →
    needs:human/required-set (distinct from the generic CAP)."""
    pr_number = item["number"]
    expected_head = item.get("head_sha", "")

    if not REAL_MERGE:
        # WOULD-MERGE ADVISORY: stamp the advisory the seat clears. The advisory label is the
        # maker-marker; re-driving is idempotent (additive label + maker-marker comment dedup).
        await _label(repo, pr_number, LABEL_WOULD_MERGE)
        await post_markered_comment(
            repo, pr_number, MARKER_WOULD_MERGE,
            "%s\n\nThe reconciler WOULD merge PR #%d now: review passed, the merge-ref guard is "
            "clean, and the risk tier is within the auto-merge cap. Auto-merge is HELD until the "
            "post-merge regression checker (task #76) exists — green CI + LLM review cannot reach "
            "PAST the merge boundary (#1188). The seat clears this advisory to merge." % (
                MARKER_WOULD_MERGE, pr_number))
        return {"transition": "merge", "number": pr_number, "result": "would_merge_advisory"}

    # REAL_MERGE ON — re-read the live PR for the authoritative head + mergeability (close the
    # read→PUT TOCTOU). The merge is CONDITIONAL on the expected-head sha (GitHub 409 on mismatch).
    pr = _json_body(await gh("GET", _ipath(repo, "/pulls/%d" % pr_number)))
    live_head = _pr_head_sha(pr) if isinstance(pr, dict) else ""
    if isinstance(pr, dict) and pr.get("merged"):
        await _close_source_issue_for_pr(repo, pr)
        return {"transition": "merge", "number": pr_number, "result": "already_merged"}
    if not (live_head and expected_head and _shas_equal(live_head, expected_head)):
        # The head advanced under us (a fix / human push) — STRIP merge:approved and re-evaluate.
        log("merge: PR #%d head moved (expected %r live %r) -> strip approval, re-evaluate"
            % (pr_number, expected_head, live_head))
        await _unlabel(repo, pr_number, LABEL_MERGE_APPROVED)
        return {"transition": "merge", "number": pr_number, "result": "head_moved"}
    if isinstance(pr, dict) and pr.get("mergeable") is False:
        # Master moved (conflicting) — strip approval; the next pass re-drives rebase.
        await _unlabel(repo, pr_number, LABEL_MERGE_APPROVED)
        return {"transition": "merge", "number": pr_number, "result": "conflicting"}

    # Conditional PUT with expected-head sha (the merge ref TOCTOU close).
    merge_resp = await gh("PUT", _ipath(repo, "/pulls/%d/merge" % pr_number),
                          {"merge_method": MERGE_METHOD, "sha": expected_head})
    st = _status(merge_resp)
    if st == 200:
        await _close_source_issue_for_pr(repo, pr)
        return {"transition": "merge", "number": pr_number, "result": "merged"}
    if st == 409:
        # Head moved between the read and the PUT — strip approval, re-evaluate next pass.
        await _unlabel(repo, pr_number, LABEL_MERGE_APPROVED)
        return {"transition": "merge", "number": pr_number, "result": "sha_mismatch_409"}
    if st in (403, 405, 422):
        # A NON-retryable block (branch protection / a newly-required check / a 'not mergeable'
        # 422 — none of which a re-PUT resolves): distinct durable escalation, NOT a strip-and-retry
        # (which would flap merge:approved forever with no terminal label and evade the dead-man).
        await _label(repo, pr_number, NEEDS_HUMAN_REQUIRED_SET)
        return {"transition": "merge", "number": pr_number, "result": "required_set", "status": st}
    # Any OTHER failure (e.g. a transient the gh() retries already exhausted) — strip approval and
    # bump the lap so a persistently-failing merge is BOUNDED (at CAP → needs:human/livelock) rather
    # than an unbounded strip→re-review→re-approve→re-fail flap the dead-man can't catch.
    await _unlabel(repo, pr_number, LABEL_MERGE_APPROVED)
    livelocked = await _bump_lap(repo, pr_number, item.get("labels") or frozenset())
    return {"transition": "merge", "number": pr_number,
            "result": "livelock" if livelocked else "merge_failed", "status": st}


async def _close_source_issue_for_pr(repo, pr):
    """Best-effort close of the source issue a merged PR resolved (close-before-strip #1208). The
    reconciler has no explicit issue→PR linkage in the PR object beyond the branch name, so derive
    the issue number from the conventional ``dev-pipeline/issue-<n>`` branch; if it can't be
    derived, skip (the dead-man + seat catch an unclosed issue)."""
    ref = ((pr.get("head") or {}).get("ref") or "") if isinstance(pr, dict) else ""
    m = re.search(r"issue-(\d+)$", ref)
    if not m:
        log("merge: could not derive source issue from branch %r -> skipping close" % ref)
        return
    await _close_source_issue(repo, int(m.group(1)))


async def _do_escalate(repo, item):
    """ESCALATE transition (the relocated gap — the catch-all ``else`` of ``owner()``). A blackboard
    tuple no branch foresaw is DETECTABLE within one pass: stamp ``needs:human/stuck`` (terminal;
    owner() then vetoes it) so an unforeseen state surfaces as an alarm, never a silent freeze. This
    is the structural guarantee the gate()-suspending monolith lacked.

    The escalate path is, by construction, the one reached when the AUTHORS' state model was WRONG —
    exactly when a thin label is least useful. So it ALSO posts a markered comment dumping the
    normalized item (the offending tuple: kind, labels, ci_verdict, mergeable signals) so the seat
    sees WHY it is stuck and can act, mirroring the rebase/merge-guard escalation markers. The
    mailbox must not be an empty envelope."""
    number = item["number"]
    # The diagnostic snapshot — the fields owner() reads, so the seat can see the unforeseen tuple.
    diag = {k: (sorted(item[k]) if isinstance(item.get(k), frozenset) else item.get(k))
            for k in ("kind", "labels", "closed", "draft", "needs_rebase", "ci_verdict",
                      "human_blocked", "risk_tier", "shovel_ready", "approved", "dispatched",
                      "has_open_pr")
            if k in item}
    await post_markered_comment(
        repo, number, MARKER_STUCK,
        "%s\n\nThe dev-pipeline reconciler reached its catch-all `escalate` branch for this item — "
        "a blackboard state no `owner()` branch foresaw. It is stamped `needs:human/stuck` (terminal "
        "until the seat clears it). The normalized item the reconciler saw:\n\n```json\n%s\n```"
        % (MARKER_STUCK, json.dumps(diag, indent=2, sort_keys=True)))
    await _label(repo, number, NEEDS_HUMAN_STUCK)
    log("escalate: unforeseen state on %s #%d -> needs:human/stuck" % (item.get("kind"), number))
    return {"transition": "escalate", "number": number, "result": "stuck"}


# ─── the reconciler (one pass: read snapshot once → owner() → ONE transition → exit) ──

async def main(input):
    payload = _unwrap(input)
    repo = payload.get("repo") or DEFAULT_REPO
    global _MODEL
    _MODEL = payload.get("model") or DEFAULT_MODEL
    if not repo:
        return {"state": "error",
                "reason": "no repo provided (input.repo) and no DEFAULT_REPO configured"}

    # S1 — read the COMPLETE blackboard once (open issues + open PRs), one snapshot per pass.
    # Stateless: the working set is recomputed from scratch every pass. ONLY pipeline:v2 items are
    # touched (step 6: the monolith no-op-skips v2; the reconciler ONLY touches v2 — mechanically
    # can't both act).
    phase("scan")
    raw_issues = await gh_paginated(_ipath(repo, "/issues"))
    if not isinstance(raw_issues, list):
        raw_issues = []
    open_prs = await gh_paginated(_ipath(repo, "/pulls"))
    if not isinstance(open_prs, list):
        open_prs = []

    # Normalize the blackboard. Issues first (cheap: labels only), then PRs (each PR costs a few
    # reads — mergeable, CI verdict, reviews — so we normalize ONLY pipeline:v2 PRs). A v2 routing
    # label gates membership; un-adopted (v1) items are invisible to this reconciler.
    items = []
    for issue in raw_issues:
        if _is_pull_request(issue):
            continue
        if LABEL_PIPELINE_V2 not in _label_names(issue):
            continue
        items.append(_build_eligible_issue(repo, issue, open_prs))
    for pr in open_prs:
        if LABEL_PIPELINE_V2 not in _label_names(pr):
            continue
        items.append(await _read_pr_blackboard(repo, pr))

    # S2 — owner() over the whole snapshot → the ONE highest-priority actionable item.
    phase("owner")
    decision = select_actionable(items)
    if decision is None:
        # A no-op pass: nothing actionable. The common scheduled-fire-into-quiet-board case.
        phase("summary")
        summary = {"state": "done", "repo": repo, "scanned_items": len(items),
                   "transition": None, "result": "noop"}
        log("reconciler pass: noop -", json.dumps(summary))
        return summary

    item, transition = decision

    # S3 — drive EXACTLY ONE transition, then EXIT. The next cron pass re-evaluates from scratch.
    phase(transition)
    if transition == "build":
        outcome = await _do_build(repo, item)
    elif transition == "rebase":
        outcome = await _do_rebase(repo, item)
    elif transition == "ci":
        outcome = await _do_ci(repo, item)
    elif transition == "review":
        outcome = await _do_review(repo, item)
    elif transition == "merge":
        outcome = await _do_merge(repo, item)
    else:  # "escalate" — the relocated gap
        outcome = await _do_escalate(repo, item)

    phase("summary")
    summary = {"state": "done", "repo": repo, "scanned_items": len(items),
               "transition": transition, "number": item.get("number"),
               "outcome": outcome}
    log("reconciler pass complete:", json.dumps(summary))
    return summary


# the per-pass model, set once in main() from input/default (a module global the transition
# helpers read so the long signatures stay focused on the blackboard item).
_MODEL = None
'''


def build_reconciler_script(
    *,
    implement_agent_id: str = DEFAULT_IMPLEMENT_AGENT_ID,
    review_agent_id: str = DEFAULT_REVIEW_AGENT_ID,
    fix_agent_id: str = DEFAULT_FIX_AGENT_ID,
    ci_agent_id: str = DEFAULT_CI_AGENT_ID,
    risk_agent_id: str = DEFAULT_RISK_AGENT_ID,
    github_server: str = "github",
    base_branch: str = "master",
    repo: str | None = None,
    merge_sentinels: list[str] | None = None,
    max_ci_iters: int = 3,
    max_review_iters: int = 3,
    max_rebase_attempts: int = 2,
    auto_merge_max_tier: int = AUTO_MERGE_MAX_TIER,
    max_laps: int = MAX_LAPS,
    max_items_per_pass: int = 200,
    merge_method: str = "squash",
    real_merge: bool = False,
    default_model: str | None = None,
) -> str:
    """Return the production reconciler workflow source.

    Defaults match a standard deployment (the named ``dev-*`` judgment agents, a ``github`` http
    server bound to ``https://api.github.com``, the #1158 tier-gate at ``auto_merge_max_tier=3``,
    and — critically — ``real_merge=False`` (WOULD-MERGE ADVISORY mode, chairman decision 2: the
    reconciler advises, the seat clears the advisory to merge, until the post-merge checker task #76
    exists). Flip ``real_merge=True`` ONLY after #76 ships.

    ``repo`` is the default blackboard the scheduled fire scans (the cron trigger can also pass
    ``repo``); ``merge_sentinels`` is the list of shell commands the fail-closed merge-ref guard
    runs; ``base_branch`` is the repo's default branch.
    """
    header = _render_constants(
        implement_agent_id=implement_agent_id,
        review_agent_id=review_agent_id,
        fix_agent_id=fix_agent_id,
        ci_agent_id=ci_agent_id,
        risk_agent_id=risk_agent_id,
        github_server=github_server,
        base_branch=base_branch,
        merge_sentinels=merge_sentinels or [],
        max_ci_iters=max_ci_iters,
        max_review_iters=max_review_iters,
        max_rebase_attempts=max_rebase_attempts,
        auto_merge_max_tier=auto_merge_max_tier,
        max_laps=max_laps,
        max_items_per_pass=max_items_per_pass,
        merge_method=merge_method,
        real_merge=real_merge,
        default_model=default_model,
    )
    # The DEFAULT_REPO constant (the scheduled fire's default blackboard) is rendered separately so
    # it threads through like triage's.
    header += "\n" + _py("DEFAULT_REPO", repo)
    # Assemble by splicing five source strings, each authored once (module-level defs resolve names
    # at CALL time, so the order between body/owner/lib/helpers is free):
    #   - _BODY: the imports + blackboard normalization + transitions + the reconciler main().
    #   - RECONCILER_OWNER: the total function owner() + select_actionable() (the completeness
    #     guarantee; also exec'd directly by the property-tests).
    #   - DEV_PIPELINE_LIB: the proven scripted/agentic helpers, spliced VERBATIM (one source of
    #     truth with the monolith — no drift).
    #   - GH_BODY_HELPERS (#1294): _json_body / gh_paginated (fail LOUD on truncated/unparseable 2xx).
    #   - COMMENT_IDEMPOTENCY_HELPERS (#1292): the maker-marker comment-POST dedup.
    return (
        header
        + "\n"
        + _BODY
        + RECONCILER_OWNER
        + DEV_PIPELINE_LIB
        + GH_BODY_HELPERS
        + COMMENT_IDEMPOTENCY_HELPERS
    )


def build_reconciler_fixture_script(
    *,
    implement_agent_id: str,
    review_agent_id: str,
    fix_agent_id: str,
    ci_agent_id: str,
    risk_agent_id: str,
    repo: str | None = None,
    max_laps: int = 4,
    max_items_per_pass: int = 50,
    real_merge: bool = False,
) -> str:
    """The CI fixture variant: a tight lap cap (4) + real generated agent ids, otherwise the
    identical script shape. Driven by ``tests/integration/test_wf_dev_pipeline_reconciler_fixture.py``
    against the host with simulated agent/tool returns."""
    return build_reconciler_script(
        implement_agent_id=implement_agent_id,
        review_agent_id=review_agent_id,
        fix_agent_id=fix_agent_id,
        ci_agent_id=ci_agent_id,
        risk_agent_id=risk_agent_id,
        repo=repo,
        max_laps=max_laps,
        max_items_per_pass=max_items_per_pass,
        real_merge=real_merge,
    )


def owner_source() -> str:
    """The pure ``owner()`` + ``select_actionable()`` source string (with the label-prefix constants
    it reads), assembled so a test can ``exec`` it into a namespace and call ``owner()`` DIRECTLY —
    fast, over the entire enumerated cross-product, with no host drive. This is the SAME text spliced
    into the production script (one source of truth), so the completeness + idempotency property-
    tests exercise exactly the function that ships."""
    consts = "\n".join(
        [
            _py("NEEDS_HUMAN_PREFIX", NEEDS_HUMAN_PREFIX),
            _py("LABEL_REVIEWED_PREFIX", LABEL_REVIEWED_PREFIX),
            _py("LABEL_RISK_PREFIX", LABEL_RISK_PREFIX),
            _py("LABEL_MERGE_APPROVED", LABEL_MERGE_APPROVED),
            _py("AUTO_MERGE_MAX_TIER", AUTO_MERGE_MAX_TIER),
        ]
    )
    return consts + "\n" + RECONCILER_OWNER


# ─── deploy surface (the tool + http_server envelope a WorkflowCreate needs) ──
#
# ``build_reconciler_script`` returns only the workflow *script string*. A deployed workflow ALSO
# needs its tool + http_server surface declared on the ``WorkflowCreate``, or the first
# ``tool('bash')`` / ``tool('http_request')`` call errors at runtime. Exporting the surface here
# keeps the two from drifting (#1135). Same union as the monolith — the reconciler clones/edits
# (bash + the editing tools, via the implement/fix child agents) AND reaches GitHub REST+GraphQL.
REQUIRED_TOOLS: list[ToolSpec] = [
    ToolSpec(type="bash"),
    ToolSpec(type="read"),
    ToolSpec(type="write"),
    ToolSpec(type="edit"),
    ToolSpec(type="glob"),
    ToolSpec(type="grep"),
    ToolSpec(type="http_request"),
]


def _github_http_server(*, name: str, base_url: str) -> HttpServerSpec:
    return HttpServerSpec(
        name=name,
        base_url=base_url,
        description="GitHub REST + GraphQL API (auth resolved from the bound vault's GITHUB_TOKEN).",
        routes=[
            HttpRouteSpec(
                # The full REST verb set, exactly as the monolith: DELETE is load-bearing (the
                # _unlabel strips), PATCH is load-bearing (#1208 _close_source_issue closes the
                # source issue), PUT is the merge call.
                path_pattern="/repos/**",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
                allow_query=True,
                description="Issues, PRs, labels, statuses, refs, reviews; list endpoints "
                "paginate via ?per_page/?page.",
            ),
            HttpRouteSpec(
                path_pattern="/graphql",
                methods=["POST"],
                description="GraphQL mutations (mark-ready: markPullRequestReadyForReview).",
            ),
        ],
    )


REQUIRED_HTTP_SERVERS: list[HttpServerSpec] = [
    _github_http_server(name="github", base_url="https://api.github.com")
]


def build_reconciler_workflow_create(
    *,
    name: str,
    description: str | None = None,
    github_server: str = "github",
    github_base_url: str = "https://api.github.com",
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` payload for the production reconciler.

    Bundles the script (``build_reconciler_script``) with the tool + http_server surface it
    requires, so a deployer can POST one object instead of hand-assembling the surface — and so the
    declared surface can never drift from the script that needs it (#1135).

    ``github_server`` names the http_server and is threaded into the script's ``GITHUB_SERVER``
    constant; ``github_base_url`` is the http_server's ``base_url`` (the credential-resolution key).
    Remaining keyword args are forwarded verbatim to ``build_reconciler_script`` — note
    ``real_merge`` defaults OFF (WOULD-MERGE ADVISORY mode); a deployer flips it ON only post-#76.
    """
    script = build_reconciler_script(github_server=github_server, **script_kwargs)
    return WorkflowCreate(
        name=name,
        description=description,
        script=script,
        tools=list(REQUIRED_TOOLS),
        http_servers=[_github_http_server(name=github_server, base_url=github_base_url)],
    )
