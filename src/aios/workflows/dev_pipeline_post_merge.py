"""The post-merge reconciliation checker — the safety backstop that lets the dev-pipeline
reconciler safely AUTO-MERGE (task #76; aios#49/#111, the post-merge residual of
``architecture/dev-pipeline-reconciler-design.md`` §5 + row 12).

WHY THIS EXISTS (the honest residue the reconciler can NOT reach). The dev-pipeline
reconciler's maker≠checker split (``dev-implement`` ≠ ``dev-review``) is STRUCTURAL
separation, NOT detection-independence: the builder and the reviewer run on the SAME
correlated LLM family, so neither catches a semantic regression they both miss. And every
pre-merge check — CI, the LLM review, the mechanical merge-guard — runs on the PR's OWN
``refs/pull/N/merge`` ref. NONE of them reaches PAST the merge boundary: a PR that passed
CI AND review can still break ``master`` once it actually lands (the #1188 post-merge
regression class), and the reconciler's own merge branch leaves a stuck issue OPEN if its
best-effort close failed. Green CI cannot reach past the merge boundary.

So this is the POST-MERGE TWIN of ``owner()``'s pre-merge gate — maker≠checker ACROSS the
merge boundary. It is a SEPARATE cron-fired stateless workflow (the gate-reaper /
telemetry-observer dead-man idiom), NOT a stage of the reconciler, precisely because its
verdict must draw from an UNCORRELATED, off-the-run substrate: the DURABLE GitHub state
(the merged PR, its provenance labels, the source issue's open/closed state, and the live
post-merge ``master`` CI), corroborated by the run journal. A reconciler run that merged a
PR cannot vouch for what that merge did to master — the check that backs the merge boundary
MUST come from a different stage than the one that produced the merge (the
substrate-different-verdict invariant).

─── THE THREE POST-MERGE VERIFICATIONS (each drawn from durable, off-the-run state) ────

For every recently-merged dev-pipeline (``pipeline:v2``) PR not yet checked, verify:

  (a) THE SOURCE ISSUE ENDED CLOSED. The reconciler's merge branch closes the source
      issue best-effort (``_close_source_issue``, close-before-strip #1208); a failed close
      leaves the issue OPEN ∧ ``dispatched`` (safe, but a silent stuck issue). We derive the
      source issue from the merged PR's ``dev-pipeline/issue-<n>`` branch and confirm
      ``state == closed``. An OPEN source issue for a merged PR → ``needs:human/issue-open``.

  (b) THE MERGE DID NOT LEAVE MASTER RED. We read the LIVE post-merge CI verdict on the
      base branch (``BASE_BRANCH``) via the deterministic Checks + combined-status API
      (``_read_ci`` — NO agent, NO model). A genuinely-red master after the merge landed →
      ``needs:human/master-red`` (the #1188 post-merge-regression alarm). ``master`` red is
      attributed to whichever merged-PR sweep observes it; the seat triages which merge
      broke it. (A still-running / unreadable master CI is INDETERMINATE — re-checked next
      sweep — never a silent green.)

  (c) THE PROVENANCE WAS LEGITIMATE (the PR was merged THROUGH the pipeline's gate). A
      ``pipeline:v2`` PR that MERGED but carries NO gate provenance — no
      ``reviewed:green@<merge_head>`` and no in-cap ``risk:tier-N`` — was merged WITHOUT the
      maker≠checker review the auto-merge authority is scoped to. That is exactly the
      bypass the post-merge checker exists to catch: a tier>cap PR, or one a human/automation
      force-merged outside the gate, must NOT pass silently → ``needs:human/bad-provenance``.
      Corroborated (best-effort, since #1397 made the run journal run-readable) against
      ``list_runs``: a dev-pipeline run SHOULD have driven the merge; its absence is folded
      into the provenance detail, never the sole verdict (the GitHub labels are the durable,
      always-present provenance; the run journal is the secondary witness).

On ANY violation → stamp the durable ``needs:human/*`` label + post ONE idempotent markered
comment (NEVER silently — the same queryable-mailbox discipline as the reconciler's
escalations) and the seat / the reconciler's Step-8 dead-man owns it from there. A clean PR
is stamped ``post-merge:checked@<merge_sha>`` so it is verified EXACTLY ONCE (bounded,
idempotent: a re-sweep skips an already-checked merge head).

─── DETERMINISM (replay-stable; the cron envelope's frozen clock if a clock is needed) ──

Authored with the EXACT ``gate_reaper.py`` / ``dev_pipeline_reconciler.py`` builder idiom:
an exported ``build_post_merge_checker_script(...) -> str`` returns workflow SOURCE (a
prepended constants header + a static ``_BODY`` of pure-stdlib ``re``/``json``, value-domain
I/O, deterministic emit order → replay-stable), plus ``REQUIRED_TOOLS`` /
``REQUIRED_HTTP_SERVERS`` and ``build_post_merge_checker_workflow_create(...)`` bundling
script + surface so the declared surface can't drift from the script (#1135). The body
imports neither ``datetime`` nor ``time``: the only "now"-like input it needs is the
recency window, and it reads recency off GitHub's own ``merged_at`` ordering (``sort=updated
direction=desc``) rather than a wall clock, so it never desyncs replay.

The list reads go through the shared ``gh_paginated`` (#1294/#1323): a truncated / unparseable
2xx page is fatal-loud ``cannot-determine``, NEVER a silent under-count read as "no
regressions". A post-merge checker that under-counts the merges it judges is exactly the
look-green-while-checking-less failure the backstop exists to prevent.

PROVENANCE SUBSTRATE SHARED VERBATIM WITH THE RECONCILER (one source of truth, no drift):
the gate provenance the checker validates is the SAME ``reviewed:green@<sha>`` /
``risk:tier-N`` / ``pipeline:v2`` / ``AUTO_MERGE_MAX_TIER`` vocabulary the reconciler's
``owner()`` + review branch STAMP — imported from ``dev_pipeline_reconciler`` so a relabel on
one side can never silently desync the post-merge verification on the other. The CI read
(``_read_ci``) + the source-issue branch convention (``dev-pipeline/issue-<n>``) come from the
shared ``dev_pipeline_lib`` for the same reason.
"""

from __future__ import annotations

from typing import Any

from aios.models.agents import HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.comment_idempotency import COMMENT_IDEMPOTENCY_HELPERS
from aios.workflows.dev_pipeline_lib import DEV_PIPELINE_LIB

# Reused VERBATIM from the reconciler so the post-merge checker validates EXACTLY the gate
# provenance the reconciler STAMPS — a relabel on one side can never silently desync the
# verification on the other (the maker≠checker-across-the-merge-boundary invariant has to
# read the same labels the merge branch wrote).
from aios.workflows.dev_pipeline_reconciler import (
    AUTO_MERGE_MAX_TIER,
    LABEL_PIPELINE_V2,
    LABEL_REVIEWED_PREFIX,
    LABEL_RISK_PREFIX,
)
from aios.workflows.gh_body import GH_BODY_HELPERS

# ─── the post-merge escalation mailbox (the relocated escalation, terminal labels) ──
# Same queryable-mailbox discipline as the reconciler: a violation becomes a DURABLE
# terminal ``needs:human/*`` label read off the board, never a silent pass and never a
# suspended run. The seat / the reconciler's Step-8 dead-man + the reaper own them.
NEEDS_HUMAN_ISSUE_OPEN = "needs:human/issue-open"  # (a) merged PR, source issue still OPEN
NEEDS_HUMAN_MASTER_RED = "needs:human/master-red"  # (b) the merge left master RED (#1188)
NEEDS_HUMAN_BAD_PROVENANCE = "needs:human/bad-provenance"  # (c) merged WITHOUT the gate

# The single idempotency marker: stamped on a PR whose merge head has passed all three
# checks, so a clean merge is verified EXACTLY ONCE (a re-sweep skips an already-checked
# merge sha). ``post-merge:checked@<merge_sha>`` keys on the SHA so a (pathological)
# re-merge of the same PR onto a new sha is re-checked.
LABEL_CHECKED_PREFIX = "post-merge:checked@"

# The escalation label a seat/ops sweep can grep for every post-merge regression at a glance
# (and clear on resolution) — the analog of the reaper's ``autodev:reaper-escalated``.
DEFAULT_ESCALATED_LABEL = "autodev:post-merge-regression"

# The verdict vocabulary for the per-sweep summary — EXACTLY these three (matches the reaper).
VERDICTS: tuple[str, ...] = ("ok", "regression-found", "cannot-determine")

# The three post-merge verification classes (the design-of-record's §5 residue).
CHECK_CLASSES: tuple[str, ...] = (
    "issue-open",  # (a) source issue stayed open after merge
    "master-red",  # (b) the merge left master red (#1188)
    "bad-provenance",  # (c) merged without the maker≠checker gate
)

# A non-terminal run status is a LIVE run; the complement is terminal. Mirrors
# models/workflows.TERMINAL_RUN_STATUSES. For provenance corroboration we look at the
# TERMINAL (completed) runs that drove an issue — a successful pipeline run is positive
# provenance evidence.
TERMINAL_RUN_STATUSES: tuple[str, ...] = ("completed", "errored", "cancelled")

# How many recently-updated closed PRs to scan per sweep (the merge backlog horizon). Sized
# above the per-interval merge rate so no merge is missed between sweeps; the
# ``post-merge:checked@`` marker makes re-scanning an already-checked merge a cheap no-op.
DEFAULT_SCAN_LIMIT = 100


def _py(name: str, value: Any) -> str:
    """One ``NAME = <repr>`` constant line for the prepended header (mirrors dev/triage/reaper)."""
    return f"{name} = {value!r}"


def _render_constants(
    *,
    github_server: str,
    base_branch: str,
    escalated_label: str,
    auto_merge_max_tier: int,
    scan_limit: int,
    require_run_provenance: bool,
    dev_pipeline_workflow_id: str | None,
    default_repo: str | None,
) -> str:
    lines = [
        _py("GITHUB_SERVER", github_server),
        _py("BASE_BRANCH", base_branch),
        _py("ESCALATED_LABEL", escalated_label),
        _py("AUTO_MERGE_MAX_TIER", auto_merge_max_tier),
        _py("SCAN_LIMIT", scan_limit),
        # When True, the ABSENCE of a driving dev-pipeline run in the journal is itself a
        # provenance VIOLATION (the strictest posture); default False — the run journal is a
        # secondary witness folded into the detail, and the durable GitHub gate labels are the
        # primary provenance (always present on a gate-merged PR, no workflow-id wiring needed).
        _py("REQUIRE_RUN_PROVENANCE", require_run_provenance),
        _py("DEV_PIPELINE_WORKFLOW_ID", dev_pipeline_workflow_id),
        _py("DEFAULT_REPO", default_repo),
        # the reconciler's provenance vocabulary (imported, NOT re-spelled — one source of truth)
        _py("LABEL_PIPELINE_V2", LABEL_PIPELINE_V2),
        _py("LABEL_REVIEWED_PREFIX", LABEL_REVIEWED_PREFIX),
        _py("LABEL_RISK_PREFIX", LABEL_RISK_PREFIX),
        # the post-merge checker's own escalation + marker labels
        _py("NEEDS_HUMAN_ISSUE_OPEN", NEEDS_HUMAN_ISSUE_OPEN),
        _py("NEEDS_HUMAN_MASTER_RED", NEEDS_HUMAN_MASTER_RED),
        _py("NEEDS_HUMAN_BAD_PROVENANCE", NEEDS_HUMAN_BAD_PROVENANCE),
        _py("LABEL_CHECKED_PREFIX", LABEL_CHECKED_PREFIX),
        _py("TERMINAL_RUN_STATUSES", list(TERMINAL_RUN_STATUSES)),
        _py("CHECK_CLASSES", list(CHECK_CLASSES)),
        # Stable comment markers — each is BOTH the comment's first line AND the maker-marker
        # the idempotency guard scans for (aios#1292): a posted comment is its own "already
        # done" marker, so an at-least-once replay never double-posts.
        _py("MARKER_ISSUE_OPEN", "## Post-merge: source issue stayed OPEN after merge"),
        _py(
            "MARKER_MASTER_RED", "## Post-merge: master is RED after this merge (#1188 regression)"
        ),
        _py(
            "MARKER_BAD_PROVENANCE",
            "## Post-merge: a pipeline PR was merged WITHOUT the maker≠checker gate",
        ),
    ]
    return "\n".join(lines)


# The static checker body — references the prepended constants AND the shared lib helpers
# (DEV_PIPELINE_LIB: gh/_ipath/_with_query/_status/_json_body via the body splice, _read_ci,
# _resolve_sha1, _close-issue plumbing) + GH_BODY_HELPERS + COMMENT_IDEMPOTENCY_HELPERS.
# Pure stdlib (re/json), value-domain I/O, bounded loops, no datetime/time: replay-stable.
_BODY = r'''
import json
import re

# NOTE: the proven scripted helpers (gh + retry, gh_paginated #1294, _ipath/_with_query/
# _status, the deterministic CI read _read_ci #1316, _resolve_sha1 #1178, post_comment_once
# #1292) live in DEV_PIPELINE_LIB / GH_BODY_HELPERS / COMMENT_IDEMPOTENCY_HELPERS, spliced in
# below. Module-level defs resolve names at CALL time, so this body can call them freely even
# though their defs are appended after this block.


# ─── input-envelope unwrap (cron fire OR bare fixture/arm-time shape) ─────────

def _config(input):
    """The checker's config — ``{repo, dev_pipeline_workflow_id?, scan_limit?}`` — out of
    EITHER the bare fixture/arm-time shape OR the cron WorkflowAction envelope
    ``{"trigger": ..., "input": <template>}`` (a cron fire carries the author's template
    verbatim under ``input``)."""
    if isinstance(input, dict) and "trigger" in input and "input" in input:
        return input.get("input") or {}
    return input or {}


def _label_names(obj):
    """The set of label names on an issue/PR (GitHub returns label objects with a ``name``;
    tolerate bare strings defensively)."""
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


def _merge_sha(pr):
    """The commit SHA the PR merged AS (``merge_commit_sha``) — the durable identity of THIS
    merge, what the ``post-merge:checked@<sha>`` marker keys on. Falls back to the head sha if
    GitHub omits it (older payloads). Returns "" when neither is present (the merge is then
    un-keyable and the checker treats it as unverifiable → re-check next sweep)."""
    if not isinstance(pr, dict):
        return ""
    sha = pr.get("merge_commit_sha")
    if isinstance(sha, str) and sha:
        return sha
    return _pr_head_sha(pr)


def _reviewed_sha(labels):
    """The sha a ``reviewed:green@<sha>`` label records the reconciler reviewed (None if
    unstamped). The provenance check (c) requires SOME reviewed-green stamp on a merged
    ``pipeline:v2`` PR — its absence means the PR merged without the gate's review (a re-pushed
    PR legitimately carries the OLD reviewed:green@<sha> too, since the reconciler stamps them
    additively; presence-of-any is the correct backstop bar, the reconciler enforced
    reviewed-FOR-head at merge time).

    Picks the LEXICOGRAPHICALLY-SMALLEST matching sha when several are present, NOT 'first in
    set-iteration order': ``labels`` is a frozenset and Python's str hashing is PYTHONHASHSEED-
    salted, so an iteration-order pick would be non-deterministic ACROSS replays in fresh
    processes — a replay-stability hazard the moment this value is ever surfaced or compared.
    ``min()`` is a stable, content-only pick."""
    shas = [n[len(LABEL_REVIEWED_PREFIX):] for n in labels
            if isinstance(n, str) and n.startswith(LABEL_REVIEWED_PREFIX)
            and n[len(LABEL_REVIEWED_PREFIX):]]
    return min(shas) if shas else None


def _risk_tier(labels):
    """The single-valued ``risk:tier-N`` the reconciler stamped, or None if unstamped."""
    for n in labels:
        if isinstance(n, str) and n.startswith(LABEL_RISK_PREFIX):
            try:
                return int(n[len(LABEL_RISK_PREFIX):])
            except (ValueError, TypeError):
                continue
    return None


def _already_checked(labels, merge_sha):
    """True if THIS merge sha already carries ``post-merge:checked@<sha>`` — a clean merge is
    verified exactly once; a re-sweep skips it (idempotent, bounded). A merge onto a DIFFERENT
    sha (a re-merge) is not skipped (the marker keys on the sha)."""
    if not (isinstance(merge_sha, str) and merge_sha):
        return False
    want = LABEL_CHECKED_PREFIX + merge_sha
    return want in labels


def _source_issue_number(pr):
    """The source issue number from the merged PR's ``dev-pipeline/issue-<n>`` branch (the
    pipeline's only durable issue→PR linkage; PR bodies carry no ``Closes #N``). Returns an int
    or None (an un-derivable branch → the issue-closed check (a) is SKIPPED for this PR, not a
    false violation: we never invent an issue number to fail on)."""
    ref = ((pr.get("head") or {}).get("ref") or "") if isinstance(pr, dict) else ""
    m = re.search(r"issue-(\d+)$", ref)
    return int(m.group(1)) if m else None


# ─── the off-the-run run-journal witness (provenance corroboration, since #1397) ──

async def _list_runs(args):
    """One ``list_runs`` read (account-scoped to this run's own account; #1397 made the run
    journal run-readable). Returns the raw tool result the caller branches on."""
    return await tool("list_runs", args)


def _run_issue_numbers(runs, repo):
    """The set of issue numbers a list of dev-pipeline runs drove (from each run's input
    envelope ``{repo, issue_number}`` or the trigger-wrapped form). Used as the provenance
    witness: a merged PR's source issue appearing here means a dev-pipeline run DID drive it."""
    out = set()
    if not isinstance(runs, list):
        return out
    for run in runs:
        if not isinstance(run, dict):
            continue
        inp = run.get("input")
        if isinstance(inp, dict) and "trigger" in inp and "input" in inp:
            inp = inp.get("input")
        if not isinstance(inp, dict):
            continue
        r = inp.get("repo")
        num = inp.get("issue_number")
        if r == repo and isinstance(num, int):
            out.add(num)
    return out


async def _completed_run_issues(repo, workflow_id, limit):
    """The set of issue numbers driven by a COMPLETED dev-pipeline run — positive provenance
    evidence (a completed run is a successful build→merge, exactly the legitimate path). Returns
    ``(issues, ok)``: ``ok`` False when the read degraded/truncated (the witness is then
    unavailable and the provenance check falls back to the durable GitHub labels alone — never a
    false 'no run' violation). Best-effort; the durable gate labels are the primary provenance."""
    if not isinstance(workflow_id, str) or not workflow_id:
        return (set(), False)
    resp = await _list_runs({"workflow_id": workflow_id, "status": "completed",
                             "limit": limit, "account_wide": True})
    if not isinstance(resp, dict) or "error" in resp:
        return (set(), False)
    runs = resp.get("runs")
    if not isinstance(runs, list):
        return (set(), False)
    if len(runs) >= limit:
        # A full page: more completed runs may exist unseen → the witness is unprovable, so we
        # mark it unavailable (the check falls back to the always-present GitHub gate labels).
        return (_run_issue_numbers(runs, repo), False)
    return (_run_issue_numbers(runs, repo), True)


# ─── the merged-PR substrate (recently-merged pipeline PRs to verify) ─────────

async def _recently_merged_v2_prs(repo, limit):
    """The recently-MERGED dev-pipeline (``pipeline:v2``) PRs to verify this sweep — the most-
    recently-updated CLOSED PRs (``sort=updated direction=desc``), filtered to the ones that
    actually MERGED (``merged_at`` set) and carry the pipeline provenance label.

    This is a SINGLE bounded page (``per_page=min(limit,100)``), NOT a full ``gh_paginated``
    walk: the closed-PR list is the repo's whole merge HISTORY, and we deliberately verify only
    the recent slice each sweep (the ``post-merge:checked@`` marker makes re-scanning an
    already-checked merge a no-op, and a sweep cadence above the merge rate guarantees no merge
    falls off the recent slice unchecked). ``_json_body`` still fails LOUD on a truncated /
    unparseable 2xx page, so a half-read never reads as 'no merges to check'.

    Returns the list, or None on a non-2xx read (the read is then the caller's branch — distinct
    from [] = proven-empty). ``limit`` is a positive int (coerced once in ``main()``)."""
    per_page = min(limit, 100)
    resp = await gh("GET", _with_query(_ipath(repo, "/pulls"),
                                       state="closed", sort="updated", direction="desc",
                                       per_page=per_page))
    if _status(resp) != 200:
        return None
    rows = _json_body(resp)  # raises loud on a truncated / unparseable 2xx page
    if not isinstance(rows, list):
        return None
    out = []
    for pr in rows:
        if not isinstance(pr, dict):
            continue
        if not pr.get("merged_at"):
            continue  # closed-but-not-merged PRs never merged anything → nothing to verify
        if LABEL_PIPELINE_V2 not in _label_names(pr):
            continue  # only pipeline-owned merges are in scope
        out.append(pr)
    return out


# ─── the three verifications (each drawn from durable, off-the-run state) ──────

async def _check_issue_closed(repo, pr):
    """(a) THE SOURCE ISSUE ENDED CLOSED. Derive the issue from the merged PR's
    ``dev-pipeline/issue-<n>`` branch and confirm ``state == closed``. Returns a finding dict
    on a violation (merged PR, issue still OPEN) or None (closed / not-derivable / unreadable).
    An unreadable issue read is INDETERMINATE (returns None this sweep, re-checked next), never
    a false violation."""
    issue_number = _source_issue_number(pr)
    if issue_number is None:
        return None  # no derivable source issue — skip (a), don't invent a violation
    resp = await gh("GET", _ipath(repo, "/issues/%d" % issue_number))
    if _status(resp) != 200:
        log("post-merge: issue #%d read non-2xx (%r) — indeterminate this sweep"
            % (issue_number, _status(resp)))
        return None
    issue = _json_body(resp)
    state = issue.get("state") if isinstance(issue, dict) else None
    if state == "open":
        return {"class": "issue-open", "issue": issue_number,
                "detail": "PR #%d MERGED but its source issue #%d is still OPEN — the "
                          "reconciler's best-effort close did not stick (the issue reads as "
                          "live to a dispatch sweep)." % (pr.get("number"), issue_number)}
    return None


async def _check_master_green(repo):
    """(b) THE MERGE DID NOT LEAVE MASTER RED. Read the LIVE post-merge CI verdict on
    ``BASE_BRANCH`` via the deterministic Checks + combined-status API (``_read_ci`` — no agent,
    no model). Returns ``(finding_or_None, verdict_str)`` where verdict_str ∈
    {green,red,no_ci,indeterminate} for the per-sweep summary. A genuinely-RED master → a
    finding (the #1188 alarm). A still-running / unreadable master CI is INDETERMINATE
    (re-checked next sweep) — never coerced to a silent green NOR a false red."""
    head = await _resolve_sha1(repo, BASE_BRANCH)
    if not head:
        log("post-merge: could not resolve %s head sha — master-CI indeterminate" % BASE_BRANCH)
        return (None, "indeterminate")
    verdict, read_ok = await _read_ci(repo, head)
    if not read_ok:
        # Neither CI surface read — cannot trust ANY verdict (down ≠ green). Indeterminate.
        return (None, "indeterminate")
    if verdict is None:
        # CI present but still running on master — not terminal. Re-check next sweep.
        return (None, "indeterminate")
    status = verdict.get("status")
    if status == "red":
        return ({"class": "master-red", "issue": None,
                 "detail": "master (%s @ %s) is RED after this merge landed — a PR that passed "
                           "CI + review broke master post-merge (#1188). %s"
                           % (BASE_BRANCH, head[:12], verdict.get("detail", ""))},
                "red")
    return (None, status)  # green / no_ci


def _check_provenance(repo, pr, completed_issues, witness_ok):
    """(c) THE PROVENANCE WAS LEGITIMATE (merged THROUGH the gate). A merged ``pipeline:v2`` PR
    MUST carry the reconciler's durable gate stamp: a ``reviewed:green@<sha>`` AND an in-cap
    ``risk:tier-N`` (1..AUTO_MERGE_MAX_TIER). Its absence means the PR merged WITHOUT the
    maker≠checker review the auto-merge authority is scoped to (a tier>cap PR, or a force-merge
    outside the gate). Returns a finding dict on a violation, else None.

    The GitHub gate labels are the PRIMARY (always-present-on-a-gate-merge) provenance. The run
    journal is a SECONDARY witness: a completed dev-pipeline run for the source issue corroborates
    the merge; its absence is folded into the detail (and, only when ``REQUIRE_RUN_PROVENANCE`` is
    set AND the witness is available, is itself a violation). ``witness_ok`` False (the journal
    read degraded) NEVER manufactures a violation — we fall back to the labels alone."""
    labels = frozenset(_label_names(pr))
    reviewed = _reviewed_sha(labels)
    tier = _risk_tier(labels)
    number = pr.get("number")
    issue_number = _source_issue_number(pr)

    in_cap_tier = isinstance(tier, int) and 1 <= tier <= AUTO_MERGE_MAX_TIER
    has_gate = (reviewed is not None) and in_cap_tier
    if not has_gate:
        missing = []
        if reviewed is None:
            missing.append("no reviewed:green@<sha> stamp")
        if not isinstance(tier, int):
            missing.append("no risk:tier-N stamp")
        elif not (1 <= tier <= AUTO_MERGE_MAX_TIER):
            missing.append("risk:tier-%d is ABOVE the auto-merge cap (%d)"
                           % (tier, AUTO_MERGE_MAX_TIER))
        return {"class": "bad-provenance", "issue": issue_number,
                "detail": "pipeline:v2 PR #%d MERGED without the reconciler's maker≠checker "
                          "gate (%s) — it bypassed the pre-merge review the auto-merge authority "
                          "is scoped to." % (number, "; ".join(missing))}

    # The gate labels are present. OPTIONAL run-journal corroboration: a completed dev-pipeline
    # run for the source issue is positive evidence. Only treat its ABSENCE as a violation when
    # explicitly required AND the witness was readable (else fall back to the labels alone).
    if REQUIRE_RUN_PROVENANCE and witness_ok and issue_number is not None \
            and issue_number not in completed_issues:
        return {"class": "bad-provenance", "issue": issue_number,
                "detail": "pipeline:v2 PR #%d carries the gate labels but NO completed "
                          "dev-pipeline run drove its source issue #%d in the run journal — "
                          "the gate labels may have been applied out-of-band."
                          % (number, issue_number)}
    return None


async def _escalate(repo, number, label, marker, detail):
    """Post ONE structured escalation comment (idempotent via the maker-marker guard) + stamp
    the per-class ``needs:human/*`` label AND the grep-able ``ESCALATED_LABEL`` so a seat/ops
    sweep finds every post-merge regression at a glance. A label already present is a GitHub
    no-op; the comment dedup means an at-least-once replay never double-posts."""
    await post_markered_comment(repo, number, marker, marker + "\n\n" + detail)
    await _label(repo, number, label)
    await _label(repo, number, ESCALATED_LABEL)


# ─── the post-merge checker (one sweep: read merges → 3 checks each → escalate/mark) ──

async def main(input):
    phase("config")
    cfg = _config(input)
    repo = cfg.get("repo") or DEFAULT_REPO
    workflow_id = cfg.get("dev_pipeline_workflow_id", DEV_PIPELINE_WORKFLOW_ID)
    # Coerce the scan cap to a positive int ONCE here (the single source of truth), so a cron
    # config that delivers ``scan_limit`` as a JSON string never reaches a ``len(runs) >= limit``
    # comparison downstream (TypeError) — both the merged-PR scan and the run-journal witness read
    # receive a clean int. A non-numeric / non-positive value falls back to the rendered default.
    limit = SCAN_LIMIT
    raw_limit = cfg.get("scan_limit", SCAN_LIMIT)
    try:
        parsed_limit = int(raw_limit)
        if parsed_limit > 0:
            limit = parsed_limit
    except (TypeError, ValueError):
        log("post-merge: non-numeric scan_limit %r — using default %d" % (raw_limit, SCAN_LIMIT))
    if not isinstance(repo, str) or not repo:
        return {"verdict": "cannot-determine", "scanned": 0, "found": [], "checked": [],
                "reason": "no repo provided (input.repo) and no DEFAULT_REPO configured"}

    # ── the merged-PR substrate (recently-merged pipeline:v2 PRs) ──
    phase("scan")
    merged = await _recently_merged_v2_prs(repo, limit)
    if merged is None:
        # A non-2xx / truncated merged-PR read taints the whole sweep — fail loud
        # cannot-determine, NEVER a silent under-count read as "no regressions".
        log("post-merge: merged-PR read degraded")
        return {"verdict": "cannot-determine", "scanned": 0, "found": [], "checked": [],
                "reason": "merged-PR list read failed (non-2xx / truncated)"}

    # ── the run-journal provenance witness (read ONCE, reused across PRs) ──
    phase("provenance-witness")
    completed_issues, witness_ok = await _completed_run_issues(repo, workflow_id, limit)

    # ── read master's post-merge CI verdict ONCE per sweep (it is a property of master, not of
    # a single PR; a red master is attributed to whichever merged-PR sweep observes it — the
    # seat triages which merge broke it). ──
    phase("master-ci")
    master_finding, master_verdict = await _check_master_green(repo)

    found = []
    checked = []
    phase("verify")
    for pr in merged:
        number = pr.get("number")
        if not isinstance(number, int):
            continue
        labels = frozenset(_label_names(pr))
        msha = _merge_sha(pr)
        if _already_checked(labels, msha):
            continue  # this merge sha was already verified clean — idempotent skip
        # Skip a PR a human is already handling for a post-merge violation (any needs:human/*
        # we stamp) — re-escalating a PR the seat is on is noise. A fresh needs:human means
        # un-cleared; we still re-stamp idempotently (the comment dedup prevents a double-post).

        pr_findings = []
        # (a) source issue closed
        f_issue = await _check_issue_closed(repo, pr)
        if f_issue is not None:
            f_issue["pr"] = number
            await _escalate(repo, number, NEEDS_HUMAN_ISSUE_OPEN, MARKER_ISSUE_OPEN,
                            f_issue["detail"])
            pr_findings.append(f_issue)
        # (b) master not red — attribute the (sweep-wide) red master to each unchecked merge
        if master_finding is not None:
            f_master = dict(master_finding)
            f_master["pr"] = number
            await _escalate(repo, number, NEEDS_HUMAN_MASTER_RED, MARKER_MASTER_RED,
                            f_master["detail"])
            pr_findings.append(f_master)
        # (c) provenance legitimate
        f_prov = _check_provenance(repo, pr, completed_issues, witness_ok)
        if f_prov is not None:
            f_prov["pr"] = number
            await _escalate(repo, number, NEEDS_HUMAN_BAD_PROVENANCE, MARKER_BAD_PROVENANCE,
                            f_prov["detail"])
            pr_findings.append(f_prov)

        if pr_findings:
            found.extend(pr_findings)
            # A violated merge is NOT stamped checked — it stays pull-able for the seat (and a
            # re-sweep re-confirms / clears once resolved). The needs:human/* label is the
            # durable record.
        else:
            # All three clean → mark this merge sha verified-once (bounded, idempotent). We do
            # NOT mark when master is INDETERMINATE this sweep (a still-running master CI must be
            # re-checked next sweep before we declare the merge clean) — only when master is a
            # settled green/no_ci AND issue+provenance passed.
            if master_verdict in ("green", "no_ci") and msha:
                await _label(repo, number, LABEL_CHECKED_PREFIX + msha)
                checked.append(number)

    # ── the structured per-sweep summary (verdict ∈ ok / regression-found / cannot-determine) ──
    phase("summary")
    if found:
        verdict = "regression-found"
        reason = None
    elif master_verdict == "indeterminate" and merged:
        # We scanned merges but could not settle master's CI — not a clean bill of health.
        verdict = "cannot-determine"
        reason = "master CI did not reach a terminal verdict this sweep"
    else:
        verdict = "ok"
        reason = None
    summary = {"verdict": verdict, "repo": repo, "scanned": len(merged),
               "master_verdict": master_verdict, "witness_ok": witness_ok,
               "found": found, "checked": checked}
    if reason is not None:
        summary["reason"] = reason
    log("post-merge checker verdict:", verdict, "scanned:", len(merged),
        "found:", len(found), "checked:", len(checked))
    return summary
'''


def build_post_merge_checker_script(
    *,
    github_server: str = "github",
    base_branch: str = "master",
    escalated_label: str = DEFAULT_ESCALATED_LABEL,
    auto_merge_max_tier: int = AUTO_MERGE_MAX_TIER,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    require_run_provenance: bool = False,
    dev_pipeline_workflow_id: str | None = None,
    repo: str | None = None,
) -> str:
    """Return the production post-merge-checker workflow source.

    The post-merge TWIN of the reconciler's ``owner()`` pre-merge gate (task #76): a separate
    cron-fired stateless workflow that verifies, from DURABLE off-the-run GitHub state, that
    every recently-merged ``pipeline:v2`` PR (a) closed its source issue, (b) did not leave
    ``base_branch`` red, and (c) carried the reconciler's maker≠checker gate provenance — and
    escalates any violation via a ``needs:human/*`` label + idempotent comment.

    ``auto_merge_max_tier`` is imported from the reconciler so the provenance check enforces the
    SAME tier-cap the merge branch did. ``require_run_provenance`` (default OFF) makes the
    absence of a driving dev-pipeline run in the journal a violation; left off, the run journal
    is a secondary witness and the durable GitHub gate labels are the primary provenance.
    """
    header = _render_constants(
        github_server=github_server,
        base_branch=base_branch,
        escalated_label=escalated_label,
        auto_merge_max_tier=auto_merge_max_tier,
        scan_limit=scan_limit,
        require_run_provenance=require_run_provenance,
        dev_pipeline_workflow_id=dev_pipeline_workflow_id,
        default_repo=repo,
    )
    # Splice five source strings, each authored once (module-level defs resolve names at CALL
    # time, so the order between body/lib/helpers is free):
    #   - _BODY: the unwrap + merged-PR scan + the three verifications + the checker main().
    #   - DEV_PIPELINE_LIB: gh/_ipath/_with_query/_status, the deterministic _read_ci (#1316),
    #     _resolve_sha1 (#1178), _label/_unlabel, post_markered_comment — spliced VERBATIM (one
    #     source of truth with the reconciler + monolith; no drift).
    #   - GH_BODY_HELPERS (#1294): _json_body / gh_paginated (fail LOUD on truncated/unparseable).
    #   - COMMENT_IDEMPOTENCY_HELPERS (#1292): the maker-marker comment-POST dedup.
    return header + "\n" + _BODY + DEV_PIPELINE_LIB + GH_BODY_HELPERS + COMMENT_IDEMPOTENCY_HELPERS


def build_post_merge_checker_fixture_script(
    *,
    repo: str | None = None,
    scan_limit: int = 50,
    require_run_provenance: bool = False,
    dev_pipeline_workflow_id: str | None = None,
) -> str:
    """The CI fixture variant: a tighter scan cap, otherwise the identical script shape. Driven
    by ``tests/integration/test_wf_dev_pipeline_post_merge_fixture.py`` against the host with
    simulated GitHub / list_runs returns."""
    return build_post_merge_checker_script(
        repo=repo,
        scan_limit=scan_limit,
        require_run_provenance=require_run_provenance,
        dev_pipeline_workflow_id=dev_pipeline_workflow_id,
    )


# ─── deploy surface (the tool + http_server envelope a WorkflowCreate needs) ──
#
# The post-merge checker READS GitHub (merged PRs, the source issue's state, master's CI) and
# the run journal (list_runs, provenance witness, run-callable since #1397), and MUTATES only
# the escalation comment + label set. It needs NO bash / read / write / edit / agent / gate —
# and STRUCTURALLY cannot merge/close/resolve anything (no PUT/PATCH route, no gate), which
# makes "a checker that only escalates, never acts on the merge it judges" a structural property
# (the maker≠checker-across-the-merge-boundary invariant: a different stage AND a different,
# read+escalate-only surface than the reconciler's effector).
REQUIRED_TOOLS: list[ToolSpec] = [
    ToolSpec(type="http_request"),
    ToolSpec(type="list_runs"),
]


def _github_http_server(*, name: str, base_url: str) -> HttpServerSpec:
    return HttpServerSpec(
        name=name,
        base_url=base_url,
        description="GitHub REST API (auth resolved from the bound vault's GITHUB_TOKEN).",
        routes=[
            HttpRouteSpec(
                # GET (list merged PRs, read the source issue's state, read master's CI surfaces,
                # read a comment thread), POST (the escalation comment + the needs:human/* +
                # escalated labels). NO PUT/PATCH/DELETE: the checker NEVER merges, closes,
                # unlabels, or resolves a gate — it only escalates. That the surface CANNOT
                # mutate the merge it judges is the structural maker≠checker-across-the-boundary
                # guarantee.
                path_pattern="/repos/**",
                methods=["GET", "POST"],
                allow_query=True,
                description="Merged PRs, issues, labels, comments, commit CI (GET reads; POST "
                "comments + labels); list endpoints paginate via ?state/?sort/?per_page/?page.",
            ),
        ],
    )


REQUIRED_HTTP_SERVERS: list[HttpServerSpec] = [
    _github_http_server(name="github", base_url="https://api.github.com")
]


def build_post_merge_checker_workflow_create(
    *,
    name: str = "dev-pipeline-post-merge-checker",
    description: str | None = None,
    github_server: str = "github",
    github_base_url: str = "https://api.github.com",
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` for the production post-merge checker.

    Bundles the script (``build_post_merge_checker_script``) with its tool + http_server surface
    (``REQUIRED_TOOLS = [http_request, list_runs]`` and the GET·POST-only ``github`` server — NO
    PUT/PATCH/DELETE, NO gate), so a deployer POSTs one object and the declared surface can never
    drift from the script that needs it (#1135). The deploy-time wiring (a ``CronSource`` →
    ``WorkflowAction`` firing every N min with the ``{repo, dev_pipeline_workflow_id}`` input
    template, on a session whose agent surface SUPERSETS this surface — the launcher-clamp rule)
    lives outside this object.
    """
    return WorkflowCreate(
        name=name,
        description=description,
        script=build_post_merge_checker_script(github_server=github_server, **script_kwargs),
        tools=list(REQUIRED_TOOLS),
        http_servers=[_github_http_server(name=github_server, base_url=github_base_url)],
    )
