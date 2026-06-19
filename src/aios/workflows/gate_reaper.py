"""Standing reaper / gate dead-man (aios#1386) — make PR abandonment impossible.

Abandonment is possible today: a parked gate with no resume, a DIRTY PR outside any
run, or an unchecked best-effort terminal label write leaves a zombie with no owner.
The dev-pipeline ``run_completion`` auto-recovery fires only on ``status=errored``
(crashed runs), NOT on parked/DIRTY — so a run that *parked* and was forgotten has no
armed alarm. The dev-pipeline freeze of 2026-06-18 was caught only by the chairman
noticing ~5h later. This module stands the armed alarm: a SEPARATE scheduled workflow
that backstops the pipeline so nothing is silently abandoned.

─── THE UNCORRELATED SUBSTRATE (why this is a separate cron workflow) ──────────────

The reaper's verdict draws ONLY from the live GitHub/run artifacts — an uncorrelated,
off-the-run substrate. A run that parked CANNOT un-park itself, so the alarm must live
OUTSIDE any pipeline run. This mirrors the ``observer_sweep`` dead-man (aios#1326): a
``run_completion`` trigger auto-disables after consecutive failures and cannot fire on
its own non-firing — silence ≠ health. So the reaper is a CRON-fired (``CronSource`` →
``WorkflowAction``) standing workflow that re-derives an abandonment verdict every sweep
from two off-the-run substrates that no parked run controls:

  * GitHub: open issues labelled ``autodev:in-progress``; open PRs and their
    ``mergeable_state`` (the CONFLICTING/DIRTY signal).
  * The run journal: ``list_runs`` over the watched dev-pipeline workflow — which runs
    are LIVE (non-terminal: ``pending``/``running``/``suspended``) and, for the
    suspended ones, how long they have been parked (``updated_at``).

The reaper correlates the two: an issue/PR that GitHub says is in-flight but that NO
live run is driving is an abandoned zombie. A correlation across two substrates neither
of which a parked run can rewrite is the load-bearing property — the issue's
"uncorrelated, off-the-run substrate" requirement.

─── THE THREE ABANDONMENT CLASSES (the issue's Spec, detect + act each sweep) ──────

  1. ``autodev:in-progress`` issue with NO live/active run → the pipeline claimed the
     issue then died/parked-and-forgot. BOUNDED RE-DRIVE: surface a re-drive
     recommendation; after K prior reaper passes (counted off the reaper's own marker
     comments — an off-the-run tally), ESCALATE-WITH-OWNER instead of recommending
     re-drive forever (no infinite re-dispatch).
  2. A gate open (a ``suspended`` run) PAST a staleness threshold (> N hours) →
     ESCALATE to the seat/ops owner. The reaper NEVER auto-resolves an approval gate
     (the issue's hard rule): it can only escalate. Approval is a human decision; a
     reaper that resolved gates would be a rubber stamp, defeating the gate.
  3. An open PR in CONFLICTING/DIRTY ``mergeable_state`` with NO driving run → re-drive
     into the rebase stage (Issue A) or, past the re-drive cap, escalate.

─── WHY THE REAPER ESCALATES (and does not itself re-dispatch) ─────────────────────

Re-DISPATCH — actually launching a fresh dev-pipeline run — rides the dispatch trigger
(#75), which the issue puts explicitly OUT OF SCOPE, and the in-run rebase capability
rides Issue A. So the reaper's ACTION is to make the abandonment IMPOSSIBLE TO MISS, not
to silently re-launch: it posts ONE idempotent, structured escalation comment per class
per issue/PR and stamps an escalation label. The structured per-sweep summary records
both what was FOUND and what was ACTED, and whether a finding is a re-drive
recommendation (below the cap) or an owner escalation (gate, or cap reached). When the
dispatch trigger (#75) lands, the re-drive recommendation is the signal it consumes; the
reaper's contract — detect within one sweep, never silently abandon — holds today.

─── DETERMINISM (replay-stable; the cron envelope's frozen clock) ──────────────────

Authored with the EXACT ``dev_pipeline.py`` / ``telemetry_observer.py`` builder idiom:
an exported ``build_reaper_script(...) -> str`` returning workflow SOURCE (a prepended
frozen-band constants header + a static ``_REAPER_BODY`` of pure-stdlib ``re``/``json``,
value-domain I/O, deterministic emit order → replay-stable), plus ``REQUIRED_TOOLS`` /
``REQUIRED_HTTP_SERVERS``, and ``build_reaper_workflow_create(...) -> WorkflowCreate``
bundling script + surface so the declared surface can't drift from the script (#1135).

The ONE clock the reaper needs is gate staleness (criterion: a gate open > N hours). It
NEVER calls ``datetime.now()`` (that would desync replay). Instead it reads the FROZEN
``trigger.fired_at`` the cron fire stamps into the run envelope by value
(``harness/trigger_runner.py``: ``compose_workflow_run_input`` →
``trigger["fired_at"] = fired_at.isoformat()``) — a single, replay-stable "now" pinned
at fire time. Staleness is ``fired_at - run.updated_at`` against the frozen ``N``-hour
band; the band lives as a prepended constant (never inferred at runtime). The body parses
ISO-8601 timestamps with ``re`` only — no ``datetime`` import — so replay stays stable.

The reaper's list reads go through the shared ``gh_paginated`` (aios#1294/#1323): a
truncated/unparseable 2xx page or a page-ceiling-with-dangling-``rel=next`` is fatal-loud
``cannot-determine``, NEVER a silent under-count read as "nothing abandoned". A reaper
that under-counts the substrate it judges is exactly the look-green-while-doing-less
failure the dead-man exists to prevent.
"""

from __future__ import annotations

from typing import Any

from aios.models.agents import HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.comment_idempotency import COMMENT_IDEMPOTENCY_HELPERS
from aios.workflows.gh_body import GH_BODY_HELPERS

# ─── frozen bands (git-versioned blueprint constants; bumped with BANDS_VERSION) ──
#
# These live as constants in the prepended header and are committed in the blueprint.
# The body NEVER infers a band at runtime — gate staleness gates on this frozen N-hour
# threshold compared against the cron envelope's frozen ``trigger.fired_at``.

# Criterion 2: a gate (a suspended run) open longer than this many hours is STALE →
# escalate to the owner. Tuned so a healthy gate (resumed within a working interval) is
# never escalated, but a parked-and-forgotten gate is caught within ~one sweep of N.
DEFAULT_GATE_STALE_HOURS = 6

# Bounded re-drive (the issue's "cap at K attempts, then escalate"): how many prior
# reaper sweeps may RECOMMEND a re-drive for the same zombie before the reaper stops
# recommending and ESCALATES-WITH-OWNER instead. Counted off the reaper's own marker
# comments (an off-the-run tally), so the cap survives reaper restarts. K=3.
DEFAULT_MAX_REDRIVE_ATTEMPTS = 3

# The labels the in-flight dev pipeline claims an issue with (must match
# dev_pipeline.LABEL_IN_PROGRESS). An issue carrying this with no live run is class 1.
DEFAULT_IN_PROGRESS_LABEL = "autodev:in-progress"

# The label the reaper STAMPS on an escalated issue/PR so a seat/ops sweep can find
# every reaper escalation at a glance (and so a human can clear it on resolution).
DEFAULT_ESCALATED_LABEL = "autodev:reaper-escalated"

# Bump on ANY band change so a stored sweep summary records which band set produced it.
DEFAULT_BANDS_VERSION = "v1-2026-06-17"

# The verdict vocabulary for the per-sweep summary — EXACTLY these three.
VERDICTS: tuple[str, ...] = ("ok", "abandonment-found", "cannot-determine")

# The three abandonment classes the reaper detects (the issue's Spec).
ABANDONMENT_CLASSES: tuple[str, ...] = (
    "in-progress-no-run",  # class 1
    "stale-gate",  # class 2
    "dirty-pr-no-run",  # class 3
)

# The action the reaper takes for a finding. It NEVER auto-resolves a gate; it escalates
# (post a structured comment + stamp the label) and — below the re-drive cap, for the
# re-drivable classes — records a re-drive recommendation for the dispatch trigger (#75).
ACTIONS: tuple[str, ...] = ("redrive-recommended", "escalated")

# A non-terminal run status is a LIVE run driving its issue. Mirrors
# models/workflows.TERMINAL_RUN_STATUSES = {completed, errored, cancelled}; the
# complement {pending, running, suspended} is "live/active".
LIVE_RUN_STATUSES: tuple[str, ...] = ("pending", "running", "suspended")


def _py(name: str, value: Any) -> str:
    """One ``NAME = <repr>`` constant line for the prepended header (mirrors dev/triage)."""
    return f"{name} = {value!r}"


def _render_reaper_constants(
    *,
    gate_stale_hours: int,
    max_redrive_attempts: int,
    in_progress_label: str,
    escalated_label: str,
    github_server: str,
    bands_version: str,
) -> str:
    lines = [
        _py("GATE_STALE_HOURS", gate_stale_hours),
        _py("MAX_REDRIVE_ATTEMPTS", max_redrive_attempts),
        _py("IN_PROGRESS_LABEL", in_progress_label),
        _py("ESCALATED_LABEL", escalated_label),
        _py("GITHUB_SERVER", github_server),
        _py("BANDS_VERSION", bands_version),
        _py("LIVE_RUN_STATUSES", list(LIVE_RUN_STATUSES)),
        _py("ABANDONMENT_CLASSES", list(ABANDONMENT_CLASSES)),
        # Stable comment markers — each is BOTH the comment's first line AND the
        # maker-marker the idempotency guard scans for (aios#1292): a posted comment is
        # its own "already done" marker, so an at-least-once replay never duplicates it
        # AND the per-class marker count is the off-the-run re-drive tally.
        _py("MARKER_IN_PROGRESS", "## Reaper: in-progress issue with no live run"),
        _py("MARKER_STALE_GATE", "## Reaper: gate parked past the staleness threshold"),
        _py("MARKER_DIRTY_PR", "## Reaper: conflicting PR with no live run"),
    ]
    return "\n".join(lines)


# The static reaper body — references the prepended constants. Pure stdlib (re/json),
# value-domain I/O, bounded loops, NO datetime import (the only clock is the frozen
# ``trigger.fired_at``): replay-stable by construction.
_REAPER_BODY = r'''
import json
import re


# ─── input-envelope unwrap (cron fire OR bare fixture/arm-time shape) ─────────

def _config(input):
    """The reaper's config — ``{repo, dev_pipeline_workflow_id, gate_stale_hours?,
    max_redrive_attempts?}`` — out of EITHER the bare fixture/arm-time shape OR the
    cron WorkflowAction envelope ``{"trigger": ..., "input": <template>}`` (a cron fire
    carries the author's template verbatim under ``input``; the reaper takes its config
    from there)."""
    if isinstance(input, dict) and "trigger" in input and "input" in input:
        return input.get("input") or {}
    return input or {}


def _fired_at(input):
    """The cron fire's FROZEN timestamp (the reaper's only "now" — replay-stable). The
    cron envelope stamps ``trigger.fired_at`` (an ISO-8601 string) by value; the bare
    fixture shape may pass ``now`` instead. Returns the ISO string or None (None =
    cannot compute staleness → that class reads cannot-determine, never a false ok)."""
    if not isinstance(input, dict):
        return None
    trig = input.get("trigger")
    if isinstance(trig, dict):
        fa = trig.get("fired_at")
        if isinstance(fa, str) and fa:
            return fa
    now = input.get("now")
    if isinstance(now, str) and now:
        return now
    return None


# ─── ISO-8601 epoch-seconds (no datetime import — replay-stable pure arithmetic) ──

_TS_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})"
    r"(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)


def _is_leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def _days_before_year(y):
    """Days from 0001-01-01 to year ``y`` (proleptic Gregorian), the stdlib formula."""
    y -= 1
    return y * 365 + y // 4 - y // 100 + y // 400


_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_before_month(y, m):
    days = 0
    for i in range(1, m):
        days += _DAYS_IN_MONTH[i - 1]
        if i == 2 and _is_leap(y):
            days += 1
    return days


def _epoch_seconds(ts):
    """Parse an ISO-8601 timestamp to integer epoch-seconds (UTC), or None if it does
    not match the expected shape. Pure integer arithmetic (no datetime), so replay is
    stable. A naive timestamp (no offset) is read as UTC. The offset is applied so two
    timestamps in different zones compare correctly."""
    if not isinstance(ts, str):
        return None
    m = _TS_RE.match(ts.strip())
    if not m:
        return None
    yy, mm, dd, hh, mi, ss = (int(m.group(i)) for i in range(1, 7))
    if not (1 <= mm <= 12) or not (1 <= dd <= 31):
        return None
    ordinal = _days_before_year(yy) + _days_before_month(yy, mm) + (dd - 1)
    days_since_epoch = ordinal - 719162  # ordinal of 1970-01-01
    secs = days_since_epoch * 86400 + hh * 3600 + mi * 60 + ss
    off = m.group(7)
    if off and off != "Z":
        sign = 1 if off[0] == "+" else -1
        digits = off[1:].replace(":", "")
        oh, om = int(digits[:2]), int(digits[2:4])
        secs -= sign * (oh * 3600 + om * 60)
    return secs


def _hours_between(later_iso, earlier_iso):
    """Whole-ish hours between two ISO timestamps (later - earlier), or None if either
    fails to parse. A float; the caller compares it to the frozen GATE_STALE_HOURS."""
    a = _epoch_seconds(later_iso)
    b = _epoch_seconds(earlier_iso)
    if a is None or b is None:
        return None
    return (a - b) / 3600.0


# ─── GitHub helpers (single-shot value-returning; retry + pagination spliced in) ──

async def _gh_once(method, path, body=None):
    """One GitHub REST call through the run's bound-vault-authed http_request. A non-2xx
    or transport error is a VALUE the caller branches on, never a raise. ``path`` may
    carry a query string (the route opts into allow_query for list pagination)."""
    args = {"server_ref": GITHUB_SERVER, "path": path, "method": method}
    if body is not None:
        args["body"] = json.dumps(body)
    return await tool("http_request", args)


_TRANSIENT_ERROR_PREFIXES = ("Request timed out", "HTTP transport error")


def _is_transient(resp):
    """Retryable: a 5xx, or a genuine transport transient surfaced as {"error": ...}.
    A 4xx / broker gate rejection is deterministic and NOT retried (#1139)."""
    if not isinstance(resp, dict):
        return True
    err = resp.get("error")
    if err is not None:
        return isinstance(err, str) and err.startswith(_TRANSIENT_ERROR_PREFIXES)
    st = resp.get("status")
    return isinstance(st, int) and 500 <= st <= 599


async def gh(method, path, body=None):
    """A GitHub call with bounded transient-5xx retry (≤3). Returns the LAST result for
    the caller to branch on (a value, never a raise). Each attempt is a fresh tool()
    await (distinct call_key), so replay stays stable."""
    resp = None
    for _ in range(3):
        resp = await _gh_once(method, path, body)
        if not _is_transient(resp):
            return resp
        log("reaper gh transient failure, retrying:", method, path, _status(resp))
    return resp


def _headers(resp):
    if not isinstance(resp, dict):
        return {}
    h = resp.get("headers")
    if not isinstance(h, dict):
        return {}
    return {str(k).lower(): v for k, v in h.items()}


def _link_next_page(link_header):
    if not isinstance(link_header, str) or not link_header:
        return None
    for part in link_header.split(","):
        seg = part.strip()
        if 'rel="next"' not in seg:
            continue
        m = re.search(r"[?&]page=(\d+)", seg)
        if m:
            return int(m.group(1))
    return None


def _with_query(path, **params):
    qs = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    return path + ("?" if qs else "") + qs


def _status(resp):
    return resp.get("status") if isinstance(resp, dict) else None


# ``_json_body`` / ``gh_paginated`` (fail-loud on truncated/unparseable/under-counted
# reads) and ``post_comment_once`` (the maker-marker idempotency guard) are spliced in
# from the shared GH_BODY_HELPERS / COMMENT_IDEMPOTENCY_HELPERS sources below.


def _ipath(repo, suffix):
    return "/repos/%s%s" % (repo, suffix)


# ─── run-substrate read (the off-the-run liveness/staleness journal) ──────────

async def _list_runs(args):
    """One list_runs read (full WfRun rows minus the script/surface blobs — input,
    status, updated_at ARE present). account_wide so the reaper sees the whole
    account's dev-pipeline runs, not only ones its own session launched."""
    return await tool("list_runs", args)


def _page_runs(resp):
    """The runs list out of a list_runs return, or None if the read errored/degraded."""
    if not isinstance(resp, dict):
        return None
    if "error" in resp and "runs" not in resp:
        return None
    runs = resp.get("runs")
    if not isinstance(runs, list):
        return None
    return runs


def _run_issue_key(run):
    """The (repo, issue_number) a dev-pipeline run is driving, dug out of its input
    envelope. A run's input is EITHER the bare ``{repo, issue_number}`` or the trigger
    envelope ``{"trigger": ..., "input": {repo, issue_number}}``. Returns (repo, int) or
    None when no issue key is reachable (an un-correlatable run — surfaced, not papered
    over: it cannot vouch for any issue)."""
    if not isinstance(run, dict):
        return None
    inp = run.get("input")
    if isinstance(inp, dict) and "trigger" in inp and "input" in inp:
        inp = inp.get("input")
    if not isinstance(inp, dict):
        return None
    repo = inp.get("repo")
    num = inp.get("issue_number")
    if isinstance(repo, str) and repo and isinstance(num, int):
        return (repo, num)
    return None


async def _run_index(workflow_id, limit):
    """Build the off-the-run run index in ONE list_runs read per status:

      * ``live_keys`` — set of (repo, issue_number) with at least one LIVE (non-terminal:
        pending/running/suspended) run driving it.
      * ``suspended_runs`` — the raw suspended run rows (the open gates, for staleness).
      * ``terminal_counts`` — {(repo, issue_number): count} of TERMINAL (errored +
        cancelled) dev-pipeline runs for that issue — the off-the-run RE-DISPATCH TALLY
        that bounds the re-drive cap (each failed re-dispatch leaves a terminal run). A
        ``completed`` run is NOT counted: a completed run is success, not a failed
        re-drive (it would have merged + closed the issue, so it won't even be
        in-progress). This is the issue's "cap at K attempts, then escalate" tally drawn
        from the live run substrate, not from reaper-internal state.

    Returns ``(live_keys, suspended_runs, terminal_counts, error)``. ``error`` is a reason
    string when a read failed/degraded or a page truncated (the no-silent-degrade trip: a
    full page means runs may exist unseen, so the index is unprovable → the whole sweep is
    cannot-determine, NEVER a silent under-count that reads a live run as dead and
    re-drives a healthy issue)."""
    live_keys = set()
    suspended_runs = []
    terminal_counts = {}
    for status in list(LIVE_RUN_STATUSES) + ["errored", "cancelled"]:
        req = {"workflow_id": workflow_id, "status": status,
               "limit": limit, "account_wide": True}
        resp = await _list_runs(req)
        runs = _page_runs(resp)
        if runs is None:
            return (live_keys, suspended_runs, terminal_counts,
                    "list_runs read failed for status=%s" % status)
        if len(runs) >= limit:
            # A full page: more runs MAY exist unseen → we cannot prove the index →
            # refuse to under-count (a missed live run would wrongly re-drive a healthy
            # issue; a missed terminal run would under-count the re-drive tally).
            # aios#1323 no-silent-degrade.
            return (live_keys, suspended_runs, terminal_counts,
                    "truncated list_runs page for status=%s (>= limit %d)" % (status, limit))
        for run in runs:
            key = _run_issue_key(run)
            if status in LIVE_RUN_STATUSES:
                if key is not None:
                    live_keys.add(key)
                if status == "suspended":
                    suspended_runs.append(run)
            else:
                # errored / cancelled → a spent re-dispatch attempt for this issue.
                if key is not None:
                    terminal_counts[key] = terminal_counts.get(key, 0) + 1
    return (live_keys, suspended_runs, terminal_counts, None)


# ─── GitHub-substrate reads (in-progress issues; open PRs + mergeable_state) ──

async def _in_progress_issues(repo):
    """Open issues carrying the in-progress label. ``gh_paginated`` follows
    Link rel=next in FULL and fails loud on a truncated/under-counted read (so a
    half-read never reads as "nothing abandoned"). Returns the issue list or None when
    a non-2xx (auth/404) makes the read the caller's branch — distinct from [] (a
    proven-empty list)."""
    path = _ipath(repo, "/issues")
    # state=open + labels filter applied via query (allow_query route). PRs are also
    # "issues" on this endpoint — filter them out by the pull_request key below.
    items = await gh_paginated(_with_query(path, state="open", labels=IN_PROGRESS_LABEL))
    if not isinstance(items, list):
        return None
    return [it for it in items if isinstance(it, dict) and "pull_request" not in it]


async def _open_pulls(repo):
    """Open PRs (lean list rows — no mergeable_state). Returns the list or None."""
    items = await gh_paginated(_with_query(_ipath(repo, "/pulls"), state="open"))
    if not isinstance(items, list):
        return None
    return items


async def _pr_mergeable_state(repo, number):
    """The single-PR read's ``mergeable_state`` — only the full GET /pulls/{n} carries
    it (the list rows do not). GitHub computes mergeability ASYNCHRONOUSLY, so a freshly
    opened PR can read ``unknown`` transiently; the reaper treats ONLY the settled
    ``dirty`` (merge conflict) as the class-3 signal, and an ``unknown`` as
    not-yet-determined (it will settle by the next sweep). Returns the state string, or
    None on a non-2xx read."""
    resp = await gh("GET", _ipath(repo, "/pulls/%d" % number))
    if _status(resp) != 200:
        return None
    pr = _json_body(resp)
    if not isinstance(pr, dict):
        return None
    return pr.get("mergeable_state")


# ─── the structured per-sweep summary (found + acted — the issue's emission) ──

def _redrive_or_escalate(spent, cap, redrive_detail, escalate_detail):
    """The bounded-re-drive decision (the issue's "cap at K attempts, then escalate"):
    below the cap → ``("redrive-recommended", <redrive_detail>)``; at/above → escalate
    to the owner ``("escalated", <escalate_detail>)``. ``spent`` is the off-the-run
    terminal-run tally (spent re-dispatch attempts). Returns ``(action, detail)``."""
    if spent >= cap:
        return ("escalated", "escalated to owner: " + escalate_detail)
    return ("redrive-recommended",
            "re-drive recommended (attempt %d of cap %d): %s"
            % (spent + 1, cap, redrive_detail))


def _finding(klass, target, action, detail):
    """One structured finding row in the sweep summary. ``klass`` ∈ ABANDONMENT_CLASSES,
    ``action`` ∈ {redrive-recommended, escalated} (the reaper NEVER auto-resolves a
    gate), ``target`` is the issue/PR number, ``detail`` is the human-readable reason."""
    return {"class": klass, "target": target, "action": action, "detail": detail}


def _summary(verdict, found, acted, scanned, reason=None):
    """The structured per-sweep summary — the reaper's PRIMARY output (the issue's
    "structured summary each sweep: found + acted"). Machine-readable telemetry, not a
    log scrape."""
    out = {
        "verdict": verdict,
        "scanned": scanned,
        "found": found,
        "acted": acted,
        "bands_version": BANDS_VERSION,
    }
    if reason is not None:
        out["reason"] = reason
    return out


async def _escalate(repo, number, marker, body):
    """Post ONE structured escalation comment (idempotent via the maker-marker guard:
    a freshly-fetched thread already carrying ``marker`` is a no-op, so an at-least-once
    replay never double-posts) and stamp the escalation label so a seat/ops sweep finds
    every reaper escalation at a glance. A label already present is a GitHub no-op."""
    existing = await gh_paginated(_ipath(repo, "/issues/%d/comments" % number))
    thread = existing if isinstance(existing, list) else []
    await post_comment_once(repo, number, marker, body, thread)
    await gh("POST", _ipath(repo, "/issues/%d/labels" % number),
             {"labels": [ESCALATED_LABEL]})


# ─── entry ────────────────────────────────────────────────────────────────────

async def main(input):
    phase("config")
    cfg = _config(input)
    repo = cfg.get("repo")
    workflow_id = cfg.get("dev_pipeline_workflow_id")
    fired_at = _fired_at(input)
    gate_stale_hours = cfg.get("gate_stale_hours", GATE_STALE_HOURS)
    max_redrive = cfg.get("max_redrive_attempts", MAX_REDRIVE_ATTEMPTS)
    limit = cfg.get("limit", 200)

    if not isinstance(repo, str) or not repo or not isinstance(workflow_id, str):
        log("reaper: missing repo / dev_pipeline_workflow_id")
        return _summary("cannot-determine", [], [], 0,
                        reason="missing repo / dev_pipeline_workflow_id")

    # ── the off-the-run run index (one list_runs read per status) ──
    phase("runs")
    live_keys, suspended_runs, terminal_counts, run_err = await _run_index(workflow_id, limit)
    if run_err is not None:
        # An unprovable run index taints EVERY class (each correlates against it). Fail
        # loud cannot-determine — NEVER under-count live runs into false abandonment.
        log("reaper: run-substrate read degraded:", run_err)
        return _summary("cannot-determine", [], [], 0, reason=run_err)

    found = []
    acted = []

    # ── class 1: in-progress issue with NO live run → bounded re-drive / escalate ──
    phase("in-progress")
    issues = await _in_progress_issues(repo)
    if issues is None:
        log("reaper: in-progress issue read degraded")
        return _summary("cannot-determine", found, acted, 0,
                        reason="in-progress issue read failed (non-2xx)")
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        number = issue.get("number")
        if not isinstance(number, int):
            continue
        if (repo, number) in live_keys:
            continue  # a live run is driving it — not abandoned
        # No live run drives an in-progress issue: a zombie. Bound the re-drive on the
        # off-the-run terminal-run tally (each spent re-dispatch left an errored/cancelled
        # run): below the cap → recommend re-drive; at/above → escalate-with-owner.
        spent = terminal_counts.get((repo, number), 0)
        action, detail = _redrive_or_escalate(
            spent, max_redrive,
            "re-dispatch the dev pipeline for issue #%d" % number,
            "%d spent dev-pipeline runs for issue #%d (>= cap %d) — needs a human, not "
            "another re-dispatch" % (spent, number, max_redrive))
        body = (MARKER_IN_PROGRESS + "\n\n"
                "Issue #%d is labelled `%s` but NO live dev-pipeline run is driving it "
                "— the run crashed or parked-and-was-forgotten. %s"
                % (number, IN_PROGRESS_LABEL, detail))
        await _escalate(repo, number, MARKER_IN_PROGRESS, body)
        f = _finding("in-progress-no-run", number, action, detail)
        found.append(f)
        acted.append(f)

    # ── class 2: a gate (suspended run) parked past the staleness threshold → escalate ──
    phase("gates")
    if fired_at is None:
        # No frozen clock → cannot compute staleness; surface it, never read a parked
        # gate as fresh (a false ok). Other classes already acted; mark the verdict.
        log("reaper: no frozen fired_at — cannot compute gate staleness")
        gate_verdict_unknown = True
    else:
        gate_verdict_unknown = False
        for run in suspended_runs:
            if not isinstance(run, dict):
                continue
            updated = run.get("updated_at")
            key = _run_issue_key(run)
            number = key[1] if key is not None else None
            hours = _hours_between(fired_at, updated)
            if hours is None:
                # Unparseable timestamp on a parked gate — surface, don't paper over.
                f = _finding("stale-gate", number, "escalated",
                             "gate run %r parked but updated_at %r is unparseable — "
                             "cannot prove freshness" % (run.get("id"), updated))
                found.append(f)
                acted.append(f)
                if number is not None:
                    await _escalate(repo, number, MARKER_STALE_GATE,
                                    MARKER_STALE_GATE + "\n\n" + f["detail"])
                continue
            if hours <= gate_stale_hours:
                continue  # a fresh gate — within the threshold, not stale
            detail = ("gate parked %.1fh (> %dh threshold) on run %r — ESCALATED to the "
                      "owner; the reaper NEVER auto-resolves an approval gate"
                      % (hours, gate_stale_hours, run.get("id")))
            f = _finding("stale-gate", number, "escalated", detail)
            found.append(f)
            acted.append(f)
            if number is not None:
                await _escalate(repo, number, MARKER_STALE_GATE,
                                MARKER_STALE_GATE + "\n\n" + detail)

    # ── class 3: conflicting/DIRTY PR with no driving run → re-drive / escalate ──
    phase("dirty-prs")
    pulls = await _open_pulls(repo)
    if pulls is None:
        log("reaper: open-PR read degraded")
        return _summary("cannot-determine", found, acted, len(issues),
                        reason="open-PR read failed (non-2xx)")
    for pr in pulls:
        if not isinstance(pr, dict):
            continue
        number = pr.get("number")
        if not isinstance(number, int):
            continue
        # The list row's issue linkage: a dev-pipeline PR drives the SAME issue number
        # only by branch convention, so we correlate the PR's OWN number against the
        # live set AND treat a PR with no live run as ownerless.
        if (repo, number) in live_keys:
            continue  # a live run is driving this PR
        state = await _pr_mergeable_state(repo, number)
        if state != "dirty":
            continue  # only a settled merge-conflict (dirty) is the class-3 signal
        spent = terminal_counts.get((repo, number), 0)
        action, detail = _redrive_or_escalate(
            spent, max_redrive,
            "re-drive PR #%d into the rebase stage (Issue A)" % number,
            "%d spent runs for PR #%d (>= cap %d) — the rebase did not stick; needs a "
            "human" % (spent, number, max_redrive))
        body = (MARKER_DIRTY_PR + "\n\n"
                "PR #%d is CONFLICTING/DIRTY (merge conflict) and NO live dev-pipeline "
                "run is driving it. %s" % (number, detail))
        await _escalate(repo, number, MARKER_DIRTY_PR, body)
        f = _finding("dirty-pr-no-run", number, action, detail)
        found.append(f)
        acted.append(f)

    scanned = len(issues) + len(suspended_runs) + len(pulls)
    if gate_verdict_unknown:
        verdict = "cannot-determine"
        reason = "no frozen fired_at — gate staleness uncomputable this sweep"
    elif found:
        verdict = "abandonment-found"
        reason = None
    else:
        verdict = "ok"
        reason = None
    log("reaper verdict:", verdict, "found:", len(found), "acted:", len(acted))
    return _summary(verdict, found, acted, scanned, reason=reason)
'''


def build_reaper_script(
    *,
    gate_stale_hours: int = DEFAULT_GATE_STALE_HOURS,
    max_redrive_attempts: int = DEFAULT_MAX_REDRIVE_ATTEMPTS,
    in_progress_label: str = DEFAULT_IN_PROGRESS_LABEL,
    escalated_label: str = DEFAULT_ESCALATED_LABEL,
    github_server: str = "github",
    bands_version: str = DEFAULT_BANDS_VERSION,
) -> str:
    """Return the production ``gate_reaper`` workflow source.

    Frozen bands are prepended as constants; the body imports neither ``datetime`` nor
    ``time`` and never infers a band at runtime — gate staleness gates on the frozen
    ``GATE_STALE_HOURS`` band compared against the cron envelope's frozen
    ``trigger.fired_at``. The shared ``gh_paginated`` / ``post_comment_once`` helper
    sources are spliced in so the fail-loud-on-truncation / maker-marker-idempotency
    contracts can never drift from the dev/triage pipelines.
    """
    header = _render_reaper_constants(
        gate_stale_hours=gate_stale_hours,
        max_redrive_attempts=max_redrive_attempts,
        in_progress_label=in_progress_label,
        escalated_label=escalated_label,
        github_server=github_server,
        bands_version=bands_version,
    )
    # Splice the shared GitHub-body helpers (aios#1294/#1323: _json_body / gh_paginated
    # that fail loud on truncated/under-counted reads) and the comment-idempotency guard
    # (aios#1292: post_comment_once / the maker-marker) BEFORE the body's main(), so the
    # body's references resolve.
    return (
        header + "\n" + GH_BODY_HELPERS + "\n" + COMMENT_IDEMPOTENCY_HELPERS + "\n" + _REAPER_BODY
    )


def build_reaper_fixture_script(
    *,
    gate_stale_hours: int = DEFAULT_GATE_STALE_HOURS,
    max_redrive_attempts: int = DEFAULT_MAX_REDRIVE_ATTEMPTS,
    bands_version: str = DEFAULT_BANDS_VERSION,
) -> str:
    """The CI fixture variant — the identical script shape (the reaper takes no scan cap
    beyond its list-page limit). Driven by
    ``tests/integration/test_wf_gate_reaper_fixture.py``."""
    return build_reaper_script(
        gate_stale_hours=gate_stale_hours,
        max_redrive_attempts=max_redrive_attempts,
        bands_version=bands_version,
    )


# ─── deploy surface (the tool + http_server envelope a WorkflowCreate needs) ──
#
# The reaper reads the run journal via list_runs and reads/mutates GitHub via
# http_request (list issues/PRs, post the escalation comment, stamp the label). It needs
# NO bash / read / write / edit / agent / gate — it mutates nothing but the escalation
# comment+label, and (structurally) CANNOT resolve a gate (no resume_gate / gate in its
# surface), which is the issue's hard rule made structural: the reaper escalates approval
# gates, it never auto-resolves them.
REQUIRED_TOOLS: list[ToolSpec] = [
    ToolSpec(type="list_runs"),
    ToolSpec(type="http_request"),
]


def _github_http_server(*, name: str, base_url: str) -> HttpServerSpec:
    return HttpServerSpec(
        name=name,
        base_url=base_url,
        description="GitHub REST API (auth resolved from the bound vault's GITHUB_TOKEN).",
        routes=[
            HttpRouteSpec(
                # GET (list issues/PRs, read a PR's mergeable_state, read a thread),
                # POST (post the escalation comment, stamp the escalated label). No
                # PUT/PATCH/DELETE: the reaper never merges, never closes, never
                # unlabels — and never resolves a gate.
                path_pattern="/repos/**",
                methods=["GET", "POST"],
                # allow_query: the in-progress / open-PR list reads paginate via
                # ?state/?labels/?per_page/?page and must follow Link rel=next in full
                # so a late zombie is never silently dropped (the no-silent-degrade
                # list-read invariant, aios#1323).
                allow_query=True,
                description="Issues, PRs, labels, comments (GET reads; POST comments + labels); "
                "list endpoints paginate via ?state/?labels/?per_page/?page.",
            ),
        ],
    )


REQUIRED_HTTP_SERVERS: list[HttpServerSpec] = [
    _github_http_server(name="github", base_url="https://api.github.com")
]


def build_reaper_workflow_create(
    *,
    name: str = "gate_reaper",
    description: str | None = None,
    github_server: str = "github",
    github_base_url: str = "https://api.github.com",
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` for the production ``gate_reaper``.

    Bundles the script (``build_reaper_script``) with its tool + http_server surface
    (``REQUIRED_TOOLS = [list_runs, http_request]`` and the GET·POST-only ``github``
    server — NO PUT/PATCH/DELETE, NO resume_gate/gate), so a deployer POSTs one object
    and the declared surface can never drift from the script that needs it (#1135). The
    deploy-time wiring (a ``CronSource`` → ``WorkflowAction`` firing every 30-60 min with
    the ``{repo, dev_pipeline_workflow_id}`` input template) lives outside this object —
    the reaper reads ``trigger.fired_at`` from whatever cron the deployer arms.
    """
    return WorkflowCreate(
        name=name,
        description=description,
        script=build_reaper_script(github_server=github_server, **script_kwargs),
        tools=list(REQUIRED_TOOLS),
        http_servers=[_github_http_server(name=github_server, base_url=github_base_url)],
    )
