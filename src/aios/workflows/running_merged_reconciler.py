"""Running==merged reconciler workflow script (aios#1327, plane B2 of the
substrate-different-verdict invariant; epic #1330, roadmap item 11).

``build_running_merged_reconciler_script`` returns workflow *source code* (the
``deep_research.py`` / ``dev_pipeline.py`` builder idiom) — a fully deterministic,
**scriptable (no agent)** workflow launched by a ``CronSource → WorkflowAction``
trigger. The SHA-diff predicate has a universally-correct policy, so it is
admissible as a script per the script-vs-agent partition doctrine.

WHAT IT RECONCILES — "the merged code is running". The deploy-actor's own ledger
says only what it INTENDED to run; it is maker==checker on the comparison target
(Coolify's app-config ``git_commit_sha`` literally returns ``'HEAD'``). The only
genuinely-uncorrelated read of what is RUNNING is the eumemic-ops audit that SSHes
to the host and ``docker inspect``s the LIVE container's image. This reconciler
compares that live read against **git master HEAD** (immutable, not fleet-authored
= the uncorrelated expected substrate), per app, and routes drift.

THE TWO READS (every read in the value domain — None/str/int/bool/list/dict):

1. **Expected SHA = git master HEAD.** ``tool('http_request', GET /repos/<repo>/commits/master)``
   against GitHub (vault-bound ``GITHUB_TOKEN``); take ``.sha``. A failed / truncated
   list read is ``cannot-determine`` (the no-silent-degrade list-read invariant,
   roadmap item 1 — ``gh_paginated`` raises ``GitHubListIncomplete`` on an
   unfetched ``rel="next"``; spliced from ``gh_body``).

2. **Running SHA = ops-audit ``docker inspect LABEL org.eumemic.build_sha`` over SSH,
   ONE read path for EVERY app** (the resolution of the previously-open fork — the
   reconciler does NOT branch on app type). The worker has no HTTP server, so SSH +
   ``docker inspect`` is the sole path for it; aios-api *could* be read via
   ``GET /health.build_sha`` (which still ships from Unit A2 as the human/ad-hoc
   liveness read) but is PINNED to the same SSH read to keep the reconciler a single
   deterministic read path. Invoked via ``tool('bash')`` cloning eumemic-ops and
   running the ``coolify-running-container-fresh.sh``-derived label read.
   NOT the deploy-ledger; NOT Coolify ``git_commit_sha`` (returns ``'HEAD'``).

PER-APP, NEVER AGGREGATE. The reconciler runs the read PER APP and decides PER APP.
A read that returns ``null`` / ``unknown`` / ``__SSH_FAIL__`` / ``__MISSING__`` /
unlabelled is ``cannot-determine`` for THAT app — NEVER coerced to ``ok``, and never
collapsed into an aggregate ``ok`` that hides one app's unreadability behind
another's match.

VERDICTS (per app):
  * ``ok``               — both reads succeeded and ``running_sha == master_head``
                           (short/long normalised: compare on the running SHA's
                           length prefix).
  * ``drifted``          — both reads succeeded and differ.
  * ``cannot-determine`` — that app's running SHA is unreadable, OR the master read
                           failed / ``cannot-determine``. NEVER coerced to ``ok``.

INTERIM MODE (A-interim — ships FIRST, before A1/A2/A3 land). Until the build-SHA is
in the image, the running read bakes ``unknown`` → ``cannot-determine`` for the
running axis. The interim mode still adds value: read master HEAD and the
deploy-ledger's last track-G SHA (acknowledged maker==checker), diff them, and emit
``needs-deploy`` when ledger-SHA ≠ master-HEAD (= "master moved past the last
deploy" — the dominant drift shape), AND ``cannot-determine`` for the *running*-vs-
ledger axis, explicitly self-flagged as not-yet-substrate-different so it is never
mistaken for the real running-truth read. When A1/A2/A3 land the reconciler runs in
full mode (``interim=False``) and the interim path is gone (one right way, no alias).

ROUTING (buildable-today tier only). Any ``drifted`` / ``cannot-determine`` /
``needs-deploy`` app → file/upsert ONE GitHub issue (idempotent by a stable title +
maker-marker body, carrying the per-app verdicts). NO algedonic lane, NO
cockpit-resume (deferred — ``role_bindings`` / priority-wake absent; out of scope).

LATENCY-WINDOW (gap #3): cron can be up to its interval late, so a clean reconcile is
"no drift seen at last poll", not "zero drift". ``external_event`` / #1281 can narrow
it later; out of scope here.

IDEMPOTENCY. The side-effecting node (file/upsert the drift issue) is idempotent by
construction: it lists open issues, finds one by the stable title, and upserts a
markered comment via ``post_comment_once`` (the maker-marker replay guard) rather than
double-filing — so a crash re-drive / re-fire never double-files.

DETERMINISM. Imports only ``re`` / ``json``; all capability I/O in the value domain;
capabilities emitted in a FIXED, BOUNDED order (master read, then per-app reads in
the configured ``APPS`` order, then the single routing read/write) — so replay-with-
memo is stable.
"""

from __future__ import annotations

from typing import Any

from aios.models.agents import HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.triggers import CronSource, TriggerCreate, WorkflowAction
from aios.models.workflows import WorkflowCreate
from aios.workflows.comment_idempotency import COMMENT_IDEMPOTENCY_HELPERS
from aios.workflows.gh_body import GH_BODY_HELPERS

# ─── defaults (override per deployment) ──────────────────────────────────────
DEFAULT_REPO = "eumemic/aios"
DEFAULT_OPS_REPO = "eumemic/eumemic-ops"
# The Coolify apps reconciled, in a FIXED order (replay-stable capability ordering).
# Each carries the Coolify app uuid the ops-audit reads the running container of.
DEFAULT_APPS: tuple[dict[str, str], ...] = (
    {"name": "aios-api", "uuid": "aios-api"},
    {"name": "aios-worker", "uuid": "aios-worker"},
)
# Off-peak default cron (every 20 minutes). The latency-window known-miss class: a
# clean reconcile is "no drift seen at last poll", bounded by this interval.
DEFAULT_SCHEDULE = "*/20 * * * *"

# The stable drift-issue title (idempotency-by-title key) and the maker-marker that
# dedups the upserted comment across replays / re-fires.
DRIFT_ISSUE_TITLE = "running==merged reconciler: drift / cannot-determine"
MARKER_DRIFT = "## running-merged-reconciler"

# Sentinels the ops-audit running-SHA read emits for an unreadable container. ANY of
# these (or null/unknown/unlabelled/empty) is cannot-determine for that app.
UNREADABLE_SENTINELS: tuple[str, ...] = ("unknown", "__SSH_FAIL__", "__MISSING__")


def _py(name: str, value: Any) -> str:
    """One ``NAME = <repr>`` constant line for the prepended header."""
    return f"{name} = {value!r}"


def _render_constants(
    *,
    repo: str,
    ops_repo: str,
    apps: tuple[dict[str, str], ...],
    github_server: str,
    base_branch: str,
    interim: bool,
) -> str:
    lines = [
        _py("REPO", repo),
        _py("OPS_REPO", ops_repo),
        _py("APPS", [dict(a) for a in apps]),
        _py("GITHUB_SERVER", github_server),
        _py("BASE_BRANCH", base_branch),
        _py("INTERIM", interim),
        _py("DRIFT_ISSUE_TITLE", DRIFT_ISSUE_TITLE),
        _py("MARKER_DRIFT", MARKER_DRIFT),
        _py("UNREADABLE_SENTINELS", list(UNREADABLE_SENTINELS)),
    ]
    return "\n".join(lines)


# The static script body — references the prepended constants. Pure stdlib (re/json),
# value-domain I/O, bounded loops in a fixed order: replay-stable by construction.
_BODY = r'''
import json
import re


# ─── scripted helpers (deterministic; no LLM, no time, no random) ────────────

def _unwrap(input):
    """Accept both a bare input dict and the trigger envelope
    ``{"trigger": ..., "input": ...}`` a WorkflowAction fire delivers."""
    if isinstance(input, dict) and "trigger" in input and "input" in input:
        return input["input"] or {}
    return input or {}


def _ipath(repo, suffix):
    return "/repos/%s%s" % (repo, suffix)


async def _gh_once(method, path, body=None):
    """One GitHub REST/GraphQL call through the run's bound-vault-authed http_request.
    A non-2xx or transport error is a VALUE the caller branches on, never a raise.
    ``path`` must NOT carry a query string unless the route opts into allow_query."""
    args = {"server_ref": GITHUB_SERVER, "path": path, "method": method}
    if body is not None:
        args["body"] = json.dumps(body)
    return await tool("http_request", args)


_TRANSIENT_ERROR_PREFIXES = ("Request timed out", "HTTP transport error")


def _is_transient(resp):
    """A retryable GitHub response: a 5xx, OR a genuine transport transient. A 4xx is
    the caller's problem (auth, 404, 422) — NOT retried. A broker gate rejection
    (route-allowlist / SSRF / path) is DETERMINISTIC, not transient."""
    if not isinstance(resp, dict):
        return True
    err = resp.get("error")
    if err is not None:
        return isinstance(err, str) and err.startswith(_TRANSIENT_ERROR_PREFIXES)
    st = resp.get("status")
    return isinstance(st, int) and 500 <= st <= 599


async def gh(method, path, body=None):
    """A GitHub call with bounded transient-5xx retry (≤3). Returns the LAST result
    for the caller to branch on (a value, never a raise). Each attempt is a fresh
    ``tool()`` await, so the CallKeyer gives it a distinct ordinal and replay is
    stable (the retry decision is a pure function of the memoized result)."""
    resp = None
    for _ in range(3):
        resp = await _gh_once(method, path, body)
        if not _is_transient(resp):
            return resp
        log("gh transient failure, retrying:", method, path, _status(resp))
    return resp


def _headers(resp):
    if not isinstance(resp, dict):
        return {}
    h = resp.get("headers")
    if not isinstance(h, dict):
        return {}
    return {str(k).lower(): v for k, v in h.items()}


def _link_next_page(link_header):
    """The ``page`` number of the ``rel="next"`` URL in a GitHub Link header, or None.
    We rebuild the next path against the path WE control, so a malformed/host-bearing
    next URL never reaches the route gate."""
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
    """Append ``params`` as a query string (deterministic key order; ints/strs)."""
    qs = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    return path + ("?" if qs else "") + qs


def _status(resp):
    return resp.get("status") if isinstance(resp, dict) else None


# ``_json_body`` / ``gh_paginated`` / ``GitHubListIncomplete`` are spliced in from the
# shared ``GH_BODY_HELPERS`` source (aios#1294/#1323): a truncated / unparseable 2xx
# body or an under-counted paginated list raises (fail loud, never degrade to None/[]).
# ``post_comment_once`` is spliced from ``COMMENT_IDEMPOTENCY_HELPERS`` (aios#1292).


# ─── running-SHA read (the ops-audit docker-inspect LABEL over SSH) ──────────

def _running_sha_command(uuid):
    """The bash command that reads ONE app's LIVE running SHA — the genuinely-
    uncorrelated substrate (the live container, a different actor than the deploy
    ledger). Clones eumemic-ops and runs the audit's docker-inspect-LABEL reader,
    which SSHes to the host and inspects the RUNNING container's image
    ``LABEL org.eumemic.build_sha``. Prints exactly one line ``RUNNING_SHA=<value>``
    where <value> is the SHA, or a sentinel (``__SSH_FAIL__`` / ``__MISSING__`` /
    ``unknown`` for an unlabelled image) that the caller maps to cannot-determine.

    Re-run tolerant (rm -rf then clone) per the at-least-once bash contract. The
    token in the clone URL is an egress-proxy placeholder (as in dev_pipeline)."""
    lines = [
        "set -u",
        "D=/workspace/ops-audit",
        'rm -rf "$D"',
        "git clone --quiet --depth 1 "
        "https://x-access-token:$GITHUB_TOKEN@github.com/%s.git \"$D\" "
        "|| { echo 'RUNNING_SHA=__SSH_FAIL__'; exit 0; }" % OPS_REPO,
        'cd "$D"',
        # The audit script emits ``image|status|startedAt`` (or __SSH_FAIL__/__MISSING__)
        # and is driven by the LABEL read; we ask it for the build_sha LABEL of the
        # running container of this app's uuid and normalise its output to one line.
        'SHA="$(audit/checks/coolify-running-container-fresh.sh --app %s --emit-build-sha '
        "2>/dev/null)\" || SHA=__SSH_FAIL__" % uuid,
        '[ -z "$SHA" ] && SHA=__MISSING__',
        'echo "RUNNING_SHA=$SHA"',
    ]
    return "\n".join(lines)


def _parse_running_sha(bash_result):
    """Pull the ``RUNNING_SHA=<value>`` line out of a tool('bash') result. Returns the
    raw string value (possibly a sentinel), or None when the read itself failed to
    produce the line (non-dict result, missing line, nonzero exit with no line) —
    None is cannot-determine, never coerced to ok."""
    if not isinstance(bash_result, dict):
        return None
    out = bash_result.get("stdout")
    if not isinstance(out, str):
        return None
    val = None
    for line in out.splitlines():
        m = re.match(r"^RUNNING_SHA=(.*)$", line.strip())
        if m:
            val = m.group(1).strip()  # last wins (the command prints exactly one)
    if val is None or val == "":
        return None
    return val


def _is_unreadable_sha(value):
    """True when a running-SHA read is cannot-determine: None, empty, a sentinel
    (__SSH_FAIL__/__MISSING__/unknown), or not a hex-looking commit id."""
    if not isinstance(value, str) or not value:
        return True
    if value in UNREADABLE_SENTINELS:
        return True
    return not re.match(r"^[0-9a-fA-F]{7,64}$", value)


def _shas_match(running, master):
    """Short/long normalised SHA equality: compare on the RUNNING sha's length prefix
    (a short ``abc1234`` running id matches a full master id with that prefix). Both
    lower-cased. Caller guarantees both are non-empty readable strings."""
    r = running.lower()
    m = master.lower()
    n = min(len(r), len(m))
    return r[:n] == m[:n]


# ─── the deploy-ledger read (INTERIM mode only) ──────────────────────────────

def _ledger_sha_command():
    """INTERIM mode: read the deploy-ledger's last track-G SHA (acknowledged
    maker==checker — the deploy actor's self-report, NOT a running-truth read).
    Clones eumemic-ops and reads the ledger tail. Prints ``LEDGER_SHA=<value>`` (or
    a sentinel). This axis is self-flagged not-yet-substrate-different in routing."""
    lines = [
        "set -u",
        "D=/workspace/ops-audit",
        'rm -rf "$D"',
        "git clone --quiet --depth 1 "
        "https://x-access-token:$GITHUB_TOKEN@github.com/%s.git \"$D\" "
        "|| { echo 'LEDGER_SHA=__SSH_FAIL__'; exit 0; }" % OPS_REPO,
        'cd "$D"',
        'SHA="$(scripts/fleet ledger-last-track-g-sha 2>/dev/null)" || SHA=__SSH_FAIL__',
        '[ -z "$SHA" ] && SHA=__MISSING__',
        'echo "LEDGER_SHA=$SHA"',
    ]
    return "\n".join(lines)


def _parse_ledger_sha(bash_result):
    if not isinstance(bash_result, dict):
        return None
    out = bash_result.get("stdout")
    if not isinstance(out, str):
        return None
    val = None
    for line in out.splitlines():
        m = re.match(r"^LEDGER_SHA=(.*)$", line.strip())
        if m:
            val = m.group(1).strip()
    if val is None or val == "":
        return None
    return val


# ─── routing (file/upsert ONE idempotent drift issue) ────────────────────────

def _verdicts_need_routing(verdicts):
    """True if ANY per-app verdict is not a clean ``ok`` — i.e. drifted /
    cannot-determine / needs-deploy. A clean reconcile (every app ok) routes nothing."""
    for v in verdicts:
        if v.get("verdict") != "ok":
            return True
    return False


def _drift_body(verdicts, master_head, interim):
    """The stable, self-identifying drift-issue comment body (carries MARKER_DRIFT so
    the maker-marker guard dedups it across replays). Renders the per-app verdicts and
    both SHAs deterministically (sorted by app name)."""
    lines = [MARKER_DRIFT, ""]
    lines.append("mode: %s" % ("interim (running axis = cannot-determine)" if interim else "full"))
    lines.append("master HEAD: %s" % (master_head if master_head else "cannot-determine"))
    lines.append("")
    for v in sorted(verdicts, key=lambda x: x.get("app", "")):
        lines.append(
            "- **%s** → `%s` (running=%r master=%r%s)"
            % (
                v.get("app", "?"),
                v.get("verdict", "?"),
                v.get("running"),
                v.get("master"),
                (" note=%s" % v["note"]) if v.get("note") else "",
            )
        )
    return "\n".join(lines)


async def _find_open_issue_by_title(repo, title):
    """Find an OPEN issue whose title equals ``title`` (idempotency-by-title). Lists
    open issues with the no-silent-degrade paginated read (raises on an under-counted
    list — a partial list could miss the existing issue and double-file). Returns the
    issue dict or None."""
    # GET /repos/{repo}/issues defaults to state=open; gh_paginated adds per_page/page,
    # so we pass a CLEAN path and filter in-script (PRs surface on the issues list and
    # carry a "pull_request" key — exclude them so we never adopt a PR as the issue).
    items = await gh_paginated(_ipath(repo, "/issues"))
    if not isinstance(items, list):
        return None
    for it in items:
        if isinstance(it, dict) and it.get("title") == title and "pull_request" not in it:
            return it
    return None


async def _route_drift(repo, verdicts, master_head, interim):
    """File / upsert ONE drift issue, idempotent by title + maker-marker. If an open
    issue with DRIFT_ISSUE_TITLE exists, upsert a markered comment on it (skipped if
    the freshly-read thread already carries the marker — the replay guard); else
    create the issue carrying the marker in its body. Returns the issue number."""
    body = _drift_body(verdicts, master_head, interim)
    existing = await _find_open_issue_by_title(repo, DRIFT_ISSUE_TITLE)
    if existing is not None:
        number = int(existing["number"])
        thread = await gh_paginated(_ipath(repo, "/issues/%d/comments" % number))
        await post_comment_once(
            repo, number, MARKER_DRIFT, body,
            thread if isinstance(thread, list) else [],
        )
        return number
    created = await gh(
        "POST", _ipath(repo, "/issues"),
        {"title": DRIFT_ISSUE_TITLE, "body": body, "labels": ["deploy-drift"]},
    )
    issue = _json_body(created)
    return int(issue["number"]) if isinstance(issue, dict) and "number" in issue else None


# ─── the reconciler state machine ────────────────────────────────────────────

async def main(input):
    payload = _unwrap(input)
    repo = payload.get("repo") or REPO
    interim = bool(payload.get("interim", INTERIM))

    # ── STEP 1: expected SHA = git master HEAD (immutable, uncorrelated substrate) ──
    # The no-silent-degrade contract: a failed / truncated read is cannot-determine
    # for the master axis, which makes EVERY app cannot-determine — never coerced to ok.
    phase("read-master")
    master_head = None
    master_readable = True
    resp = await gh("GET", _ipath(repo, "/commits/%s" % BASE_BRANCH))
    if _status(resp) == 200:
        commit = _json_body(resp)  # raises loud on a truncated/unparseable 2xx body
        if isinstance(commit, dict) and isinstance(commit.get("sha"), str) and commit["sha"]:
            master_head = commit["sha"]
        else:
            master_readable = False
    else:
        master_readable = False
    if not master_readable:
        log("master read failed (status %r) -> cannot-determine for every app" % _status(resp))

    # ── STEP 2+3: per-app running read + per-app diff (NEVER aggregate) ──
    phase("read-running")
    verdicts = []
    if interim:
        # A-interim: read the deploy-ledger's last track-G SHA (acknowledged
        # maker==checker) and diff vs master. The running axis is explicitly
        # cannot-determine and self-flagged not-yet-substrate-different.
        ledger_raw = await tool("bash", {"command": _ledger_sha_command()})
        ledger_sha = _parse_ledger_sha(ledger_raw)
        for app in APPS:
            name = app["name"]
            # The needs-deploy axis (ledger vs master) — the dominant drift shape.
            if not master_readable:
                nd_verdict = "cannot-determine"
            elif _is_unreadable_sha(ledger_sha):
                nd_verdict = "cannot-determine"
            elif _shas_match(ledger_sha, master_head):
                nd_verdict = "ok"
            else:
                nd_verdict = "needs-deploy"
            # The running-vs-ledger axis is ALWAYS cannot-determine in interim mode,
            # self-flagged so it is never mistaken for the real running-truth read.
            verdicts.append({
                "app": name,
                "verdict": nd_verdict,
                "running": None,
                "master": master_head,
                "ledger": ledger_sha,
                "note": "interim: running axis cannot-determine (not-yet-substrate-different)",
            })
    else:
        # Full mode: ONE running-SHA read per app — the ops-audit docker inspect LABEL
        # over SSH. Decide PER APP; never collapse into an aggregate ok.
        for app in APPS:
            name = app["name"]
            uuid = app["uuid"]
            running_raw = await tool("bash", {"command": _running_sha_command(uuid)})
            running_sha = _parse_running_sha(running_raw)
            if not master_readable:
                verdict = "cannot-determine"
                note = "master read failed"
            elif _is_unreadable_sha(running_sha):
                verdict = "cannot-determine"
                note = "running SHA unreadable (%r)" % running_sha
            elif _shas_match(running_sha, master_head):
                verdict = "ok"
                note = ""
            else:
                verdict = "drifted"
                note = ""
            verdicts.append({
                "app": name,
                "verdict": verdict,
                "running": running_sha,
                "master": master_head,
                "note": note,
            })

    # ── STEP 5: routing (buildable-today tier: file/upsert ONE idempotent issue) ──
    phase("route")
    issue_number = None
    if _verdicts_need_routing(verdicts):
        issue_number = await _route_drift(repo, verdicts, master_head, interim)

    clean = not _verdicts_need_routing(verdicts)
    return {
        "state": "reconciled",
        "mode": "interim" if interim else "full",
        "master_head": master_head,
        "verdicts": verdicts,
        "clean": clean,
        "routed_issue": issue_number,
    }
'''


def build_running_merged_reconciler_script(
    *,
    repo: str = DEFAULT_REPO,
    ops_repo: str = DEFAULT_OPS_REPO,
    apps: tuple[dict[str, str], ...] = DEFAULT_APPS,
    github_server: str = "github",
    base_branch: str = "master",
    interim: bool = False,
) -> str:
    """Return the running==merged reconciler workflow source.

    ``interim=True`` ships FIRST (A-interim): the running axis is cannot-determine and
    the needs-deploy signal (ledger-SHA ≠ master-HEAD) is the value-add until the
    build-SHA-in-image primitive (A1/A2/A3) lands. ``interim=False`` is the full
    reconciler: one running-SHA read per app via the ops-audit docker-inspect LABEL.
    """
    header = _render_constants(
        repo=repo,
        ops_repo=ops_repo,
        apps=apps,
        github_server=github_server,
        base_branch=base_branch,
        interim=interim,
    )
    # Splice the shared GitHub-body helper (fail-loud on truncated/under-counted reads)
    # and the comment-idempotency helper (maker-marker dedup) — each authored once and
    # injected so the class fix can't drift (#1294/#1323/#1292).
    return header + "\n" + _BODY + GH_BODY_HELPERS + COMMENT_IDEMPOTENCY_HELPERS


# ─── deploy surface (the tool + http_server envelope a WorkflowCreate needs) ──

REQUIRED_TOOLS: list[ToolSpec] = [
    ToolSpec(type="bash"),
    ToolSpec(type="http_request"),
]


def _github_http_server(*, name: str, base_url: str) -> HttpServerSpec:
    return HttpServerSpec(
        name=name,
        base_url=base_url,
        description="GitHub REST API (auth resolved from the bound vault's GITHUB_TOKEN).",
        routes=[
            HttpRouteSpec(
                # GET: read master HEAD + list open issues + read a thread. POST: file
                # the drift issue / upsert its comment. allow_query: the issues/comments
                # list reads paginate via ?per_page/?page and the no-silent-degrade
                # invariant must follow Link: rel="next" to prove completeness (#1323).
                path_pattern="/repos/**",
                methods=["GET", "POST"],
                allow_query=True,
                description="Read master HEAD; list/create issues + comments; "
                "list endpoints paginate via ?per_page/?page.",
            ),
        ],
    )


REQUIRED_HTTP_SERVERS: list[HttpServerSpec] = [
    _github_http_server(name="github", base_url="https://api.github.com")
]


def build_running_merged_reconciler_workflow_create(
    *,
    name: str,
    description: str | None = None,
    github_server: str = "github",
    github_base_url: str = "https://api.github.com",
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` payload for the reconciler.

    Bundles the script with the tool + http_server surface it requires, so the declared
    surface can never drift from the script that needs it (#1135). Any remaining keyword
    args are forwarded verbatim to ``build_running_merged_reconciler_script``.
    """
    script = build_running_merged_reconciler_script(github_server=github_server, **script_kwargs)
    return WorkflowCreate(
        name=name,
        description=description,
        script=script,
        tools=list(REQUIRED_TOOLS),
        http_servers=[_github_http_server(name=github_server, base_url=github_base_url)],
    )


def build_running_merged_reconciler_trigger(
    *,
    name: str,
    workflow_id: str,
    vault_id: str,
    schedule: str = DEFAULT_SCHEDULE,
    interim: bool = False,
    workflow_version: int | None = None,
) -> TriggerCreate:
    """Return the ``CronSource → WorkflowAction`` trigger document (acceptance item 5).

    Off-peak cron (default every 20 min) → launches ``workflow_id``. ``vault_id`` binds
    the ``GITHUB_TOKEN`` the reconciler's ``http_request`` calls authenticate with. The
    run input is the standard envelope ``{"trigger": ..., "input": <input_template>}``;
    the template carries the ``interim`` flag so the trigger selects interim-vs-full
    without rebuilding the workflow. ``CronSource`` + ``WorkflowAction`` are SHIPPED.
    """
    return TriggerCreate(
        name=name,
        source=CronSource(schedule=schedule),
        action=WorkflowAction(
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            input_template={"interim": interim},
            vault_ids=[vault_id],
        ),
    )
