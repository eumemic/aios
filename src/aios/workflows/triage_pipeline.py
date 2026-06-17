"""Stateless intake-triage workflow (aios#1226) — the company's intake organ.

This workflow triages **untriaged** GitHub issues in a repo and routes each into the
issue-state model, so newly-filed issues flow into the backlog **without the CEO seat
hand-triaging**. It is fired on a schedule by a cron ``WorkflowAction`` trigger (wired
separately, after this lands), and is authored with the SAME workflow script-host
conventions as ``dev_pipeline.py`` — a versioned code artifact, not an ad-hoc script.

The exported builders return workflow *source code* (the ``dev_pipeline.py`` pattern):
``build_triage_pipeline_script`` is the production workload authored into the runtime via
``aios workflows create``; ``build_triage_pipeline_fixture_script`` is the CI variant with
a tight scan cap, driven by ``tests/integration/test_wf_triage_pipeline_fixture.py``
against the host with simulated agent/tool returns to prove the machine replays and
completes (and is idempotent — a re-run touches nothing already labeled).

DESIGN (settled in #1226 — built to this, not redesigned):

- **Stateless + idempotent.** Each run re-scans the open issues from scratch; safe to fire
  repeatedly; a no-op when nothing is untriaged; it NEVER re-touches an already-labeled
  issue. "Untriaged" = an open issue carrying NONE of the triage state-labels
  (``needs-design`` / ``shovel-ready`` / ``needs-decision`` / ``approved`` / ``dispatched``).
  The filter is applied in-script over the listed issues — so the set is recomputed every
  run and an issue labeled by a prior run drops out of the next run's working set.

- **Ephemeral judgment.** One ``agent()`` call per untriaged issue (a flavor-1 worker, no
  standing session) classifies it into EXACTLY ONE class:
    * ``shovel-ready``  — scope clear, design settled, buildable as-is → apply ``shovel-ready``.
      **NEVER apply ``approved``**: approval is a separate chairman/seat gate (the two-axis
      issue-state model — spec-readiness ⊥ approval). Triage only ever sets the
      spec-readiness axis.
    * ``needs-design``  — architecture unresolved → apply ``needs-design``. v1 just LABELS;
      the design-vet automation (design ``agent()`` → adversarial red-team ``agent()`` →
      settle-to-``shovel-ready`` or fork-to-``needs-decision``) is aios#1218 and plugs into
      this branch later. We build a clean SEAM (``_apply_classification`` dispatches per
      class), NOT the branch.
    * ``needs-decision`` — requires chairman judgment (strategy / capital / external
      commitment / genuinely ambiguous) → apply ``needs-decision`` AND post ONE concise
      comment stating *why* it needs a decision (the settled-vs-forks split where relevant).

- **Structured run summary.** The terminal ``return`` is a structured summary: per-class
  counts + the explicit list of issues escalated to ``needs-decision`` (with each one's
  reason), the scanned/untriaged counts, and any per-issue errors. So a scheduled run's
  outcome is machine-readable telemetry, not a log scrape.

CREDENTIALS / GITHUB ACCESS: this reuses ``dev_pipeline.py``'s existing GitHub access path
— a ``tool('http_request', {server_ref: GITHUB_SERVER, ...})`` against the run's bound vault
(``GITHUB_TOKEN``). No new credential path is invented. The deploy surface
(``REQUIRED_TOOLS`` / ``REQUIRED_HTTP_SERVERS`` / ``build_triage_pipeline_workflow_create``)
is exported here exactly like dev_pipeline's so the declared tool/http_server surface can
never drift from the script that needs it (#1135). Triage needs only the GitHub REST surface
(``http_request``); it has no clone/edit step, so it does NOT declare bash or the editing
tools — but the classifier child still needs read tools to inspect linked context, so the
union includes the read-family tools (``read``/``glob``/``grep``), mirroring the
attenuation reasoning in dev_pipeline (the child surface is ``agent ∩ run``).

DETERMINISM: the script imports only ``re``/``json`` (the curated allowlist — **no
``datetime``/``time``**; any timestamp is passed via input), keeps all capability I/O in the
value domain, scans issues in a fixed, bounded, sorted order, and emits capabilities in a
deterministic order — so replay-with-memo is stable. ``http_request`` rejects a query string
in ``path`` UNLESS the route opts into ``allow_query``; the issue-list endpoint paginates via
``?per_page/?page`` and follows ``Link: rel="next"`` (the ``/repos/**`` route opts in), so a
repo with >100 open issues is fully scanned.
"""

from __future__ import annotations

from typing import Any

from aios.models.agents import HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.comment_idempotency import COMMENT_IDEMPOTENCY_HELPERS

# The stable heading on the single needs-decision comment. It is BOTH the comment's first
# line AND the maker-marker the replay guard scans for: a posted comment is its own
# "already done" marker on the next read (aios#1292).
NEEDS_DECISION_MARKER = "## Triage: needs a decision"

# ─── default judgment-node agent id (override per deployment) ─────────────────
DEFAULT_TRIAGE_AGENT_ID = "triage-classify"

# ─── the triage state-labels (the issue-state model's spec-readiness axis) ────
# An open issue carrying ANY of these is already triaged → it is NOT in the untriaged
# working set and is NEVER re-touched (idempotency). ``approved`` and ``dispatched`` are
# downstream gate/claim labels, not set by triage, but their presence still means the issue
# has already left intake — so they belong in the "already triaged" guard set.
TRIAGE_STATE_LABELS: tuple[str, ...] = (
    "needs-design",
    "shovel-ready",
    "needs-decision",
    "approved",
    "dispatched",
)

# The three classes the agent may return. Triage only ever sets the SPEC-READINESS axis;
# it NEVER applies ``approved`` (a separate chairman/seat gate — the two-axis model).
TRIAGE_CLASSES: tuple[str, ...] = ("shovel-ready", "needs-design", "needs-decision")

# The output schema the classifier agent is forced to return: exactly-one class + a reason.
# ``reason`` is required so the needs-decision branch always has a "why" to post, and the
# structured summary can record it.
TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classification", "reason"],
    "properties": {
        "classification": {"type": "string", "enum": list(TRIAGE_CLASSES)},
        "reason": {"type": "string"},
    },
}

# The instruction threaded into every classifier ``agent()`` call (alongside the issue
# body/comments). Kept in the builder so it can be tuned without editing the script body, and
# so the spec-readiness ⊥ approval invariant is stated to the agent, not just enforced in code.
CLASSIFY_INSTRUCTIONS = (
    "You are the intake-triage classifier for a software company's issue tracker. "
    "Classify this single GitHub issue into EXACTLY ONE of three spec-readiness classes:\n"
    "- `shovel-ready`: scope is clear, the design is settled, and it is buildable as-is.\n"
    "- `needs-design`: the architecture/approach is unresolved and must be designed before "
    "it can be built.\n"
    "- `needs-decision`: it requires chairman judgment — strategy, capital, an external "
    "commitment, or it is genuinely ambiguous. When you pick this, your `reason` MUST "
    "concisely state WHY a decision is needed (and the settled-vs-forks split where "
    "relevant).\n"
    "Read the issue body AND its full comment thread; a later comment can supersede the "
    "body. You are setting ONLY the spec-readiness axis — do NOT consider approval; "
    "approval is a separate chairman gate. Never invent a fourth class."
)


def _py(name: str, value: Any) -> str:
    """One ``NAME = <repr>`` constant line for the prepended header (mirrors dev_pipeline)."""
    return f"{name} = {value!r}"


def _render_constants(
    *,
    triage_agent_id: str,
    github_server: str,
    repo: str | None,
    max_issues_per_run: int,
    default_model: str | None,
) -> str:
    lines = [
        _py("TRIAGE_AGENT_ID", triage_agent_id),
        _py("GITHUB_SERVER", github_server),
        _py("DEFAULT_REPO", repo),
        _py("MAX_ISSUES_PER_RUN", max_issues_per_run),
        _py("DEFAULT_MODEL", default_model),
        _py("TRIAGE_STATE_LABELS", list(TRIAGE_STATE_LABELS)),
        _py("TRIAGE_CLASSES", list(TRIAGE_CLASSES)),
        _py("NEEDS_DECISION_MARKER", NEEDS_DECISION_MARKER),
        _py("CLASSIFY_INSTRUCTIONS", CLASSIFY_INSTRUCTIONS),
        _py("TRIAGE_SCHEMA", TRIAGE_SCHEMA),
    ]
    return "\n".join(lines)


# The static script body — references the prepended constants. Pure stdlib (re/json),
# value-domain I/O, bounded loops, no datetime/time: replay-stable by construction.
_BODY = r'''
import json
import re


# ─── input ───────────────────────────────────────────────────────────────────

def _unwrap(input):
    """Accept both a bare ``{repo}`` and the trigger envelope ``{"trigger": ..., "input": ...}``
    a cron WorkflowAction fire delivers. The triage workflow takes no per-issue input — it
    rescans the whole repo — so the only field it reads is ``repo`` (falling back to the
    deployed DEFAULT_REPO when a bare fire omits it)."""
    if isinstance(input, dict) and "trigger" in input and "input" in input:
        return input["input"] or {}
    return input or {}


# ─── scripted GitHub access (reuses dev_pipeline's bound-vault http_request path) ──

async def _gh_once(method, path, body=None):
    """One GitHub REST call through the run's bound-vault-authed http_request. A non-2xx or
    transport error is a VALUE the caller branches on, never a raise. ``path`` must NOT carry
    a query string UNLESS the route opts into allow_query (the /repos/** route does)."""
    args = {"server_ref": GITHUB_SERVER, "path": path, "method": method}
    if body is not None:
        args["body"] = json.dumps(body)
    return await tool("http_request", args)


_TRANSIENT_ERROR_PREFIXES = ("Request timed out", "HTTP transport error")


def _is_transient(resp):
    """A retryable GitHub response: a 5xx status, OR a genuine transport transient. A 4xx is
    the caller's problem (auth/404/422) and a broker gate rejection ({"error": ...} that is
    NOT a transport timeout) is deterministic — neither is retried (mirrors dev_pipeline)."""
    if not isinstance(resp, dict):
        return True
    err = resp.get("error")
    if err is not None:
        return isinstance(err, str) and err.startswith(_TRANSIENT_ERROR_PREFIXES)
    st = resp.get("status")
    return isinstance(st, int) and 500 <= st <= 599


async def gh(method, path, body=None):
    """A GitHub call with bounded transient-5xx retry (≤3). Each attempt is a fresh tool()
    await (distinct call_key) so replay stays stable; returns the LAST result as a value."""
    resp = None
    for _ in range(3):
        resp = await _gh_once(method, path, body)
        if not _is_transient(resp):
            return resp
        log("gh transient failure, retrying:", method, path, _status(resp))
    return resp


def _status(resp):
    return resp.get("status") if isinstance(resp, dict) else None


def _json_body(resp):
    if not isinstance(resp, dict):
        return None
    raw = resp.get("body")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _headers(resp):
    """Lower-cased response header map (GitHub sends ``Link`` capitalised)."""
    if not isinstance(resp, dict):
        return {}
    h = resp.get("headers")
    if not isinstance(h, dict):
        return {}
    return {str(k).lower(): v for k, v in h.items()}


def _link_next_page(link_header):
    """The ``page`` number of the rel="next" URL in a GitHub Link header, or None. We extract
    only the integer page and rebuild the path ourselves, so no host-bearing URL crosses the
    route gate (mirrors dev_pipeline's gh_paginated)."""
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
    """Append ``params`` as a deterministic-order query string to a clean ``path``."""
    qs = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    return path + ("?" if qs else "") + qs


def _ipath(repo, suffix):
    return "/repos/%s%s" % (repo, suffix)


async def gh_paginated(path, per_page=100, max_pages=50):
    """GET every page of a GitHub list endpoint, following Link: rel="next". Concatenates the
    per-page JSON arrays. A non-2xx / non-list page degrades to whatever was gathered so far.
    Requires the route to allow a query string (allow_query). Replay-stable (each page is a
    distinct tool() await)."""
    items = []
    page = 1
    for _ in range(max_pages):
        resp = await gh("GET", _with_query(path, per_page=per_page, page=page))
        if _status(resp) != 200:
            break
        chunk = _json_body(resp)
        if not isinstance(chunk, list):
            break
        items.extend(chunk)
        nxt = _link_next_page(_headers(resp).get("link"))
        if nxt is None:
            break
        page = nxt
    return items


# ─── the untriaged filter (the idempotency boundary) ──────────────────────────

def _issue_label_names(issue):
    """The set of label names on an issue. GitHub returns labels as objects with a ``name``
    field (or, defensively, bare strings). A pull request shows up in the issues endpoint with
    a ``pull_request`` key — callers exclude those separately."""
    names = set()
    if not isinstance(issue, dict):
        return names
    for lab in issue.get("labels") or []:
        if isinstance(lab, dict):
            n = lab.get("name")
            if isinstance(n, str):
                names.add(n)
        elif isinstance(lab, str):
            names.add(lab)
    return names


def _is_pull_request(issue):
    """True if this "issue" is actually a pull request (the issues endpoint returns both)."""
    return isinstance(issue, dict) and "pull_request" in issue


def is_untriaged(issue):
    """An issue is UNTRIAGED iff it is a real (non-PR) open issue carrying NONE of the triage
    state-labels. This is the idempotency boundary: an issue a prior run labeled carries one of
    these and drops out of the working set, so a re-run NEVER re-touches it."""
    if not isinstance(issue, dict) or _is_pull_request(issue):
        return False
    if issue.get("state") not in (None, "open"):
        return False
    return _issue_label_names(issue).isdisjoint(TRIAGE_STATE_LABELS)


def select_untriaged(issues):
    """The untriaged subset, sorted by issue number (a deterministic, bounded scan order so
    capabilities are emitted in a stable order → replay-stable)."""
    out = [i for i in issues if is_untriaged(i)]
    out.sort(key=lambda i: int(i.get("number", 0)))
    return out


# ─── classification → label/comment (the per-class seam; #1218 plugs in here) ──

def _comment_texts(comments):
    out = []
    if isinstance(comments, list):
        for c in comments:
            if isinstance(c, dict):
                text = c.get("body")
                if isinstance(text, str) and text.strip():
                    out.append(text)
    return out


def _normalize_classification(value):
    """Map the agent's returned classification to a known class, or None if unrecognised.
    The output_schema already constrains it to the enum, but we defend against a malformed
    return rather than mislabel an issue."""
    if isinstance(value, str) and value in TRIAGE_CLASSES:
        return value
    return None


def _decision_comment(reason):
    """The single concise comment posted on a needs-decision issue, stating WHY a decision is
    needed (the settled-vs-forks split where relevant)."""
    why = (reason or "").strip() or "This issue requires chairman judgment before it can proceed."
    return (
        NEEDS_DECISION_MARKER + "\n\n"
        "Automated intake-triage routed this issue to **needs-decision** — it requires "
        "chairman judgment (strategy / capital / external commitment / genuinely ambiguous) "
        "rather than being shovel-ready or merely needing design.\n\n"
        "**Why:** %s" % why
    )


async def _apply_classification(repo, issue_number, classification, reason, existing_comments):
    """Apply the label (and, for needs-decision, the one comment) for a classification. This is
    the clean SEAM the #1218 design-vet automation plugs into: the needs-design branch is just a
    label today; later it dispatches a design-vet sub-workflow here.

    LABEL-BEFORE-COMMENT (aios#1292, hazard 1): apply the idempotent additive label FIRST and
    post the needs-decision comment ONLY after the label returns 2xx. The label then guards the
    comment — a labeled issue drops out of ``is_untriaged`` so a re-run never re-classifies and
    re-comments. If the label fails, we return immediately and post NO comment (a comment without
    a label is exactly the duplicate-spam loop this fixes).

    REPLAY DEDUP (aios#1292, hazard 2): the comment goes through ``post_comment_once``, which
    scans the already-fetched thread for ``NEEDS_DECISION_MARKER`` and skips the POST if present
    — so an at-least-once replay (POST ran, crash before journaling) never duplicates it.

    Returns ``(label_resp, comment_resp)``: the label gh() response and the comment response
    (a 2xx, a skip sentinel, or None when not needs-decision / label failed) so the caller can
    branch and count side effects correctly. Triage NEVER applies ``approved`` — only the
    spec-readiness axis. ``classification`` is one of TRIAGE_CLASSES (already normalised)."""
    label_resp = await gh("POST", _ipath(repo, "/issues/%d/labels" % issue_number),
                          {"labels": [classification]})
    if _status(label_resp) not in (200, 201):
        # Label did not stick → do NOT post the comment; the label must guard it.
        return (label_resp, None)
    comment_resp = None
    if classification == "needs-decision":
        comment_resp = await post_comment_once(
            repo, issue_number, NEEDS_DECISION_MARKER,
            _decision_comment(reason), existing_comments)
    return (label_resp, comment_resp)


# ─── the state machine ─────────────────────────────────────────────────────────

async def main(input):
    payload = _unwrap(input)
    repo = payload.get("repo") or DEFAULT_REPO
    model = payload.get("model") or DEFAULT_MODEL
    if not repo:
        return {"state": "error",
                "reason": "no repo provided (input.repo) and no DEFAULT_REPO configured"}

    # S1 — scan: list every OPEN issue in the repo (paginated; PRs filtered out below).
    # Stateless: the working set is recomputed from scratch every run.
    phase("scan")
    issues = await gh_paginated(_ipath(repo, "/issues"))
    if not isinstance(issues, list):
        issues = []
    untriaged = select_untriaged(issues)

    # A no-op when nothing is untriaged — the scheduled-fire-into-quiet-repo case.
    counts = {c: 0 for c in TRIAGE_CLASSES}
    escalated = []   # the explicit needs-decision list (issue + reason) for the summary
    errors = []      # per-issue classify/label failures (non-fatal; the run continues)

    # S2 — classify + route each untriaged issue. One ephemeral agent() per issue, in a fixed
    # sorted order, bounded by MAX_ISSUES_PER_RUN (a stateless re-run picks up any remainder).
    phase("triage")
    for issue in untriaged[:MAX_ISSUES_PER_RUN]:
        number = int(issue.get("number", 0))

        # Read the FULL comment thread so a design pass / decision that landed in comments
        # reaches the classifier (a later comment can supersede the body). A read failure
        # degrades to body-only.
        comments = await gh_paginated(_ipath(repo, "/issues/%d/comments" % number))
        comment_bodies = _comment_texts(comments if isinstance(comments, list) else [])

        # The classifier is an EPHEMERAL agent (flavor-1 worker, no standing session). An
        # agent error is recorded per-issue and the run continues — one flaky classification
        # must not abort the whole scheduled sweep.
        try:
            result = await agent(
                {"task": "triage", "repo": repo, "issue_number": number,
                 "title": issue.get("title", ""), "body": issue.get("body", ""),
                 "comments": comment_bodies, "instructions": CLASSIFY_INSTRUCTIONS},
                agent_id=TRIAGE_AGENT_ID, output_schema=TRIAGE_SCHEMA, model=model,
                label="triage-%d" % number)
        except AgentError as exc:
            log("triage agent error on #%d:" % number, exc)
            errors.append({"issue": number, "error": "agent error: %s" % exc})
            continue

        classification = _normalize_classification(
            result.get("classification") if isinstance(result, dict) else None)
        reason = (result.get("reason") if isinstance(result, dict) else "") or ""
        if classification is None:
            log("triage returned an unrecognised classification on #%d:" % number, result)
            errors.append({"issue": number, "error": "unrecognised classification: %r"
                           % (result.get("classification") if isinstance(result, dict) else result)})
            continue

        # Label-before-comment + maker-marker dedup (aios#1292). The already-fetched
        # ``comments`` thread is handed in so the replay guard can skip a duplicate POST.
        label_resp, comment_resp = await _apply_classification(
            repo, number, classification, reason,
            comments if isinstance(comments, list) else [])
        if _status(label_resp) not in (200, 201):
            log("label %r on #%d FAILED -> %r" % (classification, number, _status(label_resp)))
            errors.append({"issue": number, "error": "label apply failed (status %r)"
                           % _status(label_resp)})
            continue

        # The label stuck (the issue is now triaged and drops out of the next run's working
        # set) → count it classified regardless of the comment outcome. A comment failure on
        # an otherwise-labeled needs-decision issue is recorded as a NON-FATAL side-effect
        # note, not a classify failure — the label already guards against re-classification,
        # so this never undercounts the mutation (the MINOR fix in #1292).
        counts[classification] += 1
        if classification == "needs-decision":
            escalated.append({"issue": number, "reason": reason})
            if not _comment_posted_ok(comment_resp):
                log("needs-decision comment on #%d not posted -> %r" % (number, comment_resp))
                errors.append({"issue": number,
                               "error": "needs-decision comment not posted (status %r); "
                               "label applied, will not re-classify"
                               % _status(comment_resp)})

    # S3 — structured run summary: per-class counts + the explicit needs-decision list.
    phase("summary")
    classified_total = sum(counts.values())
    summary = {
        "state": "done",
        "repo": repo,
        "scanned": len([i for i in issues if not _is_pull_request(i)]),
        "untriaged": len(untriaged),
        "classified": classified_total,
        "counts": counts,
        "needs_decision": escalated,
        "errors": errors,
    }
    log("triage run complete:", json.dumps(summary))
    return summary
'''


def build_triage_pipeline_script(
    *,
    triage_agent_id: str = DEFAULT_TRIAGE_AGENT_ID,
    github_server: str = "github",
    repo: str | None = None,
    max_issues_per_run: int = 50,
    default_model: str | None = None,
) -> str:
    """Return the production triage-pipeline workflow source.

    Defaults match a standard deployment (the ``triage-classify`` judgment agent, a ``github``
    http server bound to ``https://api.github.com``). ``repo`` is the default repo the
    scheduled fire scans when its input omits one (the cron trigger can also pass ``repo``);
    ``max_issues_per_run`` bounds one run's fan-out (a stateless re-run picks up any remainder).
    """
    header = _render_constants(
        triage_agent_id=triage_agent_id,
        github_server=github_server,
        repo=repo,
        max_issues_per_run=max_issues_per_run,
        default_model=default_model,
    )
    # Splice the shared comment-idempotency helper (aios#1292) into the body — authored once
    # in comment_idempotency.py and injected into BOTH pipelines so the class fix can't drift.
    # It references gh/_ipath/log from the body (all defined above); module-level defs resolve
    # names at call time, so appending after the body is fine.
    return header + "\n" + _BODY + COMMENT_IDEMPOTENCY_HELPERS


def build_triage_pipeline_fixture_script(
    *,
    triage_agent_id: str,
    repo: str | None = None,
    max_issues_per_run: int = 50,
) -> str:
    """The CI fixture variant: a real generated agent id, otherwise the identical script shape.
    Driven by ``tests/integration/test_wf_triage_pipeline_fixture.py``."""
    return build_triage_pipeline_script(
        triage_agent_id=triage_agent_id,
        repo=repo,
        max_issues_per_run=max_issues_per_run,
    )


# ─── deploy surface (the tool + http_server envelope a WorkflowCreate needs) ──
#
# ``build_triage_pipeline_script`` returns only the workflow *script string*. A deployed
# workflow ALSO needs its tool + http_server surface declared on the ``WorkflowCreate``, or
# the very first ``tool('http_request')`` call errors at runtime. Exporting the surface here
# alongside the script keeps the two from drifting (#1135).
#
# Triage's own script uses ONLY ``http_request`` (no clone/edit step → no bash). The
# classifier child needs the read-family tools to inspect linked context, and the child
# surface is ``agent ∩ run`` (attenuation), so the workflow declares the UNION:
# ``http_request`` + the read tools (``read``/``glob``/``grep``). It does NOT declare
# write/edit/bash — triage never mutates a working tree.
REQUIRED_TOOLS: list[ToolSpec] = [
    ToolSpec(type="http_request"),
    ToolSpec(type="read"),
    ToolSpec(type="glob"),
    ToolSpec(type="grep"),
]


def _github_http_server(*, name: str, base_url: str) -> HttpServerSpec:
    return HttpServerSpec(
        name=name,
        base_url=base_url,
        description="GitHub REST API (auth resolved from the bound vault's GITHUB_TOKEN).",
        routes=[
            HttpRouteSpec(
                path_pattern="/repos/**",
                # Triage reads issues (GET) and applies labels + posts the needs-decision
                # comment (POST). It never deletes a label or merges, so the route is scoped
                # to GET+POST — strictly narrower than dev_pipeline's surface.
                methods=["GET", "POST"],
                # allow_query: the issue-list + comment-thread reads paginate via
                # ?per_page/?page and follow Link: rel="next", so a repo with >100 open
                # issues is fully scanned. Safe — the route only grants GET+POST.
                allow_query=True,
                description="Issues, labels, comments; list endpoints paginate via "
                "?per_page/?page.",
            ),
        ],
    )


REQUIRED_HTTP_SERVERS: list[HttpServerSpec] = [
    _github_http_server(name="github", base_url="https://api.github.com")
]


def build_triage_pipeline_workflow_create(
    *,
    name: str,
    description: str | None = None,
    github_server: str = "github",
    github_base_url: str = "https://api.github.com",
    **script_kwargs: Any,
) -> WorkflowCreate:
    """Return the complete ``WorkflowCreate`` payload for the production triage-pipeline.

    Bundles the script (``build_triage_pipeline_script``) with the tool + http_server surface
    it requires, so a deployer can POST one object instead of hand-assembling the surface from
    source — and so the declared surface can never drift from the script that needs it (#1135).

    ``github_server`` names the http_server and is threaded into the script's ``GITHUB_SERVER``
    constant so ``tool('http_request', {server_ref: ...})`` resolves; ``github_base_url`` is
    the http_server's ``base_url`` (the credential-resolution key). Remaining keyword args are
    forwarded verbatim to ``build_triage_pipeline_script``.
    """
    script = build_triage_pipeline_script(github_server=github_server, **script_kwargs)
    return WorkflowCreate(
        name=name,
        description=description,
        script=script,
        tools=list(REQUIRED_TOOLS),
        http_servers=[_github_http_server(name=github_server, base_url=github_base_url)],
    )
