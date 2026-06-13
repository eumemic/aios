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
  no-commit fix, a fail-closed merge-guard refusal, high-risk merge approval, red master.
  Each parks the run durably instead of dead-ending in autodev's ``awaiting_triage``.

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
"""

from __future__ import annotations

from typing import Any

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
        _py("IMPLEMENT_SCHEMA", IMPLEMENT_SCHEMA),
        _py("REVIEW_SCHEMA", REVIEW_SCHEMA),
        _py("FIX_SCHEMA", FIX_SCHEMA),
        _py("CI_SCHEMA", CI_SCHEMA),
        _py("RISK_SCHEMA", RISK_SCHEMA),
    ]
    return "\n".join(lines)


# The static script body — references the prepended constants. Pure stdlib (re/json),
# value-domain I/O, bounded loops: replay-stable by construction.
_BODY = r'''
import json
import re


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


def _find_blocker(body):
    scan = _strip_meta(body)
    for i, line in enumerate(scan.split("\n"), start=1):
        for pat in SPEC_BLOCKERS:
            if re.search(pat, line):
                return (pat, i, line.strip())
    return None


def spec_ok(issue, kind):
    """The scripted pre-flight gate (V1 subset of autodev validation): empty-body,
    word-count, unresolved-marker. Returns (ok, reason)."""
    body = (issue.get("body") or "").strip()
    if not body:
        return (False, "Issue body is empty -- no spec to implement.")
    words = len(body.split())
    minimum = MIN_BUG_BODY_WORDS if kind == "bug" else MIN_SPEC_BODY_WORDS
    if words < minimum:
        return (False, "Issue body is too short (%d words, minimum %d)." % (words, minimum))
    hit = _find_blocker(body)
    if hit is not None:
        return (False, "Spec contains an unresolved marker on line %d: %r. "
                       "Resolve all open questions before implementation." % (hit[1], hit[2]))
    return (True, "")


async def gh(method, path, body=None):
    """One GitHub REST/GraphQL call through the run's bound-vault-authed http_request. A
    non-2xx or transport error is a VALUE the caller branches on, never a raise. ``path``
    must NOT carry a query string (the route allowlist is path-only) — filter in-script."""
    args = {"server_ref": GITHUB_SERVER, "path": path, "method": method}
    if body is not None:
        args["body"] = json.dumps(body)
    return await tool("http_request", args)


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


def _ipath(repo, suffix):
    return "/repos/%s%s" % (repo, suffix)


def _pr_head_sha(pr):
    return (pr.get("head") or {}).get("sha", "") if isinstance(pr, dict) else ""


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


def _merge_guard_command(repo, pr_number):
    """The merge-ref guard (autodev merge_guard.py, issue #177), preserving the load-bearing
    properties: validate the commit GitHub would ACTUALLY produce on merge — the live
    refs/pull/N/merge ref — not the PR head; FAIL CLOSED on any inability to complete
    validation; a missing merge ref means a conflicted PR (GitHub omits it) => refuse. Runs
    every configured MERGE_SENTINELS command against a detached checkout of the merge ref,
    refusing on the first nonzero (path-prefix sentinel *selection* is a deferred v1
    optimisation — running all is strictly safer). `set -eu -o pipefail` makes fail-closed
    structural, not dependent on guarding each line. Re-run-tolerant (rm -rf then clone) per
    the at-least-once bash contract."""
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
    for cmd in MERGE_SENTINELS:
        lines.append("( %s ) || exit 73" % cmd)
    lines.append("echo MERGE_GUARD_OK")
    return "\n".join(lines)


def _mark_ready_query():
    return ("mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id})"
            "{pullRequest{isDraft}}}")


# ─── the state machine ───────────────────────────────────────────────────────

async def main(input):
    payload = _unwrap(input)
    repo = payload["repo"]
    issue_number = int(payload["issue_number"])
    kind = payload.get("kind", "issue")
    model = payload.get("model") or DEFAULT_MODEL
    escalations = []

    # S1 — ingest the issue
    phase("ingest")
    resp = await gh("GET", _ipath(repo, "/issues/%d" % issue_number))
    issue = _json_body(resp)
    if _status(resp) != 200 or issue is None:
        return {"state": "error", "reason": "could not read issue %d (status %r)"
                % (issue_number, _status(resp))}

    # S2 — scripted pre-flight spec gate (regex/word-count; fail-fast before any spend)
    phase("spec-gate")
    ok, reason = spec_ok(issue, kind)
    if not ok:
        log("spec gate failed:", reason)
        await gh("POST", _ipath(repo, "/issues/%d/comments" % issue_number),
                 {"body": "## Spec not ready for implementation\n\n" + reason})
        await gh("POST", _ipath(repo, "/issues/%d/labels" % issue_number),
                 {"labels": ["underspecified"]})
        return {"state": "spec_failed", "reason": reason}

    # A3 — implement (the coding agent: clone, explore, plan, TDD, self-review, push)
    phase("implement")
    impl = await agent(
        {"task": "implement", "repo": repo, "issue_number": issue_number, "kind": kind,
         "title": issue.get("title", ""), "body": issue.get("body", "")},
        agent_id=IMPLEMENT_AGENT_ID, output_schema=IMPLEMENT_SCHEMA, model=model, label="implement")
    if impl.get("escalated"):
        escalations.append("design")
        decision = await gate({"kind": "design", "issue": issue_number,
                               "question": impl.get("escalation_reason", "")})
        if not (isinstance(decision, dict) and decision.get("resolved")):
            return {"state": "escalated", "reason": "design", "escalations": escalations}
        # resumed with a settled direction -> re-implement with the chairman's answer
        impl = await agent(
            {"task": "implement", "repo": repo, "issue_number": issue_number, "kind": kind,
             "title": issue.get("title", ""), "body": issue.get("body", ""),
             "resolution": decision.get("resolution", "")},
            agent_id=IMPLEMENT_AGENT_ID, output_schema=IMPLEMENT_SCHEMA, model=model,
            label="implement-resumed")
    branch = impl["branch"]

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
            return {"state": "failed_no_pr",
                    "reason": "could not open or adopt a PR for branch %r (status %r)"
                    % (branch, _status(created))}
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
        review = await agent(
            {"task": "review", "repo": repo, "pr_number": pr_number, "head_sha": head_sha},
            agent_id=REVIEW_AGENT_ID, output_schema=REVIEW_SCHEMA, model=model,
            label="review-%d" % i)
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
        fix = await agent(
            {"task": "fix", "repo": repo, "pr_number": pr_number, "issues": review["issues"]},
            agent_id=FIX_AGENT_ID, output_schema=FIX_SCHEMA, model=model, label="fix-%d" % i)
        head_sha = fix["head_sha"]
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
    # (N watches, N-1 fixes — matches autodev's CI loop).
    ci_ok = False
    for i in range(MAX_CI_ITERS):
        ci = await agent(
            {"task": "watch_ci", "repo": repo, "pr_number": pr_number, "head_sha": head_sha},
            agent_id=CI_AGENT_ID, output_schema=CI_SCHEMA, model=model, label="ci-%d" % i)
        if ci["status"] in ("green", "no_ci"):
            ci_ok = True
            break
        if i == MAX_CI_ITERS - 1:
            break
        before = head_sha
        fix = await agent(
            {"task": "fix_ci", "repo": repo, "pr_number": pr_number,
             "detail": ci.get("detail", "")},
            agent_id=FIX_AGENT_ID, output_schema=FIX_SCHEMA, model=model, label="ci-fix-%d" % i)
        head_sha = fix["head_sha"]
        if before and head_sha == before:
            break
    if not ci_ok:
        escalations.append("verify")
        decision = await gate({"kind": "verify", "pr": pr_number,
                               "reason": "CI failing after %d checks" % MAX_CI_ITERS})
        if not (isinstance(decision, dict) and decision.get("resolved")):
            return {"state": "escalated", "reason": "ci_exhausted", "escalations": escalations}

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
    await gh("POST", _ipath(repo, "/issues/%d/labels" % pr_number),
             {"labels": ["risk:tier-%d" % tier]})
    await gh("POST", _ipath(repo, "/issues/%d/comments" % pr_number),
             {"body": "## Risk Assessment\n\n**Tier %d**\n\n%s" % (tier, summary)})

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
        await gh("POST", _ipath(repo, "/issues/%d/comments" % pr_number),
                 {"body": "## Merge guard refused\n\n```\n%s\n```" % detail})
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
        decision = await gate({"kind": "merge_approval", "pr": pr_number, "tier": tier})
        if not (isinstance(decision, dict) and decision.get("approve")):
            return {"state": "held", "reason": "merge_approval", "pr_url": pr_url,
                    "pr_number": pr_number, "risk_tier": tier, "escalations": escalations}
    merge_resp = await gh("PUT", _ipath(repo, "/pulls/%d/merge" % pr_number),
                          {"merge_method": MERGE_METHOD})
    merged = _status(merge_resp) == 200
    if not merged:
        # At-least-once: a crash-re-driven PUT against an already-merged PR returns 405/409.
        # Confirm against GitHub before declaring failure, so a real merge is never lost.
        confirm = _json_body(await gh("GET", _ipath(repo, "/pulls/%d" % pr_number)))
        if isinstance(confirm, dict) and confirm.get("merged"):
            merged = True
        else:
            return {"state": "merge_failed", "pr_url": pr_url, "pr_number": pr_number,
                    "risk_tier": tier, "reason": "merge call returned %r" % _status(merge_resp),
                    "escalations": escalations}

    # A15 — post-merge master-CI watch (the D-2 missing state). Red OR an errored watch =>
    # HALT at a gate; the merge is a committed fact, so we never discard it by failing here.
    phase("post-merge")
    try:
        master = await agent(
            {"task": "watch_ci", "repo": repo, "ref": BASE_BRANCH},
            agent_id=CI_AGENT_ID, output_schema=CI_SCHEMA, model=model, label="master-ci")
        master_red = master["status"] == "red"
        master_detail = master.get("detail", "")
    except (AgentError, KeyError, TypeError) as exc:
        master_red = True
        master_detail = "master-CI watch failed: %s" % exc
    if master_red:
        escalations.append("master_red")
        await gate({"kind": "master_red", "repo": repo, "detail": master_detail})

    return {"state": "done", "pr_url": pr_url, "pr_number": pr_number, "merged": merged,
            "risk_tier": tier, "escalations": escalations}
'''


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
    return header + "\n" + _BODY


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
