"""Work-state reconciler, phase 1 (OBSERVE-ONLY) — the durable aios workflow form.

Registered as an aios workflow and fired on a cron. It is a DETERMINISTIC workflow:
it makes **no model calls and spawns no agents**. Every step is mechanical
computation — enumerate, join, classify, hash — per the intelligence-vs-computation
doctrine: reserve intelligence for judgment, not for a join.

Declared surface (the launcher's agent surface must SUPERSET this, or the run is
silently attenuated — the #1386/#2043 launcher-clamp trap):

    tools:        http_request, list_runs
    http_servers: github (GET /repos/**)

Input::

    {"repos": ["eumemic/aios", ...],          # optional, defaults below
     "workflow_ids": ["wf_01KV4YGV4PSGP08TJBAY32J2VK"],
     "previous_hash": "<the last disagreement_hash>"}   # for change detection

Output: the report dict from ``aios.reconcilers.work_state.ReconcileReport.to_dict``,
plus ``changed`` / ``wake_seat``.

WRITES NOTHING. Only ``GET`` is ever issued (``_gh`` hard-refuses any other verb),
and the github http_server this runs under is declared GET-only. No label edit, no
comment on a reconciled item, no re-dispatch, no close. Phase 2 (demotion) is gated
on reviewing a week of phase-1 data.

FAIL-LOUD: every read is checked. A failed read produces ``verdict: ALARM`` with
``counts: null`` — never an empty list rendered as health. That failure (an empty
result from a broken query reading as health) happened twice on 2026-07-25 and is
the bug class this whole effort exists to kill.

The classification logic here is a deliberate MIRROR of
``aios.reconcilers.work_state`` (the script host cannot import ``aios.*``);
``tests/unit/test_work_state_reconciler.py`` asserts the two agree on the same
fixtures, so they cannot drift apart silently.
"""

import hashlib
import json

DEFAULT_REPOS = [
    "eumemic/aios",
    "eumemic/eumemic-ops",
    "eumemic/eumemic-company",
    "eumemic/aios-console",
    "eumemic/autodev",
]
DEV_PIPELINE_WORKFLOW_ID = "wf_01KV4YGV4PSGP08TJBAY32J2VK"

LIVE = {"pending", "running"}
SUSPENDED = {"suspended"}
TERMINAL = {"completed", "errored", "cancelled"}
CLASSES = ["ZOMBIE", "DEAD", "MISLABELLED", "LAGGING"]

PIPELINE_LABELS = {
    "dispatched",
    "approved",
    "hold",
    "paused",
    "escalated",
    "blocked",
    "ci-loop-exhausted",
    "merge:approved",
}
PIPELINE_PREFIXES = ("needs:human/", "autodev:", "pipeline:")

MAX_ISSUE_PAGES = 50
MAX_RUN_PAGES = 200


class ReadFailure(Exception):
    """A source read failed. NEVER degrades to an empty list."""


def is_pipeline_label(label):
    if label in PIPELINE_LABELS:
        return True
    for prefix in PIPELINE_PREFIXES:
        if label.startswith(prefix):
            return True
    return False


async def _gh(path, method="GET"):
    """One GitHub GET through the declared ``github`` http_server.

    Refuses any other verb outright: phase 1 is observe-only and this is the single
    door to GitHub, so the constraint is enforced at the door rather than by
    reviewer vigilance over call sites.
    """
    if method != "GET":
        raise ReadFailure("phase-1 reconciler is OBSERVE-ONLY: refusing %s %s" % (method, path))
    result = await tool(  # noqa: F821 - injected capability
        "http_request", {"server_ref": "github", "path": path, "method": "GET"}
    )
    # tool() returns errors as VALUES. An unchecked error here would parse as an
    # empty page — the exact silent-degradation this reconciler exists to detect.
    if not isinstance(result, dict):
        raise ReadFailure("http_request returned %r for %s" % (type(result), path))
    if "error" in result:
        raise ReadFailure("GET %s failed: %s" % (path, result["error"]))
    if result.get("truncated"):
        raise ReadFailure("GET %s response was TRUNCATED — a cut body is not a page" % path)
    status = result.get("status")
    if not isinstance(status, int) or not (200 <= status < 300):
        raise ReadFailure("GET %s → HTTP %s" % (path, status))
    body = result.get("body")
    try:
        parsed = json.loads(body) if body else None
    except Exception as exc:
        raise ReadFailure("GET %s returned unparseable JSON: %s" % (path, exc))
    link = ""
    for pair in result.get("headers") or []:
        if len(pair) == 2 and pair[0].lower() == "link":
            link = pair[1]
    return parsed, link


def _next_link(link_header):
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) >= 2 and 'rel="next"' in "".join(segments[1:]):
            return segments[0].strip().strip("<>")
    return None


def _path_of(url):
    """Reduce an absolute GitHub URL to the path+query the http_server route expects."""
    marker = "api.github.com"
    if marker in url:
        return url.split(marker, 1)[1]
    return url


async def _read_items(repos):
    """Open issues/PRs carrying any pipeline state label, across ``repos``.

    Enumerates ALL open items and filters locally: the LAGGING class is defined by
    the ABSENCE of ``dispatched``, so a server-side ``?labels=dispatched`` filter
    would be structurally incapable of finding it — the same defect as the stall
    detectors that could not see the zombies because they trusted the lying label.
    """
    items = []
    exhaustive = True
    for repo in repos:
        path = "/repos/%s/issues?state=open&per_page=100&sort=created&direction=desc" % repo
        pages = 0
        while path is not None:
            if pages >= MAX_ISSUE_PAGES:
                exhaustive = False
                break
            body, link = await _gh(path)
            if not isinstance(body, list):
                raise ReadFailure("issue list for %s was not a JSON array" % repo)
            pages += 1
            for row in body:
                labels = sorted(
                    [
                        x.get("name", "") if isinstance(x, dict) else str(x)
                        for x in (row.get("labels") or [])
                    ]
                )
                if not any(is_pipeline_label(x) for x in labels):
                    continue
                items.append(
                    {
                        "repo": repo,
                        "number": row.get("number"),
                        "kind": "pull_request" if "pull_request" in row else "issue",
                        "title": row.get("title", ""),
                        "labels": labels,
                        "url": row.get("html_url", ""),
                        "updated_at": row.get("updated_at", ""),
                    }
                )
            nxt = _next_link(link)
            path = _path_of(nxt) if nxt else None
    return items, exhaustive


def _normalise_repo(raw, default_owner="eumemic"):
    """Canonicalise a repo reference from ``run.input``; ``None`` when unreadable.

    Never guesses: a wrong guess manufactures a phantom 'agree', which is exactly
    the false-health this reconciler exists to prevent.
    """
    if isinstance(raw, dict):
        owner = raw.get("owner")
        name = raw.get("name") or raw.get("repo")
        if isinstance(owner, str) and isinstance(name, str) and owner and name:
            return "%s/%s" % (owner.strip("/"), name.strip("/"))
        raw = name
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if "github.com" in text:
        tail = text.split("github.com", 1)[1].lstrip(":/")
        parts = [p for p in tail.split("/") if p]
        if len(parts) >= 2:
            name = parts[1]
            if name.endswith(".git"):
                name = name[:-4]
            return "%s/%s" % (parts[0], name)
        return None
    if text.endswith(".git"):
        text = text[:-4]
    parts = [p for p in text.strip("/").split("/") if p]
    if len(parts) == 1:
        return "%s/%s" % (default_owner, parts[0])
    if len(parts) == 2:
        return "%s/%s" % (parts[0], parts[1])
    return None


def _issue_number(raw):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip().lstrip("#")
        if text.isdigit() and int(text) > 0:
            return int(text)
    return None


def _run_key(run):
    inp = run.get("input")
    if not isinstance(inp, dict):
        return (None, None)
    repo = None
    for k in ("repo", "repository", "repo_full_name", "full_name"):
        if k in inp:
            repo = _normalise_repo(inp[k])
            break
    number = None
    for k in ("issue_number", "issue", "number", "issueNumber"):
        if k in inp:
            number = _issue_number(inp[k])
            break
    return (repo, number)


async def _read_runs(workflow_ids):
    """Every run of the pipeline workflow(s) — including terminal ones.

    'No run, ever' is only a sound ZOMBIE verdict if this read reaches the whole
    history, so a truncated read marks the source non-exhaustive and every derived
    count renders as a floor ('at least N') rather than a total.
    """
    runs = []
    exhaustive = True
    for workflow_id in workflow_ids:
        after = None
        pages = 0
        while True:
            if pages >= MAX_RUN_PAGES:
                exhaustive = False
                break
            args = {"workflow_id": workflow_id, "limit": 200, "account_wide": True}
            if after:
                args["after"] = after
            result = await tool("list_runs", args)  # noqa: F821 - injected capability
            if not isinstance(result, dict):
                raise ReadFailure("list_runs returned %r" % type(result))
            if "error" in result:
                raise ReadFailure("list_runs failed: %s" % result["error"])
            rows = result.get("runs")
            if not isinstance(rows, list):
                # A missing 'runs' key is a CONTRACT failure, not an empty page.
                raise ReadFailure("list_runs response has no 'runs' list")
            pages += 1
            runs.extend(rows)
            if len(rows) < 200:
                break
            after = rows[-1].get("id")
            if not after:
                break
    return runs, exhaustive


def classify(item, runs):
    """Classify one item against its runs. ``None`` == the label agrees.

    Precedence: live > suspended > terminal. One live run means work really is
    happening. With none live, a SUSPENDED run is parked at a gate — a different
    condition from running, so MISLABELLED, not 'agree' (folding parked into agree
    is how a gate-parked item hides behind `dispatched` forever).
    """
    dispatched = "dispatched" in item["labels"]
    live = [r for r in runs if r.get("status") in LIVE]
    suspended = [r for r in runs if r.get("status") in SUSPENDED]
    terminal = [r for r in runs if r.get("status") in TERMINAL]
    unknown = [
        r
        for r in runs
        if r.get("status") not in LIVE
        and r.get("status") not in SUSPENDED
        and r.get("status") not in TERMINAL
    ]

    def mk(classification, detail, subject, caveats=()):
        return {
            "classification": classification,
            "repo": item["repo"],
            "number": item["number"],
            "kind": item["kind"],
            "title": item["title"],
            "url": item["url"],
            "labels": [x for x in item["labels"] if is_pipeline_label(x)],
            "detail": detail,
            "run_ids": [r.get("id", "") for r in subject],
            "run_statuses": sorted({r.get("status", "") for r in subject}),
            "updated_at": item["updated_at"],
            "caveats": list(caveats),
        }

    if not dispatched:
        if live:
            return mk(
                "LAGGING",
                "%d live run(s) but no `dispatched` label" % len(live),
                live,
            )
        return None
    if live:
        return None
    if unknown and not suspended and not terminal:
        return mk(
            "MISLABELLED",
            "run(s) in unrecognised status — cannot prove live; treat as parked pending triage",
            unknown,
            ("unrecognised-run-status",),
        )
    if suspended:
        return mk(
            "MISLABELLED",
            "%d run(s) suspended at a gate — parked is NOT running" % len(suspended),
            suspended,
        )
    if terminal:
        return mk("DEAD", "`dispatched` but every run is terminal", terminal)
    return mk(
        "ZOMBIE", "`dispatched` but NO run exists for this issue — nothing ever picked it up", []
    )


def build(items, items_exhaustive, runs, runs_exhaustive, failures):
    """Join + classify. **Any failure ⇒ ALARM with counts:null.**

    The counts key is ``None`` — not ``{}``, not zeros — on ALARM, so a consumer
    cannot render 'no disagreements' from a broken read. There is no count to render.
    """
    if failures:
        return {
            "verdict": "ALARM",
            "counts": None,
            "total_disagreements": None,
            "failures": failures,
            "disagreements": [],
            "unmatched_runs": [],
            "exhaustive": False,
            "items_scanned": 0,
            "runs_scanned": 0,
        }

    index = {}
    for run in runs:
        repo, number = _run_key(run)
        if repo is None or number is None:
            continue
        index.setdefault((repo, number), []).append(run)

    open_keys = {(i["repo"], i["number"]) for i in items}
    disagreements = []
    for item in sorted(items, key=lambda i: (i["repo"], i["number"])):
        verdict = classify(item, index.get((item["repo"], item["number"]), []))
        if verdict is not None:
            disagreements.append(verdict)

    unmatched = []
    for run in runs:
        repo, number = _run_key(run)
        if run.get("status") not in LIVE and run.get("status") not in SUSPENDED:
            continue
        if repo is None or number is None:
            unmatched.append(
                {
                    "run_id": run.get("id", ""),
                    "status": run.get("status", ""),
                    "repo": repo,
                    "issue_number": number,
                    "reason": "run.input carries no usable (repo, issue_number)",
                }
            )
        elif run.get("status") in LIVE and (repo, number) not in open_keys:
            unmatched.append(
                {
                    "run_id": run.get("id", ""),
                    "status": run.get("status", ""),
                    "repo": repo,
                    "issue_number": number,
                    "reason": "live run against an item that is not open in the scanned repos",
                }
            )

    order = {c: n for n, c in enumerate(CLASSES)}
    disagreements.sort(key=lambda d: (order[d["classification"]], d["repo"], d["number"]))
    counts = {c: len([d for d in disagreements if d["classification"] == c]) for c in CLASSES}
    return {
        "verdict": "OK",
        "counts": counts,
        "total_disagreements": len(disagreements),
        "failures": [],
        "disagreements": disagreements,
        "unmatched_runs": unmatched,
        "exhaustive": bool(items_exhaustive and runs_exhaustive),
        "items_scanned": len(items),
        "runs_scanned": len(runs),
    }


def disagreement_hash(report):
    """Hash of the disagreement SET — the seat-wake change detector.

    Identity = class + item + run statuses (NOT run ids), so a re-dispatch that
    yields the same verdict for the same reason does not spam the seat, while a
    real transition (suspended → errored) does. On ALARM the failure reasons are
    hashed instead, so a NEW outage still wakes the seat.
    """
    if report["verdict"] == "ALARM":
        parts = ["ALARM"] + sorted(
            ["%s:%s" % (f.get("source", ""), f.get("reason", "")) for f in report["failures"]]
        )
    else:
        parts = sorted(
            [
                "%s:%s#%s:%s"
                % (d["classification"], d["repo"], d["number"], ",".join(sorted(d["run_statuses"])))
                for d in report["disagreements"]
            ]
        )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


async def main(input):
    input = input if isinstance(input, dict) else {}
    repos = input.get("repos") or DEFAULT_REPOS
    workflow_ids = input.get("workflow_ids") or [DEV_PIPELINE_WORKFLOW_ID]
    previous_hash = input.get("previous_hash")

    phase("read-github")  # noqa: F821 - injected capability
    failures = []
    items, items_exhaustive = [], True
    try:
        items, items_exhaustive = await _read_items(repos)
    except ReadFailure as exc:
        # A failed read is recorded as a FAILURE, never swallowed into an empty
        # list. The report below will ALARM and carry counts:null.
        failures.append({"source": "github", "reason": str(exc)})

    phase("read-runs")  # noqa: F821
    runs, runs_exhaustive = [], True
    try:
        runs, runs_exhaustive = await _read_runs(workflow_ids)
    except ReadFailure as exc:
        failures.append({"source": "aios-runs", "reason": str(exc)})

    phase("join")  # noqa: F821
    report = build(items, items_exhaustive, runs, runs_exhaustive, failures)
    current_hash = disagreement_hash(report)
    report["disagreement_hash"] = current_hash
    report["changed"] = previous_hash != current_hash
    # Wake the seat when the disagreement SET changes, and ALWAYS on an ALARM —
    # silence is not health, so a broken read is never allowed to be quiet.
    report["wake_seat"] = bool(report["changed"] or report["verdict"] == "ALARM")
    report["phase"] = "1-observe-only"
    report["writes_performed"] = 0
    report["repos_scanned"] = list(repos)

    if report["verdict"] == "ALARM":
        log("ALARM: work-state reconciler could not read its sources: %s" % json.dumps(failures))  # noqa: F821
    else:
        log(  # noqa: F821
            "work-state: %s (exhaustive=%s, changed=%s)"
            % (json.dumps(report["counts"]), report["exhaustive"], report["changed"])
        )
    return report
