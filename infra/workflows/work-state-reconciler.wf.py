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
CLASSES = ["ZOMBIE", "AMBIGUOUS", "DEAD", "MISLABELLED", "LAGGING"]

# C1 — the UNION of "work is in flight" assertions, NOT the single literal
# `dispatched`. The seat stripped `dispatched` from all nine confirmed zombies on
# 2026-07-26; they still carry `autodev:in-progress` + `needs:human/build`, which
# claim exactly the same thing. Keying on one label would have reported
# "0 ZOMBIE" over nine dead issues. Mirrors IN_FLIGHT_LABELS in the library.
IN_FLIGHT_LABELS = {"dispatched", "autodev:in-progress", "design:in-progress"}
IN_FLIGHT_PREFIXES = ("needs:human/",)
NOT_IN_FLIGHT_LABELS = {"autodev:built", "autodev:failed", "hold", "paused"}

# B3 — ZOMBIE cannot prove "never picked up": list_runs filters archived_at IS NULL
# and archiving requires a TERMINAL run, so a completed-then-archived run reads as
# "no run, ever". Stated in every report rather than left to inference.
ZOMBIE_MEANS = (
    "no UNARCHIVED run exists for this item. `list_runs` cannot see archived runs "
    "(archived_at IS NULL) and archiving requires a TERMINAL run, so ZOMBIE cannot "
    "distinguish 'never picked up' from 'ran and was tidied away'."
)

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
#: Timeline pages read per issue when acquiring the linked-PR evidence (C2). A
#: TRUNCATED timeline is a FAILED read, not a short one: the page we did not read
#: is exactly where the PR that disproves a ZOMBIE would be, so stopping quietly
#: manufactures the false ZOMBIE this reconciler exists to catch.
MAX_TIMELINE_PAGES = 20
#: Redirect hops followed on the issue-LIST path (a renamed repo). Mirrors urllib's
#: silent following in the CLI so the two forms enumerate the same items.
MAX_REDIRECT_HOPS = 5

#: A GitHub 3xx on an issue read IS the transfer signal (C4). ``http_request`` does
#: NOT follow redirects (httpx defaults ``follow_redirects=False``), so a
#: transferred issue comes back as a literal 301 carrying ``Location``; the urllib
#: CLI sees the same event as a silent hop and reports ``resp.geturl()``. Different
#: mechanism, SAME note — which is what the reader drift guard pins.
REDIRECT_STATUSES = (301, 302, 303, 307, 308)


def transfer_note(repo, number, dest):
    """The C4 note. Byte-identical to the library/CLI form — pinned by the drift guard.

    68 issues changed repos on 2026-07-25. A transfer gives the issue a NEW
    (repo, number) while every ``run.input`` still holds the OLD one, so the join
    key is stale and the item reads as a FALSE ZOMBIE. Detection only in phase 1:
    we report the stale key, we do not rewrite it.
    """
    return (
        f"{repo}#{number} was REDIRECTED to {dest} — the issue appears to have been TRANSFERRED, "
        "so any run keyed to these coordinates cannot join (C4)"
    )


class ReadFailure(Exception):
    """A source read failed. NEVER degrades to an empty list."""


def _check_issue_row(repo, row):
    """Validate ONE issue row. Raises ReadFailure — never AttributeError.

    A 2xx carrying a non-object row used to reach ``row.get()`` and raise
    AttributeError OUTSIDE the ReadFailure handlers, killing the workflow instead of
    returning verdict:ALARM. A malformed row is a FAILED READ, which must alarm.
    """
    if not isinstance(row, dict):
        raise ReadFailure(
            f"issue list for {repo} contained a non-object row ({type(row).__name__})"
        )
    raw_labels = row.get("labels")
    if raw_labels is not None and not isinstance(raw_labels, list):
        raise ReadFailure(
            "issue row {}#{} has a non-list 'labels' ({})".format(
                repo, row.get("number"), type(raw_labels).__name__
            )
        )
    return raw_labels or []


def _check_run_row(row):
    """Validate ONE run row. Same contract as :func:`_check_issue_row`."""
    if not isinstance(row, dict):
        raise ReadFailure(f"list_runs returned a non-object run row ({type(row).__name__})")
    return row


def _paginate_check(rows):
    """Return the keyset cursor for the next page, or None when the page is the last.

    B2 — the mirror of B1 in the CLI. A FULL page whose last row carries no ``id``
    leaves no cursor: the run history is TRUNCATED and we cannot continue. The old
    code broke out of the loop with ``exhaustive`` still True, so a truncated read
    rendered as exhaustive — and every ZOMBIE verdict rests on "no run exists in the
    list I read". Refusing here is what keeps a partial read from reading as health.
    """
    if len(rows) < 200:
        return None
    after = rows[-1].get("id")
    if not after:
        raise ReadFailure(
            f"list_runs returned a full page ({len(rows)} rows) whose last row has no "
            "'id' — no pagination cursor, so the run history is TRUNCATED. Refusing to "
            "report a truncated read as exhaustive."
        )
    return after


def is_in_flight_label(label):
    """True iff ``label`` asserts work is in flight (a run ought to exist)."""
    if label in NOT_IN_FLIGHT_LABELS:
        return False
    if label in IN_FLIGHT_LABELS:
        return True
    return any(label.startswith(prefix) for prefix in IN_FLIGHT_PREFIXES)


def in_flight_assertions(labels):
    """Every in-flight assertion in ``labels``, sorted — the trigger set."""
    return sorted({x for x in labels if is_in_flight_label(x)})


def is_pipeline_label(label):
    if label in PIPELINE_LABELS:
        return True
    return any(label.startswith(prefix) for prefix in PIPELINE_PREFIXES)


def _header(result, name):
    """One header value out of http_request's ``[[name, value], ...]`` pair list.

    A pair LIST, not a dict, because HTTP permits repeated header names; we take the
    last occurrence, which is what a dict would have kept anyway.
    """
    found = ""
    for pair in result.get("headers") or []:
        if len(pair) == 2 and isinstance(pair[0], str) and pair[0].lower() == name:
            found = pair[1]
    return found


async def _gh(path, method="GET", allow_redirect=False):
    """One GitHub GET through the declared ``github`` http_server.

    Returns ``(parsed_body, link_header, redirect_location_or_None)``.

    Refuses any other verb outright: phase 1 is observe-only and this is the single
    door to GitHub, so the constraint is enforced at the door rather than by
    reviewer vigilance over call sites.

    C4 — ``allow_redirect=True`` turns a 3xx from an ERROR into a SIGNAL. The
    ``http_request`` tool does not follow redirects, so a TRANSFERRED issue answers
    with a literal 301 + ``Location``; without this the workflow's only options were
    "raise" (an ALARM for a routine transfer) or, worse, nothing at all — which is
    what shipped, so 68 transferred issues would have read as false ZOMBIEs. The
    urllib CLI observes the same event as a silently-followed hop and reports
    ``resp.geturl()``; the mechanisms differ, the emitted note does not.
    """
    if method != "GET":
        raise ReadFailure(f"phase-1 reconciler is OBSERVE-ONLY: refusing {method} {path}")
    result = await tool(  # noqa: F821 - injected capability
        "http_request", {"server_ref": "github", "path": path, "method": "GET"}
    )
    # tool() returns errors as VALUES. An unchecked error here would parse as an
    # empty page — the exact silent-degradation this reconciler exists to detect.
    if not isinstance(result, dict):
        raise ReadFailure(f"http_request returned {type(result)!r} for {path}")
    if "error" in result:
        raise ReadFailure("GET {} failed: {}".format(path, result["error"]))
    if result.get("truncated"):
        raise ReadFailure(f"GET {path} response was TRUNCATED — a cut body is not a page")
    status = result.get("status")
    if isinstance(status, int) and status in REDIRECT_STATUSES:
        location = _header(result, "location")
        if not allow_redirect:
            raise ReadFailure(f"GET {path} → HTTP {status} (redirect to {location!r})")
        if not location:
            # A redirect with nowhere to go is a FAILED read, not a transfer we can
            # name. Refusing beats inventing a destination.
            raise ReadFailure(
                f"GET {path} → HTTP {status} with no Location header — cannot tell where this "
                "issue moved to, and guessing would manufacture a join key"
            )
        return None, "", location
    if not isinstance(status, int) or not (200 <= status < 300):
        raise ReadFailure(f"GET {path} → HTTP {status}")
    body = result.get("body")
    try:
        parsed = json.loads(body) if body else None
    except Exception as exc:
        raise ReadFailure(f"GET {path} returned unparseable JSON: {exc}") from exc
    return parsed, _header(result, "link"), None


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


async def _fetch_linked_prs(repo, number):
    """PRs cross-referencing an issue + any redirect seen — the ACQUIRED evidence (C2/C4).

    Returns ``(linked_pr_numbers, redirect_location_or_None)``.

    This is the half the mirror was missing. Without it ``linked_pr_numbers`` was
    never populated in the workflow, so AMBIGUOUS was UNREACHABLE with real data and
    the five known-good items (company#71/#50/#135, ops#337/#331) classified ZOMBIE
    on the form that actually runs on cron. The classifier was right; the system
    never handed it the evidence. Read-only: the timeline endpoint under GET.

    A truncated timeline is a FAILED read (:data:`MAX_TIMELINE_PAGES`), because the
    unread page is exactly where the PR that disproves the ZOMBIE would be.
    """
    path = f"/repos/{repo}/issues/{number}/timeline?per_page=100"
    found = set()
    redirected_to = None
    pages = 0
    while path is not None:
        if pages >= MAX_TIMELINE_PAGES:
            raise ReadFailure(
                f"timeline for {repo}#{number} exceeded {MAX_TIMELINE_PAGES} pages — "
                "refusing to decide ZOMBIE vs AMBIGUOUS on a TRUNCATED timeline, since "
                "the unread page is exactly where the PR that disproves a ZOMBIE would be"
            )
        body, link, location = await _gh(path, allow_redirect=True)
        if location:
            return tuple(sorted(found)), location
        if not isinstance(body, list):
            raise ReadFailure(f"timeline for {repo}#{number} was not a JSON array")
        pages += 1
        for event in body:
            if not isinstance(event, dict) or event.get("event") != "cross-referenced":
                continue
            source = event.get("source")
            issue = source.get("issue") if isinstance(source, dict) else None
            if isinstance(issue, dict) and "pull_request" in issue:
                num = issue.get("number")
                if isinstance(num, int) and not isinstance(num, bool):
                    found.add(num)
        nxt = _next_link(link)
        path = _path_of(nxt) if nxt else None
    return tuple(sorted(found)), redirected_to


async def _read_items(repos, enrich_linked_prs=True):
    """Open issues/PRs across ``repos``, ENRICHED with the linked-PR evidence.

    Returns ``(items, exhaustive, notes)``.

    Enumerates ALL open items and filters locally: the LAGGING class is defined by
    the ABSENCE of ``dispatched``, so a server-side ``?labels=dispatched`` filter
    would be structurally incapable of finding it — the same defect as the stall
    detectors that could not see the zombies because they trusted the lying label.

    Two signals are ACQUIRED here rather than assumed, and the reader drift guard
    fails the build if either goes missing from one form:

    * **C2** — every item that ASSERTS in flight gets its timeline read, so
      ``linked_pr_numbers`` is evidence the production path actually fetched. A
      classifier that can only reach AMBIGUOUS when a test hands it the answer is
      the defect family this PR exists to kill: evidence asserted, not acquired.
    * **C4** — a 3xx on either read is a TRANSFER, surfaced as an explicit note
      instead of being allowed to masquerade as a zombie.
    """
    items = []
    exhaustive = True
    notes = []
    transferred = {}
    for repo in repos:
        path = f"/repos/{repo}/issues?state=open&per_page=100&sort=created&direction=desc"
        pages = 0
        hops = 0
        while path is not None:
            if pages >= MAX_ISSUE_PAGES:
                exhaustive = False
                break
            body, link, location = await _gh(path, allow_redirect=True)
            if location:
                # C4 on the LIST path: the repo itself was renamed/moved. Follow it (the
                # CLI's urllib does so silently) but SAY SO, so the join key that no
                # longer resolves is visible as data rather than as a shrinking count.
                hops += 1
                if hops > MAX_REDIRECT_HOPS:
                    raise ReadFailure(
                        f"issue list for {repo} redirected more than "
                        f"{MAX_REDIRECT_HOPS} times — refusing to chase a redirect loop"
                    )
                notes.append(transfer_note(repo, "*", location))
                path = _path_of(location)
                continue
            if not isinstance(body, list):
                raise ReadFailure(f"issue list for {repo} was not a JSON array")
            pages += 1
            for row in body:
                raw_labels = _check_issue_row(repo, row)
                labels = sorted(
                    [x.get("name", "") if isinstance(x, dict) else str(x) for x in raw_labels]
                )
                # C5: unlabelled items are KEPT. LAGGING is defined by the ABSENCE of an
                # in-flight assertion, so filtering enumeration on presence-of-some-label
                # makes a live run against an unlabelled issue structurally undetectable.
                items.append(
                    {
                        "repo": repo,
                        "number": row.get("number"),
                        "kind": "pull_request" if "pull_request" in row else "issue",
                        "title": row.get("title", ""),
                        "labels": labels,
                        "url": row.get("html_url", ""),
                        "updated_at": row.get("updated_at", ""),
                        "linked_pr_numbers": [],
                    }
                )
            nxt = _next_link(link)
            path = _path_of(nxt) if nxt else None

    if enrich_linked_prs:
        for it in items:
            # C1: enrich on the UNION of in-flight assertions. Keying this on
            # `dispatched` alone would mean the nine issues that now assert in-flight
            # via `autodev:in-progress` never had their linked PRs read, so AMBIGUOUS
            # could never fire for them — the C1 bug wearing a C2 hat.
            if it["kind"] != "issue" or not in_flight_assertions(it["labels"]):
                continue
            linked, redirected = await _fetch_linked_prs(it["repo"], it["number"])
            it["linked_pr_numbers"] = list(linked)
            if redirected:
                transferred[(it["repo"], it["number"])] = redirected

    for key in sorted(transferred):
        notes.append(transfer_note(key[0], key[1], transferred[key]))
    return items, exhaustive, notes


def _normalise_repo(raw, default_owner="eumemic"):
    """Canonicalise a repo reference from ``run.input``; ``None`` when unreadable.

    Never guesses: a wrong guess manufactures a phantom 'agree', which is exactly
    the false-health this reconciler exists to prevent.
    """
    if isinstance(raw, dict):
        owner = raw.get("owner")
        name = raw.get("name") or raw.get("repo")
        if isinstance(owner, str) and isinstance(name, str) and owner and name:
            return "{}/{}".format(owner.strip("/"), name.strip("/"))
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
            return f"{parts[0]}/{name}"
        return None
    if text.endswith(".git"):
        text = text[:-4]
    parts = [p for p in text.strip("/").split("/") if p]
    if len(parts) == 1:
        return f"{default_owner}/{parts[0]}"
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
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
                raise ReadFailure(f"list_runs returned {type(result)!r}")
            if "error" in result:
                raise ReadFailure("list_runs failed: {}".format(result["error"]))
            rows = result.get("runs")
            if not isinstance(rows, list):
                # A missing 'runs' key is a CONTRACT failure, not an empty page.
                raise ReadFailure("list_runs response has no 'runs' list")
            pages += 1
            for row in rows:
                _check_run_row(row)
            runs.extend(rows)
            after = _paginate_check(rows)
            if after is None:
                break
    return runs, exhaustive


def classify(item, runs):
    """Classify one item against its runs. ``None`` == the labels agree.

    MIRROR of ``aios.reconcilers.work_state.classify_item`` — kept in lockstep by
    the drift tests. Two things this must NOT get wrong:

    * C1 — "claims in flight" is the UNION of in-flight assertions, never the single
      literal ``dispatched``. The trigger label(s) ride on the finding.
    * C3 — precedence is live > suspended > UNKNOWN > terminal. An unrecognised
      status is never evidence of death, not even beside a terminal sibling.
    """
    labels = item["labels"]
    triggers = in_flight_assertions(labels)
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

    def mk(classification, detail, subject, caveats=(), trigger_labels=None):
        return {
            "classification": classification,
            "repo": item["repo"],
            "number": item["number"],
            "kind": item["kind"],
            "title": item["title"],
            "url": item["url"],
            "labels": [x for x in labels if is_pipeline_label(x)],
            "detail": detail,
            "run_ids": [r.get("id", "") for r in subject],
            "run_statuses": sorted({r.get("status", "") for r in subject}),
            "updated_at": item["updated_at"],
            "caveats": list(caveats),
            "trigger_labels": list(triggers if trigger_labels is None else trigger_labels),
        }

    def _statuses(rs):
        return ", ".join(sorted({r.get("status", "") for r in rs}))

    if not triggers:
        if live:
            return mk(
                "LAGGING",
                f"{len(live)} live run(s) ({_statuses(live)}) but NO in-flight label "
                "(no `dispatched`, no `autodev:in-progress`, no `needs:human/*`)",
                live,
                trigger_labels=[],
            )
        return None

    claim = ", ".join([f"`{x}`" for x in triggers])

    if live:
        return None
    if unknown:
        # C3: handled BEFORE terminal. An unknown status may be live under a name this
        # vocabulary has not learned; a terminal sibling does not make it dead.
        others = suspended + terminal
        extra = ""
        if others:
            extra = (
                f" (alongside {len(others)} run(s) in {_statuses(others)} — a terminal "
                "sibling does NOT make an unknown status dead)"
            )
        return mk(
            "MISLABELLED",
            f"{claim} and run(s) in unrecognised status ({_statuses(unknown)}) — cannot prove live; treat as "
            f"parked pending triage{extra}",
            unknown + others,
            ("unrecognised-run-status",),
        )
    if suspended:
        return mk(
            "MISLABELLED",
            f"{claim} but {len(suspended)} run(s) suspended at a gate — parked is NOT running",
            suspended,
        )
    if terminal:
        return mk(
            "DEAD",
            f"{claim} but every run is terminal ({_statuses(terminal)})",
            terminal,
        )
    linked = item.get("linked_pr_numbers") or []
    if linked:
        # C2: linked PRs mean work demonstrably happened, so this is NOT a zombie and
        # must not land in counts["ZOMBIE"]. Its own class; phase 2 acts on the class.
        prs = ", ".join([f"#{n}" for n in linked])
        return mk(
            "AMBIGUOUS",
            f"{claim} and no unarchived run — BUT PR(s) {prs} reference this issue. Work "
            "demonstrably happened, so the join key (or the run's archival) is the "
            "suspect, not the item. NEEDS TRIAGE — deliberately NOT counted as a ZOMBIE.",
            [],
            ("has-linked-prs", "excluded-from-zombie-count"),
        )
    return mk(
        "ZOMBIE",
        f"{claim} but NO unarchived run exists for this issue — nothing (visible) ever "
        "picked it up",
        [],
        ("zombie-means-no-unarchived-run",),
    )


def build(items, items_exhaustive, runs, runs_exhaustive, failures, notes=()):
    """Join + classify. **Any failure ⇒ ALARM with counts:null.**

    The counts key is ``None`` — not ``{}``, not zeros — on ALARM, so a consumer
    cannot render 'no disagreements' from a broken read. There is no count to render.
    """
    if failures:
        return {
            "verdict": "ALARM",
            "counts": None,
            "zombie_means": ZOMBIE_MEANS,
            "total_disagreements": None,
            "failures": failures,
            "notes": list(notes),
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
        if repo is None or number is None:
            # B4: terminal runs included. Dropping an unkeyable `errored` run makes its
            # issue read as ZOMBIE with no trace of the run that did exist — a
            # manufactured false ZOMBIE, the exact failure this is meant to prevent.
            unmatched.append(
                {
                    "run_id": run.get("id", ""),
                    "status": run.get("status", ""),
                    "repo": repo,
                    "issue_number": number,
                    "reason": (
                        "run.input carries no usable (repo, issue_number) — join key "
                        "unreadable (status %s); any item this run belonged to may "
                        "therefore read as a FALSE ZOMBIE" % (run.get("status") or "unknown")
                    ),
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
        "zombie_means": ZOMBIE_MEANS,
        "in_flight_labels_checked": sorted(IN_FLIGHT_LABELS)
        + [x + "*" for x in IN_FLIGHT_PREFIXES],
        "total_disagreements": len(disagreements),
        "failures": [],
        # C4 — transfer notes ride on the report, never suppressed. A stale join key
        # is DATA a triager must see; swallowing it is how a transfer masquerades as
        # a zombie.
        "notes": list(notes),
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
        parts = [
            "ALARM",
            *sorted(
                [
                    "{}:{}".format(f.get("source", ""), f.get("reason", ""))
                    for f in report["failures"]
                ]
            ),
        ]
    else:
        parts = sorted(
            [
                "{}:{}#{}:{}:{}".format(
                    d["classification"],
                    d["repo"],
                    d["number"],
                    ",".join(sorted(d["run_statuses"])),
                    ",".join(sorted(d.get("trigger_labels") or [])),
                )
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
    items, items_exhaustive, notes = [], True, []
    try:
        items, items_exhaustive, notes = await _read_items(repos)
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
    report = build(items, items_exhaustive, runs, runs_exhaustive, failures, notes)
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
        log(f"ALARM: work-state reconciler could not read its sources: {json.dumps(failures)}")  # noqa: F821
    else:
        log(  # noqa: F821
            "work-state: {} (exhaustive={}, changed={})".format(
                json.dumps(report["counts"]), report["exhaustive"], report["changed"]
            )
        )
    return report
