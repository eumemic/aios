"""Work-state reconciler, phase 1 (OBSERVE-ONLY): the I/O shell around the pure core.

Reads GitHub (open items carrying pipeline state labels) and the aios API (runs of
the pipeline workflows), hands both to :func:`aios.reconcilers.work_state.build_report`,
and prints the result. Every read is wrapped so a failure becomes a
:class:`~aios.reconcilers.work_state.SourceFailed` — which forces the report to
ALARM — and NEVER an empty list.

**This module makes no write of any kind against a reconciled issue or PR.** The
only HTTP verb it issues is ``GET``; :func:`_get` refuses anything else at the
transport level (:class:`ObserveOnlyViolation`), so a future edit that tries to
"just also fix the label" fails loudly instead of shipping. Phase 2 (demotion) is
gated on reviewing a week of phase-1 data, per the design of record.

Usage::

    GITHUB_TOKEN=... AIOS_URL=... AIOS_API_KEY=... \\
      python -m aios.reconcilers.work_state_cli --format markdown

Exit codes — chosen so a broken cron cannot look healthy:

* ``0`` — read OK (disagreements may exist; they are DATA, not an error).
* ``2`` — ALARM: a source could not be read. Never exits 0 on a failed read.
* ``3`` — read OK and ``--fail-on-disagreement`` was passed and some exist.

Credentials come from the ENVIRONMENT only, never argv (an argv-borne secret leaks
via ``ps`` / ``/proc/<pid>/cmdline``) — the rule ``scripts/reconcile_agents.py``
already follows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from aios.reconcilers.work_state import (
    ReconcileReport,
    RunRecord,
    SourceFailed,
    SourceOk,
    SourceRead,
    TransferEvidence,
    WorkItem,
    build_report,
    is_pipeline_state_label,
    parse_issue_ref,
    render_json,
    render_markdown,
    run_record_from_payload,
    stale_run_keys,
)

#: The repos whose pipeline labels are in scope (#2043).
DEFAULT_REPOS: tuple[str, ...] = (
    "eumemic/aios",
    "eumemic/eumemic-ops",
    "eumemic/eumemic-company",
    "eumemic/aios-console",
    "eumemic/autodev",
)

#: The durable dev pipeline: one run per issue, ``run.input`` = ``{repo, issue_number}``.
DEV_PIPELINE_WORKFLOW_ID = "wf_01KV4YGV4PSGP08TJBAY32J2VK"

_GITHUB_API = "https://api.github.com"
#: Synthetic response header carrying the post-redirect URL (see :func:`_get`).
#: Not a wire header — the transport's own report of where it actually landed.
_FINAL_URL_HEADER = "X-Reconciler-Final-Url"
_REQUEST_TIMEOUT_S = 30
#: Hard page ceilings. Hitting one is NOT silently ignored: the source is marked
#: non-exhaustive and every derived count renders as "at least N".
_MAX_ISSUE_PAGES = 50
_MAX_RUN_PAGES = 200
_PER_PAGE = 100
#: Ceiling on the C4 stale-key probes issued per reconcile. Hitting it is NOT
#: silently ignored: an explicit note says how many keys went UNPROBED, so a
#: transfer that we did not look for is visible as a gap rather than as a zombie.
_MAX_TRANSFER_PROBES = 400


class ObserveOnlyViolation(RuntimeError):
    """A non-GET was attempted. Phase 1 writes NOTHING; this is the enforcement."""


class ReadFailure(RuntimeError):
    """A source read failed. Always becomes a ``SourceFailed`` — never an empty list."""


# ─── HTTP (GET-only, by construction) ────────────────────────────────────────


def _get(
    url: str, headers: Mapping[str, str], *, method: str = "GET"
) -> tuple[Any, Mapping[str, str]]:
    """Issue one **GET**. Any other verb raises :class:`ObserveOnlyViolation`.

    Phase 1's hard constraint ("zero mutations of the reconciled issues/PRs") is
    enforced HERE, at the only door to the network, rather than by reviewer
    vigilance over call sites. A PATCH/POST cannot be smuggled through this module.
    """
    if method != "GET":
        raise ObserveOnlyViolation(
            f"phase-1 reconciler is OBSERVE-ONLY: refusing {method} {url}. "
            "Mutating a reconciled item is a failed build, not a feature."
        )
    req = urllib.request.Request(url, method="GET")
    # Belt-and-braces for the reviewer's `data=` hole: urllib infers POST from a
    # non-None body, so a future `Request(url, data=...)` would mutate while
    # containing none of the literals a source grep looks for. Assert the TRANSPORT.
    if req.get_method() != "GET" or req.data is not None:
        raise ObserveOnlyViolation(
            f"phase-1 reconciler is OBSERVE-ONLY: transport resolved to "
            f"{req.get_method()} (data={'set' if req.data is not None else 'None'}) for {url}"
        )
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            raw = resp.read().decode()
            body = json.loads(raw) if raw else None
            out = dict(resp.headers)
            # urllib follows a 301 SILENTLY. GitHub 301s a TRANSFERRED issue to its
            # new (repo, number) — the single signal that a run's stored join key is
            # stale (C4). Surface the final URL so the caller can SEE the redirect
            # instead of the transfer vanishing into a false ZOMBIE.
            final = resp.geturl()
            if final and final != url:
                out[_FINAL_URL_HEADER] = final
            return body, out
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise ReadFailure(f"GET {url} → HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ReadFailure(f"GET {url} failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        # A 200 with an unparseable body is a FAILED read, not an empty one.
        raise ReadFailure(f"GET {url} returned unparseable JSON: {exc}") from exc
    except TimeoutError as exc:
        raise ReadFailure(f"GET {url} timed out after {_REQUEST_TIMEOUT_S}s") from exc


def _next_link(headers: Mapping[str, str]) -> str | None:
    """Parse GitHub's ``Link: <...>; rel="next"`` header (the exhaustion driver)."""
    link = headers.get("Link") or headers.get("link")
    if not link:
        return None
    for part in link.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        if 'rel="next"' in "".join(segments[1:]):
            return segments[0].strip().strip("<>")
    return None


# ─── GitHub: open items carrying pipeline state labels ───────────────────────


def _item_from_payload(
    repo: str, payload: Mapping[str, Any], linked: tuple[int, ...] = ()
) -> WorkItem:
    raw_labels = payload.get("labels")
    labels = tuple(
        sorted(
            str(x["name"]) if isinstance(x, Mapping) else str(x)
            for x in (raw_labels if isinstance(raw_labels, list) else [])
        )
    )
    return WorkItem(
        repo=repo,
        number=int(payload["number"]),
        kind="pull_request" if "pull_request" in payload else "issue",
        title=str(payload.get("title", "")),
        labels=labels,
        html_url=str(payload.get("html_url", "")),
        updated_at=str(payload.get("updated_at", "")),
        created_at=str(payload.get("created_at", "")),
        linked_pr_numbers=linked,
    )


def transfer_note(repo: str, number: Any, dest: str) -> str:
    """The C4 note, in ONE place. Byte-identical to the workflow mirror's
    ``transfer_note`` — the reader drift guard compares the emitted strings, so the
    two forms cannot describe the same transfer differently."""
    return (
        f"{repo}#{number} was REDIRECTED to {dest} — the issue appears to have been "
        "TRANSFERRED, so any run keyed to these coordinates cannot join (C4)"
    )


def fetch_repo_items(
    repo: str,
    *,
    token: str,
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
    max_pages: int = _MAX_ISSUE_PAGES,
    keep_unlabelled: bool = True,
    notes: list[str] | None = None,
) -> tuple[list[WorkItem], bool]:
    """Every OPEN issue/PR in ``repo``.

    C5 — ``keep_unlabelled=True`` (the default) keeps items carrying NO pipeline
    label at all. The previous cut dropped them, which made LAGGING structurally
    under-detectable: a live run against an unlabelled issue could never be LAGGING;
    it was demoted to an ``unmatched_run`` reason-stringed "not open", which is
    false — the item IS open, it just has no label. Since LAGGING is defined by the
    ABSENCE of an in-flight assertion, the enumeration cannot pre-filter on
    presence-of-*some*-label either. Unlabelled items with no run classify to
    ``None`` and cost nothing.

    Enumerates all open items and filters locally rather than asking GitHub for
    ``?labels=dispatched``: the LAGGING class is defined by the ABSENCE of
    ``dispatched``, so a query narrowed to that label is structurally incapable of
    finding it. (This is the same shape as the stall detectors that could not see
    the zombies because they filtered on the very label that was lying.)

    Returns ``(items, exhaustive)``. Raises :class:`ReadFailure` on any read error —
    the caller turns that into a ``SourceFailed``. Never returns a short list quietly.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "aios-work-state-reconciler",
    }
    url: str | None = (
        f"{_GITHUB_API}/repos/{repo}/issues?state=open&per_page={_PER_PAGE}&sort=created&direction=desc"
    )
    items: list[WorkItem] = []
    pages = 0
    while url is not None:
        if pages >= max_pages:
            return items, False  # cap hit — caller must render "at least N"
        body, resp_headers = getter(url, headers)
        if resp_headers.get(_FINAL_URL_HEADER) and notes is not None:
            # C4 on the LIST path: the repo itself was renamed/moved and urllib followed
            # the 301 silently. Say so — the workflow mirror emits the same note for the
            # same event, and the reader drift guard compares them.
            notes.append(transfer_note(repo, "*", str(resp_headers[_FINAL_URL_HEADER])))
        if not isinstance(body, list):
            raise ReadFailure(f"GET {url} did not return a JSON array (got {type(body).__name__})")
        pages += 1
        for payload in body:
            if not isinstance(payload, Mapping):
                raise ReadFailure(f"GET {url} returned a non-object row")
            item = _item_from_payload(repo, payload)
            if keep_unlabelled or any(is_pipeline_state_label(x) for x in item.labels):
                items.append(item)
        url = _next_link(resp_headers)
    return items, True


def fetch_linked_prs(
    repo: str,
    number: int,
    *,
    token: str,
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
) -> tuple[tuple[int, ...], str | None]:
    """PRs cross-referencing an issue + any redirect seen, from its timeline (READ-ONLY).

    Returns ``(linked_pr_numbers, redirected_to_url_or_None)``. The second element is
    the C4 transfer signal: a non-None value means GitHub 301'd this (repo, number),
    i.e. the issue MOVED and every run keyed to the old coordinates is now unjoinable.

    Used only to qualify a ZOMBIE verdict: "no run ever" + "a PR exists" means the
    join key is suspect, and the report says so instead of asserting a zombie.
    eumemic-company#71 is the live instance of this shape.
    """
    url = f"{_GITHUB_API}/repos/{repo}/issues/{number}/timeline?per_page={_PER_PAGE}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "aios-work-state-reconciler",
    }
    found: set[int] = set()
    redirected_to: str | None = None
    next_url: str | None = url
    while next_url is not None:
        body, resp_headers = getter(next_url, headers)
        if resp_headers.get(_FINAL_URL_HEADER):
            # C4 — 68 issues were transferred between repos on 2026-07-25. A transfer
            # gives the issue a NEW (repo, number) while every run.input still holds
            # the OLD one, so the join key is stale and the item reads as a false
            # ZOMBIE (or its run as unmatched). The 301 is the only signal, and urllib
            # follows it silently. Detection only in phase 1 — we report it, we do not
            # rewrite the key.
            redirected_to = str(resp_headers[_FINAL_URL_HEADER])
        if not isinstance(body, list):
            raise ReadFailure(f"GET {next_url} did not return a JSON array")
        for event in body:
            if not isinstance(event, Mapping):
                continue
            if event.get("event") != "cross-referenced":
                continue
            source = event.get("source")
            issue = source.get("issue") if isinstance(source, Mapping) else None
            if isinstance(issue, Mapping) and "pull_request" in issue:
                num = issue.get("number")
                if isinstance(num, int):
                    found.add(num)
        next_url = _next_link(resp_headers)
    return tuple(sorted(found)), redirected_to


def read_github_items(
    repos: Sequence[str],
    *,
    token: str,
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
    enrich_linked_prs: bool = True,
) -> SourceRead:
    """Read every repo. **One repo failing fails the whole source.**

    A partial read is a LIE with the shape of a health report: the repo that could
    not be read is exactly where the disagreements might be. So there is no
    per-repo ``continue`` here — the first failure aborts into ``SourceFailed`` and
    the report ALARMs.
    """
    all_items: list[WorkItem] = []
    exhaustive = True
    notes: list[str] = []
    try:
        for repo in repos:
            items, repo_exhaustive = fetch_repo_items(repo, token=token, getter=getter, notes=notes)
            exhaustive = exhaustive and repo_exhaustive
            all_items.extend(items)
    except ReadFailure as exc:
        return SourceFailed(name="github", reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Never a bare except, and never a swallow: any unexpected read-path error
        # is still reported as a FAILED read rather than crashing into a traceback
        # a cron would report as "no output, probably fine".
        return SourceFailed(
            name="github", reason=f"unexpected read error: {type(exc).__name__}: {exc}"
        )

    transferred: dict[tuple[str, int], str] = {}
    if enrich_linked_prs:
        try:
            enriched: list[WorkItem] = []
            for item in all_items:
                # C1: enrich on the UNION of in-flight assertions. Keying this on
                # `dispatched` alone meant the nine issues that now assert in-flight
                # via `autodev:in-progress` never had their linked PRs read, so the
                # AMBIGUOUS class (C2) could never fire for them.
                if item.claims_in_flight and item.kind == "issue":
                    linked, redirected = fetch_linked_prs(
                        item.repo, item.number, token=token, getter=getter
                    )
                    if redirected:
                        transferred[item.key] = redirected
                    enriched.append(
                        WorkItem(
                            repo=item.repo,
                            number=item.number,
                            kind=item.kind,
                            title=item.title,
                            labels=item.labels,
                            html_url=item.html_url,
                            updated_at=item.updated_at,
                            created_at=item.created_at,
                            linked_pr_numbers=linked,
                        )
                    )
                else:
                    enriched.append(item)
            all_items = enriched
        except ReadFailure as exc:
            return SourceFailed(name="github-timeline", reason=str(exc))

    for (repo_name, number), dest in sorted(transferred.items()):
        notes.append(transfer_note(repo_name, number, dest))

    return SourceOk(
        name="github",
        items=tuple(all_items),
        exhaustive=exhaustive,
        notes=tuple(notes),
    )


# ─── C4: the INVERTED stale-join-key probe ───────────────────────────────────


def probe_transfer(
    repo: str,
    number: int,
    *,
    token: str,
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
) -> tuple[str, str | None]:
    """GET the STALE coordinates directly. Returns ``(destination, unparseable_reason)``.

    ``destination`` is ``""`` when the coordinates still resolve to themselves (no
    transfer). GET-only, like every other read here.

    Why this direction. A transfer is invisible from the new coordinates (they answer
    200 as themselves) and invisible from enumeration (which only ever returns the new
    pair). It is visible from exactly one place — the OLD pair, which survives only in
    ``run.input``. The first cut of C4 probed an ENUMERATED issue and waited for a
    redirect, a world production cannot present, so every one of the 68 issues that
    moved repos on 2026-07-25 still became a FALSE ZOMBIE. Hence: ask about the stale
    coordinates.

    Two forms of authoritative evidence, because urllib follows the 301 silently:

    * the transport reports a FINAL URL different from the one we asked for; or
    * the body answers with different coordinates than the ones requested
      (``url``/``html_url``/``repository_url`` + ``number``) — which is what a
      followed redirect looks like once the hop itself is invisible.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "aios-work-state-reconciler",
    }
    url = f"{_GITHUB_API}/repos/{repo}/issues/{number}"
    body, resp_headers = getter(url, headers)
    final = resp_headers.get(_FINAL_URL_HEADER)
    if final and parse_issue_ref(final) not in (None, (repo, number)):
        return str(final), None
    if isinstance(body, Mapping):
        for field in ("url", "html_url"):
            ref = parse_issue_ref(body.get(field))
            if ref is not None and ref != (repo, number):
                return str(body[field]), None
    if final and final != url:
        # A hop we cannot name is still a hop; report it verbatim rather than guess.
        return str(final), None
    return "", None


def probe_stale_keys(
    items: Sequence[WorkItem],
    runs: Sequence[RunRecord],
    *,
    token: str,
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
    max_probes: int = _MAX_TRANSFER_PROBES,
) -> tuple[tuple[TransferEvidence, ...], tuple[str, ...]]:
    """Probe every run join key that matched NO enumerated item (C4). GET-only.

    Returns ``(evidence, notes)``. A key that still resolves to itself yields nothing
    — most stale keys are simply runs against CLOSED issues, which is normal.

    A 404/410 is NOT a failure here and must not ALARM the whole reconcile: a run
    against a deleted or invisible issue is an expected shape of history, and the run
    is already reported as unmatched. Only the transfer signal is extracted.
    """
    stale = stale_run_keys(items, runs)
    notes: list[str] = []
    evidence: list[TransferEvidence] = []
    keys = list(stale.items())
    if len(keys) > max_probes:
        notes.append(
            f"C4 stale-key probe capped at {max_probes} of {len(keys)} stale join key(s) — "
            f"{len(keys) - max_probes} were NOT probed, so a transfer among them would be "
            "unreported. Counts involving ZOMBIE are therefore floors for those items."
        )
        keys = keys[:max_probes]
    for (repo, number), run_ids in keys:
        try:
            destination, _ = probe_transfer(repo, number, token=token, getter=getter)
        except ReadFailure as exc:
            # 404/410/403 on a stale coordinate is history, not a broken source.
            notes.append(
                f"C4 stale-key probe could not read {repo}#{number} ({exc}) — a transfer "
                "there would be unreported; the run(s) holding this key remain listed as "
                f"unmatched ({', '.join(run_ids)})"
            )
            continue
        if not destination:
            continue
        ref = parse_issue_ref(destination)
        evidence.append(
            TransferEvidence(
                stale_repo=repo,
                stale_number=number,
                destination=destination,
                new_repo=None if ref is None else ref[0],
                new_number=None if ref is None else ref[1],
                run_ids=run_ids,
            )
        )
    return tuple(evidence), tuple(notes)


# ─── aios: runs of the pipeline workflows ────────────────────────────────────


def read_aios_runs(
    *,
    base_url: str,
    api_key: str,
    workflow_ids: Sequence[str] = (DEV_PIPELINE_WORKFLOW_ID,),
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
    max_pages: int = _MAX_RUN_PAGES,
) -> SourceRead:
    """Every run of the pipeline workflow(s), keyed ``(repo, issue_number)`` from ``run.input``.

    Pagination follows ``next_cursor`` to exhaustion; hitting the page cap marks the
    source non-exhaustive so counts render as floors instead of being quietly wrong.

    NOTE on the ZOMBIE class: "no run, ever" is only sound if this read reaches the
    WHOLE history, which is why terminal runs are enumerated too and why a truncated
    read can never be presented as a total.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "aios-work-state-reconciler",
    }
    records: list[RunRecord] = []
    exhaustive = True
    try:
        for workflow_id in workflow_ids:
            url: str | None = (
                f"{base_url.rstrip('/')}/v1/runs?workflow_id={urllib.parse.quote(workflow_id)}"
                f"&limit={_PER_PAGE}"
            )
            pages = 0
            while url is not None:
                if pages >= max_pages:
                    exhaustive = False
                    break
                body, _ = getter(url, headers)
                if not isinstance(body, Mapping):
                    raise ReadFailure(f"GET {url} did not return the ListResponse envelope")
                rows = body.get("data")
                if not isinstance(rows, list):
                    # The envelope's rows live under 'data'; a missing key is a
                    # CONTRACT failure, not an empty page.
                    raise ReadFailure(f"GET {url} response has no list 'data' key")
                pages += 1
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise ReadFailure(f"GET {url} returned a non-object run row")
                    records.append(run_record_from_payload(row))
                cursor = body.get("next_cursor")
                has_more = body.get("has_more")
                if has_more and not (isinstance(cursor, str) and cursor):
                    # B1 — THE fail-loud break. The server said THERE IS MORE and handed
                    # us no usable cursor. The old code fell into the `else` branch,
                    # ended pagination, and reported exhaustive=True: a truncated read
                    # rendered as exhaustive, which is the 2026-07-25 failure mode with
                    # extra steps. Every ZOMBIE verdict is derived from "no run exists in
                    # the list I read", so an unread page of runs manufactures false
                    # ZOMBIEs and hides live runs from LAGGING. This is a FAILED read.
                    raise ReadFailure(
                        f"GET {url} returned has_more={has_more!r} with an unusable "
                        f"next_cursor ({cursor!r}) — the server says there is MORE run "
                        "history and gave no way to fetch it. Refusing to report a "
                        "truncated read as exhaustive."
                    )
                if has_more:
                    assert isinstance(cursor, str)
                    url = f"{base_url.rstrip('/')}/v1/runs?cursor={urllib.parse.quote(cursor)}"
                else:
                    url = None
    except ReadFailure as exc:
        return SourceFailed(name="aios-runs", reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return SourceFailed(
            name="aios-runs", reason=f"unexpected read error: {type(exc).__name__}: {exc}"
        )

    return SourceOk(name="aios-runs", items=tuple(records), exhaustive=exhaustive)


# ─── orchestration ───────────────────────────────────────────────────────────


def reconcile(
    *,
    repos: Sequence[str],
    github_token: str | None,
    aios_url: str | None,
    aios_api_key: str | None,
    workflow_ids: Sequence[str] = (DEV_PIPELINE_WORKFLOW_ID,),
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
    now: str | None = None,
    enrich_linked_prs: bool = True,
    probe_transfers: bool = True,
) -> ReconcileReport:
    """Read both sources and build the report. Missing credentials ⇒ ALARM.

    A missing token is a READ FAILURE, not a reason to skip a source: the sandbox
    sweep that "returned zero held PRs" on 2026-07-25 did so because its credential
    was rejected and the empty result read as health. That path does not exist here.
    """
    generated_at = now or datetime.now(UTC).isoformat()
    items_read: SourceRead
    if not github_token:
        items_read = SourceFailed(
            name="github",
            reason="GITHUB_TOKEN is unset — cannot read GitHub (NOT 'no disagreements')",
        )
    else:
        items_read = read_github_items(
            repos, token=github_token, getter=getter, enrich_linked_prs=enrich_linked_prs
        )

    runs_read: SourceRead
    if not aios_url or not aios_api_key:
        runs_read = SourceFailed(
            name="aios-runs",
            reason="AIOS_URL / AIOS_API_KEY unset — cannot read run state (NOT 'no disagreements')",
        )
    else:
        runs_read = read_aios_runs(
            base_url=aios_url, api_key=aios_api_key, workflow_ids=workflow_ids, getter=getter
        )

    # C4 — the INVERTED probe. It needs BOTH sources (the join keys that matched no
    # enumerated item are only knowable once both are read), so it runs here rather
    # than inside either reader. Skipped when either read failed: the report already
    # ALARMs, and probing coordinates derived from a partial read would attach
    # confident transfer evidence to an unreliable join.
    transfers: tuple[TransferEvidence, ...] = ()
    if (
        probe_transfers
        and github_token
        and isinstance(items_read, SourceOk)
        and isinstance(runs_read, SourceOk)
    ):
        items = tuple(i for i in items_read.items if isinstance(i, WorkItem))
        runs = tuple(r for r in runs_read.items if isinstance(r, RunRecord))
        try:
            transfers, probe_notes = probe_stale_keys(
                items, runs, token=github_token, getter=getter
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            transfers, probe_notes = (
                (),
                (
                    f"C4 stale-key probe failed: {type(exc).__name__}: {exc} — a transferred "
                    "issue among the stale join keys would be unreported",
                ),
            )
        if probe_notes:
            items_read = SourceOk(
                name=items_read.name,
                items=items_read.items,
                exhaustive=items_read.exhaustive,
                pages_read=items_read.pages_read,
                notes=items_read.notes + probe_notes,
            )

    return build_report(
        items_read=items_read,
        runs_read=runs_read,
        generated_at=generated_at,
        repos_scanned=repos,
        meta={"workflow_ids": list(workflow_ids), "phase": "1-observe-only", "writes_performed": 0},
        transfers=transfers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="work-state-reconciler",
        description=(
            "Phase-1 OBSERVE-ONLY work-state reconciler: joins GitHub pipeline labels "
            "against live aios run state and reports the disagreements. Writes nothing."
        ),
    )
    parser.add_argument(
        "--repo", action="append", dest="repos", help="repeatable; defaults to the org set"
    )
    parser.add_argument("--workflow-id", action="append", dest="workflow_ids")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--fail-on-disagreement",
        action="store_true",
        help="exit 3 when disagreements exist (distinct from exit 2 = ALARM/read failed)",
    )
    parser.add_argument(
        "--no-linked-prs",
        action="store_true",
        help="skip the per-issue timeline read used to qualify ZOMBIE verdicts",
    )
    parser.add_argument(
        "--no-transfer-probe",
        action="store_true",
        help=(
            "skip the C4 stale-join-key probe (GET of the coordinates no enumerated item "
            "matched); a transferred issue then reads as a ZOMBIE"
        ),
    )
    args = parser.parse_args(argv)

    report = reconcile(
        repos=tuple(args.repos or DEFAULT_REPOS),
        github_token=os.environ.get("GITHUB_TOKEN"),
        aios_url=os.environ.get("AIOS_URL"),
        aios_api_key=os.environ.get("AIOS_API_KEY"),
        workflow_ids=tuple(args.workflow_ids or (DEV_PIPELINE_WORKFLOW_ID,)),
        enrich_linked_prs=not args.no_linked_prs,
        probe_transfers=not args.no_transfer_probe,
    )

    out = render_json(report) if args.format == "json" else render_markdown(report)
    print(out)

    if report.alarmed:
        print(
            "\nFATAL: work-state reconciler could not read its sources — this is an ALARM, "
            "not a clean run.",
            file=sys.stderr,
        )
        return 2
    if args.fail_on_disagreement and report.disagreements:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - thin entry point
    raise SystemExit(main())
