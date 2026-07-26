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
    WorkItem,
    build_report,
    is_pipeline_state_label,
    render_json,
    render_markdown,
    run_record_from_payload,
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
_REQUEST_TIMEOUT_S = 30
#: Hard page ceilings. Hitting one is NOT silently ignored: the source is marked
#: non-exhaustive and every derived count renders as "at least N".
_MAX_ISSUE_PAGES = 50
_MAX_RUN_PAGES = 200
_PER_PAGE = 100


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
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            raw = resp.read().decode()
            body = json.loads(raw) if raw else None
            return body, dict(resp.headers)
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


def fetch_repo_items(
    repo: str,
    *,
    token: str,
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
    max_pages: int = _MAX_ISSUE_PAGES,
) -> tuple[list[WorkItem], bool]:
    """Every OPEN issue/PR in ``repo`` carrying at least one pipeline state label.

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
        if not isinstance(body, list):
            raise ReadFailure(f"GET {url} did not return a JSON array (got {type(body).__name__})")
        pages += 1
        for payload in body:
            if not isinstance(payload, Mapping):
                raise ReadFailure(f"GET {url} returned a non-object row")
            item = _item_from_payload(repo, payload)
            if any(is_pipeline_state_label(x) for x in item.labels):
                items.append(item)
        url = _next_link(resp_headers)
    return items, True


def fetch_linked_prs(
    repo: str,
    number: int,
    *,
    token: str,
    getter: Callable[[str, Mapping[str, str]], tuple[Any, Mapping[str, str]]] = _get,
) -> tuple[int, ...]:
    """PRs cross-referencing an issue, from its timeline (READ-ONLY).

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
    next_url: str | None = url
    while next_url is not None:
        body, resp_headers = getter(next_url, headers)
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
    return tuple(sorted(found))


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
    try:
        for repo in repos:
            items, repo_exhaustive = fetch_repo_items(repo, token=token, getter=getter)
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

    if enrich_linked_prs:
        try:
            enriched: list[WorkItem] = []
            for item in all_items:
                if item.is_dispatched and item.kind == "issue":
                    linked = fetch_linked_prs(item.repo, item.number, token=token, getter=getter)
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

    return SourceOk(name="github", items=tuple(all_items), exhaustive=exhaustive)


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
                if has_more and isinstance(cursor, str) and cursor:
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

    return build_report(
        items_read=items_read,
        runs_read=runs_read,
        generated_at=generated_at,
        repos_scanned=repos,
        meta={"workflow_ids": list(workflow_ids), "phase": "1-observe-only", "writes_performed": 0},
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
    args = parser.parse_args(argv)

    report = reconcile(
        repos=tuple(args.repos or DEFAULT_REPOS),
        github_token=os.environ.get("GITHUB_TOKEN"),
        aios_url=os.environ.get("AIOS_URL"),
        aios_api_key=os.environ.get("AIOS_API_KEY"),
        workflow_ids=tuple(args.workflow_ids or (DEV_PIPELINE_WORKFLOW_ID,)),
        enrich_linked_prs=not args.no_linked_prs,
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
