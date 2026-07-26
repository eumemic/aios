"""Work-state reconciliation, phase 1 (OBSERVE-ONLY): the pure join + classify core.

The defect this exists to kill (eumemic/aios#2043, design of record:
``architecture/work-state-reconciliation.md`` in eumemic-company):

    **A GitHub label is an assertion that nothing re-checks against the world.**

``dispatched`` is a sticker. It stays stuck whether or not any machine ever picked
the work up — measured 2026-07-25, 14 issues across the org carried ``dispatched``
with no run and no PR, to 12 days old. The stall detectors could not see them
*because they read labels too*: a detector that consumes the same asserted state it
is meant to check is not a detector.

The truth already exists. The dev pipeline is a durable aios workflow with ONE RUN
PER ISSUE: ``run.input`` carries ``{repo, issue_number}`` and ``run.status`` carries
the live state. Nothing joined the two. **This module is the join.**

Layering
--------
This module is PURE: dataclasses, a join, a classifier, a hash. No network, no
clock, no ``os.environ`` — every input is passed in, so the whole classification
surface is unit-testable offline and the reconciler is deterministic and
re-runnable (per #2043: "no dependence on prior in-memory state beyond a
change-detection hash"). The I/O shell lives in
:mod:`aios.reconcilers.work_state_cli`; the durable-workflow form lives in
``infra/workflows/work-state-reconciler.wf.py``.

Fail-loud, structurally (the acceptance criterion that outranks the feature)
----------------------------------------------------------------------------
An empty result from a BROKEN query rendering as health happened twice on
2026-07-25 and is the exact bug class this effort exists to kill. So the report is
**not** a list of disagreements — it is a :class:`ReconcileReport` whose
:attr:`~ReconcileReport.verdict` is ``ALARM`` whenever any source read failed, and
whose per-class counts are ``None`` (not ``0``) in that case. There is no way to
render "no disagreements" from a failed read, because there is no count to render:
``counts`` does not exist unless every source is :class:`SourceOk`. "Read failed"
and "read succeeded and found nothing" are different types here, not different
values of the same type.

The four classes (#2043)
------------------------
=================================  ================================  ==============
Label says                         Runs say                          Classification
=================================  ================================  ==============
``dispatched``                     a live run exists                 agree
``dispatched``                     no run, ever                      ZOMBIE
``dispatched``                     run terminal                      DEAD
``dispatched``                     run suspended at a gate           MISLABELLED
no ``dispatched``                  a live run exists                 LAGGING
=================================  ================================  ==============

*Parked is NOT running*: a ``suspended`` run is a **different condition** from a
running one (it is waiting on a human at a gate), so it gets its own class rather
than being folded into "agree" — that fold is how a gate-parked item hides inside
a `dispatched` label forever.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# ─── run-status vocabulary (mirrors aios.models.workflows.WfRunStatus) ───────
#
# Deliberately restated as plain frozensets rather than imported: the workflow-run
# form of this reconciler executes in the credential-free script host, which may
# not import ``aios.*``. The drift guard is a unit test that asserts these agree
# with ``WfRunStatus`` / ``TERMINAL_RUN_STATUSES``, so a new status can never
# silently fall through the classifier.

#: Statuses that mean "a machine is actually on this, right now".
LIVE_RUN_STATUSES: frozenset[str] = frozenset({"pending", "running"})
#: Parked at a durable ``gate()`` — waiting on a human. NOT running.
SUSPENDED_RUN_STATUSES: frozenset[str] = frozenset({"suspended"})
#: Terminal: the run is over, whatever the label still claims.
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "errored", "cancelled"})

#: The label whose truth this reconciler checks.
DISPATCHED_LABEL = "dispatched"

#: Labels that assert something about pipeline state. An item carrying any of
#: these is in scope for enumeration (the ``dispatched`` join is the phase-1
#: check; the rest are carried as evidence for the phase-2 gate work).
PIPELINE_STATE_LABEL_PREFIXES: tuple[str, ...] = ("needs:human/", "autodev:", "pipeline:")
PIPELINE_STATE_LABELS: frozenset[str] = frozenset(
    {
        DISPATCHED_LABEL,
        "approved",
        "hold",
        "paused",
        "escalated",
        "blocked",
        "ci-loop-exhausted",
        "merge:approved",
    }
)

Classification = Literal["ZOMBIE", "DEAD", "MISLABELLED", "LAGGING"]

#: Stable ordering for rendering + the change-detection hash.
CLASSES: tuple[Classification, ...] = ("ZOMBIE", "DEAD", "MISLABELLED", "LAGGING")


def is_pipeline_state_label(label: str) -> bool:
    """True iff ``label`` asserts something about pipeline work state."""
    return label in PIPELINE_STATE_LABELS or label.startswith(PIPELINE_STATE_LABEL_PREFIXES)


# ─── inputs ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkItem:
    """One open GitHub issue or PR, as read (never as written — phase 1 writes nothing)."""

    repo: str  # canonical "owner/name"
    number: int
    kind: Literal["issue", "pull_request"]
    title: str
    labels: tuple[str, ...]
    html_url: str
    updated_at: str  # ISO-8601, verbatim from GitHub
    created_at: str = ""
    #: PRs cross-referencing this issue. Evidence only — a ZOMBIE that HAS linked
    #: PRs is flagged (``has_linked_prs``) rather than silently trusted, because
    #: "no run" plus "a PR exists" means the join key is wrong, not that the work
    #: never happened. eumemic-company#71 is exactly this shape and is the
    #: designed-in check on the join logic.
    linked_pr_numbers: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[str, int]:
        return (self.repo, self.number)

    @property
    def is_dispatched(self) -> bool:
        return DISPATCHED_LABEL in self.labels

    @property
    def pipeline_labels(self) -> tuple[str, ...]:
        return tuple(sorted(x for x in self.labels if is_pipeline_state_label(x)))


@dataclass(frozen=True)
class RunRecord:
    """One aios run of a pipeline workflow, reduced to what the join needs."""

    run_id: str
    status: str
    repo: str | None  # canonical "owner/name" parsed from run.input, None if unparseable
    issue_number: int | None
    workflow_id: str | None = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def key(self) -> tuple[str, int] | None:
        if self.repo is None or self.issue_number is None:
            return None
        return (self.repo, self.issue_number)

    @property
    def is_live(self) -> bool:
        return self.status in LIVE_RUN_STATUSES

    @property
    def is_suspended(self) -> bool:
        return self.status in SUSPENDED_RUN_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES


# ─── source reads: "failed" and "found nothing" are DIFFERENT TYPES ──────────


@dataclass(frozen=True)
class SourceOk:
    """A read that SUCCEEDED. ``items`` may legitimately be empty.

    ``exhaustive=False`` means pagination stopped at a safety cap, so every count
    derived from it is a floor — rendered "at least N", never as a total (#2043:
    "paginate to exhaustion, or explicitly report 'at least N'").
    """

    name: str
    items: tuple[Any, ...]
    exhaustive: bool = True
    pages_read: int = 0

    ok: Literal[True] = True


@dataclass(frozen=True)
class SourceFailed:
    """A read that FAILED. Carries no items — not an empty list, no items *at all*.

    This is the type-level enforcement of fail-loud: a failed read cannot be
    iterated into "0 disagreements", because it has nothing to iterate.
    """

    name: str
    reason: str

    ok: Literal[False] = False


SourceRead = SourceOk | SourceFailed


# ─── outputs ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Disagreement:
    """One item whose label and whose runs do not agree."""

    classification: Classification
    repo: str
    number: int
    kind: Literal["issue", "pull_request"]
    title: str
    html_url: str
    labels: tuple[str, ...]
    detail: str
    run_ids: tuple[str, ...] = ()
    run_statuses: tuple[str, ...] = ()
    updated_at: str = ""
    #: Non-fatal qualifiers a human should see before acting (phase 2 will act).
    caveats: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, int]:
        return (self.repo, self.number)

    def identity(self) -> str:
        """The change-detection identity: class + item + the run statuses behind it.

        Run *ids* are excluded so a re-dispatch that produces the same verdict for
        the same reason does not spam the seat; statuses are included so a
        transition (suspended → errored) DOES.
        """
        return (
            f"{self.classification}:{self.repo}#{self.number}:{','.join(sorted(self.run_statuses))}"
        )


@dataclass(frozen=True)
class UnmatchedRun:
    """A live run whose ``(repo, issue_number)`` matches no OPEN item we enumerated.

    Not one of the four classes — it is either a run against a closed issue or a
    join-key mismatch. Reported (never swallowed) so a broken join surfaces as data
    instead of as a quietly-shrinking LAGGING count.
    """

    run_id: str
    status: str
    repo: str | None
    issue_number: int | None
    reason: str


@dataclass(frozen=True)
class ReconcileReport:
    """The reconciler's output. Read :attr:`verdict` FIRST — always."""

    #: ``OK`` = every source read succeeded (disagreements may be 0 — real health).
    #: ``ALARM`` = at least one source read FAILED; counts are meaningless and are
    #: therefore absent. Never collapse these two.
    verdict: Literal["OK", "ALARM"]
    disagreements: tuple[Disagreement, ...] = ()
    unmatched_runs: tuple[UnmatchedRun, ...] = ()
    failures: tuple[SourceFailed, ...] = ()
    #: Sources that read OK but stopped short of exhaustion; their counts are floors.
    truncated_sources: tuple[str, ...] = ()
    items_scanned: int = 0
    runs_scanned: int = 0
    repos_scanned: tuple[str, ...] = ()
    generated_at: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def alarmed(self) -> bool:
        return self.verdict == "ALARM"

    @property
    def exhaustive(self) -> bool:
        """False when any source stopped at a cap — every count is then a floor."""
        return not self.truncated_sources

    @property
    def counts(self) -> Mapping[Classification, int] | None:
        """Per-class counts, or **None** when the run alarmed.

        ``None`` — not ``{}``, not zeros. A caller that wants to print "0
        disagreements" is forced to handle the ALARM case first; there is no
        count to print otherwise. This is the whole anti-pattern, closed at the
        type level.
        """
        if self.alarmed:
            return None
        return {c: sum(1 for d in self.disagreements if d.classification == c) for c in CLASSES}

    def by_class(self, classification: Classification) -> tuple[Disagreement, ...]:
        return tuple(d for d in self.disagreements if d.classification == classification)

    def disagreement_hash(self) -> str:
        """Stable hash of the disagreement SET — the seat-wake change detector.

        Sorted, so ordering churn from the API never fires a wake; identity-based
        (see :meth:`Disagreement.identity`), so the same finding on the same item
        for the same reason is the same hash tomorrow. On ALARM the hash folds in
        the failure reasons, so a persistent outage does not read as "unchanged,
        nothing to say" — a NEW failure wakes the seat.
        """
        parts = sorted(d.identity() for d in self.disagreements)
        if self.alarmed:
            parts = ["ALARM", *sorted(f"{f.name}:{f.reason}" for f in self.failures)]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        counts = self.counts
        return {
            "verdict": self.verdict,
            "counts": None if counts is None else {k: counts[k] for k in CLASSES},
            "total_disagreements": None if self.alarmed else len(self.disagreements),
            "exhaustive": self.exhaustive,
            "truncated_sources": list(self.truncated_sources),
            "items_scanned": self.items_scanned,
            "runs_scanned": self.runs_scanned,
            "repos_scanned": list(self.repos_scanned),
            "generated_at": self.generated_at,
            "disagreement_hash": self.disagreement_hash(),
            "failures": [{"source": f.name, "reason": f.reason} for f in self.failures],
            "disagreements": [
                {
                    "classification": d.classification,
                    "repo": d.repo,
                    "number": d.number,
                    "kind": d.kind,
                    "title": d.title,
                    "url": d.html_url,
                    "labels": list(d.labels),
                    "detail": d.detail,
                    "run_ids": list(d.run_ids),
                    "run_statuses": list(d.run_statuses),
                    "updated_at": d.updated_at,
                    "caveats": list(d.caveats),
                }
                for d in self.disagreements
            ],
            "unmatched_runs": [
                {
                    "run_id": u.run_id,
                    "status": u.status,
                    "repo": u.repo,
                    "issue_number": u.issue_number,
                    "reason": u.reason,
                }
                for u in self.unmatched_runs
            ],
            "meta": dict(self.meta),
        }


# ─── the join key ────────────────────────────────────────────────────────────


def normalise_repo(raw: Any, *, default_owner: str = "eumemic") -> str | None:
    """Canonicalise a repo reference from ``run.input`` to ``owner/name``.

    Accepts ``"eumemic/aios"``, ``"aios"`` (owner defaulted), a full GitHub URL, and
    a ``{"owner": ..., "name": ...}`` object, because ``run.input`` is arbitrary JSON
    written by several dispatchers over months. Returns ``None`` when the value
    cannot be understood — the caller must then report the run as unmatched rather
    than guess, since a wrong guess manufactures a phantom "agree" and re-creates
    exactly the bug we are killing.
    """
    if isinstance(raw, Mapping):
        owner = raw.get("owner")
        name = raw.get("name") or raw.get("repo")
        if isinstance(owner, str) and isinstance(name, str) and owner and name:
            return f"{owner.strip('/')}/{name.strip('/')}"
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
            return f"{parts[0]}/{parts[1].removesuffix('.git')}"
        return None
    text = text.strip("/").removesuffix(".git")
    parts = [p for p in text.split("/") if p]
    if len(parts) == 1:
        return f"{default_owner}/{parts[0]}"
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def _coerce_issue_number(raw: Any) -> int | None:
    """``run.input`` numbers arrive as int, ``"337"``, or ``"#337"``. Never guess."""
    if isinstance(raw, bool):  # bool is an int subclass; a bool is not an issue number
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip().lstrip("#")
        if text.isdigit():
            n = int(text)
            return n if n > 0 else None
    return None


#: The keys a dispatcher may have used for the issue number in ``run.input``.
_ISSUE_NUMBER_KEYS = ("issue_number", "issue", "number", "issueNumber")
_REPO_KEYS = ("repo", "repository", "repo_full_name", "full_name")


def run_record_from_payload(
    payload: Mapping[str, Any], *, default_owner: str = "eumemic"
) -> RunRecord:
    """Build a :class:`RunRecord` from an aios run dict (``list_runs`` / ``GET /v1/runs``).

    A run whose input carries no usable ``(repo, issue_number)`` yields a record with
    ``repo``/``issue_number`` ``None`` — it joins to nothing and is reported as
    unmatched. It is never dropped: a silently-dropped run is a missing "agree",
    which manufactures a false ZOMBIE.
    """
    raw_input = payload.get("input")
    inp: Mapping[str, Any] = raw_input if isinstance(raw_input, Mapping) else {}
    repo_raw: Any = None
    for k in _REPO_KEYS:
        if k in inp:
            repo_raw = inp[k]
            break
    issue_raw: Any = None
    for k in _ISSUE_NUMBER_KEYS:
        if k in inp:
            issue_raw = inp[k]
            break
    return RunRecord(
        run_id=str(payload.get("id", "")),
        status=str(payload.get("status", "")),
        repo=normalise_repo(repo_raw, default_owner=default_owner),
        issue_number=_coerce_issue_number(issue_raw),
        workflow_id=(
            payload.get("workflow_id") if isinstance(payload.get("workflow_id"), str) else None
        ),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
    )


def index_runs(runs: Iterable[RunRecord]) -> dict[tuple[str, int], list[RunRecord]]:
    """Group runs by ``(repo, issue_number)``. Unkeyable runs are excluded (and must
    be reported separately by the caller — see :func:`build_report`)."""
    index: dict[tuple[str, int], list[RunRecord]] = {}
    for run in runs:
        key = run.key
        if key is None:
            continue
        index.setdefault(key, []).append(run)
    return index


# ─── the classifier ──────────────────────────────────────────────────────────


def _statuses(runs: Sequence[RunRecord]) -> tuple[str, ...]:
    return tuple(sorted({r.status for r in runs}))


def classify_item(item: WorkItem, runs: Sequence[RunRecord]) -> Disagreement | None:
    """Classify ONE item against every run keyed to it. ``None`` == the label agrees.

    Precedence is deliberate and is the heart of the design: **live > suspended >
    terminal**. One live run means work really is happening, whatever the other runs
    say. With no live run, a *suspended* run means parked-at-a-gate — a different
    condition from running, hence MISLABELLED rather than "agree" (folding parked
    into agree is how a gate-parked item hides behind ``dispatched`` forever). Only
    with no live and no suspended run is the item DEAD/ZOMBIE.
    """
    live = [r for r in runs if r.is_live]
    suspended = [r for r in runs if r.is_suspended]
    terminal = [r for r in runs if r.is_terminal]
    unknown = [r for r in runs if not (r.is_live or r.is_suspended or r.is_terminal)]

    def _mk(
        classification: Classification,
        detail: str,
        subject: Sequence[RunRecord],
        caveats: tuple[str, ...] = (),
    ) -> Disagreement:
        return Disagreement(
            classification=classification,
            repo=item.repo,
            number=item.number,
            kind=item.kind,
            title=item.title,
            html_url=item.html_url,
            labels=item.pipeline_labels,
            detail=detail,
            run_ids=tuple(r.run_id for r in subject),
            run_statuses=_statuses(subject),
            updated_at=item.updated_at,
            caveats=caveats,
        )

    if not item.is_dispatched:
        # Label says nothing; runs say something. The projection lags reality.
        if live:
            return _mk(
                "LAGGING",
                f"{len(live)} live run(s) ({', '.join(_statuses(live))}) but no `dispatched` label",
                live,
            )
        return None

    # From here: the item claims `dispatched`.
    if live:
        return None  # agree — a machine really is on it

    if unknown and not (suspended or terminal):
        # An unrecognised status is NOT evidence of death. Refuse to classify rather
        # than guess a zombie: a wrong DEAD here is a false alarm that trains people
        # to ignore the report.
        return _mk(
            "MISLABELLED",
            "run(s) in unrecognised status "
            f"({', '.join(_statuses(unknown))}) — cannot prove live; treat as parked pending triage",
            unknown,
            caveats=("unrecognised-run-status",),
        )

    if suspended:
        return _mk(
            "MISLABELLED",
            f"{len(suspended)} run(s) suspended at a gate — parked is NOT running",
            suspended,
        )

    if terminal:
        by_status = ", ".join(_statuses(terminal))
        return _mk(
            "DEAD",
            f"`dispatched` but every run is terminal ({by_status})",
            terminal,
        )

    caveats: tuple[str, ...] = ()
    if item.linked_pr_numbers:
        # "No run ever" AND "a PR exists" cannot both be true of healthy work — the
        # join key is wrong, or the work predates the workflow. Flag it; do not
        # quietly count it as a zombie.
        caveats = ("has-linked-prs",)
    detail = "`dispatched` but NO run exists for this issue — nothing ever picked it up"
    if item.linked_pr_numbers:
        detail += (
            f" (but PR(s) {', '.join('#' + str(n) for n in item.linked_pr_numbers)} reference it —"
            " verify the join key before believing this)"
        )
    return _mk("ZOMBIE", detail, (), caveats=caveats)


# ─── the report builder ──────────────────────────────────────────────────────


def build_report(
    *,
    items_read: SourceRead,
    runs_read: SourceRead,
    generated_at: str = "",
    repos_scanned: Sequence[str] = (),
    meta: Mapping[str, Any] | None = None,
) -> ReconcileReport:
    """Join the two sources and classify. **Any failed source ⇒ ALARM, no counts.**

    This function is the choke point where "empty" could have become "healthy", and
    it is where that is made impossible: it inspects the SOURCE OBJECTS, not their
    contents. If either is a :class:`SourceFailed` it returns an ALARM report
    immediately — no classification is attempted, no zeros are produced, and
    :attr:`ReconcileReport.counts` is ``None``. A caller cannot accidentally print
    health, because there is nothing healthy-shaped to print.
    """
    failures = tuple(s for s in (items_read, runs_read) if isinstance(s, SourceFailed))
    if failures:
        return ReconcileReport(
            verdict="ALARM",
            failures=failures,
            repos_scanned=tuple(repos_scanned),
            generated_at=generated_at,
            meta=dict(meta or {}),
        )

    assert isinstance(items_read, SourceOk) and isinstance(runs_read, SourceOk)
    items = tuple(i for i in items_read.items if isinstance(i, WorkItem))
    runs = tuple(r for r in runs_read.items if isinstance(r, RunRecord))
    if len(items) != len(items_read.items) or len(runs) != len(runs_read.items):
        # A shape we did not expect is a READ failure, not an empty page.
        return ReconcileReport(
            verdict="ALARM",
            failures=(
                SourceFailed(
                    name="join",
                    reason=(
                        "source payload contained rows of an unexpected type "
                        f"(items {len(items)}/{len(items_read.items)}, "
                        f"runs {len(runs)}/{len(runs_read.items)})"
                    ),
                ),
            ),
            repos_scanned=tuple(repos_scanned),
            generated_at=generated_at,
            meta=dict(meta or {}),
        )

    run_index = index_runs(runs)
    items_by_key = {i.key: i for i in items}

    disagreements: list[Disagreement] = []
    for item in sorted(items, key=lambda i: (i.repo, i.number)):
        verdict = classify_item(item, run_index.get(item.key, []))
        if verdict is not None:
            disagreements.append(verdict)

    unmatched: list[UnmatchedRun] = []
    for run in runs:
        key = run.key
        if key is None:
            if run.is_live or run.is_suspended:
                unmatched.append(
                    UnmatchedRun(
                        run_id=run.run_id,
                        status=run.status,
                        repo=run.repo,
                        issue_number=run.issue_number,
                        reason="run.input carries no usable (repo, issue_number) — join key unreadable",
                    )
                )
            continue
        if run.is_live and key not in items_by_key:
            unmatched.append(
                UnmatchedRun(
                    run_id=run.run_id,
                    status=run.status,
                    repo=run.repo,
                    issue_number=run.issue_number,
                    reason="live run against an item that is not open (closed, or outside the scanned repos)",
                )
            )

    truncated = tuple(s.name for s in (items_read, runs_read) if not s.exhaustive)

    order = {c: n for n, c in enumerate(CLASSES)}
    disagreements.sort(key=lambda d: (order[d.classification], d.repo, d.number))

    return ReconcileReport(
        verdict="OK",
        disagreements=tuple(disagreements),
        unmatched_runs=tuple(unmatched),
        failures=(),
        truncated_sources=truncated,
        items_scanned=len(items),
        runs_scanned=len(runs),
        repos_scanned=tuple(repos_scanned),
        generated_at=generated_at,
        meta=dict(meta or {}),
    )


# ─── rendering ───────────────────────────────────────────────────────────────


def render_markdown(report: ReconcileReport, *, limit_per_class: int = 50) -> str:
    """Human-readable summary. An ALARM renders as an ALARM — never as a clean bill.

    Note the ordering rule this encodes: the verdict line comes FIRST and the counts
    are only printed when there are counts. A reader skimming the top of the report
    cannot mistake a broken read for a quiet week.
    """
    lines: list[str] = []
    if report.alarmed:
        lines.append("## 🚨 WORK-STATE RECONCILER: ALARM — THE READ FAILED")
        lines.append("")
        lines.append(
            "**This is NOT a report of zero disagreements.** One or more sources could not be "
            "read, so nothing was classified and no count exists. Treat the pipeline state as "
            "UNKNOWN until this is fixed."
        )
        lines.append("")
        for failure in report.failures:
            lines.append(f"- **{failure.name}** — {failure.reason}")
        lines.append("")
        lines.append(f"_generated {report.generated_at or 'n/a'}_")
        return "\n".join(lines)

    counts = report.counts
    assert counts is not None
    total = sum(counts.values())
    at_least = "" if report.exhaustive else "at least "
    lines.append("## Work-state reconciler (phase 1, observe-only)")
    lines.append("")
    lines.append(
        f"Read OK. Scanned {at_least}{report.items_scanned} open item(s) across "
        f"{len(report.repos_scanned)} repo(s) against {at_least}{report.runs_scanned} run(s)."
    )
    if not report.exhaustive:
        lines.append("")
        lines.append(
            "> ⚠️ **Counts are FLOORS, not totals** — pagination stopped at a safety cap for: "
            + ", ".join(report.truncated_sources)
            + ". Every number below is 'at least N'."
        )
    lines.append("")
    lines.append("| class | count | meaning |")
    lines.append("|---|---:|---|")
    meaning = {
        "ZOMBIE": "`dispatched`, no run ever — nothing picked it up",
        "DEAD": "`dispatched`, every run terminal",
        "MISLABELLED": "`dispatched`, run suspended at a gate (parked ≠ running)",
        "LAGGING": "live run, no `dispatched` label — the projection lags",
    }
    for c in CLASSES:
        lines.append(f"| **{c}** | {at_least}{counts[c]} | {meaning[c]} |")
    lines.append(f"| _total_ | {at_least}{total} | |")
    lines.append("")
    lines.append(f"`disagreement_hash`: `{report.disagreement_hash()[:16]}`")
    lines.append("")

    for c in CLASSES:
        rows = report.by_class(c)
        if not rows:
            continue
        lines.append(f"### {c} ({at_least}{len(rows)})")
        lines.append("")
        for d in rows[:limit_per_class]:
            caveat = f" ⚠️ _{', '.join(d.caveats)}_" if d.caveats else ""
            lines.append(f"- [{d.repo}#{d.number}]({d.html_url}) — {d.title}{caveat}")
            lines.append(f"  - {d.detail}")
            if d.run_ids:
                lines.append(f"  - runs: {', '.join(d.run_ids)} ({', '.join(d.run_statuses)})")
        if len(rows) > limit_per_class:
            lines.append(f"- …and {len(rows) - limit_per_class} more")
        lines.append("")

    if report.unmatched_runs:
        lines.append(f"### Unmatched live runs ({len(report.unmatched_runs)})")
        lines.append("")
        lines.append(
            "_Not a disagreement class — live runs whose join key matched no open item. "
            "Reported so a broken join surfaces as data instead of a shrinking count._"
        )
        lines.append("")
        for u in report.unmatched_runs[:limit_per_class]:
            where = f"{u.repo}#{u.issue_number}" if u.repo else "(unparseable input)"
            lines.append(f"- `{u.run_id}` [{u.status}] → {where} — {u.reason}")
        lines.append("")

    lines.append(
        "_Phase 1 is OBSERVE-ONLY: this reconciler performed **zero** writes against any "
        "listed item — no label edits, no comments, no re-dispatch, no closes._"
    )
    lines.append("")
    lines.append(f"_generated {report.generated_at or 'n/a'}_")
    return "\n".join(lines)


def render_json(report: ReconcileReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=False)
