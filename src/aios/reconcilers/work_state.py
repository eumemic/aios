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

#: The label the FIRST cut of this reconciler checked — kept as a named constant
#: because it is still the most common assertion, NOT because it is the only one.
DISPATCHED_LABEL = "dispatched"

#: **Every label that asserts "work is in flight on this item".**
#:
#: This is the C1 fix and it is the whole point of the tool. The first cut tested
#: membership of exactly ONE literal (``dispatched``) — so when the seat STRIPPED
#: that label from all nine confirmed zombies at 2026-07-26T03:25 as remediation,
#: the reconciler would have classified all nine as AGREEING and reported
#: "Read OK. 0 ZOMBIE" while nine issues sat dead. They still carry
#: ``autodev:in-progress`` + ``needs:human/build``, which assert exactly the same
#: thing: *a machine has this*.
#:
#: A tool built BECAUSE a label is an assertion nothing re-checks must not itself
#: trust a single label. So the claim "in flight" is the UNION below, and every
#: finding reports WHICH assertion triggered it (:attr:`Disagreement.trigger_labels`).
IN_FLIGHT_LABELS: frozenset[str] = frozenset(
    {
        DISPATCHED_LABEL,
        "autodev:in-progress",
        "design:in-progress",
    }
)

#: ``needs:human/<gate>`` asserts "the pipeline reached a gate and is parked there",
#: which is a claim that a run EXISTS. With no run at all it is exactly as false as
#: a stuck ``dispatched``, so the whole family counts as an in-flight assertion.
IN_FLIGHT_LABEL_PREFIXES: tuple[str, ...] = ("needs:human/",)

#: Labels that assert the work is DONE / handed off rather than in flight. Listed
#: so the union above can never quietly swallow them (``autodev:built`` means a PR
#: exists — that is not a claim that a machine is on it right now).
NOT_IN_FLIGHT_LABELS: frozenset[str] = frozenset(
    {"autodev:built", "autodev:failed", "hold", "paused"}
)


def is_in_flight_label(label: str) -> bool:
    """True iff ``label`` asserts that work is in flight (a run should exist)."""
    if label in NOT_IN_FLIGHT_LABELS:
        return False
    return label in IN_FLIGHT_LABELS or label.startswith(IN_FLIGHT_LABEL_PREFIXES)


def in_flight_assertions(labels: Iterable[str]) -> tuple[str, ...]:
    """Every in-flight assertion carried by ``labels``, sorted (the trigger set)."""
    return tuple(sorted({x for x in labels if is_in_flight_label(x)}))


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

Classification = Literal["ZOMBIE", "DEAD", "MISLABELLED", "LAGGING", "AMBIGUOUS"]

#: Stable ordering for rendering + the change-detection hash.
#:
#: ``AMBIGUOUS`` (C2) is a FIRST-CLASS class, not a footnote: an item asserting
#: in-flight with no run BUT with linked PRs cannot be a zombie and must not be
#: counted as one. A caveat does not undo a count — phase 2 will act on the class,
#: so the class has to be right. ``counts["ZOMBIE"]`` therefore excludes them.
CLASSES: tuple[Classification, ...] = ("ZOMBIE", "AMBIGUOUS", "DEAD", "MISLABELLED", "LAGGING")

#: What the ZOMBIE class can and cannot prove (B3). ``list_runs`` filters
#: ``archived_at IS NULL`` (``db/queries/__init__.py``) and archiving requires a
#: TERMINAL run (``services/workflows.py``), so a run that completed and was
#: archived reads as "no run, ever". ZOMBIE therefore means **no UNARCHIVED run**,
#: and every report says so out loud rather than letting a reader infer "nothing
#: ever picked it up".
ZOMBIE_MEANS = (
    "no UNARCHIVED run exists for this item. `list_runs` cannot see archived runs "
    "(archived_at IS NULL) and archiving requires a TERMINAL run, so ZOMBIE cannot "
    "distinguish 'never picked up' from 'ran and was tidied away'."
)


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
        """Legacy single-label test. **Not** what the classifier keys on — see
        :attr:`claims_in_flight`. Retained only because the phrase "carries
        `dispatched`" still appears in triage conversation."""
        return DISPATCHED_LABEL in self.labels

    @property
    def in_flight_labels(self) -> tuple[str, ...]:
        """The in-flight assertions this item actually carries (C1).

        This is the trigger set: the reconciler reports WHICH label made the claim,
        so a reader can see that ``autodev:in-progress`` + ``needs:human/build``
        asserted "a machine has this" just as loudly as ``dispatched`` did.
        """
        return in_flight_assertions(self.labels)

    @property
    def claims_in_flight(self) -> bool:
        """True iff ANY label asserts work is in flight. The classifier's predicate."""
        return bool(self.in_flight_labels)

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
    #: Non-fatal observations from the read itself — e.g. the C4 transfer signal
    #: (a GitHub 301 on an issue read). Surfaced in the report so a stale join key
    #: is visible as DATA rather than silently becoming a false ZOMBIE.
    notes: tuple[str, ...] = ()

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
    #: WHICH in-flight assertion(s) triggered this finding (C1). Empty for LAGGING,
    #: whose trigger is the ABSENCE of any such assertion.
    trigger_labels: tuple[str, ...] = ()

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
            f"{self.classification}:{self.repo}#{self.number}"
            f":{','.join(sorted(self.run_statuses))}"
            f":{','.join(sorted(self.trigger_labels))}"
        )


@dataclass(frozen=True)
class UnmatchedRun:
    """A run that joined to no OPEN item we enumerated.

    Not one of the classes — it is a run against a closed issue, or a join-key
    mismatch. Reported (never swallowed) so a broken join surfaces as data instead
    of as a quietly-shrinking LAGGING count.

    B4: this now includes **terminal** runs whose join key is unreadable. The
    previous cut reported only live/suspended ones, so an ``errored`` run with an
    unparseable ``run.input`` vanished — and its issue was then classified ZOMBIE
    with no trace of the run that actually existed. That is a manufactured false
    ZOMBIE, which is the exact failure the PR body claimed to prevent
    ("unkeyable runs are never skipped"). The claim is now true.
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
    #: Non-fatal read observations (C4 transfers, etc.). Never suppressed.
    notes: tuple[str, ...] = ()
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
            # B3: stated in EVERY machine-readable report, not just the markdown, so a
            # phase-2 consumer cannot read ZOMBIE as "nothing ever picked it up".
            "zombie_means": ZOMBIE_MEANS,
            "in_flight_labels_checked": sorted(IN_FLIGHT_LABELS)
            + [p + "*" for p in IN_FLIGHT_LABEL_PREFIXES],
            "total_disagreements": None if self.alarmed else len(self.disagreements),
            "exhaustive": self.exhaustive,
            "truncated_sources": list(self.truncated_sources),
            "notes": list(self.notes),
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
                    "trigger_labels": list(d.trigger_labels),
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
    """Classify ONE item against every run keyed to it. ``None`` == the labels agree.

    **What "the label claims work is in flight" means (C1).** Not ``dispatched``.
    The UNION in :data:`IN_FLIGHT_LABELS` / :data:`IN_FLIGHT_LABEL_PREFIXES`:
    ``dispatched``, ``autodev:in-progress``, ``design:in-progress``, and every
    ``needs:human/<gate>``. Keying on one literal string is how the first cut of
    this module would have reported "0 ZOMBIE" the morning after the seat stripped
    ``dispatched`` from nine dead issues that still carried
    ``autodev:in-progress`` + ``needs:human/build``. The trigger is recorded on the
    finding so a reader sees which assertion was doing the lying.

    Precedence is deliberate and is the heart of the design: **live > suspended >
    unknown > terminal**. One live run means work really is happening, whatever the
    other runs say. With no live run, a *suspended* run means parked-at-a-gate — a
    different condition from running, hence MISLABELLED rather than "agree". With
    neither, an UNRECOGNISED status outranks a terminal one (C3): an unknown status
    may well BE live under a new name, so it can never be evidence of death — not
    even alongside a corpse. Only with no live, no suspended and no unknown run is
    the item DEAD/ZOMBIE/AMBIGUOUS.
    """
    live = [r for r in runs if r.is_live]
    suspended = [r for r in runs if r.is_suspended]
    terminal = [r for r in runs if r.is_terminal]
    unknown = [r for r in runs if not (r.is_live or r.is_suspended or r.is_terminal)]
    triggers = item.in_flight_labels

    def _mk(
        classification: Classification,
        detail: str,
        subject: Sequence[RunRecord],
        caveats: tuple[str, ...] = (),
        trigger_labels: tuple[str, ...] = triggers,
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
            trigger_labels=trigger_labels,
        )

    if not item.claims_in_flight:
        # No label asserts work is in flight; the runs say otherwise. The projection
        # lags reality.
        if live:
            return _mk(
                "LAGGING",
                f"{len(live)} live run(s) ({', '.join(_statuses(live))}) but NO in-flight "
                "label (no `dispatched`, no `autodev:in-progress`, no `needs:human/*`)",
                live,
                trigger_labels=(),
            )
        return None

    # From here: the item ASSERTS work is in flight (via `triggers`).
    claim = ", ".join(f"`{x}`" for x in triggers)

    if live:
        return None  # agree — a machine really is on it

    if unknown:
        # C3: an unrecognised status is NEVER evidence of death — not even when a
        # terminal sibling exists, because the unknown run may be live under a
        # status name this vocabulary has not learned yet. Handled BEFORE terminal
        # for exactly that reason. A false DEAD is a false alarm, and false alarms
        # train people to ignore the report.
        others = [r for r in (suspended + terminal) if r]
        extra = (
            f" (alongside {len(others)} run(s) in {', '.join(_statuses(others))} — a terminal"
            " sibling does NOT make an unknown status dead)"
            if others
            else ""
        )
        return _mk(
            "MISLABELLED",
            f"{claim} and run(s) in unrecognised status ({', '.join(_statuses(unknown))}) — "
            f"cannot prove live; treat as parked pending triage{extra}",
            unknown + others,
            caveats=("unrecognised-run-status",),
        )

    if suspended:
        return _mk(
            "MISLABELLED",
            f"{claim} but {len(suspended)} run(s) suspended at a gate — parked is NOT running",
            suspended,
        )

    if terminal:
        by_status = ", ".join(_statuses(terminal))
        return _mk(
            "DEAD",
            f"{claim} but every run is terminal ({by_status})",
            terminal,
        )

    # No run at all — subject to the B3 caveat that we can only see UNARCHIVED runs.
    if item.linked_pr_numbers:
        # C2: "no run ever" AND "a PR exists" cannot both be true of healthy work, so
        # this is NOT a zombie — it is a join-key question. It gets its OWN class,
        # excluded from counts["ZOMBIE"], because a footnote does not undo a count and
        # phase 2 will act on the class, not the caveat.
        prs = ", ".join("#" + str(n) for n in item.linked_pr_numbers)
        return _mk(
            "AMBIGUOUS",
            f"{claim} and no unarchived run — BUT PR(s) {prs} reference this issue. "
            "Work demonstrably happened, so the join key (or the run's archival) is the "
            "suspect, not the item. NEEDS TRIAGE — deliberately NOT counted as a ZOMBIE.",
            (),
            caveats=("has-linked-prs", "excluded-from-zombie-count"),
        )

    return _mk(
        "ZOMBIE",
        f"{claim} but NO unarchived run exists for this issue — nothing (visible) ever "
        "picked it up",
        (),
        caveats=("zombie-means-no-unarchived-run",),
    )


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
            # B4: EVERY unkeyable run is reported, terminal ones included. The old
            # live/suspended filter silently dropped an `errored` run with an
            # unparseable input — and its issue was then classified ZOMBIE with no
            # trace of the run that did exist. A manufactured false ZOMBIE with no
            # trace in the output is the exact failure this section claims to prevent.
            unmatched.append(
                UnmatchedRun(
                    run_id=run.run_id,
                    status=run.status,
                    repo=run.repo,
                    issue_number=run.issue_number,
                    reason=(
                        f"run.input carries no usable (repo, issue_number) — join key "
                        f"unreadable (status {run.status or 'unknown'}); any item this run "
                        "belonged to may therefore read as a FALSE ZOMBIE"
                    ),
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
    notes = tuple(n for src in (items_read, runs_read) for n in src.notes)

    order = {c: n for n, c in enumerate(CLASSES)}
    disagreements.sort(key=lambda d: (order[d.classification], d.repo, d.number))

    return ReconcileReport(
        verdict="OK",
        disagreements=tuple(disagreements),
        unmatched_runs=tuple(unmatched),
        failures=(),
        truncated_sources=truncated,
        notes=notes,
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
    if report.notes:
        for note in report.notes:
            lines.append(f"> ⚠️ {note}")
        lines.append("")
    lines.append("| class | count | meaning |")
    lines.append("|---|---:|---|")
    meaning = {
        "ZOMBIE": "claims in-flight, **no UNARCHIVED run** — nothing visible picked it up",
        "AMBIGUOUS": "claims in-flight, no run, **but linked PRs exist** — NOT a zombie; triage",
        "DEAD": "claims in-flight, every run terminal",
        "MISLABELLED": "claims in-flight, run suspended at a gate (parked ≠ running)",
        "LAGGING": "live run, but NO in-flight label — the projection lags",
    }
    for c in CLASSES:
        lines.append(f"| **{c}** | {at_least}{counts[c]} | {meaning[c]} |")
    lines.append(f"| _total_ | {at_least}{total} | |")
    lines.append("")
    # B3, stated LOUDLY and unconditionally — not as a footnote a triager may miss.
    lines.append(f"> ⚠️ **What ZOMBIE can prove:** {ZOMBIE_MEANS}")
    lines.append("")
    lines.append(
        "> **In-flight assertions checked** (a finding names which one triggered it): "
        + ", ".join(f"`{x}`" for x in sorted(IN_FLIGHT_LABELS))
        + ", "
        + ", ".join(f"`{x}*`" for x in IN_FLIGHT_LABEL_PREFIXES)
        + ". Keying on `dispatched` alone is how nine dead issues read as healthy."
    )
    lines.append("")
    lines.append(f"`disagreement_hash`: `{report.disagreement_hash()[:16]}`")
    lines.append("")

    for c in CLASSES:
        rows = report.by_class(c)
        if not rows:
            continue
        lines.append(f"### {c} ({at_least}{len(rows)})")
        lines.append("")
        if c == "AMBIGUOUS":
            lines.append(
                "_Deliberately NOT counted as ZOMBIE: each of these has linked PRs, so work "
                "demonstrably happened and the join key (or run archival) is the suspect._"
            )
            lines.append("")
        for d in rows[:limit_per_class]:
            caveat = f" ⚠️ _{', '.join(d.caveats)}_" if d.caveats else ""
            lines.append(f"- [{d.repo}#{d.number}]({d.html_url}) — {d.title}{caveat}")
            if d.trigger_labels:
                lines.append(
                    f"  - triggered by: {', '.join('`' + x + '`' for x in d.trigger_labels)}"
                )
            lines.append(f"  - {d.detail}")
            if d.run_ids:
                lines.append(f"  - runs: {', '.join(d.run_ids)} ({', '.join(d.run_statuses)})")
        if len(rows) > limit_per_class:
            lines.append(f"- …and {len(rows) - limit_per_class} more")
        lines.append("")

    if report.unmatched_runs:
        lines.append(f"### Unmatched runs ({len(report.unmatched_runs)})")
        lines.append("")
        lines.append(
            "_Not a disagreement class — runs whose join key matched no open item (terminal "
            "ones included, since a dropped run manufactures a false ZOMBIE). "
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
