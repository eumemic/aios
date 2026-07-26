"""Unit tests for the phase-1 work-state reconciler (#2043).

The reconciler exists because **a GitHub label is an assertion nothing re-checks**.
These tests protect the two properties that make it worth having:

1. **The classifier is right** — the four classes (ZOMBIE / DEAD / MISLABELLED /
   LAGGING), including the precedence rules (live > suspended > terminal) and the
   "parked is NOT running" rule.
2. **It fails LOUD** — a failed read can never render as health. This is the
   acceptance criterion that outranks the feature, so it gets the most tests:
   every source-failure path is asserted to produce ``verdict == "ALARM"`` **and**
   ``counts is None``, and the "0 disagreements" rendering is asserted to be
   reachable ONLY from a successful read.

The workflow-script mirror in ``infra/workflows/work-state-reconciler.wf.py`` is
tested against the SAME fixtures as the library core, so the two cannot drift.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from aios.models.workflows import TERMINAL_RUN_STATUSES, WfRunStatus
from aios.reconcilers.work_state import (
    CLASSES,
    LIVE_RUN_STATUSES,
    SUSPENDED_RUN_STATUSES,
    Disagreement,
    ReconcileReport,
    RunRecord,
    SourceFailed,
    SourceOk,
    TransferEvidence,
    WorkItem,
    build_report,
    classify_item,
    index_runs,
    is_pipeline_state_label,
    normalise_repo,
    parse_issue_ref,
    render_json,
    render_markdown,
    run_record_from_payload,
    stale_run_keys,
)
from aios.reconcilers.work_state import (
    TERMINAL_RUN_STATUSES as RECON_TERMINAL,
)
from aios.reconcilers.work_state_cli import (
    DEFAULT_REPOS,
    ObserveOnlyViolation,
    ReadFailure,
    _get,
    _next_link,
    fetch_repo_items,
    probe_stale_keys,
    read_aios_runs,
    read_github_items,
    reconcile,
)

# ─── fixtures ────────────────────────────────────────────────────────────────


def item(
    number: int,
    *,
    repo: str = "eumemic/aios",
    labels: tuple[str, ...] = ("dispatched",),
    kind: str = "issue",
    linked: tuple[int, ...] = (),
) -> WorkItem:
    return WorkItem(
        repo=repo,
        number=number,
        kind=kind,  # type: ignore[arg-type]
        title=f"item {number}",
        labels=labels,
        html_url=f"https://github.com/{repo}/issues/{number}",
        updated_at="2026-07-20T00:00:00Z",
        linked_pr_numbers=linked,
    )


def run(
    status: str, *, repo: str | None = "eumemic/aios", number: int | None = 1, run_id: str = "run_1"
) -> RunRecord:
    return RunRecord(run_id=run_id, status=status, repo=repo, issue_number=number)


def ok_report(items: list[WorkItem], runs: list[RunRecord], **kw: Any) -> ReconcileReport:
    return build_report(
        items_read=SourceOk(name="github", items=tuple(items)),
        runs_read=SourceOk(name="aios-runs", items=tuple(runs)),
        **kw,
    )


# ─── (1) the four classes ────────────────────────────────────────────────────


def test_zombie_dispatched_with_no_run_ever() -> None:
    """The headline defect: `dispatched` for days, nothing ever picked it up."""
    report = ok_report([item(2000)], [])
    assert report.verdict == "OK"
    (d,) = report.disagreements
    assert d.classification == "ZOMBIE"
    assert d.run_ids == ()
    # B3: the wording must say UNARCHIVED — `list_runs` cannot see archived runs, so
    # ZOMBIE cannot distinguish "never picked up" from "ran and was tidied away".
    assert "NO unarchived run exists" in d.detail
    assert "zombie-means-no-unarchived-run" in d.caveats
    assert d.trigger_labels == ("dispatched",)


def test_dead_dispatched_but_run_terminal() -> None:
    for status in sorted(RECON_TERMINAL):
        report = ok_report([item(1)], [run(status)])
        (d,) = report.disagreements
        assert d.classification == "DEAD", status
        assert d.run_statuses == (status,)


def test_mislabelled_dispatched_but_run_suspended_at_a_gate() -> None:
    """Parked is NOT running — a suspended run is its own condition, never 'agree'."""
    report = ok_report([item(1)], [run("suspended")])
    (d,) = report.disagreements
    assert d.classification == "MISLABELLED"
    assert "parked is NOT running" in d.detail


def test_lagging_live_run_but_no_dispatched_label() -> None:
    report = ok_report([item(1, labels=("approved",))], [run("running")])
    (d,) = report.disagreements
    assert d.classification == "LAGGING"


@pytest.mark.parametrize("status", sorted(LIVE_RUN_STATUSES))
def test_dispatched_with_a_live_run_agrees(status: str) -> None:
    """The healthy case must produce NO disagreement — otherwise the report is noise."""
    assert ok_report([item(1)], [run(status)]).disagreements == ()


def test_no_label_and_no_live_run_is_not_a_disagreement() -> None:
    assert ok_report([item(1, labels=("hold",))], [run("completed")]).disagreements == ()


# ─── (2) precedence: live > suspended > terminal ─────────────────────────────


def test_one_live_run_outranks_terminal_and_suspended_siblings() -> None:
    """A re-dispatch leaves dead runs behind; one live run still means work IS happening."""
    runs = [
        run("errored", run_id="r1"),
        run("suspended", run_id="r2"),
        run("running", run_id="r3"),
    ]
    assert classify_item(item(1), runs) is None


def test_suspended_outranks_terminal_when_nothing_is_live() -> None:
    verdict = classify_item(item(1), [run("completed", run_id="r1"), run("suspended", run_id="r2")])
    assert verdict is not None and verdict.classification == "MISLABELLED"
    assert verdict.run_ids == ("r2",)  # the report points at the SUSPENDED run, not the corpse


def test_unrecognised_status_is_never_asserted_dead() -> None:
    """An unknown status is not evidence of death; a false DEAD trains people to ignore us."""
    verdict = classify_item(item(1), [run("quiescing")])
    assert verdict is not None
    assert verdict.classification == "MISLABELLED"
    assert "unrecognised-run-status" in verdict.caveats


# ─── (3) the join key ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eumemic/aios", "eumemic/aios"),
        ("aios", "eumemic/aios"),
        ("https://github.com/eumemic/eumemic-ops", "eumemic/eumemic-ops"),
        ("git@github.com:eumemic/aios.git", "eumemic/aios"),
        ({"owner": "eumemic", "name": "autodev"}, "eumemic/autodev"),
        ("", None),
        (None, None),
        (17, None),
        ("a/b/c", None),
    ],
)
def test_normalise_repo(raw: Any, expected: str | None) -> None:
    assert normalise_repo(raw) == expected


@pytest.mark.parametrize(
    ("payload", "key"),
    [
        ({"repo": "eumemic/aios", "issue_number": 2000}, ("eumemic/aios", 2000)),
        ({"repo": "aios", "issue": "#2000"}, ("eumemic/aios", 2000)),
        ({"repository": "eumemic/aios", "number": "2000"}, ("eumemic/aios", 2000)),
        ({"repo": "eumemic/aios", "issue_number": True}, None),  # bool is not a number
        ({"repo": "eumemic/aios"}, None),
        ({}, None),
    ],
)
def test_run_record_key_extraction(payload: dict[str, Any], key: tuple[str, int] | None) -> None:
    record = run_record_from_payload({"id": "r", "status": "running", "input": payload})
    assert record.key == key


def test_run_with_unreadable_input_is_reported_never_dropped() -> None:
    """A silently-dropped run is a missing 'agree', which manufactures a false ZOMBIE."""
    orphan = RunRecord(run_id="r9", status="running", repo=None, issue_number=None)
    report = ok_report([item(1)], [run("running"), orphan])
    assert report.disagreements == ()
    assert [u.run_id for u in report.unmatched_runs] == ["r9"]
    assert index_runs([orphan]) == {}


def test_live_run_against_a_non_open_item_is_surfaced_not_swallowed() -> None:
    report = ok_report([], [run("running", number=4242)])
    assert report.verdict == "OK"
    (unmatched,) = report.unmatched_runs
    assert unmatched.issue_number == 4242
    assert "not open" in unmatched.reason


# ─── (4) FAIL LOUD — the criterion that outranks the feature ─────────────────


def test_failed_github_read_alarms_and_has_NO_counts() -> None:
    report = build_report(
        items_read=SourceFailed(name="github", reason="HTTP 401"),
        runs_read=SourceOk(name="aios-runs", items=()),
    )
    assert report.verdict == "ALARM"
    assert report.counts is None  # NOT {}, NOT zeros — there is nothing to render
    assert report.to_dict()["counts"] is None
    assert report.to_dict()["total_disagreements"] is None


def test_failed_runs_read_alarms() -> None:
    report = build_report(
        items_read=SourceOk(name="github", items=(item(1),)),
        runs_read=SourceFailed(name="aios-runs", reason="connection refused"),
    )
    assert report.verdict == "ALARM"
    assert report.counts is None
    assert report.disagreements == ()  # no classification is even attempted


def test_successful_empty_read_is_health_and_is_DISTINGUISHABLE_from_failure() -> None:
    """'Read failed' vs 'read succeeded and found nothing' must never collapse."""
    healthy = ok_report([], [])
    assert healthy.verdict == "OK"
    assert healthy.counts == dict.fromkeys(CLASSES, 0)
    broken = build_report(
        items_read=SourceFailed(name="github", reason="boom"),
        runs_read=SourceOk(name="aios-runs", items=()),
    )
    assert broken.counts is None
    assert healthy.disagreement_hash() != broken.disagreement_hash()


def test_markdown_of_an_alarm_never_reads_as_a_clean_bill() -> None:
    md = render_markdown(
        build_report(
            items_read=SourceFailed(name="github", reason="HTTP 403 rate limited"),
            runs_read=SourceOk(name="aios-runs", items=()),
        )
    )
    assert "ALARM" in md
    assert "NOT a report of zero disagreements" in md
    assert "HTTP 403 rate limited" in md
    # The healthy vocabulary must be ABSENT — no table, no per-class zeros.
    assert "| **ZOMBIE** |" not in md
    assert "Read OK" not in md


def test_missing_credentials_alarm_rather_than_skip_a_source() -> None:
    """The 2026-07-25 failure exactly: a rejected credential rendered as 'nothing held'."""
    report = reconcile(repos=["eumemic/aios"], github_token=None, aios_url=None, aios_api_key=None)
    assert report.verdict == "ALARM"
    assert report.counts is None
    assert {f.name for f in report.failures} == {"github", "aios-runs"}


def test_http_error_becomes_a_source_failure_not_an_empty_page() -> None:
    def boom(url: str, headers: Any) -> Any:
        raise ReadFailure(f"GET {url} → HTTP 500")

    assert isinstance(read_github_items(["eumemic/aios"], token="t", getter=boom), SourceFailed)
    assert isinstance(read_aios_runs(base_url="http://x", api_key="k", getter=boom), SourceFailed)


def test_one_repo_failing_fails_the_WHOLE_github_source() -> None:
    """A partial read is a lie shaped like health: the unread repo is where the zombies are."""
    calls: list[str] = []

    def getter(url: str, headers: Any) -> Any:
        calls.append(url)
        if "eumemic-ops" in url:
            raise ReadFailure("HTTP 404")
        return [], {}

    result = read_github_items(["eumemic/aios", "eumemic/eumemic-ops"], token="t", getter=getter)
    assert isinstance(result, SourceFailed)
    assert "404" in result.reason


def test_unparseable_body_is_a_read_failure_not_an_empty_result() -> None:
    def getter(url: str, headers: Any) -> Any:
        return {"not": "a list"}, {}

    assert isinstance(read_github_items(["eumemic/aios"], token="t", getter=getter), SourceFailed)


def test_runs_envelope_without_data_key_is_a_contract_failure() -> None:
    def getter(url: str, headers: Any) -> Any:
        return {"unexpected": []}, {}

    result = read_aios_runs(base_url="http://x", api_key="k", getter=getter)
    assert isinstance(result, SourceFailed)
    assert "data" in result.reason


def test_unexpected_row_shape_alarms_rather_than_silently_dropping_rows() -> None:
    report = build_report(
        items_read=SourceOk(name="github", items=(item(1), "not an item")),
        runs_read=SourceOk(name="aios-runs", items=()),
    )
    assert report.verdict == "ALARM"
    assert report.counts is None


def test_cli_exit_code_is_2_on_alarm_and_never_0() -> None:
    from aios.reconcilers import work_state_cli

    code = work_state_cli.main(["--repo", "eumemic/aios", "--format", "json"])
    assert code == 2  # no credentials in the test env ⇒ ALARM ⇒ non-zero


# ─── (5) pagination: exhaustion, or an explicit floor ────────────────────────


def test_page_cap_marks_the_source_non_exhaustive_and_counts_render_as_floors() -> None:
    def getter(url: str, headers: Any) -> Any:
        return [
            {
                "number": 1,
                "title": "t",
                "labels": [{"name": "dispatched"}],
                "html_url": "u",
                "updated_at": "",
            }
        ], {"Link": '<https://api.github.com/next>; rel="next"'}

    items, exhaustive = fetch_repo_items("eumemic/aios", token="t", getter=getter, max_pages=3)
    assert exhaustive is False
    assert len(items) == 3

    report = build_report(
        items_read=SourceOk(name="github", items=tuple(items), exhaustive=False),
        runs_read=SourceOk(name="aios-runs", items=()),
    )
    assert report.exhaustive is False
    assert report.truncated_sources == ("github",)
    md = render_markdown(report)
    assert "FLOORS, not totals" in md
    assert "at least" in md


def test_link_header_pagination_is_followed_to_exhaustion() -> None:
    pages = {
        "https://api.github.com/repos/eumemic/aios/issues?state=open&per_page=100&sort=created&direction=desc": (
            [
                {
                    "number": 1,
                    "title": "a",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ],
            {"Link": '<https://api.github.com/p2>; rel="next"'},
        ),
        "https://api.github.com/p2": (
            [
                {
                    "number": 2,
                    "title": "b",
                    "labels": [{"name": "approved"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ],
            {},
        ),
    }

    def getter(url: str, headers: Any) -> Any:
        return pages[url]

    items, exhaustive = fetch_repo_items("eumemic/aios", token="t", getter=getter)
    assert exhaustive is True
    assert [i.number for i in items] == [1, 2]


def test_next_link_parsing() -> None:
    header = '<https://api.github.com/a?page=2>; rel="next", <https://api.github.com/a?page=9>; rel="last"'
    assert _next_link({"Link": header}) == "https://api.github.com/a?page=2"
    assert _next_link({}) is None
    assert _next_link({"Link": '<https://x>; rel="last"'}) is None


def test_unlabelled_open_items_are_KEPT_so_lagging_is_detectable() -> None:
    """C5. LAGGING is defined by the ABSENCE of an in-flight assertion, so enumeration
    can neither query by label NOR drop items that carry none. Dropping them made a
    live run against an unlabelled issue structurally impossible to classify LAGGING —
    it was demoted to an ``unmatched_run`` stamped "not open", which is simply false."""
    seen: list[str] = []

    def getter(url: str, headers: Any) -> Any:
        seen.append(url)
        return [
            {
                "number": 1,
                "title": "x",
                "labels": [{"name": "bug"}],
                "html_url": "",
                "updated_at": "",
            },
            {
                "number": 2,
                "title": "y",
                "labels": [{"name": "dispatched"}],
                "html_url": "",
                "updated_at": "",
            },
        ], {}

    items, _ = fetch_repo_items("eumemic/aios", token="t", getter=getter)
    assert [i.number for i in items] == [1, 2]
    assert all("labels=" not in url for url in seen)
    # And the payoff: a live run against the UNLABELLED issue #1 is now LAGGING.
    report = ok_report(list(items), [run("running", number=1)])
    (lagging,) = report.by_class("LAGGING")
    assert lagging.number == 1
    assert lagging.trigger_labels == ()  # its trigger is the ABSENCE of an assertion

    # Opting out still works for callers that only want labelled items.
    only_labelled, _ = fetch_repo_items(
        "eumemic/aios", token="t", getter=getter, keep_unlabelled=False
    )
    assert [i.number for i in only_labelled] == [2]


# ─── (6) change detection ────────────────────────────────────────────────────


def test_hash_is_stable_under_ordering_and_run_id_churn() -> None:
    a = ok_report([item(1), item(2)], [run("errored", number=1, run_id="r1")])
    b = ok_report([item(2), item(1)], [run("errored", number=1, run_id="r99")])
    assert a.disagreement_hash() == b.disagreement_hash()


def test_hash_changes_when_a_run_transitions() -> None:
    parked = ok_report([item(1)], [run("suspended")])
    dead = ok_report([item(1)], [run("errored")])
    assert parked.disagreement_hash() != dead.disagreement_hash()


def test_hash_changes_when_an_item_joins_or_leaves_the_set() -> None:
    one = ok_report([item(1)], [])
    two = ok_report([item(1), item(2)], [])
    assert one.disagreement_hash() != two.disagreement_hash()


def test_alarm_hash_is_driven_by_failure_reasons_so_a_new_outage_wakes_the_seat() -> None:
    def alarm(reason: str) -> ReconcileReport:
        return build_report(
            items_read=SourceFailed(name="github", reason=reason),
            runs_read=SourceOk(name="aios-runs", items=()),
        )

    assert alarm("HTTP 500").disagreement_hash() == alarm("HTTP 500").disagreement_hash()
    assert alarm("HTTP 500").disagreement_hash() != alarm("HTTP 403").disagreement_hash()


# ─── (7) the eumemic-company#71 shape: a ZOMBIE with linked PRs ──────────────


def test_linked_prs_make_it_AMBIGUOUS_and_EXCLUDED_from_the_zombie_count() -> None:
    """C2. 'No run ever' + 'a PR exists' cannot both be true of healthy work, so these
    are NOT zombies. The first cut attached a caveat and returned ZOMBIE anyway, which
    inflated the headline ~36% (5 of 14) and — the part that matters — left phase 2
    acting on the CLASS while the disclaimer sat in a footnote. A caveat does not undo
    a count."""
    report = ok_report([item(71, repo="eumemic/eumemic-company", linked=(72, 73))], [])
    (d,) = report.disagreements
    assert d.classification == "AMBIGUOUS"
    assert "has-linked-prs" in d.caveats
    assert "excluded-from-zombie-count" in d.caveats
    assert "#72" in d.detail and "NOT counted as a ZOMBIE" in d.detail
    counts = report.counts
    assert counts is not None
    assert counts["ZOMBIE"] == 0
    assert counts["AMBIGUOUS"] == 1
    md = render_markdown(report)
    assert "⚠️" in md
    assert "AMBIGUOUS (1)" in md
    assert "ZOMBIE (" not in md  # empty classes are not rendered at all


def test_the_five_known_good_items_are_not_counted_as_zombies() -> None:
    """The exact five the reviewer named, with their real linked PRs."""
    known_good = {
        ("eumemic/eumemic-company", 71): (67, 68, 69, 100, 101),
        ("eumemic/eumemic-company", 50): (96, 204, 206),
        ("eumemic/eumemic-company", 135): (212,),
        ("eumemic/eumemic-ops", 337): (1977, 1979, 2016),
        ("eumemic/eumemic-ops", 331): (1995, 2041),
    }
    items = [item(n, repo=r, linked=prs) for (r, n), prs in known_good.items()]
    report = ok_report(items, [])
    counts = report.counts
    assert counts is not None
    assert counts["ZOMBIE"] == 0, "known-good items must never inflate the ZOMBIE count"
    assert counts["AMBIGUOUS"] == 5
    assert {(d.repo, d.number) for d in report.by_class("AMBIGUOUS")} == set(known_good)


# ─── (8) OBSERVE-ONLY: the phase-1 hard constraint ──────────────────────────


def test_transport_refuses_any_verb_other_than_GET() -> None:
    with pytest.raises(ObserveOnlyViolation):
        _get("https://api.github.com/repos/eumemic/aios/issues/1", {}, method="PATCH")
    with pytest.raises(ObserveOnlyViolation):
        _get("https://api.github.com/repos/eumemic/aios/issues/1", {}, method="POST")


def test_reconciler_source_contains_no_mutating_github_call() -> None:
    """A grep-level guard: the reconciler must never grow a write path."""
    for path in (
        Path("src/aios/reconcilers/work_state.py"),
        Path("src/aios/reconcilers/work_state_cli.py"),
        Path("infra/workflows/work-state-reconciler.wf.py"),
    ):
        source = path.read_text()
        for verb in ('"POST"', '"PATCH"', '"PUT"', '"DELETE"'):
            offenders = [
                line
                for line in source.splitlines()
                if verb in line and "refus" not in line.lower() and "method !=" not in line
            ]
            assert not offenders, f"{path} may mutate: {offenders}"


def test_report_records_zero_writes() -> None:
    report = reconcile(repos=[], github_token=None, aios_url=None, aios_api_key=None)
    assert report.meta["writes_performed"] == 0
    assert report.meta["phase"] == "1-observe-only"


# ─── (9) drift guards ────────────────────────────────────────────────────────


def test_status_vocabulary_matches_the_models_source_of_truth() -> None:
    """The reconciler restates WfRunStatus (the script host can't import aios.*);
    this test is what stops a new status from silently falling through."""
    from typing import get_args

    assert RECON_TERMINAL == TERMINAL_RUN_STATUSES
    assert set(get_args(WfRunStatus)) == LIVE_RUN_STATUSES | SUSPENDED_RUN_STATUSES | RECON_TERMINAL


def test_default_repos_cover_the_org_set_named_in_2043() -> None:
    assert set(DEFAULT_REPOS) >= {
        "eumemic/aios",
        "eumemic/eumemic-ops",
        "eumemic/eumemic-company",
        "eumemic/aios-console",
        "eumemic/autodev",
    }


def test_pipeline_label_predicate() -> None:
    assert is_pipeline_state_label("dispatched")
    assert is_pipeline_state_label("needs:human/review")
    assert not is_pipeline_state_label("bug")


# ─── (10) the workflow-script mirror agrees with the library core ───────────


def _load_wf_module() -> Any:
    path = Path("infra/workflows/work-state-reconciler.wf.py")
    spec = importlib.util.spec_from_file_location("wf_work_state", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENARIOS: list[tuple[str, tuple[str, ...], list[str]]] = [
    ("ZOMBIE", ("dispatched",), []),
    ("DEAD", ("dispatched",), ["completed"]),
    ("DEAD", ("dispatched",), ["errored", "cancelled"]),
    ("MISLABELLED", ("dispatched",), ["suspended"]),
    ("MISLABELLED", ("dispatched",), ["completed", "suspended"]),
    ("LAGGING", ("approved",), ["running"]),
    ("", ("dispatched",), ["running"]),
    ("", ("dispatched",), ["errored", "running"]),
    ("", ("approved",), ["completed"]),
    # FIX ROUND — the mirror must agree on every NEW rule too, or the two forms drift.
    ("ZOMBIE", ("autodev:in-progress",), []),  # C1
    ("ZOMBIE", ("needs:human/build",), []),  # C1
    ("ZOMBIE", ("approved", "autodev:in-progress", "needs:human/build"), []),  # the nine
    ("", ("autodev:built",), []),  # C1: done is not in-flight
    ("MISLABELLED", ("dispatched",), ["errored", "quiesced"]),  # C3
    ("MISLABELLED", ("dispatched",), ["suspended", "quiesced"]),  # C3
    ("", ("dispatched",), ["quiesced", "running"]),  # C3: live still wins
    ("LAGGING", (), ["running"]),  # C5: no labels at all
]


@pytest.mark.parametrize(("expected", "labels", "statuses"), _SCENARIOS)
def test_workflow_script_mirror_classifies_identically(
    expected: str, labels: tuple[str, ...], statuses: list[str]
) -> None:
    wf = _load_wf_module()
    lib_verdict = classify_item(
        item(1, labels=labels), [run(s, run_id=f"r{n}") for n, s in enumerate(statuses)]
    )
    wf_verdict = wf.classify(
        {
            "repo": "eumemic/aios",
            "number": 1,
            "kind": "issue",
            "title": "item 1",
            "labels": list(labels),
            "url": "",
            "updated_at": "",
        },
        [{"id": f"r{n}", "status": s} for n, s in enumerate(statuses)],
    )
    assert (lib_verdict.classification if lib_verdict else "") == expected
    assert (wf_verdict["classification"] if wf_verdict else "") == expected
    # The trigger set is part of the contract now, so it must not drift either.
    assert list(lib_verdict.trigger_labels if lib_verdict else []) == (
        wf_verdict["trigger_labels"] if wf_verdict else []
    )


def test_workflow_script_mirror_agrees_on_AMBIGUOUS() -> None:
    """C2 in both forms: linked PRs mean NOT a zombie, in the library and the mirror."""
    wf = _load_wf_module()
    lib = classify_item(item(71, repo="eumemic/eumemic-company", linked=(67, 68)), [])
    wf_verdict = wf.classify(
        {
            "repo": "eumemic/eumemic-company",
            "number": 71,
            "kind": "issue",
            "title": "item 71",
            "labels": ["dispatched"],
            "url": "",
            "updated_at": "",
            "linked_pr_numbers": [67, 68],
        },
        [],
    )
    assert lib is not None
    assert lib.classification == "AMBIGUOUS"
    assert wf_verdict["classification"] == "AMBIGUOUS"
    assert "excluded-from-zombie-count" in wf_verdict["caveats"]


def test_workflow_script_alarms_with_null_counts_on_failure() -> None:
    wf = _load_wf_module()
    report = wf.build([], True, [], True, [{"source": "github", "reason": "HTTP 500"}])
    assert report["verdict"] == "ALARM"
    assert report["counts"] is None
    assert report["total_disagreements"] is None


def test_workflow_script_and_library_agree_on_the_hash() -> None:
    """Both forms must produce the SAME change-detection hash for the same world."""
    wf = _load_wf_module()
    items = [item(2000), item(337, repo="eumemic/eumemic-ops")]
    runs = [run("suspended", number=2000, run_id="r1")]
    lib = ok_report(items, runs)
    wf_report = wf.build(
        [
            {
                "repo": i.repo,
                "number": i.number,
                "kind": i.kind,
                "title": i.title,
                "labels": list(i.labels),
                "url": i.html_url,
                "updated_at": i.updated_at,
            }
            for i in items
        ],
        True,
        [
            {
                "id": r.run_id,
                "status": r.status,
                "input": {"repo": r.repo, "issue_number": r.issue_number},
            }
            for r in runs
        ],
        True,
        [],
    )
    assert wf_report["counts"] == dict(lib.counts or {})
    assert wf.disagreement_hash(wf_report) == lib.disagreement_hash()


def test_workflow_script_validates_against_its_declared_surface() -> None:
    """The script must pass create-time validation with exactly the tools it declares."""
    from aios.models.agents import ToolSpec
    from aios.workflows.script_validation import validate_workflow_script

    script = Path("infra/workflows/work-state-reconciler.wf.py").read_text()
    validate_workflow_script(script, [ToolSpec(type="http_request"), ToolSpec(type="list_runs")])


def test_workflow_script_makes_no_model_or_agent_calls() -> None:
    """#2043: the reconciler is MECHANICAL. Intelligence is reserved for judgment."""
    script = Path("infra/workflows/work-state-reconciler.wf.py").read_text()
    for forbidden in ("agent(", "llm(", "await agent", "prompt("):
        assert forbidden not in script


# ─── (11) end-to-end shape on the known-zombie fixture ──────────────────────


def test_end_to_end_on_the_known_2043_zombie_set() -> None:
    """The #2043 sanity set: if the reconciler finds nothing here, the CODE is wrong."""
    known = [
        ("eumemic/aios", 2000),
        ("eumemic/eumemic-ops", 337),
        ("eumemic/eumemic-ops", 331),
        ("eumemic/eumemic-company", 210),
        ("eumemic/eumemic-company", 199),
        ("eumemic/eumemic-company", 192),
        ("eumemic/eumemic-company", 166),
        ("eumemic/eumemic-company", 151),
        ("eumemic/eumemic-company", 147),
        ("eumemic/eumemic-company", 135),
        ("eumemic/eumemic-company", 71),
        ("eumemic/eumemic-company", 50),
        ("eumemic/aios-console", 166),
        ("eumemic/aios-console", 12),
    ]
    items = [
        item(n, repo=r, labels=("approved", "dispatched"), linked=((72,) if n == 71 else ()))
        for r, n in known
    ]
    report = ok_report(items, [run("running", repo="eumemic/aios", number=1)])
    counts = report.counts
    assert counts is not None
    # 13, not 14: #71 has a linked PR, so it is AMBIGUOUS (C2) and must NOT inflate
    # the headline. The set of FINDINGS is still all 14 — nothing is dropped.
    assert counts["ZOMBIE"] == 13
    assert counts["AMBIGUOUS"] == 1
    assert {(d.repo, d.number) for d in report.disagreements} == set(known)
    # #71 is the designed-in check on the join logic, not a silent pass.
    (suspect,) = [d for d in report.disagreements if d.number == 71]
    assert suspect.classification == "AMBIGUOUS"
    assert suspect.caveats == ("has-linked-prs", "excluded-from-zombie-count")
    payload = json.loads(render_json(report))
    assert payload["verdict"] == "OK"
    assert payload["counts"]["ZOMBIE"] == 13
    assert "ZOMBIE (13)" in render_markdown(report)
    assert "no UNARCHIVED run exists" in payload["zombie_means"]


def test_disagreements_are_sorted_by_class_then_repo_then_number() -> None:
    items = [
        item(9, labels=("approved",)),
        item(5),
        item(3, repo="eumemic/aios-console"),
    ]
    runs = [run("running", number=9)]
    got = [(d.classification, d.repo, d.number) for d in ok_report(items, runs).disagreements]
    assert got == [
        ("ZOMBIE", "eumemic/aios", 5),
        ("ZOMBIE", "eumemic/aios-console", 3),
        ("LAGGING", "eumemic/aios", 9),
    ]


def test_report_to_dict_round_trips_as_json() -> None:
    payload = json.loads(render_json(ok_report([item(1)], [run("suspended")])))
    assert payload["disagreements"][0]["classification"] == "MISLABELLED"
    assert isinstance(payload["disagreement_hash"], str)
    assert payload["exhaustive"] is True
    assert payload["failures"] == []
    # The healthy shape carries real counts; the ALARM shape carries null. Never both.
    assert payload["counts"]["MISLABELLED"] == 1


def test_disagreement_identity_excludes_run_ids() -> None:
    a = Disagreement(
        classification="DEAD",
        repo="r",
        number=1,
        kind="issue",
        title="t",
        html_url="",
        labels=(),
        detail="",
        run_ids=("r1",),
        run_statuses=("errored",),
    )
    b = a.__class__(**{**a.__dict__, "run_ids": ("r2",)})
    assert a.identity() == b.identity()


# ─── (11) FIX ROUND for the second review ───────────────────────────────────
#
# Each test below names the finding it locks down. These are the regressions the
# reviewer reproduced, so they get explicit tests rather than incidental coverage.


# ---- C1: the headline. In-flight is a UNION of assertions, not one literal. ----

#: The NINE confirmed zombies with the labels they ACTUALLY carry as of
#: 2026-07-26T03:25, after the seat stripped `dispatched` from every one of them as
#: remediation. If the classifier keys on `dispatched`, this set reports "0 ZOMBIE"
#: while nine issues sit dead — the exact failure the reviewer reproduced.
_NINE_AFTER_THE_STRIP: list[tuple[str, int]] = [
    ("eumemic/aios", 2000),
    ("eumemic/eumemic-ops", 337),
    ("eumemic/eumemic-ops", 331),
    ("eumemic/eumemic-company", 210),
    ("eumemic/eumemic-company", 199),
    ("eumemic/eumemic-company", 192),
    ("eumemic/eumemic-company", 166),
    ("eumemic/eumemic-company", 151),
    ("eumemic/eumemic-company", 147),
]
_LABELS_AFTER_THE_STRIP = ("approved", "autodev:in-progress", "needs:human/build")


def test_C1_the_nine_zombies_are_still_found_after_dispatched_was_stripped() -> None:
    """THE headline regression. `dispatched` is gone from all nine; they still assert
    'a machine has this' via `autodev:in-progress` + `needs:human/build`. A tool built
    BECAUSE a label is an assertion nothing re-checks must not itself trust one label."""
    items = [item(n, repo=r, labels=_LABELS_AFTER_THE_STRIP) for r, n in _NINE_AFTER_THE_STRIP]
    report = ok_report(items, [])
    counts = report.counts
    assert counts is not None
    assert counts["ZOMBIE"] == 9, "keying on `dispatched` alone would report 0 here"
    assert {(d.repo, d.number) for d in report.by_class("ZOMBIE")} == set(_NINE_AFTER_THE_STRIP)
    # And the report must say WHICH assertion lied, not just that something did.
    for d in report.by_class("ZOMBIE"):
        assert d.trigger_labels == ("autodev:in-progress", "needs:human/build")
        assert "`autodev:in-progress`" in d.detail
    assert "triggered by:" in render_markdown(report)


@pytest.mark.parametrize(
    ("labels", "expect_in_flight"),
    [
        (("dispatched",), True),
        (("autodev:in-progress",), True),
        (("design:in-progress",), True),
        (("needs:human/build",), True),
        (("needs:human/review",), True),
        (("needs:human/anything-at-all",), True),
        (("approved",), False),
        (("autodev:built",), False),  # a PR exists — done, not in flight
        (("autodev:failed",), False),
        (("hold",), False),
        ((), False),
    ],
)
def test_C1_in_flight_vocabulary(labels: tuple[str, ...], expect_in_flight: bool) -> None:
    assert item(1, labels=labels).claims_in_flight is expect_in_flight
    verdict = classify_item(item(1, labels=labels), [])
    assert (verdict is not None and verdict.classification == "ZOMBIE") is expect_in_flight


def test_C1_a_done_label_alone_is_not_an_in_flight_claim() -> None:
    """`autodev:built` means a PR exists. That is not a claim a machine is on it now,
    so a built item with no live run is NOT a zombie."""
    assert classify_item(item(1, labels=("autodev:built",)), []) is None


# ---- C3: unknown status outranks terminal; never evidence of death. ----


def test_C3_unknown_status_beside_a_terminal_run_is_not_DEAD() -> None:
    """`if unknown and not (suspended or terminal)` made [unknown, errored] return DEAD,
    contradicting the documented contract that an unrecognised status is never evidence
    of death. The unknown run may well BE live under a status name we have not learned."""
    d = classify_item(item(1), [run("errored", run_id="r1"), run("quiesced", run_id="r2")])
    assert d is not None
    assert d.classification == "MISLABELLED"
    assert "unrecognised-run-status" in d.caveats
    assert "quiesced" in d.detail
    assert "terminal sibling does NOT make an unknown status dead" in d.detail


def test_C3_unknown_status_beside_a_suspended_run_is_still_not_DEAD() -> None:
    d = classify_item(item(1), [run("suspended", run_id="r1"), run("quiesced", run_id="r2")])
    assert d is not None
    assert d.classification == "MISLABELLED"
    assert "unrecognised-run-status" in d.caveats


def test_C3_a_live_run_still_outranks_everything() -> None:
    assert (
        classify_item(item(1), [run("quiesced", run_id="r1"), run("running", run_id="r2")]) is None
    )


# ---- B1 / B2: a truncated read must NEVER render as exhaustive. ----


def test_B1_has_more_with_an_unusable_cursor_is_a_READ_FAILURE() -> None:
    """The server said THERE IS MORE and gave no cursor. The old code ended pagination
    and reported exhaustive=True — a truncated read rendered as exhaustive, which is the
    2026-07-25 failure mode this PR exists to kill. Every ZOMBIE rests on 'no run in the
    list I read', so an unread page manufactures false ZOMBIEs."""
    bad_cursors: tuple[object, ...] = (None, "", 0, {})
    for bad_cursor in bad_cursors:

        def getter(url: str, headers: Any, _c: Any = bad_cursor) -> Any:
            return (
                {
                    "data": [{"id": "r1", "status": "completed"}],
                    "has_more": True,
                    "next_cursor": _c,
                },
                {},
            )

        source = read_aios_runs(base_url="http://x", api_key="k", getter=getter)
        assert isinstance(source, SourceFailed), f"cursor={bad_cursor!r} must FAIL the read"
        assert "has_more" in source.reason and "unusable" in source.reason
        # ...and it must surface as ALARM with counts None, never as a clean report.
        report = build_report(items_read=SourceOk(name="github", items=()), runs_read=source)
        assert report.verdict == "ALARM"
        assert report.counts is None


def test_B2_workflow_full_page_without_an_id_cursor_is_a_READ_FAILURE() -> None:
    """The mirror of B1 in the workflow form: a full page whose last row carries no `id`
    leaves no keyset cursor, so the history is truncated. Breaking the loop with
    exhaustive=True is the same lie in a different file."""
    wf = _load_wf_module()
    rows = [{"id": f"r{n}", "status": "completed"} for n in range(199)]
    rows.append({"status": "completed"})
    assert len(rows) == 200
    with pytest.raises(wf.ReadFailure) as excinfo:
        wf._paginate_check(rows)
    assert "TRUNCATED" in str(excinfo.value)


# ---- B3: ZOMBIE means "no UNARCHIVED run", and every report says so. ----


def test_B3_every_report_states_what_zombie_can_actually_prove() -> None:
    """`list_runs` filters archived_at IS NULL and archiving requires a TERMINAL run, so
    a completed-then-archived run reads as 'no run, ever'. The class cannot distinguish
    'never picked up' from 'tidied away', so it must not imply the former."""
    report = ok_report([item(2000)], [])
    payload = json.loads(render_json(report))
    assert "no UNARCHIVED run exists" in payload["zombie_means"]
    assert "archived" in payload["zombie_means"]
    md = render_markdown(report)
    assert "What ZOMBIE can prove" in md
    assert "no UNARCHIVED run exists" in md
    # The per-finding text must not overclaim either.
    (d,) = report.disagreements
    assert "NO unarchived run exists" in d.detail
    assert "zombie-means-no-unarchived-run" in d.caveats


def test_B3_workflow_form_states_it_too() -> None:
    wf = _load_wf_module()
    report = wf.build([], True, [], True, [])
    assert "no UNARCHIVED run exists" in report["zombie_means"]


# ---- B4: unkeyable runs are NEVER skipped — the PR body's claim, made true. ----


def test_B4_terminal_runs_with_an_unreadable_join_key_are_reported() -> None:
    """Only live/suspended unkeyable runs reached unmatched_runs, so an `errored` run
    with an unparseable input vanished — and its issue was then classified ZOMBIE with no
    trace of the run that existed. A manufactured false ZOMBIE, silently."""
    for status in ("completed", "errored", "cancelled"):
        report = ok_report([item(2000)], [run(status, repo=None, number=None, run_id="r_lost")])
        assert [u.run_id for u in report.unmatched_runs] == ["r_lost"], status
        (u,) = report.unmatched_runs
        assert "FALSE ZOMBIE" in u.reason
        assert status in u.reason
        assert "Unmatched runs" in render_markdown(report)


def test_B4_workflow_form_reports_terminal_unkeyable_runs_too() -> None:
    wf = _load_wf_module()
    report = wf.build([], True, [{"id": "r_lost", "status": "errored", "input": {}}], True, [])
    assert [u["run_id"] for u in report["unmatched_runs"]] == ["r_lost"]
    assert "FALSE ZOMBIE" in report["unmatched_runs"][0]["reason"]


# ---- malformed rows ALARM instead of killing the workflow (first review) ----


def test_malformed_issue_row_is_a_read_failure_not_an_AttributeError() -> None:
    """A 2xx carrying a non-object row reached row.get() and raised AttributeError
    OUTSIDE the ReadFailure handlers, killing the workflow instead of returning
    verdict:ALARM. A malformed row is a FAILED READ."""
    wf = _load_wf_module()
    for bad in ("a string", 42, None, ["nested"]):
        with pytest.raises(wf.ReadFailure):
            wf._check_issue_row("eumemic/aios", bad)


def test_malformed_labels_field_is_a_read_failure() -> None:
    wf = _load_wf_module()
    with pytest.raises(wf.ReadFailure):
        wf._check_issue_row("eumemic/aios", {"number": 1, "labels": "dispatched"})


def test_malformed_run_row_is_a_read_failure() -> None:
    wf = _load_wf_module()
    for bad in ("a string", 42, None):
        with pytest.raises(wf.ReadFailure):
            wf._check_run_row(bad)


# ---- C4: a transferred issue is DETECTED, not silently turned into a zombie. ----


def test_C4_the_INVERTED_probe_finds_a_transfer_from_a_REALIZABLE_world() -> None:
    """THE C4 fixture, rebuilt so it describes a world production can actually present.

    The world, exactly as GitHub serves it after a completed transfer:

    * a RUN whose ``input`` names the OLD coordinates (company#71) — 68 issues moved
      repos on 2026-07-25, so every one of their runs carries a stale key right now;
    * enumeration that returns ONLY the NEW coordinates (ops#900). The OLD pair is
      *never enumerated* — that is the whole point, and it is why the previous fixture
      (start from an enumerated issue, follow a redirect on its timeline) asserted a
      world that cannot occur. Agreement on an impossible fixture is not evidence.

    So the probe is INVERTED: the reconciler takes the run join keys that matched NO
    enumerated item and GETs those STALE coordinates. company#71 redirects to ops#900,
    which is authoritative evidence of the transfer — and ops#900, which would
    otherwise be a textbook false ZOMBIE (in-flight label, no run keyed to it), is
    kept OUT of counts['ZOMBIE'].
    """
    world = _github_world(
        # Enumeration NEVER returns the old pair. Only the new coordinates exist.
        issues={
            "eumemic/eumemic-ops": [
                {
                    "number": 900,
                    "title": "moved here from company#71",
                    "labels": [{"name": "autodev:in-progress"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ],
            "eumemic/eumemic-company": [],
        },
        timelines={"eumemic/eumemic-ops/issues/900": []},
        moved={
            "eumemic/eumemic-company/issues/71": (
                "https://api.github.com/repos/eumemic/eumemic-ops/issues/900"
            )
        },
    )
    # The run still holds the OLD coordinates. This is the stale join key.
    runs = [
        {
            "id": "run_stale",
            "status": "completed",
            "input": {"repo": "eumemic/eumemic-company", "issue_number": 71},
        }
    ]
    source, transfers, paths = _cli_reader(
        world, ["eumemic/eumemic-ops", "eumemic/eumemic-company"], runs
    )

    # 1. The probe actually asked about the STALE coordinates — the one place a
    #    completed transfer is visible.
    assert "/repos/eumemic/eumemic-company/issues/71" in paths, (
        "the reconciler never probed the stale join key, so a real transfer is invisible"
    )

    # 2. The redirect is read as authoritative evidence, resolved to NEW coordinates.
    (evidence,) = transfers
    assert (evidence.stale_repo, evidence.stale_number) == ("eumemic/eumemic-company", 71)
    assert (evidence.new_repo, evidence.new_number) == ("eumemic/eumemic-ops", 900)
    assert evidence.run_ids == ("run_stale",)

    report = build_report(
        items_read=source,
        runs_read=SourceOk(name="aios-runs", items=tuple(run_record_from_payload(r) for r in runs)),
        transfers=transfers,
    )

    # 3. THE payoff: the issue at the new coordinates is NOT a zombie.
    counts = report.counts or {}
    assert counts["ZOMBIE"] == 0, "a transferred issue STILL became a false ZOMBIE"
    assert counts["AMBIGUOUS"] == 1
    (d,) = report.disagreements
    assert d.classification == "AMBIGUOUS"
    assert (d.repo, d.number) == ("eumemic/eumemic-ops", 900)
    assert "transferred-stale-join-key" in d.caveats
    assert "excluded-from-zombie-count" in d.caveats
    assert "eumemic/eumemic-company#71" in d.detail
    assert "run_stale" in d.detail

    # 4. And the stale key is visible as DATA and as prose.
    assert any("STALE JOIN KEY" in n for n in report.notes)
    payload = json.loads(render_json(report))
    assert payload["transfers"][0]["stale_number"] == 71
    assert payload["transfers"][0]["new_number"] == 900
    assert "TRANSFERRED" in render_markdown(report)


def test_C4_the_OLD_probe_direction_could_not_have_seen_this_world() -> None:
    """Why the inversion was necessary, pinned as a test rather than left as a claim.

    In the realizable world the OLD coordinates are never enumerated, so a
    detector that only inspects ENUMERATED items — the previous design — issues no
    request that could ever observe the transfer. Here: enumeration touches only
    ops#900 and its timeline, and neither answers with a redirect. Without the probe
    of the stale key, ops#900 is a ZOMBIE.
    """
    world = _github_world(
        issues={
            "eumemic/eumemic-ops": [
                {
                    "number": 900,
                    "title": "moved",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ]
        },
        timelines={"eumemic/eumemic-ops/issues/900": []},
        moved={
            "eumemic/eumemic-company/issues/71": (
                "https://api.github.com/repos/eumemic/eumemic-ops/issues/900"
            )
        },
    )
    runs = [
        {
            "id": "run_stale",
            "status": "completed",
            "input": {"repo": "eumemic/eumemic-company", "issue_number": 71},
        }
    ]
    # Enumeration-only (no stale-key probe): NOTHING redirects, so no transfer is seen.
    source, _transfers, enum_paths = _cli_reader(world, ["eumemic/eumemic-ops"], [])
    assert not any("issues/71" in p for p in enum_paths)
    blind = build_report(
        items_read=source,
        runs_read=SourceOk(name="aios-runs", items=tuple(run_record_from_payload(r) for r in runs)),
    )
    assert (blind.counts or {})["ZOMBIE"] == 1, (
        "this fixture must reproduce the FALSE ZOMBIE, or it is not testing the defect"
    )

    # With the inverted probe, the same world yields AMBIGUOUS instead.
    _s, transfers, _p = _cli_reader(world, ["eumemic/eumemic-ops"], runs)
    fixed = build_report(
        items_read=_s,
        runs_read=SourceOk(name="aios-runs", items=tuple(run_record_from_payload(r) for r in runs)),
        transfers=transfers,
    )
    assert (fixed.counts or {})["ZOMBIE"] == 0
    assert (fixed.counts or {})["AMBIGUOUS"] == 1


def test_C4_stale_run_keys_is_exactly_the_probe_list() -> None:
    """The probe list is the runs that joined to nothing — not the enumerated items."""
    items = [item(900, repo="eumemic/eumemic-ops")]
    runs = [
        run("completed", repo="eumemic/eumemic-company", number=71, run_id="r_a"),
        run("errored", repo="eumemic/eumemic-company", number=71, run_id="r_b"),
        run("running", repo="eumemic/eumemic-ops", number=900, run_id="r_joined"),
        run("errored", repo=None, number=None, run_id="r_unkeyable"),
    ]
    assert stale_run_keys(items, runs) == {("eumemic/eumemic-company", 71): ("r_a", "r_b")}


def test_C4_a_stale_key_that_resolves_to_ITSELF_is_not_a_transfer() -> None:
    """Most stale keys are runs against CLOSED issues. That must NOT read as a transfer.

    Without this, "probe the stale keys" could be satisfied by something that reports a
    transfer for every closed issue — which would move real zombies out of the count
    and bury the defect the tool exists to surface.
    """
    world = _github_world(
        issues={
            "eumemic/aios": [
                {
                    "number": 2000,
                    "title": "z",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ]
        },
        timelines={"eumemic/aios/issues/2000": []},
    )
    runs = [
        {"id": "r_closed", "status": "completed", "input": {"repo": "eumemic/aios", "issue": 111}}
    ]
    source, transfers, paths = _cli_reader(world, ["eumemic/aios"], runs)
    assert "/repos/eumemic/aios/issues/111" in paths
    assert transfers == ()
    report = build_report(
        items_read=source,
        runs_read=SourceOk(name="aios-runs", items=tuple(run_record_from_payload(r) for r in runs)),
        transfers=transfers,
    )
    assert (report.counts or {})["ZOMBIE"] == 1, "a real zombie must stay a zombie"
    assert (report.counts or {})["AMBIGUOUS"] == 0


def test_C4_probe_read_error_is_a_NOTE_not_an_ALARM() -> None:
    """A 404 on a stale coordinate is HISTORY (deleted/invisible issue), not an outage.

    It must not ALARM the whole reconcile — but it must not be silent either, since an
    unprobed key is a transfer we did not look for.
    """

    def getter(url: str, headers: Any) -> Any:
        raise ReadFailure(f"GET {url} → HTTP 404: gone")

    transfers, notes = probe_stale_keys(
        [item(1)],
        [run("completed", repo="eumemic/aios", number=4242, run_id="r_x")],
        token="t",
        getter=getter,
    )
    assert transfers == ()
    assert any("404" in n and "eumemic/aios#4242" in n for n in notes)


def test_C4_probe_cap_is_reported_as_a_gap_not_ignored() -> None:
    """Hitting the probe ceiling says so. An unprobed key is an unseen transfer."""

    def getter(url: str, headers: Any) -> Any:
        return {"number": 1, "url": url}, {}

    runs = [run("completed", repo="eumemic/aios", number=n, run_id=f"r{n}") for n in range(10, 20)]
    transfers, notes = probe_stale_keys([], runs, token="t", getter=getter, max_probes=3)
    assert transfers == ()
    assert any("capped at 3 of 10" in n and "NOT probed" in n for n in notes)


def test_C4_a_transfer_OUTSIDE_the_scanned_repos_still_emits_its_note() -> None:
    """The destination may be a repo we do not scan. There is then no item to caveat, so
    the note is the ONLY trace the stale key leaves — and it must survive."""
    evidence = TransferEvidence(
        stale_repo="eumemic/eumemic-company",
        stale_number=71,
        destination="https://api.github.com/repos/other-org/private/issues/5",
        new_repo="other-org/private",
        new_number=5,
        run_ids=("run_stale",),
    )
    report = build_report(
        items_read=SourceOk(name="github", items=()),
        runs_read=SourceOk(name="aios-runs", items=()),
        transfers=(evidence,),
    )
    assert report.verdict == "OK"
    assert any("STALE JOIN KEY" in n for n in report.notes)
    assert "other-org/private" in "\n".join(report.notes)


# ---- ITEM 2: the change detector must SEE a transfer-caveat change. ----


def _transfer(dest_number: int = 900, run_id: str = "run_stale") -> TransferEvidence:
    return TransferEvidence(
        stale_repo="eumemic/eumemic-company",
        stale_number=71,
        destination=f"https://api.github.com/repos/eumemic/eumemic-ops/issues/{dest_number}",
        new_repo="eumemic/eumemic-ops",
        new_number=dest_number,
        run_ids=(run_id,),
    )


def test_ITEM2_transfer_notes_are_folded_into_the_disagreement_hash() -> None:
    """A transfer caveat appearing must WAKE the seat.

    A stale key whose destination lies outside the scanned repos produces no
    disagreement at all, so a hash computed only over disagreements would read a
    newly-discovered transfer as "unchanged, nothing to say" and stay silent about the
    exact fact a triager needs.
    """

    def report(transfers: tuple[TransferEvidence, ...]) -> ReconcileReport:
        return build_report(
            items_read=SourceOk(name="github", items=()),
            runs_read=SourceOk(name="aios-runs", items=()),
            transfers=transfers,
        )

    none_seen = report(())
    one_seen = report((_transfer(),))
    moved_elsewhere = report((_transfer(dest_number=901),))
    other_run = report((_transfer(run_id="run_other"),))

    assert none_seen.disagreement_hash() != one_seen.disagreement_hash(), (
        "a NEW transfer caveat did not change the hash — the seat would never be woken"
    )
    assert one_seen.disagreement_hash() != moved_elsewhere.disagreement_hash()
    assert one_seen.disagreement_hash() != other_run.disagreement_hash()
    # Stable for the same evidence: it must not spam the seat every cron tick.
    assert one_seen.disagreement_hash() == report((_transfer(),)).disagreement_hash()
    # Order of evidence is not a change.
    a = report((_transfer(), _transfer(dest_number=901)))
    b = report((_transfer(dest_number=901), _transfer()))
    assert a.disagreement_hash() == b.disagreement_hash()


def test_ITEM2_transfer_notes_change_the_hash_on_the_ALARM_path_too() -> None:
    """An outage must not mask a newly-discovered stale key."""

    def alarm(transfers: tuple[TransferEvidence, ...]) -> ReconcileReport:
        return build_report(
            items_read=SourceFailed(name="github", reason="HTTP 500"),
            runs_read=SourceOk(name="aios-runs", items=()),
            transfers=transfers,
        )

    assert alarm(()).verdict == "ALARM"
    assert alarm(()).disagreement_hash() != alarm((_transfer(),)).disagreement_hash()


def test_ITEM2_both_forms_agree_on_the_hash_WITH_transfers() -> None:
    """The mirror must fold transfer notes in identically, or the two wake differently."""
    wf = _load_wf_module()
    evidence = _transfer()
    lib = build_report(
        items_read=SourceOk(name="github", items=()),
        runs_read=SourceOk(name="aios-runs", items=()),
        transfers=(evidence,),
    )
    wf_report = wf.build(
        [],
        True,
        [],
        True,
        [],
        [],
        [
            {
                "stale_repo": evidence.stale_repo,
                "stale_number": evidence.stale_number,
                "destination": evidence.destination,
                "new_repo": evidence.new_repo,
                "new_number": evidence.new_number,
                "run_ids": list(evidence.run_ids),
            }
        ],
    )
    assert wf.disagreement_hash(wf_report) == lib.disagreement_hash()
    # And the mirror's hash moves when the evidence does.
    bare = wf.build([], True, [], True, [], [], [])
    assert wf.disagreement_hash(bare) != wf.disagreement_hash(wf_report)


def test_C4_parse_issue_ref_never_guesses() -> None:
    assert parse_issue_ref("https://api.github.com/repos/eumemic/aios/issues/12") == (
        "eumemic/aios",
        12,
    )
    assert parse_issue_ref("https://github.com/eumemic/aios/issues/12") == ("eumemic/aios", 12)
    for bad in (None, "", "https://api.github.com/repos/eumemic/aios", "nonsense", 7):
        assert parse_issue_ref(bad) is None


def test_C4_a_repo_RENAME_on_the_list_path_is_still_surfaced() -> None:
    """The list-path redirect IS realizable (a repo rename/move) and is retained.

    Distinct from the issue-transfer case: here the redirect happens on a request we
    genuinely make, so following it silently while saying nothing would hide that the
    enumerated coordinates are not the ones we asked for.
    """
    from aios.reconcilers.work_state_cli import _FINAL_URL_HEADER

    def getter(url: str, headers: Any) -> Any:
        return [], {_FINAL_URL_HEADER: "https://api.github.com/repos/eumemic/renamed/issues"}

    source = read_github_items(
        repos=["eumemic/oldname"], token="t", getter=getter, enrich_linked_prs=False
    )
    assert isinstance(source, SourceOk)
    assert any("TRANSFERRED" in n and "renamed" in n for n in source.notes)


# ---- the drift guard now asserts the TRANSPORT, not source text. ----


def test_drift_guard_catches_a_data_kwarg_that_would_silently_become_a_POST() -> None:
    """`urllib.request.Request(url, data=...)` infers POST from a non-None body and
    contains none of the literals a source grep looks for. Assert the TRANSPORT."""
    import urllib.request

    real_request = urllib.request.Request

    class SneakyRequest(real_request):  # type: ignore[misc,valid-type]
        def __init__(self, url: str, **kw: Any) -> None:
            super().__init__(url, data=b"{}", **{k: v for k, v in kw.items() if k != "method"})

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(urllib.request, "Request", SneakyRequest)
        with pytest.raises(ObserveOnlyViolation) as excinfo:
            _get("https://api.github.com/repos/eumemic/aios/issues/1", {})
        assert "OBSERVE-ONLY" in str(excinfo.value)
    finally:
        monkey.undo()


# ═══════════════════════════════════════════════════════════════════════════
# (13) READER PARITY — the drift guard that would have CAUGHT C2 and C4.
#
# The previous guard only ever called `classify()` / `build()`, so it compared
# the two forms' arithmetic while never comparing what they can ACQUIRE. That is
# precisely how C2 and C4 shipped: the library could fetch linked PRs and detect a
# transfer, the workflow could not, and every classifier-level assertion still
# passed because the test HANDED the workflow the evidence its production path
# could never obtain.
#
# So the guard now drives BOTH READ PATHS over ONE recorded GitHub world and
# requires: identical items, identical classifications, identical hashes, identical
# notes — AND that each form actually issued the request that acquires the evidence.
# A capability present in one form and absent in the other now FAILS THE BUILD.
# ═══════════════════════════════════════════════════════════════════════════


def _github_world(
    *,
    issues: dict[str, list[dict[str, Any]]],
    timelines: dict[str, list[dict[str, Any]]] | None = None,
    redirects: dict[str, str] | None = None,
    moved: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One recorded GitHub, replayable through EITHER transport.

    Keyed by the API path (never the full URL) because the two forms address the
    same endpoint differently: the CLI builds absolute ``https://api.github.com/...``
    URLs for urllib, the workflow passes a bare path to the ``github`` http_server.
    Same bytes, same endpoints, two transports — which is the only way a comparison
    between them means anything.
    """
    return {
        "issues": issues,
        "timelines": timelines or {},
        "redirects": redirects or {},
        # C4 — ``moved`` records TRANSFERS as GitHub actually presents them:
        # ``{"owner/name/issues/N": "<url of the NEW coordinates>"}``. The OLD pair
        # answers with a redirect; the NEW pair is what enumeration returns. This is
        # the realizable world — see ``_resolve``.
        "moved": moved or {},
    }


def _path_of(url: str) -> str:
    return url.split("api.github.com", 1)[1] if "api.github.com" in url else url


def _resolve(world: dict[str, Any], path: str) -> tuple[str, Any]:
    """Resolve one recorded request. Returns ``(kind, payload)``.

    ``kind`` is ``"redirect"`` (payload = destination URL) or ``"ok"`` (payload = the
    JSON body). Shared by both adapters so neither form can be fed a different world.
    """
    for prefix, dest in world["redirects"].items():
        if path.startswith(prefix):
            return "redirect", dest
    if "/timeline" in path:
        key = path.split("/repos/", 1)[1].split("/timeline")[0]  # "owner/name/issues/N"
        return "ok", world["timelines"].get(key, [])
    if "/issues?" in path:
        repo = path.split("/repos/", 1)[1].split("/issues?")[0]
        return "ok", world["issues"].get(repo, [])
    if "/issues/" in path:
        # A single-issue read — the C4 stale-key PROBE. ``moved`` entries redirect (a
        # transfer); anything else answers as ITSELF, which is what a stale key
        # belonging to a merely-closed issue does.
        key = path.split("/repos/", 1)[1]  # "owner/name/issues/N"
        if key in world["moved"]:
            return "redirect", world["moved"][key]
        repo, number = key.split("/issues/")
        return "ok", {
            "number": int(number),
            "url": f"https://api.github.com/repos/{repo}/issues/{number}",
            "html_url": f"https://github.com/{repo}/issues/{number}",
            "state": "closed",
            "labels": [],
        }
    raise AssertionError(f"the recorded world has no entry for {path!r}")


def _cli_reader(
    world: dict[str, Any], repos: list[str], runs: list[dict[str, Any]] | None = None
) -> tuple[SourceOk, tuple[TransferEvidence, ...], list[str]]:
    """Drive the LIBRARY/CLI read path over ``world``, INCLUDING the C4 stale-key probe.

    Returns ``(source, transfers, paths_requested)``. ``runs`` are raw run payloads, so
    the probe list is derived exactly as production derives it — from join keys that
    matched no enumerated item — rather than being handed to the test.
    """
    from aios.reconcilers.work_state_cli import _FINAL_URL_HEADER

    seen: list[str] = []

    def getter(url: str, headers: Any) -> Any:
        path = _path_of(url)
        seen.append(path)
        kind, payload = _resolve(world, path)
        if kind == "redirect":
            # urllib FOLLOWS the 301 silently and reports where it landed; that
            # synthetic header is the CLI's transfer signal.
            return [], {_FINAL_URL_HEADER: payload}
        return payload, {}

    source = read_github_items(repos, token="t", getter=getter, enrich_linked_prs=True)
    assert isinstance(source, SourceOk), getattr(source, "reason", source)
    records = [run_record_from_payload(r) for r in (runs or [])]
    items = [i for i in source.items if isinstance(i, WorkItem)]
    transfers, probe_notes = probe_stale_keys(items, records, token="t", getter=getter)
    if probe_notes:
        source = SourceOk(
            name=source.name,
            items=source.items,
            exhaustive=source.exhaustive,
            notes=source.notes + probe_notes,
        )
    return source, transfers, seen


def _wf_reader(
    world: dict[str, Any], repos: list[str], runs: list[dict[str, Any]] | None = None
) -> tuple[Any, list[str]]:
    """Drive the WORKFLOW read path over the SAME ``world``, INCLUDING the C4 probe.

    Returns ``((wf, items, exhaustive, notes, transfers), paths)``.
    """
    wf = _load_wf_module()
    seen: list[str] = []

    async def fake_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert name == "http_request"
        assert args["method"] == "GET", "OBSERVE-ONLY: the mirror may only ever GET"
        assert "body" not in args, "OBSERVE-ONLY: a GET carries no request body"
        path = args["path"]
        seen.append(path)
        kind, payload = _resolve(world, path)
        if kind == "redirect":
            # http_request does NOT follow redirects, so the transfer arrives as a
            # literal 301 + Location. Different mechanism from urllib, same event.
            return {"status": 301, "headers": [["Location", payload]], "body": ""}
        return {"status": 200, "headers": [], "body": json.dumps(payload)}

    wf.tool = fake_tool
    items, exhaustive, notes = asyncio.run(wf._read_items(repos))
    transfers, probe_notes = asyncio.run(wf._probe_stale_keys(items, runs or []))
    notes = list(notes) + list(probe_notes)
    return (wf, items, exhaustive, notes, transfers), seen


def _assert_forms_agree(
    world: dict[str, Any], repos: list[str], runs: list[dict[str, Any]] | None = None
) -> tuple[Any, Any]:
    """THE guard. Both forms read the same world; every observable must match.

    Compares the ACQUIRED evidence, the classifications, the notes and the hash. A
    capability that exists in only one form cannot survive this. ``runs`` are raw run
    payloads fed to BOTH forms, so the C4 stale-key probe list is derived by each form
    from the same join, never handed to it.
    """
    runs = runs or []
    source, cli_transfers, cli_paths = _cli_reader(world, repos, runs)
    (wf, wf_items, wf_exhaustive, wf_notes, wf_transfers), wf_paths = _wf_reader(world, repos, runs)

    lib_report = build_report(
        items_read=source,
        runs_read=SourceOk(name="aios-runs", items=tuple(run_record_from_payload(r) for r in runs)),
        transfers=cli_transfers,
    )
    wf_report = wf.build(wf_items, wf_exhaustive, runs, True, [], wf_notes, wf_transfers)

    # 1. The same ITEMS, with the same ACQUIRED linked-PR evidence.
    assert [(i.repo, i.number) for i in source.items] == [
        (i["repo"], i["number"]) for i in wf_items
    ]
    assert [list(i.linked_pr_numbers) for i in source.items] == [
        list(i["linked_pr_numbers"]) for i in wf_items
    ], "linked-PR evidence differs between the two forms"

    # 2. The same CLASSIFICATIONS and the same COUNTS.
    assert dict(lib_report.counts or {}) == wf_report["counts"]
    assert [(d.classification, d.repo, d.number) for d in lib_report.disagreements] == [
        (d["classification"], d["repo"], d["number"]) for d in wf_report["disagreements"]
    ]

    # 3. The same NOTES (C4 transfers) — byte for byte.
    assert sorted(lib_report.notes) == sorted(wf_report["notes"])

    # 3b. The same C4 stale-key EVIDENCE, as structured data. Notes are prose; this is
    #     what phase 2 will act on, so it must match field for field.
    assert [
        (e.stale_repo, e.stale_number, e.new_repo, e.new_number, list(e.run_ids))
        for e in cli_transfers
    ] == [
        (t["stale_repo"], t["stale_number"], t["new_repo"], t["new_number"], t["run_ids"])
        for t in wf_transfers
    ], "the two forms disagree about which join keys are STALE"

    # 4. The same change-detection HASH.
    assert wf.disagreement_hash(wf_report) == lib_report.disagreement_hash()

    # 5. Both forms actually WENT AND GOT IT. Equal outputs from unequal effort is
    #    exactly the shape C2 shipped in: one form fetched the evidence, the other
    #    was handed it by a test.
    assert sorted(cli_paths) == sorted(wf_paths), (
        "the two forms issued DIFFERENT requests for the same world — one of them is "
        f"not acquiring evidence it appears to have.\nCLI: {sorted(cli_paths)}\n"
        f"WF : {sorted(wf_paths)}"
    )
    return lib_report, wf_report


def test_drift_guard_C2_both_forms_FETCH_the_linked_pr_evidence() -> None:
    """C2, as a SYSTEM test: nothing is injected. Both forms must go and get it.

    The five known-good items with their real linked PRs. The workflow used never to
    fetch a timeline at all, so `linked_pr_numbers` stayed empty and AMBIGUOUS was
    UNREACHABLE with real data — all five classified ZOMBIE on the form that runs on
    cron. The old test passed anyway because it INJECTED `linked_pr_numbers` into
    `classify()`. Here the only source of that evidence is the recorded timeline, so
    a form that does not fetch it CANNOT reach AMBIGUOUS.
    """
    known_good = {
        ("eumemic/eumemic-company", 71): [67, 68, 69, 100, 101],
        ("eumemic/eumemic-company", 50): [96, 204, 206],
        ("eumemic/eumemic-company", 135): [212],
        ("eumemic/eumemic-ops", 337): [1977, 1979, 2016],
        ("eumemic/eumemic-ops", 331): [1995, 2041],
    }
    issues: dict[str, list[dict[str, Any]]] = {}
    timelines: dict[str, list[dict[str, Any]]] = {}
    for (repo, number), prs in known_good.items():
        issues.setdefault(repo, []).append(
            {
                "number": number,
                "title": f"item {number}",
                # The REAL current labels: `dispatched` was stripped on 2026-07-26.
                "labels": [{"name": "approved"}, {"name": "autodev:in-progress"}],
                "html_url": "",
                "updated_at": "",
            }
        )
        timelines[f"{repo}/issues/{number}"] = [
            {"event": "cross-referenced", "source": {"issue": {"number": n, "pull_request": {}}}}
            for n in prs
        ] + [
            {"event": "labeled"},
            {"event": "cross-referenced", "source": {"issue": {"number": 9}}},
        ]

    world = _github_world(issues=issues, timelines=timelines)
    repos = ["eumemic/eumemic-company", "eumemic/eumemic-ops"]
    lib_report, wf_report = _assert_forms_agree(world, repos)

    # The payoff, in BOTH forms, from FETCHED evidence only.
    assert wf_report["counts"]["ZOMBIE"] == 0, (
        "the five known-good items must not be zombies in the WORKFLOW form — this is "
        "the assertion the injected-input test could never make"
    )
    assert wf_report["counts"]["AMBIGUOUS"] == 5
    assert (lib_report.counts or {})["ZOMBIE"] == 0
    assert (lib_report.counts or {})["AMBIGUOUS"] == 5
    assert {(d["repo"], d["number"]) for d in wf_report["disagreements"]} == set(known_good)
    for d in wf_report["disagreements"]:
        assert d["classification"] == "AMBIGUOUS"
        assert "excluded-from-zombie-count" in d["caveats"]
        # The ACQUIRED PR numbers are named in the finding, not merely counted.
        for n in known_good[(d["repo"], d["number"])]:
            assert f"#{n}" in d["detail"]


def test_C2_workflow_read_path_actually_requests_the_timeline() -> None:
    """The fetch itself, pinned. `_read_items` must issue the timeline GET.

    Stated separately from the parity guard because this is the exact regression:
    the mirror's read path never asked for a timeline, so no amount of classifier
    correctness could produce an AMBIGUOUS from real workflow data.
    """
    world = _github_world(
        issues={
            "eumemic/eumemic-company": [
                {
                    "number": 135,
                    "title": "t",
                    "labels": [{"name": "autodev:in-progress"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ]
        },
        timelines={
            "eumemic/eumemic-company/issues/135": [
                {
                    "event": "cross-referenced",
                    "source": {"issue": {"number": 212, "pull_request": {}}},
                }
            ]
        },
    )
    (wf, items, _, _, _), paths = _wf_reader(world, ["eumemic/eumemic-company"])
    assert any("/issues/135/timeline" in p for p in paths), (
        "the workflow read path did not FETCH the linked-PR evidence"
    )
    assert items[0]["linked_pr_numbers"] == [212]
    assert wf.classify(items[0], [])["classification"] == "AMBIGUOUS"


def test_C2_a_zombie_stays_a_zombie_when_the_fetch_finds_no_linked_pr() -> None:
    """The enrichment must not turn everything AMBIGUOUS. An empty timeline ⇒ ZOMBIE.

    Without this, 'fetch the timeline' could be satisfied by a call that always
    reports linkage, which would bury the nine real zombies instead of the five
    known-good items. Both forms, same world.
    """
    world = _github_world(
        issues={
            "eumemic/aios": [
                {
                    "number": 2000,
                    "title": "t",
                    "labels": [{"name": "autodev:in-progress"}, {"name": "needs:human/build"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ]
        },
        timelines={"eumemic/aios/issues/2000": [{"event": "labeled"}]},
    )
    lib_report, wf_report = _assert_forms_agree(world, ["eumemic/aios"])
    assert wf_report["counts"]["ZOMBIE"] == 1
    assert wf_report["counts"]["AMBIGUOUS"] == 0
    assert (lib_report.counts or {})["ZOMBIE"] == 1
    (d,) = wf_report["disagreements"]
    assert d["trigger_labels"] == ["autodev:in-progress", "needs:human/build"]


def test_C2_a_truncated_timeline_is_a_READ_FAILURE_not_a_quiet_zombie() -> None:
    """A timeline we could not read to the end cannot decide ZOMBIE vs AMBIGUOUS.

    The unread page is exactly where the PR that disproves the ZOMBIE would be, so
    stopping quietly manufactures the false ZOMBIE this whole reconciler exists to
    catch. Fail loud instead.
    """
    wf = _load_wf_module()

    async def fake_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        path = args["path"]
        if "/timeline" in path:
            return {
                "status": 200,
                "headers": [
                    [
                        "Link",
                        "<https://api.github.com/repos/eumemic/aios/issues/1/timeline"
                        '?page=2>; rel="next"',
                    ]
                ],
                "body": "[]",
            }
        return {
            "status": 200,
            "headers": [],
            "body": json.dumps(
                [
                    {
                        "number": 1,
                        "title": "t",
                        "labels": [{"name": "dispatched"}],
                        "html_url": "",
                        "updated_at": "",
                    }
                ]
            ),
        }

    wf.tool = fake_tool
    with pytest.raises(wf.ReadFailure) as excinfo:
        asyncio.run(wf._read_items(["eumemic/aios"]))
    assert "TRUNCATED" in str(excinfo.value)


# ---- C4 in the WORKFLOW form. ----


def test_C4_workflow_form_finds_a_transfer_from_the_SAME_REALIZABLE_world() -> None:
    """C4 in the mirror, on the world production can actually present.

    Same fixture shape as the library test: a run naming the OLD coordinates
    (company#71) and an enumeration that returns ONLY the new ones (ops#900). The old
    mirror fixture put a redirect on the TIMELINE of an ENUMERATED issue — a world that
    cannot occur, because after a transfer the old pair is never enumerated and the new
    pair resolves normally. The mirror must invert the probe exactly as the library does
    or the form that actually runs on cron keeps manufacturing false ZOMBIEs.
    """
    world = _github_world(
        issues={
            "eumemic/eumemic-ops": [
                {
                    "number": 900,
                    "title": "moved here from company#71",
                    "labels": [{"name": "autodev:in-progress"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ]
        },
        timelines={"eumemic/eumemic-ops/issues/900": []},
        moved={
            "eumemic/eumemic-company/issues/71": (
                "https://api.github.com/repos/eumemic/eumemic-ops/issues/900"
            )
        },
    )
    runs = [
        {
            "id": "run_stale",
            "status": "completed",
            "input": {"repo": "eumemic/eumemic-company", "issue_number": 71},
        }
    ]
    (wf, items, _, notes, transfers), paths = _wf_reader(world, ["eumemic/eumemic-ops"], runs)

    # The mirror asked about the STALE coordinates — the inverted probe, in the form
    # that runs on cron.
    assert "/repos/eumemic/eumemic-company/issues/71" in paths, (
        "the mirror never probed the stale join key, so a real transfer is invisible to it"
    )
    assert transfers == [
        {
            "stale_repo": "eumemic/eumemic-company",
            "stale_number": 71,
            "destination": "https://api.github.com/repos/eumemic/eumemic-ops/issues/900",
            "new_repo": "eumemic/eumemic-ops",
            "new_number": 900,
            "run_ids": ["run_stale"],
        }
    ]

    report = wf.build(items, True, runs, True, [], notes, transfers)
    assert report["verdict"] == "OK"
    assert report["counts"]["ZOMBIE"] == 0, "the mirror STILL turned a transfer into a zombie"
    assert report["counts"]["AMBIGUOUS"] == 1
    (d,) = report["disagreements"]
    assert d["classification"] == "AMBIGUOUS"
    assert (d["repo"], d["number"]) == ("eumemic/eumemic-ops", 900)
    assert "transferred-stale-join-key" in d["caveats"]
    assert "excluded-from-zombie-count" in d["caveats"]
    assert "eumemic/eumemic-company#71" in d["detail"]
    assert any("STALE JOIN KEY" in n for n in report["notes"])


def test_C4_both_forms_agree_on_the_REALIZABLE_transfer_world() -> None:
    """Parity on the C4 signal from two different transports, on the realizable world.

    urllib follows the 301 silently and reports `geturl()`; `http_request` does not
    follow and returns a literal 301 + Location. The transports differ by construction —
    the stale-key EVIDENCE, the notes, the classifications and the HASH must not.
    """
    world = _github_world(
        issues={
            "eumemic/eumemic-ops": [
                {
                    "number": 900,
                    "title": "moved",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ],
            "eumemic/eumemic-company": [],
        },
        timelines={"eumemic/eumemic-ops/issues/900": []},
        moved={
            "eumemic/eumemic-company/issues/71": (
                "https://api.github.com/repos/eumemic/eumemic-ops/issues/900"
            )
        },
    )
    runs = [
        {
            "id": "run_stale",
            "status": "completed",
            "input": {"repo": "eumemic/eumemic-company", "issue_number": 71},
        }
    ]
    lib_report, wf_report = _assert_forms_agree(
        world, ["eumemic/eumemic-ops", "eumemic/eumemic-company"], runs
    )
    assert any("STALE JOIN KEY" in n for n in wf_report["notes"])
    assert wf_report["counts"]["ZOMBIE"] == 0
    assert wf_report["counts"]["AMBIGUOUS"] == 1
    assert (lib_report.counts or {})["AMBIGUOUS"] == 1


def test_C4_mirror_stale_key_that_resolves_to_itself_is_not_a_transfer() -> None:
    """A stale key belonging to a merely-CLOSED issue must not read as a transfer."""
    world = _github_world(
        issues={
            "eumemic/aios": [
                {
                    "number": 2000,
                    "title": "z",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ]
        },
        timelines={"eumemic/aios/issues/2000": []},
    )
    runs = [
        {"id": "r_closed", "status": "errored", "input": {"repo": "eumemic/aios", "issue": 111}}
    ]
    _lib_report, wf_report = _assert_forms_agree(world, ["eumemic/aios"], runs)
    assert wf_report["counts"]["ZOMBIE"] == 1, "a real zombie must stay a zombie in the mirror"
    assert wf_report["counts"]["AMBIGUOUS"] == 0


def test_C4_mirror_probe_read_error_is_a_NOTE_not_an_ALARM() -> None:
    """A 404 on a stale coordinate is history, not an outage — but it is never silent."""
    wf = _load_wf_module()

    async def fake_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": 404, "headers": [], "body": "{}"}

    wf.tool = fake_tool
    transfers, notes = asyncio.run(
        wf._probe_stale_keys(
            [],
            [
                {
                    "id": "r_x",
                    "status": "completed",
                    "input": {"repo": "eumemic/aios", "issue": 4242},
                }
            ],
        )
    )
    assert transfers == []
    assert any("404" in n and "eumemic/aios#4242" in n for n in notes)


def test_C4_mirror_probe_cap_is_reported_as_a_gap() -> None:
    """Hitting the ceiling says how many keys went UNPROBED, in the mirror too."""
    wf = _load_wf_module()

    async def fake_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        path = args["path"]
        return {
            "status": 200,
            "headers": [],
            "body": json.dumps({"number": 1, "url": f"https://api.github.com{path}"}),
        }

    wf.tool = fake_tool
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(wf, "MAX_TRANSFER_PROBES", 3)
        runs = [
            {"id": f"r{n}", "status": "completed", "input": {"repo": "eumemic/aios", "issue": n}}
            for n in range(10, 20)
        ]
        transfers, notes = asyncio.run(wf._probe_stale_keys([], runs))
    finally:
        monkey.undo()
    assert transfers == []
    assert any("capped at 3 of 10" in n and "NOT probed" in n for n in notes)


def test_C4_a_redirect_without_a_location_is_a_read_failure_not_a_guess() -> None:
    """A redirect with nowhere to go cannot be named. Refuse rather than invent."""
    wf = _load_wf_module()

    async def fake_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": 301, "headers": [], "body": ""}

    wf.tool = fake_tool
    with pytest.raises(wf.ReadFailure) as excinfo:
        asyncio.run(wf._read_items(["eumemic/aios"]))
    assert "no Location" in str(excinfo.value)


def test_C4_a_non_redirect_error_status_still_ALARMS() -> None:
    """Making 3xx a signal must not make 4xx/5xx one. A 404 is still a FAILED read."""
    wf = _load_wf_module()

    async def fake_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": 404, "headers": [], "body": "{}"}

    wf.tool = fake_tool
    with pytest.raises(wf.ReadFailure) as excinfo:
        asyncio.run(wf._read_items(["eumemic/aios"]))
    assert "404" in str(excinfo.value)


# ---- the CAPABILITY guard: present in one form, absent in the other ⇒ FAIL. ----


#: Every read-path capability that must exist in BOTH forms, with the probe that
#: proves it. The previous drift guard compared only `classify()`/`build()`, so a
#: capability could live in the library alone and every test still passed — which is
#: exactly how C2 and C4 shipped. Adding a capability to one form now requires
#: adding it to the other, or this table fails the build.
_READ_CAPABILITIES: list[tuple[str, str]] = [
    ("linked-PR enrichment (C2)", "linked_pr_numbers"),
    ("transfer/redirect detection (C4)", "transfer_note"),
]


@pytest.mark.parametrize(("capability", "marker"), _READ_CAPABILITIES)
def test_every_read_capability_exists_in_BOTH_forms(capability: str, marker: str) -> None:
    """Structural backstop to the behavioural parity guard above.

    Behaviour is the real test; this catches the case where someone deletes a
    capability from the mirror and also deletes the fixture that exercised it — a
    green build for a mirror that can no longer see what the library sees.
    """
    wf_source = Path("infra/workflows/work-state-reconciler.wf.py").read_text()
    cli_source = Path("src/aios/reconcilers/work_state_cli.py").read_text()
    lib_source = Path("src/aios/reconcilers/work_state.py").read_text()
    assert marker in wf_source, f"{capability} is MISSING from the durable workflow mirror"
    assert marker in cli_source or marker in lib_source, (
        f"{capability} is MISSING from the library core"
    )


def test_the_workflow_mirror_can_REACH_every_class_from_fetched_data() -> None:
    """Reachability, not just correctness: every class must be producible by the
    workflow's own READ PATH.

    AMBIGUOUS was unreachable in the mirror — the classifier had a branch for it that
    no production input could ever satisfy. A class that only a test can reach is a
    class that does not ship, so reachability is asserted from the reader outward.
    """
    world = _github_world(
        issues={
            "eumemic/aios": [
                # ZOMBIE: asserts in flight, no run, timeline shows no PR.
                {
                    "number": 1,
                    "title": "z",
                    "labels": [{"name": "autodev:in-progress"}],
                    "html_url": "",
                    "updated_at": "",
                },
                # AMBIGUOUS: same, but the timeline HAS a PR.
                {
                    "number": 2,
                    "title": "a",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                },
                # DEAD / MISLABELLED / LAGGING come from the run side.
                {
                    "number": 3,
                    "title": "d",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                },
                {
                    "number": 4,
                    "title": "m",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                },
                {"number": 5, "title": "l", "labels": [], "html_url": "", "updated_at": ""},
            ]
        },
        timelines={
            "eumemic/aios/issues/1": [],
            "eumemic/aios/issues/2": [
                {
                    "event": "cross-referenced",
                    "source": {"issue": {"number": 99, "pull_request": {}}},
                }
            ],
            "eumemic/aios/issues/3": [],
            "eumemic/aios/issues/4": [],
        },
    )
    (wf, items, _, notes, _t), _ = _wf_reader(world, ["eumemic/aios"])
    runs = [
        {"id": "r3", "status": "errored", "input": {"repo": "eumemic/aios", "issue_number": 3}},
        {"id": "r4", "status": "suspended", "input": {"repo": "eumemic/aios", "issue_number": 4}},
        {"id": "r5", "status": "running", "input": {"repo": "eumemic/aios", "issue_number": 5}},
    ]
    report = wf.build(items, True, runs, True, [], notes)
    assert report["counts"] == {
        "ZOMBIE": 1,
        "AMBIGUOUS": 1,
        "DEAD": 1,
        "MISLABELLED": 1,
        "LAGGING": 1,
    }, "a class the workflow's own read path cannot reach is a class that does not ship"


def test_workflow_read_path_is_OBSERVE_ONLY_end_to_end() -> None:
    """Gate 1, asserted on the READ PATH rather than by grepping the source.

    Every request the mirror issues while reading a full world — issue pages AND the
    new timeline fetches — must be a GET with no body.
    """
    wf = _load_wf_module()
    verbs: list[str] = []

    async def fake_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        verbs.append(args.get("method", ""))
        assert "body" not in args
        if "/timeline" in args["path"]:
            return {"status": 200, "headers": [], "body": "[]"}
        return {
            "status": 200,
            "headers": [],
            "body": json.dumps(
                [
                    {
                        "number": 1,
                        "title": "t",
                        "labels": [{"name": "dispatched"}],
                        "html_url": "",
                        "updated_at": "",
                    }
                ]
            ),
        }

    wf.tool = fake_tool
    asyncio.run(wf._read_items(["eumemic/aios"]))
    assert verbs and set(verbs) == {"GET"}
    with pytest.raises(wf.ReadFailure):
        asyncio.run(wf._gh("/repos/eumemic/aios/issues/1", method="PATCH"))
