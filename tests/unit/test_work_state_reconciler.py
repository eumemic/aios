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
    WorkItem,
    build_report,
    classify_item,
    index_runs,
    is_pipeline_state_label,
    normalise_repo,
    render_json,
    render_markdown,
    run_record_from_payload,
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


def test_C4_a_redirect_on_the_issue_read_is_surfaced_as_a_transfer() -> None:
    """68 issues were transferred between repos on 2026-07-25. A transfer gives the issue
    a NEW (repo, number) while every run.input still holds the OLD one, so the join key
    is stale and the item reads as a false ZOMBIE. urllib follows the 301 silently, so the
    transport reports where it actually landed and the report says so."""
    from aios.reconcilers.work_state_cli import _FINAL_URL_HEADER

    def getter(url: str, headers: Any) -> Any:
        if "/issues?" in url or url.endswith("/issues"):
            return [
                {
                    "number": 71,
                    "title": "moved",
                    "labels": [{"name": "dispatched"}],
                    "html_url": "",
                    "updated_at": "",
                }
            ], {}
        return [], {
            _FINAL_URL_HEADER: "https://api.github.com/repos/eumemic/eumemic-ops/issues/900"
        }

    source = read_github_items(
        repos=["eumemic/eumemic-company"], token="t", getter=getter, enrich_linked_prs=True
    )
    assert isinstance(source, SourceOk)
    assert any("TRANSFERRED" in n for n in source.notes)
    assert any("eumemic-ops/issues/900" in n for n in source.notes)
    report = build_report(items_read=source, runs_read=SourceOk(name="aios-runs", items=()))
    assert any("TRANSFERRED" in n for n in report.notes)
    assert "TRANSFERRED" in render_markdown(report)
    assert any("TRANSFERRED" in n for n in json.loads(render_json(report))["notes"])


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
