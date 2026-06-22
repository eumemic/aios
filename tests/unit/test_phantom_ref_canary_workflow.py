"""Load-bearing-property checks for the phantom-ref canary workflow.

``.github/workflows/phantom-ref-canary.yml`` is a scheduled guard: weekly it runs
the offline scanner (``src/aios/_drift/forward_refs.py``), checks each referenced
issue's live GitHub state, and files a drift issue for the closed/missing ones.
This module pins the canary's load-bearing properties so a careless future edit
can't silently disable or weaken it — e.g. dropping the schedule, defaulting an
API error to "open" (and dropping a real phantom), appending instead of replacing
the issue body, auto-closing it, or flagging PR-typed refs.

Pure-Python: parses the workflow YAML with PyYAML; no DB, no Docker, no network.

PyYAML gotcha: the bare mapping key ``on:`` parses as the boolean ``True`` (the
"Norway problem"), so the trigger mapping is resolved via ``doc.get(True)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # types-PyYAML not in the dep set

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WF = _REPO_ROOT / ".github" / "workflows" / "phantom-ref-canary.yml"

_ISSUE_TITLE = "Phantom issue references detected on master"
_SCANNER_PATH = "src/aios/_drift/forward_refs.py"


def _doc() -> dict[Any, Any]:
    # Keys are `str`, except `on:` which PyYAML parses as the bool `True` (the
    # Norway problem) — hence an `Any`-keyed mapping rather than `dict[str, ...]`.
    doc: dict[Any, Any] = yaml.safe_load(_WF.read_text())
    return doc


def _triggers(doc: dict[Any, Any]) -> dict[str, Any]:
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict), f"{_WF} has no `on:` trigger mapping; resolved {triggers!r}."
    return triggers


def _canary_run(doc: dict[Any, Any]) -> str:
    """The single ``run:`` step body (the scan + check + file-drift-issue logic)."""
    runs = [s["run"] for s in doc["jobs"]["canary"]["steps"] if "run" in s]
    assert len(runs) == 1, f"expected exactly one run step in the canary job, found {len(runs)}"
    run = runs[0]
    assert isinstance(run, str)
    return run


def test_workflow_parses_to_nonempty_dict() -> None:
    assert _WF.exists(), (
        f"{_WF} is missing; the phantom-ref canary must exist so stale forward "
        f"references are caught on a schedule."
    )
    doc = _doc()
    assert isinstance(doc, dict) and doc, f"{_WF} did not parse to a non-empty dict."


def test_runs_weekly_on_an_off_minute_cron() -> None:
    """A scheduled cron drives it, at an off-minute (not :00/:30).

    The state of a referenced issue changes outside any PR, so the verdict must be
    re-evaluated on a clock. Off-minute avoids the top-of-hour scheduler stampede.
    """
    schedule = _triggers(_doc())["schedule"]
    crons = [entry["cron"] for entry in schedule]
    assert crons, f"{_WF} on.schedule must define at least one cron; found {schedule!r}."
    for cron in crons:
        minute = cron.split()[0]
        assert minute not in {"0", "30"}, (
            f"{_WF} cron {cron!r} fires on a stampede minute; pick an off-minute (not :00/:30)."
        )


def test_supports_manual_dispatch() -> None:
    """``workflow_dispatch`` so the canary can be run on demand (and dry-run once merged)."""
    assert "workflow_dispatch" in _triggers(_doc()), (
        f"{_WF} must allow workflow_dispatch for manual/dry-run invocation."
    )


def test_permissions_are_minimal_and_sufficient() -> None:
    perms = _doc()["permissions"]
    assert perms["issues"] == "write", (
        f"{_WF} permissions.issues must be 'write' so it can file the drift issue; "
        f"found {perms.get('issues')!r}."
    )
    assert perms["contents"] == "read", (
        f"{_WF} permissions.contents must be 'read'; found {perms.get('contents')!r}."
    )


def test_concurrency_does_not_cancel_in_progress() -> None:
    """``cancel-in-progress`` must stay false so a later run can't abort one
    mid-issue-write. PyYAML parses the unquoted ``false`` as Python ``False``."""
    assert _doc()["concurrency"]["cancel-in-progress"] is False, (
        "cancel-in-progress must stay false so a scheduled run can't cancel an "
        "in-flight canary mid drift-issue write."
    )


def test_runs_the_offline_scanner() -> None:
    """The canary's verdict is driven by the committed offline scanner, not an
    ad-hoc inline grep that could drift from the fixture-tested patterns."""
    assert _SCANNER_PATH in _canary_run(_doc()), (
        f"{_WF} must invoke the offline scanner {_SCANNER_PATH!r} so its verdict tracks "
        f"the fixture-tested patterns."
    )


def test_hard_fails_on_non_404_api_error() -> None:
    """A non-404 API error must abort the run, never default the issue state.

    Defaulting a rate-limit/5xx to "open" would silently drop a real phantom; the
    only error that maps to a verdict is a clean 404 ('missing'). Pin both: an
    explicit hard-fail path AND a 404→missing branch.
    """
    run = _canary_run(_doc())
    assert "gh api" in run, f"{_WF} must query issue state via `gh api`."
    assert "not a 404" in run and "exit 1" in run, (
        f"{_WF} must hard-fail (exit 1) on an API error that is not a 404, rather than "
        f"defaulting the issue state."
    )
    assert "missing" in run, f"{_WF} must map a clean 404 to a 'missing' verdict."


def test_skips_pull_request_refs() -> None:
    """A forward ref to a PR (a merged PR is the expected end state) must be skipped,
    not flagged — pinned via the ``is_pr`` filter on the issues API ``pull_request``."""
    run = _canary_run(_doc())
    assert "pull_request" in run and "is_pr" in run, (
        f"{_WF} must distinguish PR-typed refs (via the issues API `pull_request` field) "
        f"and skip them."
    )


def test_drift_issue_body_is_replaced_not_appended() -> None:
    """The issue body is REPLACED each run (``gh issue edit --body``), never appended
    (``gh issue comment``), so a deferred phantom doesn't pile a weekly comment."""
    run = _canary_run(_doc())
    assert "gh issue edit" in run and "--body" in run, (
        f"{_WF} must update the existing drift issue with `gh issue edit --body` (replace)."
    )
    assert "gh issue comment" not in run, (
        f"{_WF} must NOT append via `gh issue comment` — the body is a replacement so a "
        f"standing phantom doesn't accrete weekly comments."
    )


def test_drift_issue_is_not_auto_closed() -> None:
    """No auto-close: a clean run is silent, but a previously-filed issue is left for a
    human to close (auto-close-on-clean flaps when a referenced issue is reopened)."""
    assert "gh issue close" not in _canary_run(_doc()), (
        f"{_WF} must never auto-close the drift issue; a clean run exits silently and "
        f"leaves any standing issue for a human."
    )


def test_drift_issue_title_and_dedicated_label() -> None:
    """The stable title is present, and the issue is filed under the dedicated
    non-incident ``drift`` label (a phantom ref is hygiene, not a production outage),
    created idempotently before use."""
    run = _canary_run(_doc())
    assert _ISSUE_TITLE in run, (
        f"{_WF} must reference the stable title {_ISSUE_TITLE!r} so reruns de-duplicate."
    )
    assert "gh label create drift" in run, f"{_WF} must create the `drift` label idempotently."
    assert "gh issue create" in run and "--label drift" in run, (
        f"{_WF} must file the issue under the dedicated `drift` label, not the incident queue."
    )
