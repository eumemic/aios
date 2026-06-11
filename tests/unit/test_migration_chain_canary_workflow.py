"""Load-bearing-property checks for the migration-chain canary workflow.

``.github/workflows/migration-chain-canary.yml`` is a post-merge guard: it
runs ``tests/unit/test_migration_chain.py`` on every migration-touching push to
``master`` and opens an incident issue if the alembic chain broke (see the
issue title constant below). This module pins the canary's load-bearing
properties so a careless future edit can't silently disable it — e.g. dropping
the ``master`` trigger, broadening the run to the whole suite (coupling the
verdict to unrelated flakes), or removing the failure-path issue creation.

Pure-Python: parses the workflow YAML with PyYAML; no DB, no Docker, no CI.

PyYAML gotcha: the bare mapping key ``on:`` parses as the boolean ``True`` (the
"Norway problem"), so the trigger mapping is resolved via ``doc.get(True)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # types-PyYAML not in the dep set

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WF = _REPO_ROOT / ".github" / "workflows" / "migration-chain-canary.yml"

_ISSUE_TITLE = "ALEMBIC CHAIN BROKEN on master"
_CHAIN_TEST_PATH = "tests/unit/test_migration_chain.py"


def _doc() -> dict[Any, Any]:
    # Keys are `str`, except `on:` which PyYAML parses as the bool `True` (the
    # Norway problem) — hence an `Any`-keyed mapping rather than `dict[str, ...]`.
    doc: dict[Any, Any] = yaml.safe_load(_WF.read_text())
    return doc


def _triggers(doc: dict[Any, Any]) -> dict[str, Any]:
    # Resolve the trigger mapping under either the string key or the bool True.
    # A missing mapping fails the calling test loudly when it indexes the result.
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict), f"{_WF} has no `on:` trigger mapping; resolved {triggers!r}."
    return triggers


def _run_steps(doc: dict[Any, Any]) -> list[str]:
    return [s["run"] for s in doc["jobs"]["canary"]["steps"] if "run" in s]


def test_workflow_parses_to_nonempty_dict() -> None:
    """The canary workflow exists and parses to a non-empty mapping."""
    assert _WF.exists(), (
        f"{_WF} is missing; the migration-chain canary workflow must exist so a "
        f"broken alembic chain on master is caught post-merge."
    )
    doc = _doc()
    assert isinstance(doc, dict) and doc, f"{_WF} did not parse to a non-empty dict."


def test_triggers_on_push_to_master() -> None:
    """The canary must run on pushes to master (post-merge verdict)."""
    triggers = _triggers(_doc())
    branches = triggers["push"]["branches"]
    assert "master" in branches, (
        f"{_WF} on.push.branches must include 'master' so the chain test runs after "
        f"every merge; found {branches!r}."
    )


def test_paths_filter_includes_migrations() -> None:
    """The trigger paths filter must include 'migrations/**'."""
    paths = _triggers(_doc())["push"]["paths"]
    assert "migrations/**" in paths, (
        f"{_WF} on.push.paths must include 'migrations/**' so a migration merge fires "
        f"the canary; found {paths!r}."
    )


def test_paths_filter_includes_test_and_self() -> None:
    """The paths filter must include the chain test AND the canary workflow itself.

    A change to either the chain test or this workflow must re-run the canary so
    the guard can't be quietly weakened without an immediate verdict.
    """
    paths = _triggers(_doc())["push"]["paths"]
    assert _CHAIN_TEST_PATH in paths, (
        f"{_WF} on.push.paths must include {_CHAIN_TEST_PATH!r}; found {paths!r}."
    )
    self_path = ".github/workflows/migration-chain-canary.yml"
    assert self_path in paths, (
        f"{_WF} on.push.paths must include {self_path!r} so edits to the canary re-run it; "
        f"found {paths!r}."
    )


def test_permissions_are_minimal_and_sufficient() -> None:
    """Permissions: issues:write (to open the incident) and contents:read."""
    doc = _doc()
    perms = doc["permissions"]
    assert perms["issues"] == "write", (
        f"{_WF} permissions.issues must be 'write' so the failure path can open the "
        f"incident issue; found {perms.get('issues')!r}."
    )
    assert perms["contents"] == "read", (
        f"{_WF} permissions.contents must be 'read'; found {perms.get('contents')!r}."
    )


def test_runs_only_the_chain_test_not_the_suite() -> None:
    """Exactly the chain test runs — never the whole ``tests/unit`` suite.

    Mutation-resistant core: broadening the run to ``pytest tests/unit`` would
    couple the canary's verdict to unrelated test flakes and slow it down. We
    assert (a) some step runs the specific chain-test file, and (b) NO run step
    invokes the suite broadly (``pytest tests/unit`` followed by whitespace).

    The suite-level regex ``pytest\\s+tests/unit\\s`` requires whitespace
    immediately after ``unit``; the chain-test path has ``/`` there, so it does
    not match ``pytest tests/unit/test_migration_chain.py``.
    """
    run_steps = _run_steps(_doc())
    assert any(f"pytest {_CHAIN_TEST_PATH}" in run for run in run_steps), (
        f"{_WF} must have a step running 'pytest {_CHAIN_TEST_PATH}'; run steps were {run_steps!r}."
    )
    suite_re = re.compile(r"pytest\s+tests/unit\s")
    offenders = [run for run in run_steps if suite_re.search(run)]
    assert not offenders, (
        f"{_WF} must NOT run the whole tests/unit suite (couples the canary to unrelated "
        f"flakes); offending run step(s): {offenders!r}."
    )


def test_issue_title_constant_present() -> None:
    """The exact incident issue title appears in a run step."""
    run_steps = _run_steps(_doc())
    assert any(_ISSUE_TITLE in run for run in run_steps), (
        f"{_WF} must reference the incident title {_ISSUE_TITLE!r} in a run step so the "
        f"opened issue is recognizable and de-duplicated."
    )


def test_incident_label_used_and_created() -> None:
    """The incident is labeled, and the label is created idempotently first."""
    run_steps = _run_steps(_doc())
    assert any("--label incident" in run for run in run_steps), (
        f"{_WF} must pass '--label incident' when creating the issue; run steps were {run_steps!r}."
    )
    assert any("gh label create incident" in run for run in run_steps), (
        f"{_WF} must 'gh label create incident' (idempotent guard) so issue creation "
        f"can't fail on a missing label; run steps were {run_steps!r}."
    )


def test_issue_step_runs_only_on_failure() -> None:
    """The step that creates the incident must be gated on ``failure()``."""
    doc = _doc()
    steps = doc["jobs"]["canary"]["steps"]
    title_steps = [s for s in steps if "run" in s and _ISSUE_TITLE in s["run"]]
    assert title_steps, f"{_WF} has no step whose run contains the incident title {_ISSUE_TITLE!r}."
    for step in title_steps:
        assert step.get("if") == "failure()", (
            f"{_WF}: the incident-creating step must have if: failure() (else it runs on "
            f"green and spams issues); found if={step.get('if')!r}."
        )


def test_incident_body_is_actionable() -> None:
    """The workflow references the run id and commit sha for an actionable issue."""
    text = _WF.read_text()
    assert "github.run_id" in text, (
        f"{_WF} must reference github.run_id so the incident issue links the failing run."
    )
    assert "github.sha" in text, (
        f"{_WF} must reference github.sha so the incident issue names the offending commit."
    )
