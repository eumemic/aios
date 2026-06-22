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


def _failure_step(doc: dict[Any, Any]) -> dict[str, Any]:
    """The step whose ``run`` body contains the incident title — i.e. the
    failure-path step that opens/updates the incident issue."""
    steps = doc["jobs"]["canary"]["steps"]
    return next(s for s in steps if "run" in s and _ISSUE_TITLE in s["run"])


def _failure_step_run(doc: dict[Any, Any]) -> str:
    """The ``run`` string of the failure-path step."""
    run = _failure_step(doc)["run"]
    assert isinstance(run, str)
    return run


def _logical_commands(run: str) -> list[str]:
    """Join shell line-continuations and drop comment lines, yielding one
    string per logical command."""
    joined = run.replace("\\\n", " ")
    return [
        line.strip()
        for line in joined.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


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
    assert there is exactly one pytest step and the set of test paths it names
    is precisely ``{tests/unit/test_migration_chain.py}`` — nothing more.

    This exact-token-set check fails for end-of-line ``tests/unit`` (no trailing
    whitespace), for the whole suite, and for any extra test file added to the
    run.
    """
    pytest_steps = [r for r in _run_steps(_doc()) if "pytest" in r]
    assert len(pytest_steps) == 1, f"expected exactly one pytest step, found {len(pytest_steps)}"
    test_paths = set(re.findall(r"tests/\S+", pytest_steps[0]))
    assert test_paths == {_CHAIN_TEST_PATH}, (
        f"canary must run ONLY the chain test; found test paths {test_paths}"
    )


def test_issue_title_constant_present() -> None:
    """The exact incident issue title appears in a run step."""
    run_steps = _run_steps(_doc())
    assert any(_ISSUE_TITLE in run for run in run_steps), (
        f"{_WF} must reference the incident title {_ISSUE_TITLE!r} in a run step so the "
        f"opened issue is recognizable and de-duplicated."
    )


def test_incident_issue_is_created_with_the_incident_label() -> None:
    """The incident is labeled, and the label is created idempotently first.

    Pinning ``--label incident`` on the ``gh issue create`` command specifically
    (not merely somewhere in the run) is load-bearing: ``--label incident`` also
    appears in a comment and in the ``gh issue list --label incident`` lookup, so
    a future edit that drops it from ``gh issue create`` alone would leave new
    issues unlabeled — and the label-based lookup would then never find them,
    spawning a fresh duplicate on every failure.
    """
    doc = _doc()
    fail_run = _failure_step_run(doc)  # the run of the step whose body contains TITLE
    commands = _logical_commands(fail_run)

    # The label must be created idempotently before use.
    assert any(c.startswith("gh label create incident") for c in commands), (
        "failure step must idempotently create the incident label"
    )

    # The CREATE command specifically must carry --label incident, else new
    # issues are unlabeled and the label-based lookup spawns duplicates.
    create_cmds = [c for c in commands if c.startswith("gh issue create")]
    assert len(create_cmds) == 1, f"expected one `gh issue create`, found {len(create_cmds)}"
    assert "--label incident" in create_cmds[0], "gh issue create must apply the incident label"


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
    """The failure step wires run id and commit sha into its ``env:`` mapping.

    Structural (not text-substring): the failure step is the step whose ``run``
    contains the issue title. Asserting against its ``env:`` mapping can't be
    satisfied by the top-of-file security comment that merely mentions these
    contexts — the env wiring must actually be present.
    """
    doc = _doc()
    fail_step = _failure_step(doc)
    env = fail_step.get("env", {})
    assert "github.run_id" in env.get("RUN_URL", ""), (
        "failure step must wire RUN_URL to github.run_id"
    )
    assert "github.sha" in env.get("COMMIT_SHA", ""), (
        "failure step must wire COMMIT_SHA to github.sha"
    )


def test_concurrency_does_not_cancel_in_progress() -> None:
    """``cancel-in-progress`` must stay false (load-bearing per the workflow).

    PyYAML parses the unquoted ``false`` as Python ``False``.
    """
    doc = _doc()
    assert doc["concurrency"]["cancel-in-progress"] is False, (
        "cancel-in-progress must stay false so a later push can't cancel an "
        "in-flight canary mid-issue-creation"
    )
    assert doc["concurrency"]["group"] == "${{ github.workflow }}-${{ github.ref }}", (
        "concurrency group must serialize per-ref (workflow + ref), else "
        "cancel-in-progress:false no longer guarantees one canary per ref"
    )
