"""Unit tests for the triage-pipeline deploy-surface export (aios#1226).

``build_triage_pipeline_script`` returns only the workflow *script string*, but a deployed
workflow ALSO needs its tool + http_server surface declared on the ``WorkflowCreate`` —
otherwise the first ``tool('http_request')`` call errors with "tool 'http_request' is not in
the workflow's declared tools" (run_tools.py).

This module pins the exported deploy surface so it can't drift from the script that needs it:

- ``REQUIRED_TOOLS`` — ``http_request`` (the script's own tool) plus the read-family tools
  (``read``/``glob``/``grep``) the classifier child needs (the child surface is
  ``agent ∩ run``). Triage never mutates a working tree, so it does NOT declare bash/write/edit
  — a strictly narrower surface than dev_pipeline's.
- ``REQUIRED_HTTP_SERVERS`` — the ``github`` spec with a single ``/repos/**`` route scoped to
  ``GET,POST`` (read issues, apply labels, post comments — no DELETE/PUT/PATCH) with
  ``allow_query`` for issue-list / comment-thread pagination.
- ``build_triage_pipeline_workflow_create(...)`` — the complete ``WorkflowCreate`` payload.
"""

from __future__ import annotations

from aios.models.agents import HttpServerSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.triage_pipeline import (
    REQUIRED_HTTP_SERVERS,
    REQUIRED_TOOLS,
    TRIAGE_CLASSES,
    TRIAGE_STATE_LABELS,
    build_triage_pipeline_script,
    build_triage_pipeline_workflow_create,
)

# ─── REQUIRED_TOOLS ────────────────────────────────────────────────────────────


def test_required_tools_is_http_request_plus_read_family() -> None:
    got = {t.type for t in REQUIRED_TOOLS}
    assert got == {"http_request", "read", "glob", "grep"}


def test_required_tools_includes_http_request() -> None:
    assert "http_request" in {t.type for t in REQUIRED_TOOLS}


def test_required_tools_excludes_mutating_tools() -> None:
    # Triage has no clone/edit step — it must NOT carry bash/write/edit (least privilege).
    got = {t.type for t in REQUIRED_TOOLS}
    for forbidden in ("bash", "write", "edit"):
        assert forbidden not in got, f"{forbidden} should not be in the triage surface"


def test_required_tools_have_no_duplicates() -> None:
    types = [t.type for t in REQUIRED_TOOLS]
    assert len(types) == len(set(types))


# ─── REQUIRED_HTTP_SERVERS ─────────────────────────────────────────────────────


def test_single_github_server() -> None:
    assert len(REQUIRED_HTTP_SERVERS) == 1
    server = REQUIRED_HTTP_SERVERS[0]
    assert server.name == "github"
    assert server.base_url == "https://api.github.com"


def test_repos_route_is_get_post_only() -> None:
    server = REQUIRED_HTTP_SERVERS[0]
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert len(repos) == 1
    # Triage only reads issues (GET) and applies labels / posts comments (POST). No DELETE,
    # PUT, or PATCH — strictly narrower than dev_pipeline's full verb set.
    assert set(repos[0].methods or []) == {"GET", "POST"}


def test_repos_route_allows_query_for_pagination() -> None:
    server = REQUIRED_HTTP_SERVERS[0]
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert repos[0].allow_query is True


def test_single_repos_route_only() -> None:
    # No /graphql route — triage uses no GraphQL mutation (unlike dev_pipeline's mark-ready).
    server = REQUIRED_HTTP_SERVERS[0]
    assert {r.path_pattern for r in server.routes} == {"/repos/**"}


# ─── build_triage_pipeline_workflow_create ──────────────────────────────────────


def test_workflow_create_is_valid() -> None:
    wc = build_triage_pipeline_workflow_create(name="triage-pipeline")
    assert isinstance(wc, WorkflowCreate)
    assert wc.name == "triage-pipeline"


def test_workflow_create_carries_the_script() -> None:
    wc = build_triage_pipeline_workflow_create(name="triage-pipeline")
    assert wc.script == build_triage_pipeline_script()
    assert "tool(" in wc.script


def test_workflow_create_carries_required_tools() -> None:
    wc = build_triage_pipeline_workflow_create(name="triage-pipeline")
    assert {t.type for t in wc.tools} == {t.type for t in REQUIRED_TOOLS}


def test_workflow_create_carries_required_http_servers() -> None:
    wc = build_triage_pipeline_workflow_create(name="triage-pipeline")
    assert len(wc.http_servers) == 1
    server = wc.http_servers[0]
    assert isinstance(server, HttpServerSpec)
    assert server.name == "github"
    assert {r.path_pattern for r in server.routes} == {"/repos/**"}


def test_workflow_create_forwards_build_kwargs() -> None:
    wc = build_triage_pipeline_workflow_create(name="triage-pipeline", max_issues_per_run=7)
    assert wc.script == build_triage_pipeline_script(max_issues_per_run=7)


def test_workflow_create_github_server_name_matches_script() -> None:
    wc = build_triage_pipeline_workflow_create(name="triage-pipeline", github_server="gh")
    assert any(isinstance(s, HttpServerSpec) and s.name == "gh" for s in wc.http_servers)
    assert "'gh'" in wc.script or '"gh"' in wc.script


# ─── the triage state-label / class invariants (the two-axis model) ─────────────


def test_state_labels_are_the_five_intake_labels() -> None:
    assert set(TRIAGE_STATE_LABELS) == {
        "needs-design",
        "shovel-ready",
        "needs-decision",
        "approved",
        "dispatched",
    }


def test_classes_are_exactly_the_three_spec_readiness_classes() -> None:
    assert set(TRIAGE_CLASSES) == {"shovel-ready", "needs-design", "needs-decision"}


def test_triage_never_sets_approved() -> None:
    # Triage only sets the spec-readiness axis; approval is a separate chairman gate.
    assert "approved" not in TRIAGE_CLASSES


def test_script_constant_excludes_datetime_import() -> None:
    # Restricted import allowlist: no datetime/time (a timestamp must be passed via input).
    script = build_triage_pipeline_script()
    assert "import datetime" not in script
    assert "import time" not in script
