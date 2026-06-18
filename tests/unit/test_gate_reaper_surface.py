"""Unit tests for the standing-reaper deploy-surface export (aios#1386 / #1135).

``build_reaper_script`` returns only the workflow *script string*, but a deployed
workflow ALSO needs its tool + http_server surface declared on the ``WorkflowCreate``.
This module pins the exported deploy surface so it can't drift from the script that needs
it — AND pins the issue's load-bearing structural rule: the reaper can NEVER auto-resolve
an approval gate, because ``resume_gate`` / ``gate`` are absent from its surface and the
GitHub route grants only GET/POST (no merge/close/unlabel).
"""

from __future__ import annotations

from aios.models.workflows import WorkflowCreate
from aios.workflows.gate_reaper import (
    REQUIRED_HTTP_SERVERS,
    REQUIRED_TOOLS,
    build_reaper_script,
    build_reaper_workflow_create,
)
from aios.workflows.script_validation import validate_workflow_script


def test_required_tools_are_exactly_list_runs_and_http_request() -> None:
    # The reaper reads the run journal (list_runs) and reads/mutates GitHub
    # (http_request) — nothing else. No bash/read/write/edit, and CRUCIALLY no
    # resume_gate/gate: it CANNOT auto-resolve a gate, only escalate.
    assert {t.type for t in REQUIRED_TOOLS} == {"list_runs", "http_request"}


def test_surface_cannot_resolve_a_gate() -> None:
    types = {t.type for t in REQUIRED_TOOLS}
    assert "resume_gate" not in types
    assert "gate" not in types


def test_required_tools_have_no_duplicates() -> None:
    types = [t.type for t in REQUIRED_TOOLS]
    assert len(types) == len(set(types))


def test_github_server_is_get_post_only() -> None:
    assert len(REQUIRED_HTTP_SERVERS) == 1
    server = REQUIRED_HTTP_SERVERS[0]
    assert server.name == "github"
    assert server.base_url == "https://api.github.com"
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert len(repos) == 1
    methods = repos[0].methods or []
    # GET (list issues/PRs, read mergeable_state, read a thread) + POST (escalation
    # comment + label). NO PUT/PATCH/DELETE: the reaper never merges, closes, or unlabels.
    assert set(methods) == {"GET", "POST"}
    for forbidden in ("PUT", "PATCH", "DELETE"):
        assert forbidden not in methods
    # list reads paginate via query — the route must opt into allow_query so a late
    # zombie is never silently dropped past page 1.
    assert repos[0].allow_query is True


def test_build_workflow_create_bundles_script_and_surface() -> None:
    wc = build_reaper_workflow_create()
    assert isinstance(wc, WorkflowCreate)
    assert wc.name == "gate_reaper"
    assert {t.type for t in wc.tools} == {"list_runs", "http_request"}
    assert len(wc.http_servers) == 1
    assert wc.http_servers[0].name == "github"
    assert wc.script == build_reaper_script()


def test_script_validates_against_its_declared_surface() -> None:
    # The #1285 create-time validator: compiles, defines async def main(input), and
    # REQUIRED_TOOLS is a superset of the script's literal tool() names.
    validate_workflow_script(script=build_reaper_script(), tools=list(REQUIRED_TOOLS))


def test_script_imports_no_clock() -> None:
    # The only "now" the reaper may read is the frozen trigger.fired_at (replay-stable).
    # A datetime/time import would let a wall clock leak in and desync replay.
    src = build_reaper_script()
    assert "import datetime" not in src
    assert "import time" not in src
    assert "datetime.now" not in src
    assert "time.time" not in src
