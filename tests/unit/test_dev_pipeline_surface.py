"""Unit tests for the dev-pipeline deploy-surface export (#1135).

``build_dev_pipeline_script`` returns only the workflow *script string*, but a deployed
workflow ALSO needs its tool + http_server surface declared on the ``WorkflowCreate`` —
otherwise the first ``tool('bash')`` call errors with "tool 'bash' is not in the
workflow's declared tools" (run_tools.py).

This module pins the exported deploy surface so it can't drift from the script that
needs it:

- ``REQUIRED_TOOLS`` — the UNION of the script's own tools (``bash``/``http_request``)
  AND every named agent's tools (``read``/``write``/``edit``/``glob``/``grep``). The
  child-agent surface is ``agent ∩ run``, so declaring only ``[bash, http_request]``
  on the workflow would strip the editing tools from the implement/fix agents.
- ``REQUIRED_HTTP_SERVERS`` — the two-route ``github`` spec: ``/repos/**`` with
  ``GET,POST,PUT,DELETE,PATCH`` (DELETE is load-bearing for the success-path unlabel;
  PATCH is load-bearing for the post-merge issue-close, #1208) and a separate
  ``/graphql`` POST route (mark-ready mutation).
- ``build_dev_pipeline_workflow_create(...)`` — the complete ``WorkflowCreate`` payload
  (script + tools + http_servers) so the surface can't drift from the script.
"""

from __future__ import annotations

from aios.models.workflows import WorkflowCreate
from aios.workflows.dev_pipeline import (
    REQUIRED_HTTP_SERVERS,
    REQUIRED_TOOLS,
    build_dev_pipeline_script,
    build_dev_pipeline_workflow_create,
)

# ─── REQUIRED_TOOLS: the union of script + agent tools ─────────────────────────


def test_required_tools_is_the_full_union() -> None:
    # The script uses bash + http_request; the implement/fix agents need the editing
    # tools (read/write/edit/glob/grep). The export must emit the union so the child
    # agents (agent ∩ run) keep their editing surface.
    got = {t.type for t in REQUIRED_TOOLS}
    assert got == {"bash", "read", "write", "edit", "glob", "grep", "http_request"}


def test_required_tools_includes_scripts_own_tools() -> None:
    types = {t.type for t in REQUIRED_TOOLS}
    assert "bash" in types
    assert "http_request" in types


def test_required_tools_includes_agent_editing_tools() -> None:
    types = {t.type for t in REQUIRED_TOOLS}
    for editing in ("read", "write", "edit", "glob", "grep"):
        assert editing in types, f"{editing} missing — implement/fix agents would be crippled"


def test_required_tools_have_no_duplicates() -> None:
    types = [t.type for t in REQUIRED_TOOLS]
    assert len(types) == len(set(types))


# ─── REQUIRED_HTTP_SERVERS: the two-route github spec ──────────────────────────


def test_required_http_servers_has_single_github_server() -> None:
    assert len(REQUIRED_HTTP_SERVERS) == 1
    server = REQUIRED_HTTP_SERVERS[0]
    assert server.name == "github"
    assert server.base_url == "https://api.github.com"


def test_repos_route_allows_delete() -> None:
    server = REQUIRED_HTTP_SERVERS[0]
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert len(repos) == 1
    methods = repos[0].methods
    assert methods is not None
    # DELETE is load-bearing: the success-path _unlabel(autodev:in-progress) issues
    # DELETE /repos/.../labels/...; omitting it silently failed every unlabel.
    # PATCH is load-bearing too: _close_source_issue (#1188) closes the source issue on
    # merge with PATCH /repos/.../issues/{n}. Omitting PATCH made the broker reject the
    # close as a route-allowlist mismatch (a deterministic {"error": ...}), so the close
    # never fired — the merged issue stayed OPEN with `dispatched` stripped → re-dispatch
    # loop (#1208). The strip (DELETE) succeeded while the close (PATCH) was silently denied.
    assert set(methods) == {"GET", "POST", "PUT", "DELETE", "PATCH"}


def test_repos_route_allows_patch_for_issue_close() -> None:
    # #1208: closing the source issue on merge is a PATCH /repos/.../issues/{n}. If the
    # route omits PATCH the broker denies the call before it reaches GitHub, so the issue
    # is never closed (regression from #1188 — strip fired, close did not).
    server = REQUIRED_HTTP_SERVERS[0]
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert "PATCH" in (repos[0].methods or [])


def test_repos_route_allows_query_for_pagination() -> None:
    # #1156: the comment-thread read must follow ?per_page/?page pagination past the first
    # 30 comments; the /repos/** route opts into allow_query so the query reaches GitHub.
    server = REQUIRED_HTTP_SERVERS[0]
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert repos[0].allow_query is True


def test_graphql_route_is_separate_and_post_only() -> None:
    server = REQUIRED_HTTP_SERVERS[0]
    graphql = [r for r in server.routes if r.path_pattern == "/graphql"]
    assert len(graphql) == 1
    assert graphql[0].methods == ["POST"]
    # GraphQL takes its query in the POST body, not the URL — it does not opt into allow_query.
    assert graphql[0].allow_query is False


def test_two_distinct_routes() -> None:
    server = REQUIRED_HTTP_SERVERS[0]
    patterns = {r.path_pattern for r in server.routes}
    assert patterns == {"/repos/**", "/graphql"}


# ─── build_dev_pipeline_workflow_create: the complete payload ──────────────────


def test_workflow_create_is_a_valid_workflow_create() -> None:
    wc = build_dev_pipeline_workflow_create(name="dev-pipeline")
    assert isinstance(wc, WorkflowCreate)
    assert wc.name == "dev-pipeline"


def test_workflow_create_carries_the_script() -> None:
    wc = build_dev_pipeline_workflow_create(name="dev-pipeline")
    assert wc.script == build_dev_pipeline_script()
    assert "tool(" in wc.script


def test_workflow_create_carries_required_tools() -> None:
    wc = build_dev_pipeline_workflow_create(name="dev-pipeline")
    assert {t.type for t in wc.tools} == {t.type for t in REQUIRED_TOOLS}


def test_workflow_create_carries_required_http_servers() -> None:
    wc = build_dev_pipeline_workflow_create(name="dev-pipeline")
    assert len(wc.http_servers) == 1
    server = wc.http_servers[0]
    assert server.name == "github"
    assert {r.path_pattern for r in server.routes} == {"/repos/**", "/graphql"}


def test_workflow_create_forwards_build_kwargs_to_script() -> None:
    # Customising the script (e.g. iteration caps) must flow into the embedded script.
    wc = build_dev_pipeline_workflow_create(name="dev-pipeline", max_review_iters=7)
    assert wc.script == build_dev_pipeline_script(max_review_iters=7)


def test_workflow_create_github_server_name_matches_script_server() -> None:
    # The http_server name on the payload must equal the script's GITHUB_SERVER constant,
    # or every tool('http_request', {server_ref: GITHUB_SERVER}) fails with unknown server.
    wc = build_dev_pipeline_workflow_create(name="dev-pipeline", github_server="gh")
    assert any(s.name == "gh" for s in wc.http_servers)
    assert "'gh'" in wc.script or '"gh"' in wc.script
