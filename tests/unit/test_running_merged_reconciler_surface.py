"""Unit tests for the running==merged reconciler deploy surface + trigger (aios#1327).

Pins acceptance items 4 (the exported ``build_running_merged_reconciler_script`` builder:
imports only re/json, value-domain I/O, fixed capability order) and 5 (a
``CronSource → WorkflowAction`` trigger document targeting the reconciler, off-peak
schedule, GITHUB_TOKEN vault bound), plus the deploy surface (tools + http_server) so it
can't drift from the script that needs it (#1135) and validates at create time.
"""

from __future__ import annotations

import ast

from aios.models.agents import HttpServerSpec
from aios.models.triggers import CronSource, TriggerCreate, WorkflowAction
from aios.models.workflows import WorkflowCreate
from aios.workflows.running_merged_reconciler import (
    DEFAULT_SCHEDULE,
    REQUIRED_HTTP_SERVERS,
    REQUIRED_TOOLS,
    build_running_merged_reconciler_script,
    build_running_merged_reconciler_trigger,
    build_running_merged_reconciler_workflow_create,
)
from aios.workflows.script_validation import validate_workflow_script

# ─── the script builder (acceptance item 4) ──────────────────────────────────


def test_script_imports_only_re_and_json() -> None:
    src = build_running_merged_reconciler_script()
    tree = ast.parse(src)
    imported = {n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    # Pure stdlib re/json only — value-domain I/O, replay-stable by construction.
    assert imported == {"re", "json"}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]


def test_script_defines_async_main() -> None:
    tree = ast.parse(build_running_merged_reconciler_script())
    assert any(isinstance(n, ast.AsyncFunctionDef) and n.name == "main" for n in tree.body)


def test_script_validates_against_declared_surface() -> None:
    # create-time validation: the declared surface is a superset of the AST-required one.
    src = build_running_merged_reconciler_script()
    validate_workflow_script(src, list(REQUIRED_TOOLS))


def test_interim_variant_validates_too() -> None:
    src = build_running_merged_reconciler_script(interim=True)
    validate_workflow_script(src, list(REQUIRED_TOOLS))
    assert "INTERIM = True" in src


# ─── REQUIRED_TOOLS / REQUIRED_HTTP_SERVERS (the deploy surface) ──────────────


def test_required_tools_are_bash_and_http_request() -> None:
    # The reconciler is scriptable (no agent): it only needs bash (ops-audit SSH read)
    # and http_request (GitHub). No editing tools — there are no child agents.
    assert {t.type for t in REQUIRED_TOOLS} == {"bash", "http_request"}


def test_required_http_server_is_github_with_repos_route() -> None:
    assert len(REQUIRED_HTTP_SERVERS) == 1
    server = REQUIRED_HTTP_SERVERS[0]
    assert server.name == "github"
    assert server.base_url == "https://api.github.com"
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert len(repos) == 1
    # GET (read master HEAD + list issues/comments) and POST (file issue / upsert comment).
    assert set(repos[0].methods or []) == {"GET", "POST"}
    # allow_query: the no-silent-degrade paginated reads add ?per_page/?page (#1323).
    assert repos[0].allow_query is True


# ─── build_*_workflow_create (the complete payload) ──────────────────────────


def test_workflow_create_bundles_script_and_surface() -> None:
    wc = build_running_merged_reconciler_workflow_create(name="running-merged-reconciler")
    assert isinstance(wc, WorkflowCreate)
    assert wc.name == "running-merged-reconciler"
    assert wc.script == build_running_merged_reconciler_script()
    assert {t.type for t in wc.tools} == {"bash", "http_request"}
    assert len(wc.http_servers) == 1
    assert isinstance(wc.http_servers[0], HttpServerSpec)
    assert wc.http_servers[0].name == "github"


def test_workflow_create_forwards_interim_kwarg() -> None:
    wc = build_running_merged_reconciler_workflow_create(name="rmr-interim", interim=True)
    assert wc.script == build_running_merged_reconciler_script(interim=True)


# ─── the CronSource → WorkflowAction trigger document (acceptance item 5) ─────


def test_trigger_is_cron_to_workflow_action() -> None:
    tc = build_running_merged_reconciler_trigger(
        name="rmr-cron", workflow_id="wf_abc", vault_id="vault_github"
    )
    assert isinstance(tc, TriggerCreate)
    assert isinstance(tc.source, CronSource)
    assert tc.source.schedule == DEFAULT_SCHEDULE  # off-peak default (every 20 min)
    assert isinstance(tc.action, WorkflowAction)
    assert tc.action.workflow_id == "wf_abc"
    # GITHUB_TOKEN vault bound so the reconciler's http_request calls authenticate.
    assert tc.action.vault_ids == ["vault_github"]
    # the input envelope carries the interim flag (selects interim vs full at fire time)
    assert tc.action.input_template == {"interim": False}


def test_trigger_interim_flag_threads_into_template() -> None:
    tc = build_running_merged_reconciler_trigger(
        name="rmr-cron", workflow_id="wf_abc", vault_id="vault_github", interim=True
    )
    assert tc.action.input_template == {"interim": True}


def test_trigger_custom_schedule_and_version_pin() -> None:
    tc = build_running_merged_reconciler_trigger(
        name="rmr-cron",
        workflow_id="wf_abc",
        vault_id="vault_github",
        schedule="*/30 * * * *",
        workflow_version=3,
    )
    assert tc.source.schedule == "*/30 * * * *"
    assert tc.action.workflow_version == 3
