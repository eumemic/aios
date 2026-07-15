from __future__ import annotations

from aios.models.workflows import WfRunCreate
from aios.tools.invoke_session import _CallWorkflowArgs
from aios.workflows.determinism import content_hash
from aios.workflows.wf_script_host import agent


def test_http_run_defaults_to_fresh_workspace() -> None:
    run = WfRunCreate(workflow_id="wf_1", environment_id="env_1")
    assert run.workspace == "fresh"


def test_session_workflow_call_defaults_shared_and_allows_fresh() -> None:
    assert _CallWorkflowArgs(workflow_id="wf_1").workspace == "shared"
    assert _CallWorkflowArgs(workflow_id="wf_1", workspace="fresh").workspace == "fresh"


def test_agent_workspace_mode_enters_call_identity() -> None:
    shared = agent("work")
    fresh = agent("work", workspace="fresh")
    assert shared._spec["workspace"] == "shared"
    assert fresh._spec["workspace"] == "fresh"
    assert content_hash(shared._capability_id, shared._spec) != content_hash(
        fresh._capability_id, fresh._spec
    )
