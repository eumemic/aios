import inspect

from aios.services import workflows as workflows_service
from aios.tools.invoke_session import _CallAgentArgs, _CallWorkflowArgs
from aios.workflows import service as workflow_runtime_service
from aios.workflows.wf_script_host import agent, invoke_workflow


def test_agent_spawn_surfaces_share_existing_true_default() -> None:
    assert agent("x")._spec["auto_archive_on_completion"] is True
    assert _CallAgentArgs(agent_id="agt_1").auto_archive_on_completion is True


def test_agent_workflow_parameter_mutates_both_directions() -> None:
    assert agent("x", auto_archive_on_completion=True)._spec["auto_archive_on_completion"] is True
    assert agent("x", auto_archive_on_completion=False)._spec["auto_archive_on_completion"] is False


def test_run_duals_do_not_expose_session_lifetime_choice() -> None:
    assert "auto_archive_on_completion" not in invoke_workflow("wf_1", "x")._spec
    assert "auto_archive_on_completion" not in _CallWorkflowArgs.model_fields
    assert (
        "auto_archive_on_completion"
        not in inspect.signature(workflows_service.launch_awaited_run).parameters
    )
    assert (
        "auto_archive_on_completion"
        not in inspect.signature(workflow_runtime_service.create_run).parameters
    )
