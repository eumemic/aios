from aios.tools.invoke_session import _CallAgentArgs, _CallWorkflowArgs
from aios.workflows.wf_script_host import agent, invoke_workflow


def test_agent_spawn_surfaces_share_existing_true_default() -> None:
    assert agent("x")._spec["auto_archive_on_completion"] is True
    assert _CallAgentArgs(agent_id="agt_1").auto_archive_on_completion is True


def test_agent_workflow_parameter_mutates_both_directions() -> None:
    assert agent("x", auto_archive_on_completion=True)._spec["auto_archive_on_completion"] is True
    assert agent("x", auto_archive_on_completion=False)._spec["auto_archive_on_completion"] is False


def test_run_dual_preserves_its_existing_non_archiving_default_and_mutates() -> None:
    assert invoke_workflow("wf_1", "x")._spec["auto_archive_on_completion"] is False
    assert _CallWorkflowArgs(workflow_id="wf_1").auto_archive_on_completion is False
    assert (
        invoke_workflow("wf_1", "x", auto_archive_on_completion=True)._spec[
            "auto_archive_on_completion"
        ]
        is True
    )
    assert (
        _CallWorkflowArgs(
            workflow_id="wf_1", auto_archive_on_completion=True
        ).auto_archive_on_completion
        is True
    )
