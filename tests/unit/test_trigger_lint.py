from __future__ import annotations

from aios.services.trigger_lint import lint_unconditional_wake


def test_cron_wake_owner_warns() -> None:
    warnings = lint_unconditional_wake(source_kind="cron", action_kind="wake_owner")
    assert len(warnings) == 1
    assert "every fire" in warnings[0]


def test_unconditional_sandbox_wake_warns() -> None:
    assert lint_unconditional_wake(
        source_kind="cron",
        action_kind="sandbox_command",
        command='FINDING=x; tool wake_self \'{"content":"found"}\'',
    )


def test_guarded_sandbox_wakes_do_not_warn() -> None:
    for command in (
        'if [ -n "$FINDING" ]; then tool wake_self \'{"content":"found"}\'; fi',
        '[ -n "$FINDING" ] && tool wake_self \'{"content":"found"}\'',
        'case "$STATE" in bad) tool wake_self \'{"content":"found"}\';; esac',
    ):
        assert not lint_unconditional_wake(
            source_kind="cron", action_kind="sandbox_command", command=command
        )


def test_unconditional_workflow_wake_warns() -> None:
    script = """\nasync def main(input):\n    await tool("wake_self", {"content": "found"})\n"""
    assert lint_unconditional_wake(
        source_kind="cron", action_kind="workflow", workflow_script=script
    )


def test_guarded_workflow_wake_does_not_warn() -> None:
    script = """\nasync def main(input):\n    if input.get("finding"):\n        await tool("wake_self", {"content": "found"})\n"""
    assert not lint_unconditional_wake(
        source_kind="cron", action_kind="workflow", workflow_script=script
    )


def test_non_recurring_source_does_not_warn() -> None:
    assert not lint_unconditional_wake(source_kind="one_shot", action_kind="wake_owner")
