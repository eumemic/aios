from __future__ import annotations

import shutil
import subprocess

import pytest

from aios.services.trigger_lint import (
    OBSERVED_WAKE_WARNING,
    SELF_DISABLING_EXIT_WARNING,
    lint_self_disabling_exit,
    lint_unconditional_wake,
    observed_wake_is_noisy,
)


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


def test_observed_wake_backstop() -> None:
    assert observed_wake_is_noisy([True] * 5)
    assert observed_wake_is_noisy([True] * 10 + [False])
    assert not observed_wake_is_noisy([False, *([True] * 9)])
    assert observed_wake_is_noisy([True] * 4)
    assert not observed_wake_is_noisy([False, *([True] * 4)])


def test_observed_wake_warning_does_not_overstate_history() -> None:
    assert observed_wake_is_noisy([True] * 4)
    assert "at least five consecutive fires" not in OBSERVED_WAKE_WARNING


# ─── #2402: a monitor must not be able to disable itself ─────────────────────

_HEARTBEAT_V3 = (
    'if curl -fsS --retry 3 --retry-delay 5 --max-time 25 "https://hc-ping.com/u"; '
    "then exit 0; else\n"
    '   curl -fsS --max-time 15 "https://hc-ping.com/u/fail" || true\n'
    '   echo "HEARTBEAT FAILED - signalled /fail" >&2\n'
    "   exit 1\n"
    "fi\n"
)

_HEARTBEAT_FIXED = (
    'if curl -fsS --retry 3 --retry-delay 5 --max-time 25 "https://hc-ping.com/u"; '
    "then exit 0; fi\n"
    'curl -fsS --max-time 15 "https://hc-ping.com/u/fail" || true\n'
    'echo "heartbeat ping failed; signalled /fail, exiting 0 so the trigger survives" >&2\n'
    "exit 0\n"
)


def test_self_disabling_heartbeat_warns() -> None:
    """The exact v3 shape that auto-disabled all three dead-man emitters."""
    warnings = lint_self_disabling_exit(
        source_kind="cron", action_kind="sandbox_command", command=_HEARTBEAT_V3
    )
    assert warnings == [SELF_DISABLING_EXIT_WARNING]
    assert "auto-disable" in SELF_DISABLING_EXIT_WARNING
    assert "exit 0" in SELF_DISABLING_EXIT_WARNING


def test_fixed_heartbeat_does_not_warn() -> None:
    assert not lint_self_disabling_exit(
        source_kind="cron", action_kind="sandbox_command", command=_HEARTBEAT_FIXED
    )


def test_other_nonzero_exit_forms_warn() -> None:
    for command in ("check || exit 2", "exit 127", "false; exit  1;", "[ -f x ] || { exit 3; }"):
        assert lint_self_disabling_exit(
            source_kind="cron", action_kind="sandbox_command", command=command
        ), command


def test_exit_lookalikes_do_not_warn() -> None:
    for command in (
        "exit 0",
        "exit",
        "echo exit_code=1",
        "echo 'on_exit 1'",
        "curl https://x/exit/1",
        "exit 10x",
    ):
        assert not lint_self_disabling_exit(
            source_kind="cron", action_kind="sandbox_command", command=command
        ), command


def test_self_disabling_lint_scope() -> None:
    # Standing reactive sources also count failures toward auto-disable.
    assert lint_self_disabling_exit(
        source_kind="run_completion", action_kind="sandbox_command", command="exit 1"
    )
    # A one-shot fires once and self-deletes — there is no breaker to trip.
    assert not lint_self_disabling_exit(
        source_kind="one_shot", action_kind="sandbox_command", command="exit 1"
    )
    # Non-sandbox actions have no exit code.
    assert not lint_self_disabling_exit(source_kind="cron", action_kind="wake_owner")


async def test_trigger_write_path_surfaces_self_disabling_warning() -> None:
    """The service lint that feeds create/update ``warnings`` includes the new check."""
    from typing import Any, cast

    from aios.models.triggers import CronSource, SandboxCommandAction
    from aios.services.triggers import _lint_trigger

    warnings = await _lint_trigger(
        cast(Any, None),
        CronSource(kind="cron", schedule="*/5 * * * *"),
        SandboxCommandAction(command=_HEARTBEAT_V3),
        account_id="acc_x",
    )
    assert warnings == [SELF_DISABLING_EXIT_WARNING]


_EXIT_CASES = [
    "exit -1",
    "exit 256",
    "exit 512",
    "exit 1",
    "exit 0",
    "exit 257",
    "exit '1'",
    'exit "2"',
    "exit +1",
    "exit -256",
    "exit '-1'",
    'exit "+256"',
    "exit 010",
    "exit 99999999999999999999",
]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("cmd", _EXIT_CASES)
def test_warns_iff_bash_status_nonzero(cmd: str) -> None:
    status = subprocess.run(["bash", "-c", cmd], capture_output=True).returncode
    warned = bool(
        lint_self_disabling_exit(source_kind="cron", action_kind="sandbox_command", command=cmd)
    )
    assert warned == (status != 0)


def test_mismatched_quotes_do_not_match() -> None:
    assert not lint_self_disabling_exit(
        source_kind="cron", action_kind="sandbox_command", command="exit '1\""
    )


@pytest.mark.parametrize(
    "cmd",
    [
        "exit " + "1" * 4301,
        "exit -" + "9" * 5000,
        "exit " + "0" * 5000 + "1",
        "exit '" + "7" * 4400 + "'",
    ],
)
def test_huge_exit_literal_never_raises(cmd: str) -> None:
    # bash: status 2 (out of range) or 1 — non-zero either way; must warn, not raise.
    assert lint_self_disabling_exit(
        source_kind="cron", action_kind="sandbox_command", command=cmd
    ) == [SELF_DISABLING_EXIT_WARNING]


def test_huge_zero_padded_zero_is_status_zero() -> None:
    cmd = "exit " + "0" * 5000
    assert (
        lint_self_disabling_exit(source_kind="cron", action_kind="sandbox_command", command=cmd)
        == []
    )
