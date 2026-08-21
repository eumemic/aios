"""Regression tests for permanent diagnostics in slow CI pytest lanes."""

from __future__ import annotations

from pathlib import Path

_WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "code-validation.yml"


def test_slow_ci_pytest_invocations_report_durations() -> None:
    workflow = _WORKFLOW.read_text()
    slow_invocations = [
        line.strip()
        for line in workflow.splitlines()
        if "uv run pytest" in line
        and ("tests/integration" in line or "tests/e2e" in line)
        and '"docker and perf"' not in line
    ]

    assert len(slow_invocations) == 5
    assert all("--durations=25" in invocation for invocation in slow_invocations)


def test_integration_pytest_invocations_dump_stalled_test_stacks() -> None:
    workflow = _WORKFLOW.read_text()
    integration_invocations = [
        line.strip() for line in workflow.splitlines() if "uv run pytest tests/integration" in line
    ]

    assert len(integration_invocations) == 2
    assert all(
        "-o faulthandler_timeout=120" in invocation for invocation in integration_invocations
    )
