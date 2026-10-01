"""Blind-is-not-healthy contract for scheduled monitors (#2317).

A monitor that cannot read its source must not produce the same outcome as one
that read it and found nothing. The retirement monitors' CLIs exit 2
(``EXIT_BLIND``) when they cannot compute a verdict; these checks pin that the
workflows driving them do not swallow that exit with ``|| true`` (which, with no
JSON produced, skips every alert step and renders the run green).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _step_running(workflow: str, module: str) -> str:
    doc: dict[Any, Any] = yaml.safe_load((_WORKFLOWS / workflow).read_text())
    runs = [
        step["run"]
        for job in doc["jobs"].values()
        for step in job["steps"]
        if isinstance(step.get("run"), str) and f"python -m {module}" in step["run"]
    ]
    assert len(runs) == 1, f"{workflow}: expected one step running {module}, found {len(runs)}"
    run: str = runs[0]
    return run


@pytest.mark.parametrize(
    ("workflow", "module"),
    [
        ("retirement-aging-sla.yml", "aios.retirements.aging"),
        ("retirement-residue-rescan.yml", "aios.retirements.residue_scan"),
    ],
)
def test_monitor_step_fails_loud_when_blind(workflow: str, module: str) -> None:
    run = _step_running(workflow, module)
    cli_line = next(line for line in run.splitlines() if f"python -m {module}" in line)
    assert "|| true" not in cli_line, (
        f"{workflow} swallows every exit of {module} -- a blind run (exit 2) renders green"
    )
    assert "rc=$?" in cli_line, f"{workflow} must capture {module}'s exit code"
    assert "BLIND" in run and "exit 1" in run, (
        f"{workflow} must fail the step with a BLIND message on a non-verdict exit"
    )
