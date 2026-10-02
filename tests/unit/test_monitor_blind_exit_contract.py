"""Blind-is-not-healthy contract for scheduled monitors (#2317).

A monitor that cannot read its source must not produce the same outcome as one
that read it and found nothing. The retirement monitors' CLIs exit 2
(``EXIT_BLIND``) when they cannot compute a verdict; these checks pin that the
workflows driving them do not swallow that exit with ``|| true`` (which, with no
JSON produced, skips every alert step and renders the run green).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
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


_STEP_CASES = [
    ("retirement-residue-rescan.yml", "aios.retirements.residue_scan"),
    ("retirement-aging-sla.yml", "aios.retirements.aging"),
]


def _run_step_with_cli(
    tmp_path: Path, workflow: str, module: str, cli_script: str
) -> subprocess.CompletedProcess[str]:
    """Execute the workflow step's real ``run`` body with the CLI swapped out.

    ``uv run python -m <module>`` is replaced by ``python3 -c <cli_script>``;
    a ``uv`` shim on PATH makes the step's later ``uv run python ...`` lines
    run under the test interpreter.
    """
    run = _step_running(workflow, module)
    stub = f"{sys.executable} -c {shlex.quote(cli_script)}"
    body = run.replace(f"uv run python -m {module}", stub)
    assert body != run
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "uv"
    shim.write_text(
        f'#!/usr/bin/env bash\n[ "$1" = run ] && shift\n[ "$1" = python ] && shift\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    shim.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
    }
    return subprocess.run(
        ["bash", "-c", body],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "cli_script",
    [
        "import sys; sys.exit(1)",  # uncaught-exception-style exit, no payload
        "import sys; sys.exit(0)",  # exit 0, no payload
        "import sys; print('not json'); sys.exit(1)",
        "import sys; print('{\"enabled\": true}'); sys.exit(0)",  # missing list key
        "import sys; print('[]'); sys.exit(0)",
    ],
)
@pytest.mark.parametrize(("workflow", "module"), _STEP_CASES)
def test_monitor_step_fails_blind_without_parseable_payload(
    tmp_path: Path, workflow: str, module: str, cli_script: str
) -> None:
    proc = _run_step_with_cli(tmp_path, workflow, module, cli_script)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"{workflow} step went green with no verdict payload:\n{out}"
    assert "BLIND" in out, out


@pytest.mark.parametrize(
    ("workflow", "module", "payload", "rc", "expected"),
    [
        (
            "retirement-residue-rescan.yml",
            "aios.retirements.residue_scan",
            {"enabled": False, "findings": []},
            0,
            ["enabled=False", "count=0"],
        ),
        (
            "retirement-residue-rescan.yml",
            "aios.retirements.residue_scan",
            {
                "enabled": True,
                "findings": [{"table": "t", "column": "c", "token": "x", "count": 2}],
            },
            1,
            ["enabled=True", "count=1"],
        ),
        ("retirement-aging-sla.yml", "aios.retirements.aging", {"breaches": []}, 0, ["count=0"]),
        (
            "retirement-aging-sla.yml",
            "aios.retirements.aging",
            {"breaches": [{"domain": "d"}]},
            1,
            ["count=1"],
        ),
    ],
)
def test_monitor_step_passes_real_verdict_through(
    tmp_path: Path,
    workflow: str,
    module: str,
    payload: dict[str, Any],
    rc: int,
    expected: list[str],
) -> None:
    script = f"import json, sys; print(json.dumps({payload!r})); sys.exit({rc})"
    proc = _run_step_with_cli(tmp_path, workflow, module, script)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    outputs = (tmp_path / "out").read_text().split()
    for line in expected:
        assert line in outputs, outputs
