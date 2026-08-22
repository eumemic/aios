"""Regression coverage for the Docker E2E runner resource budget (#1967)."""

from __future__ import annotations

from pathlib import Path

_WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "code-validation.yml"


def test_docker_e2e_runs_serially_and_retries_only_recorded_failures() -> None:
    workflow = _WORKFLOW.read_text()
    docker_step = workflow.split("- name: E2E tests (docker shard)", 1)[1].split(
        "- name: E2E perf backstop", 1
    )[0]

    command = 'uv run pytest tests/e2e -q -m "docker and not perf" --durations=25'
    assert docker_step.count(command) == 2
    assert "-n " not in docker_step
    assert "--dist" not in docker_step
    assert f"{command} --lf" in docker_step
    assert "if !" in docker_step
    assert "FLAKE_RETRY e2e-docker" in docker_step
