"""Regression coverage for the Docker E2E runner resource budget (#1967)."""

from __future__ import annotations

from pathlib import Path

_WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "code-validation.yml"


def test_docker_e2e_runs_serially_without_blanket_retry() -> None:
    workflow = _WORKFLOW.read_text()
    docker_step = workflow.split("- name: E2E tests (docker shard)", 1)[1].split(
        "- name: E2E perf backstop", 1
    )[0]

    assert '-m "docker and not perf" --durations=25' in docker_step
    assert "-n " not in docker_step
    assert "--dist" not in docker_step
    assert "--lf" not in docker_step
    assert "FLAKE_RETRY" not in docker_step
