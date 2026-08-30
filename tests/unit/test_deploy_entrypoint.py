"""Negative controls for the repository-owned candidate admission boundary."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
ENTRYPOINT = ROOT / "docker" / "deploy-entrypoint.sh"
DOCKERFILE = ROOT / "Dockerfile"


def _fake_aios(tmp_path: Path, *, migration_status: int) -> tuple[Path, Path]:
    calls = tmp_path / "calls"
    binary = tmp_path / "aios"
    binary.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$1" >> "$CALL_LOG"\n'
        'if [ "$1" = migrate ]; then\n'
        f"  exit {migration_status}\n"
        "fi\n"
        'printf "%s" "$1" > "$STARTED_MARKER"\n'
    )
    binary.chmod(0o755)
    return binary, calls


def _deploy(
    tmp_path: Path,
    *,
    migration_status: int,
    service: str = "api",
    candidate: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    _, calls = _fake_aios(tmp_path, migration_status=migration_status)
    marker = tmp_path / "service-started"
    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALL_LOG": str(calls),
        "STARTED_MARKER": str(marker),
    }
    command = [str(ENTRYPOINT)]
    if candidate:
        command.append("--candidate")
    command.extend(["aios", service])
    result = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, marker, calls


def test_exhausted_migration_failure_rejects_api_candidate(tmp_path: Path) -> None:
    """A final LockNotAvailable exit is load-bearing at candidate startup."""
    deployment, candidate_started, calls = _deploy(tmp_path, migration_status=73, candidate=True)

    assert deployment.returncode == 73
    assert not candidate_started.exists()
    assert calls.read_text().splitlines() == ["migrate"]


def test_successful_migration_starts_api_candidate(tmp_path: Path) -> None:
    deployment, candidate_started, calls = _deploy(tmp_path, migration_status=0, candidate=True)

    assert deployment.returncode == 0
    assert candidate_started.read_text() == "api"
    assert calls.read_text().splitlines() == ["migrate", "api"]


def test_worker_never_owns_migrations(tmp_path: Path) -> None:
    deployment, worker_started, calls = _deploy(tmp_path, migration_status=74, service="worker")

    assert deployment.returncode == 0
    assert worker_started.read_text() == "worker"
    assert calls.read_text().splitlines() == ["worker"]


def test_ordinary_api_start_allows_application_rollback(tmp_path: Path) -> None:
    """An older compatible image starts after a candidate-only DB revision."""
    deployment, rollback_started, calls = _deploy(tmp_path, migration_status=1)

    assert deployment.returncode == 0
    assert rollback_started.read_text() == "api"
    assert calls.read_text().splitlines() == ["api"]


def test_candidate_mode_rejects_non_api_commands(tmp_path: Path) -> None:
    deployment, service_started, calls = _deploy(
        tmp_path, migration_status=0, service="worker", candidate=True
    )

    assert deployment.returncode == 64
    assert not service_started.exists()
    assert not calls.exists()


def test_image_installs_and_opts_api_into_candidate_boundary() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "COPY --chmod=755 docker/deploy-entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["aios-deploy-entrypoint"]' in dockerfile
    assert 'CMD ["--candidate", "aios", "api"]' in dockerfile
    assert 'CMD ["aios", "worker"]' in dockerfile
