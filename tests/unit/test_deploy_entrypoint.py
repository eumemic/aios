"""Negative control for the repository-owned candidate promotion boundary."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
ENTRYPOINT = ROOT / "docker" / "deploy-entrypoint.sh"
DOCKERFILE = ROOT / "Dockerfile"


def _fake_aios(tmp_path: Path, *, migration_status: int) -> Path:
    binary = tmp_path / "aios"
    binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = migrate ]; then\n'
        f"  exit {migration_status}\n"
        "fi\n"
        'printf "%s" "$1" > "$STARTED_MARKER"\n'
    )
    binary.chmod(0o755)
    return binary


def _deploy(tmp_path: Path, *, migration_status: int) -> tuple[subprocess.CompletedProcess[str], Path]:
    _fake_aios(tmp_path, migration_status=migration_status)
    marker = tmp_path / "candidate-started"
    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "STARTED_MARKER": str(marker),
    }
    result = subprocess.run(
        [str(ENTRYPOINT), "aios", "api"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, marker


def test_exhausted_migration_failure_rejects_candidate(tmp_path: Path) -> None:
    """A final LockNotAvailable exit is load-bearing at candidate startup.

    Status 73 stands in for ``aios migrate`` exhausting its lock retries.  The
    boundary must remain non-successful and must never exec the candidate API;
    therefore the candidate cannot become healthy or be promoted.
    """
    deployment, candidate_started = _deploy(tmp_path, migration_status=73)

    assert deployment.returncode == 73
    assert not candidate_started.exists()


def test_successful_migration_starts_candidate(tmp_path: Path) -> None:
    deployment, candidate_started = _deploy(tmp_path, migration_status=0)

    assert deployment.returncode == 0
    assert candidate_started.read_text() == "api"


def test_image_installs_candidate_boundary() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "COPY --chmod=755 docker/deploy-entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["aios-deploy-entrypoint"]' in dockerfile
