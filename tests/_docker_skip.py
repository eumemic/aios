"""Loud Docker-unavailable skips for the aios test suite (WP-25 X8).

Self-contained pytest plugin module: stdlib + pytest only, no aios imports,
loaded by file path (see ``tests/conftest.py``) so nothing depends on
``sys.path`` or a package ``__init__``.

Every Docker-gated skip carries ``DOCKER_SKIP_PREFIX`` in its reason, and
``pytest_terminal_summary`` prints a red banner counting them so an off-Docker
run can't silently pass with the integration/e2e tier never having run.
"""

from __future__ import annotations

import functools
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DOCKER_SKIP_PREFIX = "DOCKER-UNAVAILABLE:"


def _socket_candidates() -> list[Path]:
    """Known Docker socket paths, or the test-only ``AIOS_TEST_DOCKER_SOCKETS`` override."""
    override = os.environ.get("AIOS_TEST_DOCKER_SOCKETS")
    if override is not None:
        return [Path(p) for p in override.split(os.pathsep) if p]
    return [
        Path("/var/run/docker.sock"),
        Path.home() / ".docker" / "run" / "docker.sock",
    ]


def _docker_info_ok() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, check=False, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def probe_docker() -> tuple[bool, str]:
    """Return ``(available, probed)`` — ``probed`` describes what was tried.

    Sets ``DOCKER_HOST`` when Docker is reachable via a known socket but the
    variable is missing (the Python ``docker`` lib / testcontainers need it).
    """
    if "DOCKER_HOST" in os.environ:
        host = os.environ["DOCKER_HOST"]
        return _docker_info_ok(), f"DOCKER_HOST={host}"

    candidates = _socket_candidates()
    for sock in candidates:
        if sock.exists():
            os.environ["DOCKER_HOST"] = f"unix://{sock}"
            if _docker_info_ok():
                return True, f"DOCKER_HOST=unix://{sock}"
            del os.environ["DOCKER_HOST"]

    probed = ", ".join(str(s) for s in candidates) or "(none)"
    return False, f"DOCKER_HOST unset; sockets: {probed}"


def docker_skip_reason(probed: str) -> str:
    return (
        f"{DOCKER_SKIP_PREFIX} Docker-backed tests (integration/e2e) did NOT run. "
        f"Probed: {probed}. Fix: start Docker, or set DOCKER_HOST to your active "
        "context's endpoint (see `docker context inspect`)."
    )


def _skip_text(report: Any) -> str:
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def count_docker_skips(reports: Iterable[Any]) -> int:
    return sum(
        1
        for report in reports
        if getattr(report, "skipped", False) and DOCKER_SKIP_PREFIX in _skip_text(report)
    )


def pytest_terminal_summary(terminalreporter: Any) -> None:
    n = count_docker_skips(terminalreporter.stats.get("skipped", []))
    if n > 0:
        _, probed = probe_docker()
        terminalreporter.write_sep(
            "=",
            f"{n} Docker-backed tests SKIPPED — Docker unreachable (probed: {probed}). "
            "These did not run.",
            red=True,
            bold=True,
        )
