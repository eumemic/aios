"""WP-25 X8: Docker-unavailable skips are loud (prefixed reason + red banner)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

_MOD_NAME = "aios_tests_docker_skip"
_DOCKER_SKIP_PATH = Path(__file__).resolve().parents[1] / "_docker_skip.py"


def _load_docker_skip(path: Path) -> ModuleType:
    if _MOD_NAME in sys.modules:
        return sys.modules[_MOD_NAME]
    spec = importlib.util.spec_from_file_location(_MOD_NAME, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _report(skipped: bool, longrepr: Any) -> SimpleNamespace:
    return SimpleNamespace(skipped=skipped, longrepr=longrepr)


def test_count_docker_skips_matches_prefix_across_longrepr_shapes() -> None:
    mod = _load_docker_skip(_DOCKER_SKIP_PATH)
    reason_a = mod.docker_skip_reason("DOCKER_HOST=tcp://a:1")
    reason_b = mod.docker_skip_reason("sockets: /var/run/docker.sock")
    reports = [
        # skipif path (tuple longrepr)
        _report(True, ("t.py", 3, f"Skipped: {reason_a}")),
        # fixture pytest.skip path (tuple longrepr, different probed details)
        _report(True, ("conftest.py", 9, f"Skipped: {reason_b}")),
        # plain-string longrepr
        _report(True, f"Skipped: {reason_a}"),
        # non-Docker skips are ignored
        _report(True, ("t.py", 5, "Skipped: needs network")),
        _report(True, "Skipped: something else"),
        # not skipped at all
        _report(False, None),
    ]
    assert mod.count_docker_skips(reports) == 3


def test_docker_skip_reason_carries_prefix_and_probe() -> None:
    mod = _load_docker_skip(_DOCKER_SKIP_PATH)
    reason = mod.docker_skip_reason("DOCKER_HOST=tcp://x:2")
    assert reason.startswith(mod.DOCKER_SKIP_PREFIX)
    assert "tcp://x:2" in reason
    assert "did NOT run" in reason


_CONFTEST = """
import importlib.util
import sys

import pytest

_MOD_NAME = "aios_tests_docker_skip"


def _load_docker_skip(path):
    if _MOD_NAME in sys.modules:
        return sys.modules[_MOD_NAME]
    spec = importlib.util.spec_from_file_location(_MOD_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_ds = _load_docker_skip({path!r})
_ok, _probed = _ds.probe_docker()
pytest_terminal_summary = _ds.pytest_terminal_summary
needs_docker = pytest.mark.skipif(not _ok, reason=_ds.docker_skip_reason(_probed))


@pytest.fixture
def docker_fixture():
    if not _ok:
        pytest.skip(_ds.docker_skip_reason(_probed))
    return True
"""

_TESTS = """
from conftest import needs_docker


@needs_docker
def test_marked():
    pass


def test_fixture(docker_fixture):
    pass


def test_plain():
    pass
"""


def test_banner_end_to_end_with_and_without_xdist(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        textwrap.dedent(_CONFTEST).format(path=str(_DOCKER_SKIP_PATH))
    )
    (tmp_path / "test_x.py").write_text(textwrap.dedent(_TESTS))
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["DOCKER_HOST"] = "tcp://127.0.0.1:1"
    env["AIOS_TEST_DOCKER_SOCKETS"] = ""
    env["PYTHONPATH"] = str(tmp_path)
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    tail = ["--rootdir", str(tmp_path), str(tmp_path)]
    for extra in (["-p", "xdist.plugin", "-n", "2"], []):
        proc = subprocess.run(
            [*base, *extra, *tail],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert "2 Docker-backed tests SKIPPED" in out, out
        assert "tcp://127.0.0.1:1" in out, out
