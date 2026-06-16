"""Import-hygiene guard for the sandbox ↔ tools circular dependency.

``aios.sandbox.secret_egress_proxy`` must import cleanly even when it is the
*first* aios module loaded — i.e. standalone, before a full test run's collection
order happens to pre-break the ``sandbox.spec`` ↔ ``tools`` cycle. Run in a fresh
subprocess so the assertion is deterministic regardless of what this process has
already imported. A regression makes the module uncollectable in isolation and
risks an app-startup ImportError if import order ever shifts.
"""

from __future__ import annotations

import subprocess
import sys


def _imports_clean(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )


def test_secret_egress_proxy_imports_standalone() -> None:
    result = _imports_clean("aios.sandbox.secret_egress_proxy")
    assert result.returncode == 0, result.stderr


def test_sandbox_spec_imports_standalone() -> None:
    result = _imports_clean("aios.sandbox.spec")
    assert result.returncode == 0, result.stderr
