"""Guard the repository-wide pytest parallelism default.

The complete suite must fit a 120-second fresh-checkout review budget without
using xdist's host-dependent ``auto`` worker count. Keep the default at a fixed
three workers and group tests by scope to amortize setup while retaining balance.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _addopts() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return list(data["tool"]["pytest"]["ini_options"]["addopts"])


def _numprocesses(addopts: list[str]) -> int:
    for index, opt in enumerate(addopts):
        match = re.fullmatch(r"-n=?(\d+)", opt)
        if match:
            return int(match.group(1))
        if opt in {"-n", "--numprocesses"} and index + 1 < len(addopts):
            value = addopts[index + 1]
            if value.isdigit():
                return int(value)
    raise AssertionError(f"no explicit -n/--numprocesses in pytest addopts: {addopts!r}")


def test_default_parallelism_balances_time_and_memory_budgets() -> None:
    """Pin the worker count that completes without unbounded memory use."""
    addopts = _addopts()
    assert _numprocesses(addopts) == 3, (
        "the full suite needs three bounded workers; do not use slower -n=2 "
        "or host-dependent -n=auto"
    )
    assert "--dist=loadscope" in addopts, (
        "--dist=loadscope must be preserved to amortize setup without "
        f"sacrificing class-level balancing; addopts={addopts!r}"
    )
