"""Guard the repository-wide pytest parallelism default.

Context (PR #2354 fix round): the review parked the PR reporting that a
fresh-checkout full-suite run "exceeded the 120-second execution ceiling ...
and was terminated with exit 137". Investigation showed the exit-137 is a
SIGKILL from *memory* exhaustion, not a wall-clock timeout:

  * ``-n=auto`` (4 workers, 2 GiB cgroup): OOM-kill -> exit 137 / worker crash,
    peak RSS pinned at the 2048 MiB ceiling.
  * ``-n=3`` (the SHA-under-review default): passes in ~93-100 s but peaks at
    2047 MiB against a 2048 MiB limit -- one MiB from the OOM killer. A
    marginally heavier worker tips it into the same exit-137.
  * ``-n=2``: passes in ~96-106 s with ~90-110 MiB of headroom.

So the property "the complete repository test suite must finish successfully"
is only robust when the default worker count leaves memory headroom under a
2 GiB budget. This test pins that invariant so a future edit cannot silently
raise the default back onto the OOM edge.

The over-correction guard is explicit below: the fix must NOT degrade to serial
/ ``-n=0`` / ``-n=1`` execution (which the pyproject comment itself notes
"cannot finish within their test budget"). Parallelism with ``loadfile``
distribution must be preserved. Bounded, not disabled.
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
    for opt in addopts:
        m = re.fullmatch(r"-n=?(\d+)", opt)
        if m:
            return int(m.group(1))
        if opt in {"-n", "--numprocesses"}:
            # ``-n N`` split across two entries is not how this repo writes it,
            # but handle it rather than silently miss the value.
            idx = addopts.index(opt)
            if idx + 1 < len(addopts):
                nxt = addopts[idx + 1]
                if nxt.isdigit():
                    return int(nxt)
    raise AssertionError(f"no explicit -n/--numprocesses in pytest addopts: {addopts!r}")


def test_default_parallelism_is_bounded_for_memory_headroom() -> None:
    """Default -n must leave OOM headroom under the ~2 GiB CI worker budget.

    RED before the fix: the reviewed SHA pins -n=3, whose measured peak RSS is
    2047/2048 MiB -> exit-137 on any marginally heavier worker. GREEN after:
    -n<=2 keeps ~90 MiB of headroom and the suite finishes.
    """
    n = _numprocesses(_addopts())
    assert n <= 2, (
        f"pytest default -n={n} peaks at the 2 GiB cgroup ceiling and OOM-kills "
        "(exit 137) on heavier workers; keep it <=2 for memory headroom"
    )


def test_default_parallelism_is_not_disabled() -> None:
    """Over-correction guard + positive control.

    The degenerate 'fix' for an OOM/timeout is to stop running tests in
    parallel at all (-n=0 / -n=1 / serial), which the pyproject comment notes
    "cannot finish within their test budget". Assert parallelism survives.
    """
    addopts = _addopts()
    n = _numprocesses(addopts)
    assert n >= 2, (
        f"pytest default -n={n} disables meaningful parallelism; serial runs "
        "blow the review time budget -- keep >=2"
    )
    assert "--dist=loadfile" in addopts, (
        "--dist=loadfile must be preserved so same-file tests stay on one "
        f"worker; addopts={addopts!r}"
    )
