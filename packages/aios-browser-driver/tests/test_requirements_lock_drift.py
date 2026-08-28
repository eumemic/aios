"""``requirements.lock`` is the browser image's dependency pin (jarbot#106).

``docker/Dockerfile.browser`` installs the driver's dependencies from the
committed hashed export — NOT by resolving pyproject ranges at build time —
so the image runs the exact versions the repo's tests ran against (the
uv.lock'd worker image's property, extended to the standalone install).
This guard fails when the export drifts from the workspace ``uv.lock`` or
the package's declared dependencies.

Fix on drift, from ``packages/aios-browser-driver``:

    uv export --frozen --no-emit-project --no-dev > requirements.lock
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_requirements_lock_matches_workspace_lock() -> None:
    exported = subprocess.run(
        ["uv", "export", "--frozen", "--no-emit-project", "--no-dev"],
        cwd=_PACKAGE_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    committed = (_PACKAGE_DIR / "requirements.lock").read_text()
    assert exported == committed, (
        "packages/aios-browser-driver/requirements.lock is stale relative to uv.lock — "
        "the browser image would install different versions than the repo tests. "
        "Regenerate: cd packages/aios-browser-driver && "
        "uv export --frozen --no-emit-project --no-dev > requirements.lock"
    )
