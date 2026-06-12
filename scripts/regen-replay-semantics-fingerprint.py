#!/usr/bin/env python3
"""Regenerate the workflow replay-semantics fingerprint snapshot.

The snapshot is an epoch-stamped canary: it may only change together with a
``HOST_SEMANTICS_EPOCH`` bump in ``src/aios/workflows/determinism.py``.  This
script therefore loads the existing fixture first and refuses to overwrite it
when the rebuilt content differs while the epoch still equals the stored one —
silently regenerating in that state would mask a replay-semantics change.

Permitted regens: rebuilt content identical to the existing fixture
(idempotent re-run), epoch differing from the stored one (legitimate bump), or
fixture file absent (bootstrap).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.unit.replay_semantics_fingerprint_fixture import build_snapshot  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "replay_semantics_fingerprint.json"


class EpochNotBumpedError(Exception):
    """Rebuilt snapshot differs but ``HOST_SEMANTICS_EPOCH`` was not bumped."""


def check_regen_allowed(existing: dict[str, Any] | None, rebuilt: dict[str, Any]) -> None:
    """Raise :class:`EpochNotBumpedError` when overwriting would mask a change.

    Permitted: fixture absent (bootstrap), rebuilt identical to existing
    (idempotent re-run), or rebuilt epoch differing from the stored epoch
    (legitimate bump).  Refused: content differs while the epoch matches.
    """
    if existing is None or rebuilt == existing or rebuilt["epoch"] != existing["epoch"]:
        return
    raise EpochNotBumpedError(
        "replay-semantics fingerprint content changed but HOST_SEMANTICS_EPOCH "
        f"is still {existing['epoch']}. Replay semantics may only change together "
        "with an epoch bump: bump HOST_SEMANTICS_EPOCH in "
        "src/aios/workflows/determinism.py first, then re-run this script."
    )


def main(out: Path = OUT) -> int:
    rebuilt = build_snapshot()
    existing = json.loads(out.read_text()) if out.exists() else None
    try:
        check_regen_allowed(existing, rebuilt)
    except EpochNotBumpedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rebuilt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
