#!/usr/bin/env python3
"""Regenerate the workflow replay-semantics fingerprint snapshots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.unit.replay_semantics_fingerprint_fixture import build_snapshot

OUT = ROOT / "tests" / "fixtures" / "replay_semantics_fingerprint.json"


def main() -> None:
    snapshot = build_snapshot()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
