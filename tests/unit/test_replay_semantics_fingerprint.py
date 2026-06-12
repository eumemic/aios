"""Workflow replay-semantics epoch fingerprint canary."""

from __future__ import annotations

import json
from pathlib import Path

from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from tests.unit.replay_semantics_fingerprint_fixture import build_snapshot

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "replay_semantics_fingerprint.json"
)


def test_replay_semantics_fingerprint_matches_epoch_snapshot() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert snapshot["epoch"] == HOST_SEMANTICS_EPOCH
    assert build_snapshot() == snapshot
