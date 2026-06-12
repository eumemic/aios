"""Workflow replay-semantics epoch fingerprint canary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from aios.workflows.determinism import HOST_SEMANTICS_EPOCH
from tests.unit.replay_semantics_fingerprint_fixture import build_snapshot

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "replay_semantics_fingerprint.json"
)
_REGEN_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "regen-replay-semantics-fingerprint.py"
)


def _load_regen_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_regen_fingerprint", _REGEN_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_semantics_fingerprint_matches_epoch_snapshot() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert snapshot["epoch"] == HOST_SEMANTICS_EPOCH
    assert build_snapshot() == snapshot


def test_regen_refuses_content_change_without_epoch_bump() -> None:
    regen = _load_regen_script()
    existing = {"epoch": 1, "surface": "old"}
    rebuilt = {"epoch": 1, "surface": "new"}
    with pytest.raises(regen.EpochNotBumpedError, match="HOST_SEMANTICS_EPOCH"):
        regen.check_regen_allowed(existing, rebuilt)


def test_regen_refusal_message_points_at_determinism_module() -> None:
    regen = _load_regen_script()
    with pytest.raises(regen.EpochNotBumpedError, match=r"determinism\.py"):
        regen.check_regen_allowed({"epoch": 3, "k": "a"}, {"epoch": 3, "k": "b"})


def test_regen_permits_identical_content() -> None:
    regen = _load_regen_script()
    snapshot = {"epoch": 1, "surface": "same"}
    regen.check_regen_allowed(snapshot, dict(snapshot))


def test_regen_permits_epoch_bump() -> None:
    regen = _load_regen_script()
    regen.check_regen_allowed({"epoch": 1, "surface": "old"}, {"epoch": 2, "surface": "new"})


def test_regen_permits_bootstrap_when_fixture_absent() -> None:
    regen = _load_regen_script()
    regen.check_regen_allowed(None, {"epoch": 1, "surface": "new"})


def test_regen_main_bootstraps_then_is_idempotent_then_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    regen = _load_regen_script()
    out = tmp_path / "fixtures" / "replay_semantics_fingerprint.json"

    # Bootstrap: fixture absent → written.
    assert regen.main(out) == 0
    written = json.loads(out.read_text())
    assert written == build_snapshot()

    # Idempotent re-run: identical content → still permitted.
    assert regen.main(out) == 0
    assert json.loads(out.read_text()) == written

    # Same-epoch content drift → refused, fixture untouched.
    tampered = dict(written)
    tampered["surface"] = "tampered"
    out.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    assert regen.main(out) == 1
    assert "HOST_SEMANTICS_EPOCH" in capsys.readouterr().err
    assert json.loads(out.read_text()) == tampered

    # Stored epoch differing from the rebuilt one → legitimate bump, rewritten.
    bumped = dict(written)
    bumped["epoch"] = written["epoch"] - 1
    out.write_text(json.dumps(bumped, indent=2, sort_keys=True) + "\n")
    assert regen.main(out) == 0
    assert json.loads(out.read_text()) == written
