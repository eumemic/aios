"""The restart ledger: round-trip, corruption tolerance, and its plane-root
location (outside the five subdirs ``clear_state`` wipes)."""

from __future__ import annotations

from pathlib import Path

from aios_browser_driver.host import _LEDGER_RELPATH, _load_ledger, _save_ledger


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ".aios" / "sessions.json"
    _save_ledger(path, {"sess_a": "01BOOTA", "sess_b": "01BOOTB"})
    assert _load_ledger(path) == {"sess_a": "01BOOTA", "sess_b": "01BOOTB"}


def test_missing_file_is_an_empty_ledger(tmp_path: Path) -> None:
    assert _load_ledger(tmp_path / "absent.json") == {}


def test_corrupt_file_is_an_empty_ledger(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text("{not json", "utf-8")
    assert _load_ledger(path) == {}
    path.write_text('["a", "list"]', "utf-8")
    assert _load_ledger(path) == {}


def test_ledger_lives_at_the_plane_root_not_in_a_swept_subdir() -> None:
    # clear_state rmtrees profile/shots/frames/downloads/input; the reaper
    # sweeps shots/frames/downloads. The ledger must sit outside all five.
    assert _LEDGER_RELPATH.parts[0] == ".aios"
