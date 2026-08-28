"""The input-spool tailer: EOF arming, partial-line buffering, malformed-skip,
and inode rotation (the reaper unlinks and recreates the spool)."""

from __future__ import annotations

import json
from pathlib import Path

from aios_browser_driver.takeover.spool import SpoolTailer


def _batch(seq: int) -> str:
    return json.dumps({"grant_id": "g", "epoch": 1, "seq": seq, "events": []}) + "\n"


def test_arms_from_eof_so_pre_takeover_input_is_not_replayed(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    spool.write_text(_batch(1) + _batch(2), "utf-8")  # written before the open
    tailer = SpoolTailer(spool)
    tailer.arm()
    assert tailer.poll() == []  # nothing after EOF
    spool.write_text(spool.read_text() + _batch(3), "utf-8")
    assert [b["seq"] for b in tailer.poll()] == [3]
    tailer.close()


def test_absent_spool_is_adopted_on_first_appearance(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    tailer = SpoolTailer(spool)
    tailer.arm()  # file does not exist yet
    assert tailer.poll() == []
    spool.write_text(_batch(1), "utf-8")
    assert [b["seq"] for b in tailer.poll()] == [1]
    tailer.close()


def test_partial_line_is_buffered_until_its_newline(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    spool.write_text("", "utf-8")
    tailer = SpoolTailer(spool)
    tailer.arm()
    with spool.open("a") as fh:
        fh.write('{"grant_id": "g", "epoch": 1, "seq": 5, "eve')
    assert tailer.poll() == []  # incomplete — no newline yet
    with spool.open("a") as fh:
        fh.write('nts": []}\n')
    assert [b["seq"] for b in tailer.poll()] == [5]
    tailer.close()


def test_malformed_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    spool.write_text("", "utf-8")
    tailer = SpoolTailer(spool)
    tailer.arm()
    spool.write_text("not json at all\n" + _batch(7), "utf-8")
    assert [b["seq"] for b in tailer.poll()] == [7]  # bad line dropped, good one kept
    tailer.close()


def test_inode_rotation_drains_old_then_follows_new(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    spool.write_text("", "utf-8")
    tailer = SpoolTailer(spool)
    tailer.arm()
    spool.write_text(_batch(1), "utf-8")
    assert [b["seq"] for b in tailer.poll()] == [1]

    # The reaper unlinks and recreates the spool with a fresh inode.
    spool.unlink()
    spool.write_text(_batch(2), "utf-8")
    assert [b["seq"] for b in tailer.poll()] == [2]  # adopted the new file from its start
    tailer.close()


def test_rotation_does_not_lose_the_tail_of_the_old_file(tmp_path: Path) -> None:
    spool = tmp_path / "spool.jsonl"
    spool.write_text("", "utf-8")
    tailer = SpoolTailer(spool)
    tailer.arm()
    # Old file gains a line we never polled; then it rotates. The unpolled tail
    # must still be drained before switching.
    spool.write_text(_batch(1), "utf-8")
    spool.rename(tmp_path / "old.jsonl")
    spool.write_text(_batch(2), "utf-8")
    seqs = [b["seq"] for b in tailer.poll()]
    assert seqs == [1, 2]
    tailer.close()
