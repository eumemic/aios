"""Unit tests for :class:`SqliteAnsweredSpool` (#1234).

The spool persists ``tool_call_id`` → serialized-result so a connector
that restarts mid-window (platform send succeeded, result-POST failed)
re-POSTs the persisted result on replay instead of re-running the
side-effecting tool body.
"""

from __future__ import annotations

from pathlib import Path

from aios_connector_http.spool import SqliteAnsweredSpool


def test_add_and_load_round_trip(tmp_path: Path) -> None:
    spool = SqliteAnsweredSpool(tmp_path / "answered.sqlite")
    try:
        spool.add("call_1", "RESULT_1")
        spool.add("call_2", None)
        assert spool.load() == {"call_1": "RESULT_1", "call_2": None}
    finally:
        spool.close()


def test_add_defaults_result_to_none(tmp_path: Path) -> None:
    spool = SqliteAnsweredSpool(tmp_path / "answered.sqlite")
    try:
        spool.add("call_1")
        assert spool.load() == {"call_1": None}
    finally:
        spool.close()


def test_insert_or_ignore_keeps_first_result(tmp_path: Path) -> None:
    """A replay re-persist must not clobber the original send result."""
    spool = SqliteAnsweredSpool(tmp_path / "answered.sqlite")
    try:
        spool.add("call_1", "FIRST")
        spool.add("call_1", "SECOND")
        assert spool.load() == {"call_1": "FIRST"}
    finally:
        spool.close()


def test_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "answered.sqlite"
    spool = SqliteAnsweredSpool(path)
    spool.add("call_1", "RESULT_1")
    spool.close()

    reopened = SqliteAnsweredSpool(path)
    try:
        assert reopened.load() == {"call_1": "RESULT_1"}
    finally:
        reopened.close()
