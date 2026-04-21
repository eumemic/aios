"""Unit tests for the SSE subscriber lock helpers (issue #81)."""

from __future__ import annotations

from aios.db.sse_lock import session_lock_key


class TestSessionLockKey:
    def test_deterministic(self) -> None:
        assert session_lock_key("sess_01KAAABBBCCC") == session_lock_key("sess_01KAAABBBCCC")

    def test_different_sessions_get_different_keys(self) -> None:
        a = session_lock_key("sess_01KAAAAAAAAA")
        b = session_lock_key("sess_01KBBBBBBBBB")
        assert a != b

    def test_keys_fit_in_int32(self) -> None:
        classid, objid = session_lock_key("sess_01KAAABBBCCC")
        assert -(2**31) <= classid < 2**31
        assert -(2**31) <= objid < 2**31

    def test_empty_string_does_not_crash(self) -> None:
        session_lock_key("")
