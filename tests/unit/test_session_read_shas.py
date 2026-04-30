"""Tests for the per-session read-sha cache used by the write-tool precondition."""

from __future__ import annotations

import pytest

from aios.harness import runtime

SESSION_A = "sesn_AAA"
SESSION_B = "sesn_BBB"
STORE = "memstore_X"


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    runtime.clear_session_read_shas(SESSION_A)
    runtime.clear_session_read_shas(SESSION_B)
    yield
    runtime.clear_session_read_shas(SESSION_A)
    runtime.clear_session_read_shas(SESSION_B)


def test_miss_returns_none() -> None:
    assert runtime.get_read_sha(SESSION_A, STORE, "/x.md") is None


def test_set_then_get() -> None:
    runtime.set_read_sha(SESSION_A, STORE, "/x.md", "abc")
    assert runtime.get_read_sha(SESSION_A, STORE, "/x.md") == "abc"


def test_set_overwrites() -> None:
    runtime.set_read_sha(SESSION_A, STORE, "/x.md", "abc")
    runtime.set_read_sha(SESSION_A, STORE, "/x.md", "def")
    assert runtime.get_read_sha(SESSION_A, STORE, "/x.md") == "def"


def test_per_session_isolation() -> None:
    runtime.set_read_sha(SESSION_A, STORE, "/x.md", "abc")
    assert runtime.get_read_sha(SESSION_B, STORE, "/x.md") is None


def test_per_path_isolation() -> None:
    runtime.set_read_sha(SESSION_A, STORE, "/x.md", "abc")
    assert runtime.get_read_sha(SESSION_A, STORE, "/y.md") is None


def test_per_store_isolation() -> None:
    runtime.set_read_sha(SESSION_A, STORE, "/x.md", "abc")
    assert runtime.get_read_sha(SESSION_A, "memstore_OTHER", "/x.md") is None


def test_clear_session_read_shas() -> None:
    runtime.set_read_sha(SESSION_A, STORE, "/x.md", "abc")
    runtime.set_read_sha(SESSION_A, STORE, "/y.md", "def")
    runtime.set_read_sha(SESSION_B, STORE, "/x.md", "ghi")
    runtime.clear_session_read_shas(SESSION_A)
    assert runtime.get_read_sha(SESSION_A, STORE, "/x.md") is None
    assert runtime.get_read_sha(SESSION_A, STORE, "/y.md") is None
    assert runtime.get_read_sha(SESSION_B, STORE, "/x.md") == "ghi"


def test_clear_unknown_session_no_error() -> None:
    runtime.clear_session_read_shas("sesn_NEVER_EXISTED")  # idempotent
