"""Pure helpers in ``db.queries.trace`` (#1149), DB-free.

Pins ``_resolve_owed_response`` — the Python half of resolving an archived root
session's owed-request outcome (the SQL fetches a *written* response; this helper
applies the ``derive_response`` ``child_gone`` fallback an archived session can
never answer). This is the wiring that makes the normalizer's archived
``child_gone`` branch live instead of green-washing a failed root to ``ok``.
"""

from __future__ import annotations

from aios.db.queries.trace import _resolve_owed_response


def test_written_errored_response_passes_through() -> None:
    written = {"result": None, "is_error": True, "error": {"kind": "no_return"}}
    assert _resolve_owed_response(written, open_ids=["r1"], archived=True) == {
        "result": None,
        "is_error": True,
        "error": {"kind": "no_return"},
    }


def test_written_ok_response_passes_through() -> None:
    written = {"result": 42, "is_error": False, "error": None}
    assert _resolve_owed_response(written, open_ids=[], archived=False) == {
        "result": 42,
        "is_error": False,
        "error": None,
    }


def test_archived_with_open_request_resolves_child_gone() -> None:
    # No written response, session archived, owes a request → derive child_gone.
    assert _resolve_owed_response(None, open_ids=["r1"], archived=True) == {
        "result": None,
        "is_error": True,
        "error": {"kind": "child_gone"},
    }


def test_archived_owing_nothing_is_none() -> None:
    assert _resolve_owed_response(None, open_ids=[], archived=True) is None


def test_live_session_owing_request_is_none() -> None:
    # A live session that owes a request is still pending — not child_gone.
    assert _resolve_owed_response(None, open_ids=["r1"], archived=False) is None
