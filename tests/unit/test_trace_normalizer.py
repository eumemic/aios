"""The trace normalizer's truth table (#1149) — pure, DB-free.

Pins the locked decisions:
* ``no_return`` is a *present* ``request_response`` (``error.kind='no_return'``),
  not an absence; an absence is ``running``.
* the small enum ``{ok, errored, cancelled, suspended, running}`` + raw
  ``error_kind`` passthrough (raw discriminant preserved verbatim).
* ``suspended`` is distinct from ``running``.
* session ``errored`` derives from ``stop_reason.type``, not a status column.
"""

from __future__ import annotations

from aios.services import trace_normalizer as norm

# ─── servicer nodes (derive_response / derive_run_response output) ────────────


def test_absent_response_is_running_not_no_return() -> None:
    # Absence ⇒ still running. NEVER no_return (the locked correction).
    assert norm.normalize_response(None) == ("running", None)


def test_no_return_is_a_present_errored_response() -> None:
    resp = {"result": None, "is_error": True, "error": {"kind": "no_return"}}
    assert norm.normalize_response(resp) == ("errored", "no_return")


def test_child_gone_passes_through_verbatim() -> None:
    resp = {"result": None, "is_error": True, "error": {"kind": "child_gone"}}
    assert norm.normalize_response(resp) == ("errored", "child_gone")


def test_ok_response() -> None:
    assert norm.normalize_response({"result": 42, "is_error": False}) == ("ok", None)


def test_errored_without_error_kind_is_none() -> None:
    assert norm.normalize_response({"is_error": True, "error": None}) == ("errored", None)


# ─── root runs (wf_runs.status + run_completed payload) ──────────────────────


def test_run_root_completed_ok() -> None:
    assert norm.normalize_run_root(
        status="completed", run_completed_error=None, run_completed_is_error=False
    ) == ("ok", None)


def test_run_root_completed_but_errored_surfaces_kind() -> None:
    assert norm.normalize_run_root(
        status="completed",
        run_completed_error={"kind": "output_schema_violation"},
        run_completed_is_error=True,
    ) == ("errored", "output_schema_violation")


def test_run_root_errored_status() -> None:
    assert norm.normalize_run_root(
        status="errored",
        run_completed_error={"kind": "author_exception"},
        run_completed_is_error=True,
    ) == ("errored", "author_exception")


def test_run_root_cancelled() -> None:
    assert norm.normalize_run_root(
        status="cancelled", run_completed_error=None, run_completed_is_error=False
    ) == ("cancelled", None)


def test_run_root_suspended_is_distinct_from_running() -> None:
    suspended = norm.normalize_run_root(
        status="suspended", run_completed_error=None, run_completed_is_error=False
    )
    running = norm.normalize_run_root(
        status="running", run_completed_error=None, run_completed_is_error=False
    )
    assert suspended == ("suspended", None)
    assert running == ("running", None)
    assert suspended != running


def test_run_root_pending_is_running() -> None:
    assert norm.normalize_run_root(
        status="pending", run_completed_error=None, run_completed_is_error=False
    ) == ("running", None)


# ─── root sessions (stop_reason.type, NOT a status column) ───────────────────


def test_session_root_error_stop_reason() -> None:
    state, kind = norm.normalize_session_root(
        {"type": "error", "error": {"kind": "context_overflow"}}, owes_open_request=False
    )
    assert state == "errored"
    assert kind == "context_overflow"


def test_session_root_error_falls_back_to_finish_reason() -> None:
    state, kind = norm.normalize_session_root(
        {"type": "error", "finish_reason": "content_filter"}, owes_open_request=False
    )
    assert (state, kind) == ("errored", "content_filter")


def test_session_root_interrupt_is_cancelled() -> None:
    assert norm.normalize_session_root({"type": "interrupt"}, owes_open_request=False) == (
        "cancelled",
        None,
    )


def test_session_root_rescheduling_is_running() -> None:
    assert norm.normalize_session_root({"type": "rescheduling"}, owes_open_request=False) == (
        "running",
        None,
    )


def test_session_root_end_turn_ok_when_owes_nothing() -> None:
    assert norm.normalize_session_root({"type": "end_turn"}, owes_open_request=False) == (
        "ok",
        None,
    )


def test_session_root_end_turn_running_when_owes_request() -> None:
    assert norm.normalize_session_root({"type": "end_turn"}, owes_open_request=True) == (
        "running",
        None,
    )


def test_session_root_archived_resolves_owed_child_gone() -> None:
    state, kind = norm.normalize_session_root(
        None,
        owes_open_request=True,
        owed_request_response={"is_error": True, "error": {"kind": "child_gone"}},
        is_archived=True,
    )
    assert (state, kind) == ("errored", "child_gone")


def test_session_root_live_no_stop_reason_is_running() -> None:
    assert norm.normalize_session_root(None, owes_open_request=False) == ("running", None)
