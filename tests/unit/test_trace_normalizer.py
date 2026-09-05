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

from aios.models.sessions import Err, Ok
from aios.services import trace_normalizer as norm

# ─── servicer nodes (derive_response / derive_run_response output) ────────────


def test_absent_response_is_running_not_no_return() -> None:
    # Absence ⇒ still running. NEVER no_return (the locked correction).
    assert norm.normalize_response(None) == ("running", None)


def test_no_return_is_a_present_errored_response() -> None:
    assert norm.normalize_response(Err(error={"kind": "no_return"})) == ("errored", "no_return")


def test_child_gone_passes_through_verbatim() -> None:
    assert norm.normalize_response(Err(error={"kind": "child_gone"})) == ("errored", "child_gone")


def test_ok_response() -> None:
    assert norm.normalize_response(Ok(result=42)) == ("ok", None)


def test_errored_without_error_kind_is_none() -> None:
    assert norm.normalize_response(Err(error={})) == ("errored", None)


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
    # Never-stepped, abandoned child: stop_reason IS NULL (no step_end ran), so
    # the owed child_gone is the oldest answered request. Real production shape.
    state, kind = norm.normalize_session_root(
        None,
        owes_open_request=True,
        owed_request_response={"is_error": True, "error": {"kind": "child_gone"}},
        is_archived=True,
    )
    assert (state, kind) == ("errored", "child_gone")


def test_session_root_end_turn_archived_owed_no_return_is_errored() -> None:
    # Path A: single-Ask child stalled on its first Ask, quiescence guard wrote
    # no_return, step_end then wrote end_turn unconditionally, reclaim archived.
    # The end_turn branch must NOT short-circuit before the archived override.
    assert norm.normalize_session_root(
        {"type": "end_turn"},
        owes_open_request=False,
        owed_request_response={"is_error": True, "error": {"kind": "no_return"}},
        is_archived=True,
    ) == ("errored", "no_return")


def test_session_root_end_turn_archived_owed_child_gone_is_errored() -> None:
    # Path B: stepped child whose creation Ask was never answered; C5
    # revoked_lease archival wrote child_gone for the now-oldest answered
    # request, but step_end already wrote end_turn on the last turn.
    assert norm.normalize_session_root(
        {"type": "end_turn"},
        owes_open_request=False,
        owed_request_response={"is_error": True, "error": {"kind": "child_gone"}},
        is_archived=True,
    ) == ("errored", "child_gone")


def test_session_root_end_turn_archived_owed_recoverable_error_still_ok() -> None:
    # Regression pin for the preserved shape: a child self-emitted an error on
    # its first Ask (error_handler writes Err(error={"message": ...}) with NO
    # ``kind``), got latched errored for that step, then recovered on a later
    # step and ended cleanly. Once archived its oldest answered response is the
    # recoverable error — not a doom kind — so end_turn must still win as ok.
    # This would FAIL under a naive "any errored owed -> errored" guard, proving
    # the doom-kind gate is necessary AND sufficient.
    assert norm.normalize_session_root(
        {"type": "end_turn"},
        owes_open_request=False,
        owed_request_response={"is_error": True, "error": {"message": "boom"}},
        is_archived=True,
    ) == ("ok", None)


def test_session_root_end_turn_archived_owed_non_doom_kind_still_ok() -> None:
    # Locks the doom-kind gate from the "any kind" side: a non-doom kind on the
    # oldest answered request does not dominate end_turn, even when archived.
    # (Harness error kinds like context_overflow come with stop_reason=error,
    # not end_turn, so an end_turn + non-doom-kind-errored-owed combo can only
    # arise from the SQL's oldest-answered semantics on a recovered session.)
    assert norm.normalize_session_root(
        {"type": "end_turn"},
        owes_open_request=False,
        owed_request_response={"is_error": True, "error": {"kind": "context_overflow"}},
        is_archived=True,
    ) == ("ok", None)


def test_session_root_end_turn_archived_owed_no_return_with_open_request_is_errored() -> None:
    # The doom-kind override does not consult owes_open_request (a doom kind on
    # the oldest answered request is terminal regardless). Pins that the guard
    # fires even when owes_open_request=True — the no_return response closes the
    # request in production, but the guard must not depend on it.
    assert norm.normalize_session_root(
        {"type": "end_turn"},
        owes_open_request=True,
        owed_request_response={"is_error": True, "error": {"kind": "no_return"}},
        is_archived=True,
    ) == ("errored", "no_return")


def test_session_root_live_no_stop_reason_is_running() -> None:
    assert norm.normalize_session_root(None, owes_open_request=False) == ("running", None)


def test_session_root_end_turn_not_archived_with_doom_owed_is_ok() -> None:
    # The doom-kind override is gated on is_archived. A live (non-archived)
    # session carrying a doom-kind owed response (an artefact of the
    # oldest-answered SQL, e.g. a parent read mid-step) must still route through
    # the end_turn branch — a live session can still run again.
    assert norm.normalize_session_root(
        {"type": "end_turn"},
        owes_open_request=False,
        owed_request_response={"is_error": True, "error": {"kind": "no_return"}},
        is_archived=False,
    ) == ("ok", None)
