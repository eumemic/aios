"""Workflow-run errors must surface their reported kind in the stop message (#2049)."""

from aios.harness.loop import _terminal_error_stop_message, _workflow_error_detail

_NO_DETAIL = "the provider returned no error detail"


def test_kind_only_workflow_error_surfaces_its_kind() -> None:
    msg = _terminal_error_stop_message(_workflow_error_detail({"kind": "cancelled"}))
    assert "cancelled" in msg
    assert _NO_DETAIL not in msg


def test_kind_and_message_both_surface() -> None:
    msg = _terminal_error_stop_message(
        _workflow_error_detail({"kind": "child_gone", "message": "run archived"})
    )
    assert "child_gone" in msg
    assert "run archived" in msg
    assert _NO_DETAIL not in msg


def test_empty_error_falls_back_to_outcome() -> None:
    msg = _terminal_error_stop_message(_workflow_error_detail(None, "cancelled"))
    assert "cancelled" in msg
    assert _NO_DETAIL not in msg


def test_no_error_and_no_outcome_says_no_detail() -> None:
    msg = _terminal_error_stop_message(_workflow_error_detail(None, None))
    assert _NO_DETAIL in msg
