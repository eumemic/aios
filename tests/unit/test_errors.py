"""Unit tests for the AiosError flat-message renderer.

``to_message`` is the single-string rendering used by the model-path tool result and the
sandbox broker envelope. Its ``default=str`` is load-bearing: both call sites render it
inside an except clause, where a raise would skip the result append (a ghost tool call).
"""

from __future__ import annotations

from datetime import UTC, datetime

from aios.errors import ForbiddenError


def test_to_message_bare() -> None:
    assert ForbiddenError("denied").to_message() == "denied"


def test_to_message_with_detail() -> None:
    assert ForbiddenError("denied", detail={"x": 1}).to_message() == 'denied ({"x": 1})'


def test_to_message_is_total_over_non_serializable_detail() -> None:
    # A non-JSON-serializable detail value (datetime) must render via str(), never raise —
    # the renderer runs in an except clause where a raise would ghost the tool call.
    msg = ForbiddenError("e", detail={"when": datetime(2026, 1, 1, tzinfo=UTC)}).to_message()
    assert msg == 'e ({"when": "2026-01-01 00:00:00+00:00"})'


def test_to_message_keeps_non_ascii_raw() -> None:
    # ensure_ascii=False — agent-facing content stays raw UTF-8, not \uXXXX-escaped.
    assert ForbiddenError("e", detail={"name": "café"}).to_message() == 'e ({"name": "café"})'
