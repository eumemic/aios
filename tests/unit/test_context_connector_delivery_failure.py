"""Coverage for the model-visible connector-delivery-failure notice (#1261).

A connector appends a ``connector_delivery_failed`` lifecycle event when an
outbound the model *consciously sent* did not arrive (carrier block / delivery
failure). It is the model-visible-lifecycle channel — like the FS-loss notices
(``test_context_fs_lifecycle.py``) it must:
  * be a member of the ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist (so it both
    renders and survives windowing), and
  * render as a bracketed user-role notice at its seq position via a renderer
    distinct from the FS-loss renderer (carrier specifics ride in ``data``),
  * NOT advance ``reacting_to`` in the pure replay — the wake is produced by the
    session-targeted lifecycle *route* (``wake=True``), not by ``build_messages``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.harness.context import build_messages
from aios.models.events import MODEL_VISIBLE_LIFECYCLE_EVENTS, Event

_AT = datetime(2026, 6, 10, tzinfo=UTC)


def _msg(seq: int, role: str, content: str) -> Event:
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="message",
        data={"role": role, "content": content},
        created_at=_AT,
        orig_channel=None,
        focal_channel_at_arrival=None,
    )


def _lifecycle(seq: int, data: dict[str, Any]) -> Event:
    return Event(
        id=f"evt_{seq}",
        session_id="sess_01TEST",
        seq=seq,
        kind="lifecycle",
        data=data,
        created_at=_AT,
        orig_channel=None,
        focal_channel_at_arrival=None,
    )


def test_connector_delivery_failed_is_allowlisted() -> None:
    """The new kind is in the model-visible allowlist, so the windowing read
    carries it through and the renderer dispatches on it."""
    assert "connector_delivery_failed" in MODEL_VISIBLE_LIFECYCLE_EVENTS


def test_delivery_failure_renders_as_user_notice() -> None:
    events = [
        _msg(1, "user", "ping the customer"),
        _lifecycle(
            2,
            {
                "event": "connector_delivery_failed",
                "connector": "sms",
                "data": {"detail": "blocked by carrier", "peer": "+15550123"},
            },
        ),
    ]
    result = build_messages(events, system_prompt=None)
    notices = [
        m
        for m in result.messages
        if m["role"] == "user" and isinstance(m["content"], str) and "not delivered" in m["content"]
    ]
    assert len(notices) == 1
    content = notices[0]["content"]
    assert content.startswith("[") and content.endswith("]")
    assert "sms" in content
    assert "blocked by carrier" in content
    assert "+15550123" in content
    assert "did not receive" in content


def test_delivery_failure_uses_its_own_renderer_not_the_fs_one() -> None:
    """A delivery-failure notice must NOT borrow the FS-loss copy."""
    content = build_messages(
        [
            _msg(1, "user", "x"),
            _lifecycle(2, {"event": "connector_delivery_failed", "connector": "sms"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    assert "filesystem" not in content
    assert "not delivered" in content


def test_delivery_failure_does_not_advance_reacting_to() -> None:
    """The notice is NOT stimulus-bearing in the pure replay: the wake comes
    from the session-targeted route's ``wake=True``, not the renderer. The
    watermark stays at the last real stimulus (the user message)."""
    events = [
        _msg(1, "user", "send it"),
        _lifecycle(5, {"event": "connector_delivery_failed", "connector": "sms"}),
    ]
    result = build_messages(events, system_prompt=None)
    assert result.reacting_to == 1


def test_delivery_failure_renderer_is_total_on_sparse_data() -> None:
    """Renderer never raises even with no connector/detail — it runs inside the
    per-wake replay where a raise would brick the session."""
    content = build_messages(
        [_msg(1, "user", "x"), _lifecycle(2, {"event": "connector_delivery_failed"})],
        system_prompt=None,
    ).messages[-1]["content"]
    assert "not delivered" in content
