"""Coverage for the model-visible connector delivery/edit acks (#1341).

The success-path complement to ``connector_delivery_failed`` (#1308): a
connector appends a ``connector_message_delivered`` / ``connector_message_edited``
lifecycle event when the platform confirmed an outbound the model consciously
sent was delivered, or that an edit landed. Like the delivery-failure notice it
must:
  * be a member of the ``MODEL_VISIBLE_LIFECYCLE_EVENTS`` allowlist (so it both
    renders and survives windowing), and
  * render as a bracketed user-role notice at its seq position via a renderer
    distinct from the FS-loss / delivery-failure renderers, and
  * NOT advance ``reacting_to`` in the pure replay — acks are informational; a
    wake (if any) comes from the lifecycle *route*, not ``build_messages``,
  * be total on sparse ``data`` (``data={}`` never raises).
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


def test_ack_events_in_allowlist() -> None:
    """Both reserved ack events are in the model-visible allowlist, so the
    windowing read carries them through and the renderer dispatches on them."""
    assert "connector_message_delivered" in MODEL_VISIBLE_LIFECYCLE_EVENTS
    assert "connector_message_edited" in MODEL_VISIBLE_LIFECYCLE_EVENTS


def test_delivered_ack_renders_as_user_notice() -> None:
    events = [
        _msg(1, "user", "ping the customer"),
        _lifecycle(
            2,
            {
                "event": "connector_message_delivered",
                "connector": "sms",
                "data": {"platform_message_id": "SM123", "tool_call_id": "call_1"},
            },
        ),
    ]
    result = build_messages(events, system_prompt=None)
    notices = [
        m
        for m in result.messages
        if m["role"] == "user" and isinstance(m["content"], str) and "delivered" in m["content"]
    ]
    assert len(notices) == 1
    content = notices[0]["content"]
    assert content.startswith("[") and content.endswith("]")
    assert "sms" in content


def test_edited_ack_renders_distinct_copy() -> None:
    content = build_messages(
        [
            _msg(1, "user", "fix that typo"),
            _lifecycle(2, {"event": "connector_message_edited", "connector": "slack"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    assert content.startswith("[") and content.endswith("]")
    assert "edit" in content.lower()
    assert "slack" in content


def test_ack_uses_its_own_renderer_not_fs_or_failure() -> None:
    """An ack notice must not borrow the FS-loss or delivery-failure copy."""
    content = build_messages(
        [
            _msg(1, "user", "x"),
            _lifecycle(2, {"event": "connector_message_delivered", "connector": "sms"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    assert "filesystem" not in content
    assert "not delivered" not in content


def test_render_ack_notice_total_non_stimulus() -> None:
    """The ack is NOT stimulus-bearing in the pure replay and the renderer is
    total on sparse ``data`` (``data={}`` does not raise)."""
    events = [
        _msg(1, "user", "send it"),
        _lifecycle(5, {"event": "connector_message_delivered"}),
        _lifecycle(6, {"event": "connector_message_edited"}),
    ]
    result = build_messages(events, system_prompt=None)
    # Watermark stays at the last real stimulus (the user message).
    assert result.reacting_to == 1
    # Both rendered without raising.
    notices = [
        m for m in result.messages if m["role"] == "user" and m["content"].startswith("[Your")
    ]
    assert len(notices) == 2
