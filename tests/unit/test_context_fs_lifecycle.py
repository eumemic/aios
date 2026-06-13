"""Coverage for the FS-loss lifecycle notices in the context builder (§5.9).

The durable-session-sandbox loss events (``sandbox_fs_reset`` /
``sandbox_fs_expired`` / ``sandbox_fs_over_limit``) are the only non-``message``
events ``build_messages`` renders. They must:
  * render as a bracketed user-role notice at their seq position,
  * NOT advance the ``reacting_to`` watermark (they are not stimulus-bearing,
    so a GC/reset append never, on its own, wakes the session), and
  * leave other lifecycle events skipped (the allowlist is minimal).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aios.harness.context import build_messages
from aios.harness.window import WindowOmission
from aios.models.events import Event

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


def test_reset_event_renders_as_user_notice() -> None:
    events = [
        _msg(1, "user", "hello"),
        _lifecycle(2, {"event": "sandbox_fs_reset", "reason": "snapshot_missing"}),
    ]
    result = build_messages(events, system_prompt=None)
    notices = [
        m
        for m in result.messages
        if m["role"] == "user" and isinstance(m["content"], str) and "filesystem" in m["content"]
    ]
    assert len(notices) == 1
    assert notices[0]["content"].startswith("[")
    assert "fresh base filesystem" in notices[0]["content"]
    assert "/workspace" in notices[0]["content"]


def test_notice_does_not_advance_reacting_to() -> None:
    """The notice is NOT stimulus-bearing: the watermark stays at the last real
    stimulus (the user message), so the event never wakes the session."""
    events = [
        _msg(1, "user", "hello"),
        _lifecycle(5, {"event": "sandbox_fs_expired", "reason": "retention_ttl"}),
    ]
    result = build_messages(events, system_prompt=None)
    assert result.reacting_to == 1, "an FS-loss notice must not advance reacting_to"


def test_reset_reasons_render_distinctly() -> None:
    missing = build_messages(
        [
            _msg(1, "user", "x"),
            _lifecycle(2, {"event": "sandbox_fs_reset", "reason": "snapshot_missing"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    image_changed = build_messages(
        [
            _msg(1, "user", "x"),
            _lifecycle(2, {"event": "sandbox_fs_reset", "reason": "environment_image_changed"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    assert "could no longer be found" in missing
    assert "base image was changed" in image_changed


def test_expired_disk_pressure_vs_ttl_render_distinctly() -> None:
    ttl = build_messages(
        [
            _msg(1, "user", "x"),
            _lifecycle(2, {"event": "sandbox_fs_expired", "reason": "retention_ttl"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    pressure = build_messages(
        [
            _msg(1, "user", "x"),
            _lifecycle(2, {"event": "sandbox_fs_expired", "reason": "disk_pressure"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    assert "inactivity" in ttl
    assert "reclaim disk space" in pressure


def test_expired_account_cap_renders_distinctly() -> None:
    account_cap = build_messages(
        [
            _msg(1, "user", "x"),
            _lifecycle(2, {"event": "sandbox_fs_expired", "reason": "account_cap"}),
        ],
        system_prompt=None,
    ).messages[-1]["content"]
    # The account-cap cause must read as a quota crossing, not generic inactivity
    # or per-host disk pressure.
    assert "account" in account_cap.lower()
    assert "inactivity" not in account_cap


def test_over_limit_renders() -> None:
    msg = build_messages(
        [_msg(1, "user", "x"), _lifecycle(2, {"event": "sandbox_fs_over_limit"})],
        system_prompt=None,
    ).messages[-1]["content"]
    assert "size budget" in msg


def test_non_allowlisted_lifecycle_event_is_skipped() -> None:
    """A lifecycle event outside the allowlist (e.g. a github clone failure)
    is NOT rendered — the allowlist is deliberately minimal."""
    events = [
        _msg(1, "user", "hello"),
        _lifecycle(2, {"event": "github_clone_failed", "message": "boom"}),
    ]
    result = build_messages(events, system_prompt=None)
    assert all("boom" not in (m.get("content") or "") for m in result.messages)


def test_omission_marker_anchors_on_leading_notice() -> None:
    """Now that the windowing read carries FS-loss notices, the first retained
    event can be a notice — one in the gap between the last dropped message and
    the first retained message. The head omission marker anchors on
    ``events[0].created_at`` (#1044's non-empty-window invariant), which must
    hold whether ``events[0]`` is a message or a notice — accessing
    ``created_at`` on a lifecycle event must not raise. Pins that this no longer
    assumes a leading message."""
    events = [
        _lifecycle(7, {"event": "sandbox_fs_reset", "reason": "snapshot_missing"}),
        _msg(8, "user", "still here?"),
    ]
    omission = WindowOmission(began_at=datetime(2026, 6, 9, tzinfo=UTC), omitted_messages=4)

    # Must not raise: events[0] is a lifecycle notice, not a message.
    result = build_messages(events, system_prompt=None, omission=omission)

    # The head omission marker landed, and the notice itself still renders.
    assert result.messages[0]["role"] == "user"
    assert "messages" in result.messages[0]["content"]
    assert any(
        isinstance(m.get("content"), str) and "fresh base filesystem" in m["content"]
        for m in result.messages
    )
