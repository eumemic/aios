"""Unit tests for the SSE channel/chat_type filter (#1613).

``_passes_channel_filter`` decides whether a serialized event row is yielded on
a channel-scoped SSE/wait stream. The critical invariant: NULL-channel
lifecycle/terminal rows ALWAYS pass so ``done``/``terminated``/archive reach the
consumer even when a channel filter is active.
"""

from __future__ import annotations

from typing import Any

from aios.api.sse import _passes_channel_filter

SIGNAL_DM = "6c21718f-f095-483f-8cd6-610137d581aa"
SIGNAL_GROUP = "abcDEF123_-=="


def _msg(channel: str | None) -> dict[str, Any]:
    return {"kind": "message", "channel": channel, "data": {"role": "user"}}


def _lifecycle() -> dict[str, Any]:
    return {"kind": "lifecycle", "channel": None, "data": {"status": "terminated"}}


def test_no_filter_passes_everything() -> None:
    assert _passes_channel_filter(_msg("signal/b/A"), None, None)
    assert _passes_channel_filter(_msg(None), None, None)
    assert _passes_channel_filter(_lifecycle(), None, None)


def test_channel_filter_matches_and_excludes() -> None:
    assert _passes_channel_filter(_msg("signal/b/A"), ["signal/b/A"], None)
    assert not _passes_channel_filter(_msg("signal/b/B"), ["signal/b/A"], None)


def test_channel_filter_or_semantics() -> None:
    chans = ["signal/b/A", "signal/b/B"]
    assert _passes_channel_filter(_msg("signal/b/A"), chans, None)
    assert _passes_channel_filter(_msg("signal/b/B"), chans, None)
    assert not _passes_channel_filter(_msg("signal/b/C"), chans, None)


def test_null_channel_lifecycle_always_passes_under_filter() -> None:
    # The dropped-DM-class regression guard: a channel-scoped stream must still
    # deliver terminal/lifecycle rows so the consumer sees end-of-stream.
    assert _passes_channel_filter(_lifecycle(), ["signal/b/A"], None)
    assert _passes_channel_filter(_msg(None), ["signal/b/A"], None)
    assert _passes_channel_filter(_lifecycle(), None, "dm")


def test_chat_type_filter() -> None:
    assert _passes_channel_filter(_msg(f"signal/b/{SIGNAL_DM}"), None, "dm")
    assert not _passes_channel_filter(_msg(f"signal/b/{SIGNAL_DM}"), None, "group")
    assert _passes_channel_filter(_msg(f"signal/b/{SIGNAL_GROUP}"), None, "group")
    assert not _passes_channel_filter(_msg(f"signal/b/{SIGNAL_GROUP}"), None, "dm")


def test_channel_and_chat_type_compose() -> None:
    chan = f"signal/b/{SIGNAL_DM}"
    assert _passes_channel_filter(_msg(chan), [chan], "dm")
    assert not _passes_channel_filter(_msg(chan), [chan], "group")
