"""Group-send self-echo correlation.

signal-cli 0.14.x's JSON-RPC ``send`` returns a top-level ``timestamp``
for DM sends but a bare ``null`` for groups.  The timestamp does arrive
on the receive stream as a self-echo envelope (``sourceUuid == bot_uuid``
+ ``dataMessage.groupInfo`` + ``dataMessage.timestamp``).  ``signal_send``
registers a future before issuing the RPC and waits briefly for the
echo dispatcher to resolve it; on timeout we degrade to the
no-timestamp result shape.

These tests cover the correlation in both directions:

- ``_maybe_resolve_self_echo`` matches by ``(phone, chat_id)`` FIFO and
  prunes stale (cancelled / timed-out) waiters at the front.
- ``signal_send`` registers + awaits + degrades cleanly.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import pytest

from aios_signal.connector import SignalConnector, _SignalConnectionState
from aios_signal.daemon import GroupInfo
from tests.conftest import (
    ALICE_UUID,
    BOB_UUID,
    BOT_UUID,
    CONNECTION_ID,
    GROUP_CHAT_ID,
    GROUP_RAW_ID,
    PHONE,
)


def _state() -> _SignalConnectionState:
    return _SignalConnectionState(
        phone=PHONE,
        bot_uuid=BOT_UUID,
        contact_names={},
        groups=[
            GroupInfo(
                id=GROUP_CHAT_ID,
                name="Tea Party",
                member_uuids=[ALICE_UUID, BOB_UUID],
            )
        ],
    )


def _self_echo_envelope(
    *,
    bot_uuid: str = BOT_UUID,
    group_raw_id: str = GROUP_RAW_ID,
    timestamp_ms: int = 1700000000000,
    message: str = "hello",
) -> dict[str, Any]:
    """Shape mirrors signal-cli 0.14.2 group self-echo envelopes
    captured via direct daemon probe during PR 8 smoke."""
    return {
        "source": bot_uuid,
        "sourceUuid": bot_uuid,
        "sourceName": "SmokeBot",
        "timestamp": timestamp_ms,
        "dataMessage": {
            "timestamp": timestamp_ms,
            "message": message,
            "groupInfo": {"groupId": group_raw_id, "groupName": "Tea Party"},
        },
    }


# ── _maybe_resolve_self_echo ────────────────────────────────────────


async def test_maybe_resolve_self_echo_pops_first_waiter(connector: SignalConnector) -> None:
    """A queued waiter resolves with the envelope's timestamp."""
    state = _state()
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    connector._pending_echoes[(PHONE, GROUP_CHAT_ID)] = deque([fut])
    connector._maybe_resolve_self_echo(state, _self_echo_envelope(timestamp_ms=12345))
    assert fut.done()
    assert fut.result() == 12345
    # Empty queue is pruned so we don't leak keys.
    assert (PHONE, GROUP_CHAT_ID) not in connector._pending_echoes


async def test_maybe_resolve_self_echo_skips_stale_futures(
    connector: SignalConnector,
) -> None:
    """A cancelled waiter at the head of the queue is drained so the next
    live waiter receives the echo."""
    stale: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    stale.cancel()
    fresh: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    connector._pending_echoes[(PHONE, GROUP_CHAT_ID)] = deque([stale, fresh])

    connector._maybe_resolve_self_echo(_state(), _self_echo_envelope(timestamp_ms=999))

    assert fresh.done()
    assert fresh.result() == 999


async def test_maybe_resolve_self_echo_ignores_non_bot_sender(
    connector: SignalConnector,
) -> None:
    """Echoes from other senders are not bot self-echoes; futures stay
    unresolved so we don't latch onto unrelated inbounds."""
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    connector._pending_echoes[(PHONE, GROUP_CHAT_ID)] = deque([fut])
    env = _self_echo_envelope()
    env["sourceUuid"] = ALICE_UUID  # not the bot

    connector._maybe_resolve_self_echo(_state(), env)

    assert not fut.done()


async def test_maybe_resolve_self_echo_handles_edit_envelope(
    connector: SignalConnector,
) -> None:
    """Edits arrive on the receive stream wrapped as
    ``envelope.editMessage.dataMessage`` (nested one level deeper than
    a normal send), with the new edit timestamp at the envelope root.
    Match should still fire so chained edits get their new sent_at_ms
    back from the tool result."""
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    connector._pending_echoes[(PHONE, GROUP_CHAT_ID)] = deque([fut])

    # Build a synthetic edit envelope matching the shape captured from
    # signal-cli 0.14.2 during PR 8 smoke: top-level ``editMessage``
    # with the original message's ``targetSentTimestamp`` and a nested
    # ``dataMessage`` carrying the new content + groupInfo.
    edit_envelope: dict[str, Any] = {
        "source": BOT_UUID,
        "sourceUuid": BOT_UUID,
        "timestamp": 555,
        "editMessage": {
            "targetSentTimestamp": 111,
            "dataMessage": {
                "timestamp": 555,
                "message": "✅ Edited",
                "groupInfo": {"groupId": GROUP_RAW_ID, "groupName": "QA"},
            },
        },
    }

    connector._maybe_resolve_self_echo(_state(), edit_envelope)

    assert fut.done()
    assert fut.result() == 555


async def test_maybe_resolve_self_echo_ignores_dm_echo(connector: SignalConnector) -> None:
    """DM self-echoes carry no ``groupInfo``; the DM send path is
    already getting its timestamp from the RPC return, so we skip."""
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    connector._pending_echoes[(PHONE, GROUP_CHAT_ID)] = deque([fut])
    env = _self_echo_envelope()
    del env["dataMessage"]["groupInfo"]

    connector._maybe_resolve_self_echo(_state(), env)

    assert not fut.done()


# ── signal_send end-to-end ──────────────────────────────────────────


async def test_signal_send_group_returns_sent_at_ms_from_echo(
    connector: SignalConnector,
) -> None:
    """The end-to-end path: signal_send for a group registers a future,
    the daemon emits a self-echo via _handle_envelope's None branch,
    signal_send picks up the timestamp, returns ``sent_at_ms``."""
    sent_ts = 1700000054321

    async def _fake_send(_method: str, _params: dict[str, Any]) -> Any:
        # signal-cli 0.14.x: group send returns null at the RPC layer.
        # The echo arrives via the receive stream concurrently — simulate
        # by routing a synthetic envelope through _maybe_resolve_self_echo
        # directly.  In production the dispatcher invokes this from
        # _inbound_dispatcher when the envelope lands on any peer's
        # account stream (group self-echoes arrive on RECEIVING peers'
        # streams, not on the sender's own stream).
        connector._maybe_resolve_self_echo(_state(), _self_echo_envelope(timestamp_ms=sent_ts))
        return None

    connector._daemon.rpc.call.side_effect = _fake_send  # type: ignore[union-attr]
    connector.state[CONNECTION_ID] = _state()

    result = await connector.signal_send(
        text="hello", chat_id=GROUP_CHAT_ID, connection_id=CONNECTION_ID
    )

    assert result == {"sent_at_ms": sent_ts}


async def test_signal_send_group_falls_back_when_echo_times_out(
    connector: SignalConnector,
) -> None:
    """When the self-echo never arrives (slow network, daemon stall),
    signal_send returns ``{"status": "ok"}`` after the deadline rather
    than hanging the tool call forever."""
    connector._daemon.rpc.call.return_value = None  # type: ignore[union-attr]
    connector.state[CONNECTION_ID] = _state()

    result = await connector.signal_send(
        text="silence", chat_id=GROUP_CHAT_ID, connection_id=CONNECTION_ID
    )

    assert result == {"status": "ok"}


async def test_inbound_dispatcher_resolves_echo_on_peer_account_stream(
    connector: SignalConnector,
) -> None:
    """Regression for the post-#6 smoke gap: signal-cli emits group
    self-echoes on the receiving peers' account streams (because
    those are the streams the message actually flows through),
    *not* on the sender's own account stream.  The dispatcher must
    resolve the echo regardless of which ``account`` arrived with
    the envelope — keyed by ``sourceUuid`` matching any known bot
    rather than by per-account routing.
    """
    # Register the bot under its own account.
    connector.state[CONNECTION_ID] = _state()
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    connector._pending_echoes[(PHONE, GROUP_CHAT_ID)] = deque([fut])

    # Mock listener.messages() to emit a single self-echo envelope
    # arriving on a DIFFERENT account stream (e.g. a peer bot like
    # Ultron whose account signal-cli also serves).
    peer_account = "+19092871349"

    class _StubListener:
        async def messages(self) -> Any:
            yield peer_account, _self_echo_envelope(timestamp_ms=777)

    assert connector._daemon is not None
    connector._daemon.listener = _StubListener()  # type: ignore[assignment]

    # Dispatcher runs until listener.messages exhausts; our generator
    # yields once then completes.
    await connector._inbound_dispatcher()

    assert fut.done()
    assert fut.result() == 777


async def test_signal_send_cleans_up_echo_future_when_rpc_raises(
    connector: SignalConnector,
) -> None:
    """If ``rpc.call`` raises (e.g. signal-cli's libsignal
    ``InvalidSessionException`` when a group member has no
    established protocol session yet), the pre-registered echo
    future must be cancelled rather than left orphan at the head of
    the deque.  Without the cleanup, a subsequent successful send's
    echo would resolve the orphan with the WRONG message's
    timestamp.
    """
    from aios_signal.errors import RpcError

    connector._daemon.rpc.call.side_effect = RpcError("send failed")  # type: ignore[union-attr]
    connector.state[CONNECTION_ID] = _state()

    with pytest.raises(RpcError):
        await connector.signal_send(
            text="explodes", chat_id=GROUP_CHAT_ID, connection_id=CONNECTION_ID
        )

    # Any future registered before the raise must be done (cancelled
    # by the finally block) so the drain-stale logic in
    # _maybe_resolve_self_echo prunes it before the next send.
    queue = connector._pending_echoes.get((PHONE, GROUP_CHAT_ID), deque())
    assert all(fut.done() for fut in queue)


async def test_signal_send_dm_does_not_register_echo_future(
    connector: SignalConnector,
) -> None:
    """DMs skip the echo path entirely — signal-cli returns their
    timestamp inline, so a pending-echoes entry would leak."""
    connector._daemon.rpc.call.return_value = {"timestamp": 999}  # type: ignore[union-attr]
    connector.state[CONNECTION_ID] = _state()

    result = await connector.signal_send(text="dm", chat_id=ALICE_UUID, connection_id=CONNECTION_ID)

    assert result == {"sent_at_ms": 999}
    assert (PHONE, ALICE_UUID) not in connector._pending_echoes
