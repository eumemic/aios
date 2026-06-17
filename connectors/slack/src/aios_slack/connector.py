"""Slack connector built on the aios-connector-http SDK.

MVP slice 1/4 — the connection layer (design §3.1-§3.3).  This slice
stands up the package, the Socket-Mode transport, and the
:meth:`SlackConnector.serve_connection` lifecycle.  It does **not** yet
parse, gate, or emit inbound events: the socket listener acks the
envelope and pushes the *raw* Slack event onto a per-connection queue,
and the drain task only logs.  Parsing / mention-gating / ``emit_inbound``
land in slice B; the outbound ``@tool`` vocabulary in a later slice.

Multi-connection runtime container: one container can serve N
connections of type ``"slack"``, each bound to one Slack workspace
install (one ``bot_token`` + one ``app_token``).  Each connection gets
its own ``AsyncWebClient`` + ``AsyncSocketModeClient`` because every
bot token is a distinct Slack identity with its own socket and event
stream — there is no useful sharing across tokens.

Lifecycle:

* :meth:`serve_connection` per connection: build the Web + Socket
  clients from the connection's secrets, ``auth.test`` for identity,
  enforce the fail-closed install-identity gate (INV-5), register an
  ack-first socket listener, then race the socket connection with an
  inbound drainer in a :class:`asyncio.TaskGroup`.  On cancellation,
  the ``finally`` closes the socket client.
* :meth:`teardown` is a no-op container-wide; per-connection cleanup is
  owned by ``serve_connection``'s ``finally`` (exactly like telegram).
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

import structlog
from aios_connector_http import HttpConnector
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.socket_mode.aiohttp import SocketModeClient as AsyncSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

log = structlog.get_logger(__name__)

# The bare AsyncWebClient retries only ``ConnectionError`` — never an HTTP
# 429.  Append a rate-limit retry handler at construction so a throttle is
# absorbed transparently (honoring ``Retry-After``) inside the call rather
# than surfacing as an opaque ``is_error`` result.  ``>= 2`` per design §3.3.
_RATE_LIMIT_MAX_RETRIES = 2


@dataclass
class _SlackConnectionState:
    """Per-connection Slack plumbing + identity caches.

    ``bot_user_id`` and ``team_id`` are cached from ``auth.test``.
    ``team_id`` is the durable workspace install identity and the
    ``external_account_id`` address segment; ``bot_user_id`` is cached
    for self-filtering / mention detection in slice B but is **not** an
    address segment.
    """

    web_client: AsyncWebClient
    socket_client: AsyncSocketModeClient
    bot_user_id: str
    team_id: str
    # Slice A only enqueues the raw Slack event dict; parsing into a typed
    # ``InboundMessage`` is slice B.
    inbound_queue: asyncio.Queue[dict[str, Any]]


class SlackConnector(HttpConnector):
    connector = "slack"
    state: dict[str, _SlackConnectionState]

    # ── lifecycle ─────────────────────────────────────────────────────

    async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        """Build the Slack clients for this connection and run its loops.

        Each connection has its own ``bot_token`` + ``app_token`` (one
        Socket-Mode connection per workspace install); no sharing across
        connections.  Races the socket connection + an inbound drainer in
        a :class:`asyncio.TaskGroup`; on cancellation both stop and the
        ``finally`` closes the socket client.

        Refuses to serve (INV-5, fail-closed) when ``auth.test``'s
        ``team_id`` does not match the connection's ``external_account_id``
        — a wrong-token paste becomes a loud refusal instead of a silent
        split-brain.
        """
        bot_token = secrets.get("bot_token")
        app_token = secrets.get("app_token")
        if not bot_token:
            raise RuntimeError(
                f"slack connection {connection_id!r} requires a 'bot_token' entry in its secrets"
            )
        if not app_token:
            raise RuntimeError(
                f"slack connection {connection_id!r} requires an 'app_token' entry in its secrets"
            )

        # Build the Web client first, with the rate-limit retry handler wired
        # at construction (design §3.3 — the bare client never retries 429).
        web_client = AsyncWebClient(token=bot_token)
        web_client.retry_handlers.append(
            AsyncRateLimitErrorRetryHandler(max_retry_count=_RATE_LIMIT_MAX_RETRIES)
        )

        socket_client = AsyncSocketModeClient(app_token=app_token, web_client=web_client)

        try:
            auth = await web_client.auth_test()
            bot_user_id = str(auth["user_id"])
            team_id = str(auth["team_id"])

            # INV-5 fail-closed identity gate.  ``external_account_id`` is the
            # durable workspace install identity the connection was created
            # against; if ``auth.test`` reports a different ``team_id`` the
            # operator pasted the wrong workspace's token.  Refuse to serve.
            expected_account_id = self._external_account_id(connection_id)
            if expected_account_id is not None and team_id != expected_account_id:
                log.error(
                    "slack.install.identity_mismatch",
                    connection_id=connection_id,
                    expected_team_id=expected_account_id,
                    actual_team_id=team_id,
                    bot_user_id=bot_user_id,
                )
                await self.emit_lifecycle(
                    connection_id=connection_id,
                    event="slack.install.identity_mismatch",
                    reason="team_id_mismatch",
                    data={
                        "expected_team_id": expected_account_id,
                        "actual_team_id": team_id,
                    },
                )
                return

            state = _SlackConnectionState(
                web_client=web_client,
                socket_client=socket_client,
                bot_user_id=bot_user_id,
                team_id=team_id,
                inbound_queue=asyncio.Queue(),
            )
            self.state[connection_id] = state
            self._register_listener(connection_id, state)
            log.info(
                "slack.connection.ready",
                connection_id=connection_id,
                team_id=team_id,
                bot_user_id=bot_user_id,
            )

            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    self._run_socket(state),
                    name=f"slack-socket-{connection_id}",
                )
                tg.create_task(
                    self._drain_queue(connection_id, state),
                    name=f"slack-drain-{connection_id}",
                )
        finally:
            await self._close_socket(socket_client)

    def _external_account_id(self, connection_id: str) -> str | None:
        """Return the connection's ``external_account_id`` (Slack ``team_id``).

        Read off the SDK base class's per-connection state, populated by
        ``_on_connection_added`` before the ``serve_connection`` worker is
        spawned.  Returns ``None`` only if the slot is somehow absent (e.g.
        a directly-invoked ``serve_connection`` in a test that did not go
        through discovery) so the identity gate degrades to "no expectation"
        rather than crashing.
        """
        conn = self._connections.get(connection_id)
        return conn.external_account_id if conn is not None else None

    def _register_listener(self, connection_id: str, state: _SlackConnectionState) -> None:
        """Wire an ack-first socket listener that enqueues the raw event.

        Ack-first is mandatory (design §3.2, correctness sev 82): the async
        Socket-Mode client does NOT auto-acknowledge.  We
        ``send_socket_mode_response`` at the very top — before any parse or
        enqueue — so the ~3s ack window is never missed under load.  Missing
        it means Slack redelivers 3x then throttles/closes the socket, a
        silently-dead bot.  Parsing / gating is slice B; here we just enqueue.
        """

        async def on_request(client: AsyncSocketModeClient, req: SocketModeRequest) -> None:
            # 1. Ack FIRST, before any work.
            await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
            # 2. Then enqueue the raw envelope for off-ack-path handling.
            await state.inbound_queue.put(
                {"type": req.type, "envelope_id": req.envelope_id, "payload": req.payload}
            )

        state.socket_client.socket_mode_request_listeners.append(on_request)

    async def _run_socket(self, state: _SlackConnectionState) -> None:
        """Open the Socket-Mode connection and keep the task alive.

        ``connect()`` establishes the WebSocket and returns once the
        background monitor + message receiver are running; the listener
        fires from that receiver.  We then block forever so this task owns
        the socket's lifetime — cancellation (``removed`` / shutdown) unwinds
        into ``serve_connection``'s ``finally`` which closes the client.
        """
        await state.socket_client.connect()
        await asyncio.Event().wait()

    async def _drain_queue(self, connection_id: str, state: _SlackConnectionState) -> None:
        """Drain the per-connection inbound queue.

        Slice A only logs the raw event off the ack path; parsing, gating,
        and ``emit_inbound`` land in slice B.  Running this in its own task
        (not the listener) keeps the ack path unblocked.
        """
        while True:
            event = await state.inbound_queue.get()
            log.info(
                "slack.inbound.raw",
                connection_id=connection_id,
                team_id=state.team_id,
                event_type=event.get("type"),
                envelope_id=event.get("envelope_id"),
            )

    @staticmethod
    async def _close_socket(socket_client: AsyncSocketModeClient) -> None:
        """Best-effort Socket-Mode shutdown sequence."""
        with contextlib.suppress(Exception):
            await socket_client.disconnect()
        with contextlib.suppress(Exception):
            await socket_client.close()
