"""Slack connector built on the aios-connector-http SDK.

MVP slices 1+2/4 — the connection layer (design §3.1-§3.3) plus the
inbound decision layer (§3.4, §3.6).  Slice 1 stood up the package, the
Socket-Mode transport, and the :meth:`SlackConnector.serve_connection`
lifecycle.  Slice 2 adds inbound normalization and the four
connector-side gates: the drain task now parses each raw Slack event,
runs the self/bot-loop, cross-app/team, subtype, and mention gates (all
pure functions in :mod:`aios_slack.parse`), and on a ``FORWARD`` outcome
emits a normalized inbound via :meth:`emit_inbound`.  ``message_changed``
is routed to a non-emitting system path; mention-gated drops are
fail-quiet.  The outbound ``@tool`` vocabulary lands in a later slice.

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
from dataclasses import dataclass, field
from typing import Any

import structlog
from aios_connector_http import HttpConnector
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.socket_mode.aiohttp import SocketModeClient as AsyncSocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

from .parse import GateOutcome, InboundMessage, gate

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
    # The transport enqueues the raw Slack envelope dict
    # ``{type, envelope_id, payload}``; the drain task parses + gates it.
    inbound_queue: asyncio.Queue[dict[str, Any]]
    # ``api_app_id`` is not returned by ``auth.test``; it rides on every
    # Socket-Mode event payload.  Cached lazily from the first event so the
    # cross-app gate (§3.6 gate 2) has an expectation to enforce; until then
    # it is ``None`` and the gate fails open on the app dimension (the
    # ``team_id`` dimension still enforces).
    api_app_id: str | None = None
    # The per-thread sent-``ts`` set powering the ``bot_thread_participant``
    # implicit-mention bypass (§3.6 gate 4).  Slice C populates this from the
    # ``ts`` of every ``slack_send`` the bot makes into a thread; here it
    # starts empty and is consulted read-only by the gate.
    bot_thread_ts: set[str] = field(default_factory=set)


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
        spawned.

        The ``None`` branch is **unreachable in production**: the runtime only
        spawns ``serve_connection`` for connections it has already registered
        via ``_on_connection_added``, so the slot is always present and the
        INV-5 identity gate always has an expectation to enforce.  We keep the
        ``None``-degrades-to-"no expectation" fallback solely so a unit test
        can drive ``serve_connection`` directly without going through
        discovery; it must never silence the gate against a real install.
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
        """Drain the per-connection inbound queue: parse, gate, emit.

        Runs off the ack path (its own task, not the listener) so gating +
        ``emit_inbound`` never eat into the 3s Socket-Mode ack window.

        Each iteration's handling is wrapped in a guard so a transient
        failure (a malformed envelope, a parse/emit error) is logged and
        the next event is processed, rather than escaping into the
        :class:`asyncio.TaskGroup` and tearing down the whole connection —
        and with it the socket task — over one bad envelope.
        ``CancelledError`` is re-raised so shutdown still unwinds promptly.
        """
        while True:
            envelope = await state.inbound_queue.get()
            try:
                await self._handle_envelope(connection_id, state, envelope)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "slack.inbound.drain_error",
                    connection_id=connection_id,
                    team_id=state.team_id,
                    envelope_id=envelope.get("envelope_id"),
                )

    async def _handle_envelope(
        self,
        connection_id: str,
        state: _SlackConnectionState,
        envelope: dict[str, Any],
    ) -> None:
        """Parse one Socket-Mode envelope, run the gates, and maybe emit.

        Only ``events_api`` envelopes carrying a ``message`` event are
        actionable for slice 2; everything else (``hello``, slash
        commands, interactivity, non-message events) is logged and
        ignored.  The four gates (``aios_slack.parse.gate``) decide the
        disposition:

        * ``FORWARD``  → build the normalized inbound and ``emit_inbound``.
        * ``DROP``     → fail-quiet (mention-gated drops would be recorded
          to per-channel pending-history in slice C; for now they are
          observably logged and dropped).
        * ``DIVERT_EDIT`` → ``message_changed`` system path; non-emitting.
        """
        payload = envelope.get("payload")
        if envelope.get("type") != "events_api" or not isinstance(payload, dict):
            log.debug(
                "slack.inbound.skip_non_event",
                connection_id=connection_id,
                envelope_type=envelope.get("type"),
            )
            return

        event = payload.get("event")
        if not isinstance(event, dict) or event.get("type") != "message":
            return

        # Cache ``api_app_id`` off the first event so the cross-app gate has
        # an expectation to enforce on subsequent events.
        api_app_id = payload.get("api_app_id")
        if isinstance(api_app_id, str) and state.api_app_id is None:
            state.api_app_id = api_app_id

        decision = gate(
            event,
            bot_user_id=state.bot_user_id,
            team_id=state.team_id,
            api_app_id=state.api_app_id,
            bot_thread_ts=frozenset(state.bot_thread_ts),
        )

        if decision.outcome is GateOutcome.FORWARD and decision.message is not None:
            await self._emit_message(connection_id, state, decision.message)
            return

        log.info(
            "slack.inbound.gated",
            connection_id=connection_id,
            team_id=state.team_id,
            outcome=decision.outcome.value,
            reason=decision.reason,
            record_pending=decision.record_pending,
        )

    async def _emit_message(
        self,
        connection_id: str,
        state: _SlackConnectionState,
        msg: InboundMessage,
    ) -> None:
        """Forward a gate-passing normalized inbound to aios.

        ``chat_id`` is the bare Slack conversation id (threads share the
        channel session, §3.4).  Slack-specific extras ride
        ``connector_metadata`` under **non-reserved** keys only; ``sender``
        carries the opaque id + the (already-sanitized) display name.
        """
        sender_payload: dict[str, Any] = {
            "id": msg.sender_id,
            "display_name": msg.sender_name,
        }
        metadata = build_metadata(msg, state)
        await self.emit_inbound(
            connection_id=connection_id,
            event_id=msg.event_id,
            chat_id=msg.chat_id,
            sender=sender_payload,
            content=msg.text,
            metadata=metadata,
        )

    @staticmethod
    async def _close_socket(socket_client: AsyncSocketModeClient) -> None:
        """Best-effort Socket-Mode shutdown sequence."""
        with contextlib.suppress(Exception):
            await socket_client.disconnect()
        with contextlib.suppress(Exception):
            await socket_client.close()


def build_metadata(msg: InboundMessage, state: _SlackConnectionState) -> dict[str, Any]:
    """Stamp Slack-specific extras onto a normalized inbound (§3.4, §3.6).

    Everything Slack-specific rides ``connector_metadata`` under
    **non-reserved** keys only.  The load-bearing entry is ``thread_ts``:
    it is the model's ``slack_send(thread_ts=…)`` source read off the
    inbound metadata header (the Telegram ``reply_to_message_id`` shape) —
    threads share the channel session, so ``chat_id`` stays bare and
    ``thread_ts`` is the lone thread authority.

    ``self_mentioned`` is the derived convenience the harness renders;
    ``chat_kind`` / ``team_id`` / ``app_id`` / ``message_ts`` are the
    self-describing platform extras.
    """
    metadata: dict[str, Any] = {
        "channel": msg.chat_id,
        "chat_kind": msg.chat_kind,
        "team_id": state.team_id,
        "message_ts": msg.message_ts,
        "self_mentioned": state.bot_user_id in msg.mentions,
    }
    if msg.thread_ts is not None:
        metadata["thread_ts"] = msg.thread_ts
    if state.api_app_id is not None:
        metadata["app_id"] = state.api_app_id
    if msg.mentions:
        metadata["mentions"] = list(msg.mentions)
    if msg.edited:
        metadata["edited"] = True
        if msg.edit_ts is not None:
            metadata["edit_ts"] = msg.edit_ts
    return metadata
