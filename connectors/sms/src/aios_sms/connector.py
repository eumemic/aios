"""SMS connector built on the aios-connector-http SDK (design §3).

Scope of this slice (#1253) — the **inbound / transport layer**:

* ``setup(tg)`` stands up **one** container-wide aiohttp webhook
  listener (``POST /twilio/inbound`` + ``POST /twilio/status``) via
  ``tg.create_task`` — the signal shared-daemon fan-out shape.
* ``serve_connection`` fetches the connection's secrets, normalizes its
  ``from_number``, registers ``from_number → (connection_id, auth_token,
  queue)`` in the shared demux map, then drains its per-connection queue
  → ``emit_inbound(chat_id=From, sender={"display_name": From},
  content=Body, event_id=MessageSid)``.

``connector = "sms"`` is **provider-neutral** (design §3.7): Twilio is a
per-connection discriminator, not a connector type. Out of scope for
this slice (later #1252 children): ``sms_send``, the consent ledger,
spend/registration gates, and the status-callback → session correlation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from aiohttp import web
from aios_connector_http import HttpConnector

from .addressing import normalize_e164
from .config import Settings
from .state import SmsConnectionState
from .webhook import DemuxEntry, InboundEnvelope, WebhookListener

__all__ = ["SmsConnector"]

log = structlog.get_logger(__name__)

# The single-source inbound dedup invariant (design §3.2): the durable
# ``event_id`` is ALWAYS Twilio's ``MessageSid``. The deprecated
# ``SmsSid`` / ``SmsMessageSid`` aliases carry the same value but must
# never be the source — a test asserts this so a future refactor can't
# silently switch the key and break cross-restart dedup.
_EVENT_ID_FIELD = "MessageSid"
_FORBIDDEN_EVENT_ID_FIELDS = ("SmsSid", "SmsMessageSid")


class SmsConnector(HttpConnector):
    connector = "sms"
    state: dict[str, SmsConnectionState]

    # ── emit-retry tunables (transport errors on emit_inbound) ─────────
    #
    # Twilio already received ``200`` for every envelope in the queue, so
    # it will NOT redeliver (design §3.2). On a transient transport failure
    # (``httpx.RequestError`` — connect/timeout/protocol; no response
    # object, so ``emit_inbound``'s ``is_error`` guard is unreachable) the
    # connector holds the ONLY in-process copy and must retry before
    # dropping, or the 200-acked message is lost. The retry is bounded so a
    # sustained outage produces a bounded head-of-line stall (subsequent
    # envelopes pile up behind the held one, up to ``INBOUND_QUEUE_MAXSIZE``)
    # rather than retrying forever; after exhaustion the envelope is
    # logged (``sms.inbound.emit_dropped`` carries the ``MessageSid``) and
    # dropped, and the drain continues. A fatal 401/403 (``HTTPStatusError``,
    # NOT a ``RequestError``) still propagates and tears the connection
    # down — auth-broken is not a transient blip.
    EMIT_RETRY_MAX_ATTEMPTS: int = 3
    EMIT_RETRY_BACKOFF_INITIAL: float = 0.5
    EMIT_RETRY_BACKOFF_MAX: float = 5.0

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        super().__init__(base_url=base_url, token=token)
        self._settings = settings or Settings()
        self._listener = WebhookListener(public_base_url=self._settings.public_base_url)
        self._runner: web.AppRunner | None = None

    # ── lifecycle ─────────────────────────────────────────────────────

    async def setup(self, tg: asyncio.TaskGroup) -> None:
        """Stand up the one container-wide webhook listener.

        Spawned under the runner's TaskGroup so an unhandled crash in the
        listener tears the container down (fail hard) rather than
        silently stalling inbound delivery — the signal-daemon contract
        (design §3.2).
        """
        self._runner = await self._listener.start(self._settings.host, self._settings.port)

        async def _serve_forever() -> None:
            # The AppRunner is already serving on its own; this task just
            # keeps a TaskGroup member alive for the listener's lifetime
            # and ensures clean teardown on cancellation.
            try:
                await asyncio.Event().wait()
            finally:
                await self._listener_cleanup()

        tg.create_task(_serve_forever(), name="sms-webhook-listener")

    async def teardown(self) -> None:
        await self._listener_cleanup()

    async def _listener_cleanup(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        """Register the connection in the demux map, then drain its queue.

        ``secrets`` must carry ``from_number`` (the AIOS-owned Twilio
        number) and ``auth_token`` (the verify key). Missing → raise;
        the SDK logs the failure under ``connector.connection.serve_failed``
        and keeps the container serving its other connections.
        """
        from_number_raw = secrets.get("from_number", "").strip()
        auth_token = secrets.get("auth_token", "").strip()
        if not from_number_raw:
            raise RuntimeError(f"sms connection {connection_id!r} requires a 'from_number' secret")
        if not auth_token:
            raise RuntimeError(f"sms connection {connection_id!r} requires an 'auth_token' secret")

        # Normalize ONCE here so the demux key, the state, and the
        # verify-key lookup all key on the same canonical value the
        # webhook routes ``To`` by. Asymmetric normalization is the
        # signal ``account.strip()`` message-loss class (design §3.3).
        from_number = normalize_e164(from_number_raw)

        state = SmsConnectionState(
            connection_id=connection_id,
            from_number=from_number,
            auth_token=auth_token,
        )
        self.state[connection_id] = state
        self._listener.register(
            from_number,
            DemuxEntry(
                connection_id=connection_id,
                auth_token=auth_token,
                queue=state.inbound_queue,
            ),
        )
        log.info(
            "sms.connection.ready",
            connection_id=connection_id,
            from_number=from_number,
        )
        # The drain loop's ``finally`` tears the connection down (unregister
        # the number + drop the state/queue). Transient transport errors on
        # ``emit_inbound`` MUST NOT escape here: ``_emit_envelope`` retries
        # (``httpx.RequestError``) then logs-and-drops so a blip does not
        # discard the entire 200-acked backlog. Only cancellation, fatal
        # 401/403 (``HTTPStatusError``), or a programmer bug reaches the
        # ``finally`` — tearing down is correct for those, not for a blip.
        try:
            while True:
                envelope = await state.inbound_queue.get()
                await self._emit_envelope(connection_id, envelope)
        finally:
            self._listener.unregister(from_number)
            self.state.pop(connection_id, None)

    # ── inbound drain ─────────────────────────────────────────────────

    async def _emit_envelope(self, connection_id: str, envelope: InboundEnvelope) -> None:
        """Drain one verified inbound → ``emit_inbound``.

        Runs off the webhook's critical path (design §3.2). ``chat_id`` is
        the peer ``From`` number (E.164); ``content`` is the message
        ``Body``; ``event_id`` is the single-source ``MessageSid``.
        ``From`` is a **routing key, not a trust anchor** (design §3.3,
        §5.2): the sender display name flows only through the typed
        ``sender`` dict, and provenance is stamped unverified for the
        model.

        Transport failures (``httpx.RequestError`` — connect/timeout/
        protocol; raised by ``emit_inbound`` because no ``response`` object
        exists, so its ``is_error`` guard is unreachable) are retried in a
        bounded loop. Twilio already received ``200``, so it will not
        redeliver; the connector holds the only in-process copy and must
        retry, or the message is lost. The retry is bounded (capped
        backoff over a few attempts) so a sustained outage causes a bounded
        head-of-line stall, not a forever-retry; after exhaustion the
        envelope is logged (``sms.inbound.emit_dropped`` carries the
        ``MessageSid`` for operator recovery) and dropped, and the drain
        continues. A fatal 401/403 (``httpx.HTTPStatusError``, NOT a
        ``RequestError``) propagates and tears the connection down — the
        right response to broken auth, not to a transient blip.
        """
        params = envelope.params
        from_number = normalize_e164(params.get("From", ""))
        body = params.get("Body", "")
        event_id = params.get(_EVENT_ID_FIELD, "")
        if not from_number or not event_id:
            log.warning(
                "sms.inbound.dropped",
                connection_id=connection_id,
                reason="missing_from_or_message_sid",
                message_sid=event_id or None,
            )
            return

        backoff = self.EMIT_RETRY_BACKOFF_INITIAL
        for attempt in range(1, self.EMIT_RETRY_MAX_ATTEMPTS + 1):
            try:
                result = await self.emit_inbound(
                    connection_id=connection_id,
                    chat_id=from_number,
                    sender={"display_name": from_number},
                    content=body,
                    event_id=event_id,
                    metadata=self._inbound_metadata(params),
                    timestamp=None,
                )
            except httpx.RequestError as exc:
                # Transport failure: no response, so the server may never
                # have seen the request. Retry the same ``event_id``
                # (idempotent against the ``inbound_acks`` ledger the server
                # may already have written on the timeout family). Do NOT
                # let this escape — the drain loop's ``finally`` would
                # discard the whole 200-acked backlog.
                if attempt >= self.EMIT_RETRY_MAX_ATTEMPTS:
                    log.error(
                        "sms.inbound.emit_dropped",
                        connection_id=connection_id,
                        chat_id=from_number,
                        event_id=event_id,
                        attempts=attempt,
                        error=type(exc).__name__,
                    )
                    return
                log.warning(
                    "sms.inbound.emit_retry",
                    connection_id=connection_id,
                    chat_id=from_number,
                    event_id=event_id,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.EMIT_RETRY_BACKOFF_MAX)
                continue
            log.info(
                "sms.inbound.emitted",
                connection_id=connection_id,
                chat_id=from_number,
                event_id=event_id,
                deduped=bool(result and result.get("deduped")),
            )
            return

    @staticmethod
    def _inbound_metadata(params: dict[str, str]) -> dict[str, Any]:
        """Non-reserved metadata for one inbound.

        ``From`` is unauthenticated and trivially spoofable, so SMS sender
        provenance is stamped **explicitly unverified** toward the model
        (design §5.2): identity-gated tools must refuse on SMS-origin
        alone. We deliberately do NOT put ``sender_name`` / ``channel`` /
        ``attachments`` / ``platform_timestamp`` here — they are
        server-stripped reserved keys (design §5.5).
        """
        meta: dict[str, Any] = {"sender_verified": False}
        num_segments = params.get("NumSegments")
        if num_segments:
            meta["num_segments"] = num_segments
        return meta
