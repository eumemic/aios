"""The container-wide aiohttp webhook listener (design §3.2).

**One** shared listener per container — the signal shared-daemon
fan-out shape (design §3.2). It exposes two routes, both Twilio webhooks
carrying ``X-Twilio-Signature``, both **verified before parse**:

* ``POST /twilio/inbound`` — MO messages → enqueue on the per-connection
  queue → return ``200`` + empty TwiML immediately. ``serve_connection``
  drains the queue and calls ``emit_inbound`` off this critical path.
* ``POST /twilio/status`` — delivery status callbacks. In this slice the
  route exists and is verified, but the correlation→session surfacing is
  a later slice (design §3.5); here it acks ``204`` after verification so
  Twilio stops retrying a *verified* callback.

Routing is **route-then-verify** (design §3.2 step 2): we read the raw
body, decode the urlencoded params, route by the normalized ``To``
number (inbound) / ``From`` number (status) to the connection's cached
``auth_token``, then verify. ``To`` is a *signed* parameter, so an
attacker cannot alter it without holding the token; a misroute verifies
against the wrong connection's distinct token and fails closed.

Security posture baked in (design §5.3):

* **pre-parse body cap** (``MAX_BODY_BYTES``) before reading the form;
* **off-loop HMAC** via ``asyncio.to_thread`` so a verify flood cannot
  starve the single event loop;
* **bounded per-connection queue** that sheds on overflow;
* **uniform response** on any unverified/unroutable request so the
  endpoint is not a number-enumeration oracle;
* **cold-start → transient 5xx** (not 403) when the ``To`` number's
  connection has not yet been discovered, so Twilio retries once
  discovery completes. Property: *cold-start inbound is briefly
  unavailable per number, never briefly unauthenticated.*
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import parse_qsl

import structlog
from aiohttp import web

from .addressing import normalize_e164
from .config import (
    EMPTY_TWIML,
    MAX_BODY_BYTES,
    TWIML_CONTENT_TYPE,
)
from .verify import is_valid, reconstruct_signed_url

__all__ = [
    "TWILIO_SIGNATURE_HEADER",
    "DemuxEntry",
    "InboundEnvelope",
    "InboundQueue",
    "WebhookListener",
]

log = structlog.get_logger(__name__)

TWILIO_SIGNATURE_HEADER = "X-Twilio-Signature"


@dataclass(frozen=True, slots=True)
class InboundEnvelope:
    """A verified Twilio inbound MO message, enqueued for the drain loop.

    ``params`` is the full decoded Twilio form so the drain loop can pull
    ``From`` / ``Body`` / ``MessageSid`` (and nothing else trusts any
    field until after verification, which already happened).
    """

    connection_id: str
    params: dict[str, str]


# Bounded queue of verified inbound envelopes, one per connection.
InboundQueue = asyncio.Queue[InboundEnvelope]


@dataclass(frozen=True, slots=True)
class DemuxEntry:
    """What the listener needs to verify + route one connection's webhooks."""

    connection_id: str
    auth_token: str
    queue: InboundQueue


class WebhookListener:
    """Shared aiohttp app demuxing Twilio webhooks by number → connection.

    ``serve_connection`` calls :meth:`register` as each connection is
    discovered (and :meth:`unregister` on removal); the routes look up
    the per-number entry, verify, and enqueue.
    """

    def __init__(
        self,
        *,
        public_base_url: str | None,
        verify_fn: Callable[..., Awaitable[bool]] | None = None,
    ) -> None:
        self._public_base_url = public_base_url
        # normalized E.164 → demux entry. The same map is keyed by our
        # owned number; inbound routes by ``To`` (= our number), status
        # routes by ``From`` (= our number) — both our-number axes.
        self._by_number: dict[str, DemuxEntry] = {}
        # Indirection so tests can stub verification; defaults to the
        # off-loop HMAC path.
        self._verify_fn = verify_fn or self._verify_offloop
        self.app = web.Application(client_max_size=MAX_BODY_BYTES)
        self.app.add_routes(
            [
                web.post("/twilio/inbound", self.handle_inbound),
                web.post("/twilio/status", self.handle_status),
            ]
        )

    # ── demux registration (called by serve_connection) ───────────────

    def register(self, number: str, entry: DemuxEntry) -> None:
        self._by_number[normalize_e164(number)] = entry

    def unregister(self, number: str) -> None:
        self._by_number.pop(normalize_e164(number), None)

    def lookup(self, number: str) -> DemuxEntry | None:
        return self._by_number.get(normalize_e164(number))

    # ── verification (off-loop) ───────────────────────────────────────

    async def _verify_offloop(
        self, auth_token: str, url: str, params: dict[str, str], signature: str | None
    ) -> bool:
        """Run the synchronous HMAC off the event loop (design §5.3c)."""
        return await asyncio.to_thread(is_valid, auth_token, url, params, signature)

    def _signed_url(self, request: web.Request) -> str:
        """Reconstruct the URL Twilio signed for this request."""
        return reconstruct_signed_url(
            configured_base_url=self._public_base_url,
            path=request.path,
            query=request.query_string,
            forwarded_proto=request.headers.get("X-Forwarded-Proto"),
            forwarded_host=request.headers.get("X-Forwarded-Host"),
            keep_port=True,  # SMS over HTTPS keeps the port (design §3.2)
        )

    # ── shared request handling ───────────────────────────────────────

    async def _read_params(self, request: web.Request) -> dict[str, str] | None:
        """Read the RAW body (pre-parse cap already enforced by aiohttp's
        ``client_max_size``) and decode the urlencoded form.

        Returns ``None`` on an unreadable/oversize body. We decode the
        params ourselves from the raw bytes (rather than
        ``request.post()``) so the exact ``name=value`` pairs Twilio
        signed are what we feed the HMAC — re-serializing would break
        signature equality.
        """
        try:
            raw = await request.read()
        except web.HTTPException:
            return None
        if len(raw) > MAX_BODY_BYTES:
            return None
        text = raw.decode("utf-8", errors="replace")
        return dict(parse_qsl(text, keep_blank_values=True))

    async def _verify_request(
        self, request: web.Request, route_key: str
    ) -> tuple[DemuxEntry, dict[str, str]] | web.Response:
        """Read → route-by-number → verify. Returns the routed entry +
        params on success, or a *uniform* error :class:`web.Response`.

        ``route_key`` is the param name to route by: ``"To"`` for inbound
        (our number is the recipient), ``"From"`` for status callbacks
        (our number is the sender).
        """
        # Cheap first filter: require the signature header's presence
        # before doing any parsing work (design §5.3b).
        signature = request.headers.get(TWILIO_SIGNATURE_HEADER)
        if not signature:
            return self._forbidden()

        params = await self._read_params(request)
        if params is None:
            # Oversize / unreadable — uniform 403 (don't leak the reason).
            return self._forbidden()

        number = params.get(route_key, "")
        if not number:
            return self._forbidden()

        entry = self.lookup(number)
        if entry is None:
            # Cold start: the connection for this number is not yet
            # discovered, so its auth_token isn't in the map. Return a
            # transient 5xx (NOT 403) so Twilio retries once discovery
            # completes. Briefly unavailable, never briefly unauthenticated.
            log.info(
                "sms.webhook.cold_start",
                route_key=route_key,
                number=normalize_e164(number),
            )
            return web.Response(status=503, text="connection not yet ready")

        url = self._signed_url(request)
        valid = await self._verify_fn(entry.auth_token, url, params, signature)
        if not valid:
            log.warning(
                "sms.webhook.verify_failed",
                connection_id=entry.connection_id,
                route_key=route_key,
            )
            return self._forbidden()

        return entry, params

    @staticmethod
    def _forbidden() -> web.Response:
        """Uniform 403 for every unverified/unroutable request (design
        §5.3d) so the endpoint is not a number-enumeration oracle."""
        return web.Response(status=403, text="forbidden")

    # ── routes ────────────────────────────────────────────────────────

    async def handle_inbound(self, request: web.Request) -> web.Response:
        """``POST /twilio/inbound`` — verify → enqueue → 200 + empty TwiML.

        The handler does the minimum on the critical path: verify and
        enqueue. ``emit_inbound`` happens off this path in the drain loop
        (design §3.2). On queue overflow we **shed** (still ack 200) — a
        dropped inbound is recoverable via Twilio retry; an OOM is not.
        """
        outcome = await self._verify_request(request, route_key="To")
        if isinstance(outcome, web.Response):
            return outcome
        entry, params = outcome

        envelope = InboundEnvelope(connection_id=entry.connection_id, params=params)
        try:
            entry.queue.put_nowait(envelope)
        except asyncio.QueueFull:
            log.warning(
                "sms.webhook.queue_full_shed",
                connection_id=entry.connection_id,
                message_sid=params.get("MessageSid"),
            )
        return self._twiml_ack()

    async def handle_status(self, request: web.Request) -> web.Response:
        """``POST /twilio/status`` — verify (route by ``From``) → ack.

        Status callbacks route by the **From** number (our owned number
        on an *outbound* message) — the opposite axis from inbound
        (design §3.5). In this slice we verify and ack; the durable
        MessageSid→session correlation and the session-targeted
        delivery-failure surfacing land in a later slice.
        """
        outcome = await self._verify_request(request, route_key="From")
        if isinstance(outcome, web.Response):
            return outcome
        entry, params = outcome
        log.info(
            "sms.webhook.status",
            connection_id=entry.connection_id,
            message_sid=params.get("MessageSid"),
            message_status=params.get("MessageStatus"),
        )
        return web.Response(status=204)

    @staticmethod
    def _twiml_ack() -> web.Response:
        """200 + empty TwiML — the synchronous reply carries no message;
        the agent replies async via ``sms_send`` (design §3.2)."""
        return web.Response(status=200, text=EMPTY_TWIML, content_type=TWIML_CONTENT_TYPE)

    # ── server lifecycle ──────────────────────────────────────────────

    async def start(self, host: str, port: int) -> web.AppRunner:
        """Bind + start the aiohttp app, returning its runner for cleanup.

        Spawned from the connector's ``setup(tg)`` so the listener's
        lifetime spans every connection (design §3.2).
        """
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log.info("sms.webhook.listening", host=host, port=port)
        return runner
