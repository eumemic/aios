"""Webhook listener with the forwarded-header fallback gate (no configured
base URL) and the startup ingress self-test (design §3.2 step 3, §5.4, §6).
"""

from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from aios_sms.config import Settings
from aios_sms.connector import SmsConnector
from aios_sms.ingress import IngressPolicy
from aios_sms.selftest import run_ingress_self_test
from aios_sms.verify import compute_signature
from aios_sms.webhook import (
    TWILIO_SIGNATURE_HEADER,
    DemuxEntry,
    InboundQueue,
    WebhookListener,
)

OUR_NUMBER = "+18005551234"
PEER = "+14155550000"
AUTH_TOKEN = "test-auth-token"
ALLOWED = frozenset({"sms.example.com"})
# 127.0.0.1 is the aiohttp test client's socket peer, so trusting it lets
# us exercise the "trusted proxy" branch end-to-end through the app.
LOOPBACK_PROXY = frozenset({"127.0.0.1"})


def _listener_no_base(*, policy: IngressPolicy) -> tuple[WebhookListener, InboundQueue]:
    listener = WebhookListener(public_base_url=None, ingress_policy=policy)
    queue: InboundQueue = InboundQueue(maxsize=10)
    listener.register(
        OUR_NUMBER, DemuxEntry(connection_id="conn_1", auth_token=AUTH_TOKEN, queue=queue)
    )
    return listener, queue


async def _post(client: TestClient, params: dict[str, str], headers: dict[str, str]):
    return await client.post("/twilio/inbound", data=params, headers=headers)


async def test_fallback_trusted_proxy_and_allowed_host_verifies() -> None:
    """No configured base URL: a request from a trusted proxy with an
    allowed forwarded host reconstructs the URL and verifies."""
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=LOOPBACK_PROXY)
    listener, queue = _listener_no_base(policy=policy)
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    try:
        params = {"To": OUR_NUMBER, "From": PEER, "Body": "hi", "MessageSid": "SM1"}
        # The probe signs against the URL the listener will reconstruct:
        # https + the forwarded host + the /twilio/inbound path.
        signed_url = "https://sms.example.com/twilio/inbound"
        sig = compute_signature(AUTH_TOKEN, signed_url, params)
        resp = await _post(
            c,
            params,
            {
                TWILIO_SIGNATURE_HEADER: sig,
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "sms.example.com",
            },
        )
        assert resp.status == 200
        assert queue.qsize() == 1
    finally:
        await c.close()


async def test_fallback_untrusted_host_is_403() -> None:
    """A forwarded host NOT in allowedHosts is refused with a uniform 403
    even with an otherwise-valid signature for that host."""
    policy = IngressPolicy(allowed_hosts=ALLOWED, trusted_proxies=LOOPBACK_PROXY)
    listener, queue = _listener_no_base(policy=policy)
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    try:
        params = {"To": OUR_NUMBER, "From": PEER, "Body": "hi", "MessageSid": "SM1"}
        signed_url = "https://attacker.example/twilio/inbound"
        sig = compute_signature(AUTH_TOKEN, signed_url, params)
        resp = await _post(
            c,
            params,
            {
                TWILIO_SIGNATURE_HEADER: sig,
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "attacker.example",
            },
        )
        assert resp.status == 403
        assert queue.empty()
    finally:
        await c.close()


async def test_fallback_empty_policy_refuses_all_forwarded() -> None:
    """The default all-empty policy refuses every forwarded host (fail
    closed) — the safe default when no base URL is configured."""
    policy = IngressPolicy(allowed_hosts=frozenset(), trusted_proxies=frozenset())
    listener, queue = _listener_no_base(policy=policy)
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    try:
        params = {"To": OUR_NUMBER, "From": PEER, "Body": "hi", "MessageSid": "SM1"}
        sig = compute_signature(
            AUTH_TOKEN, "https://sms.example.com/twilio/inbound", params
        )
        resp = await _post(
            c,
            params,
            {
                TWILIO_SIGNATURE_HEADER: sig,
                "X-Forwarded-Proto": "https",
                "X-Forwarded-Host": "sms.example.com",
            },
        )
        assert resp.status == 403
        assert queue.empty()
    finally:
        await c.close()


# ── startup self-test (design §6) ─────────────────────────────────────


async def test_self_test_passes_against_correctly_configured_listener() -> None:
    """The probe POSTs a synthetic signed request through the listener's
    public URL and gets a 200 when the reconstructed signing URL matches.

    The listener under test must have its ``public_base_url`` equal to the
    live origin the probe targets — that is exactly the production
    invariant the self-test asserts.
    """
    # Bind a listener (placeholder base) just to learn a free port.
    bootstrap = WebhookListener(public_base_url="http://placeholder")
    server = TestServer(bootstrap.app)
    c = TestClient(server)
    await c.start_server()
    try:
        live_base = f"http://{c.host}:{c.port}"
        # The *serving* listener's configured base now matches its origin.
        bootstrap._public_base_url = live_base
        result = await run_ingress_self_test(bootstrap, public_base_url=live_base)
        assert result.ok, result.detail
        assert result.status == 200
    finally:
        await c.close()


async def test_self_test_fails_on_url_drift() -> None:
    """If the listener reconstructs a different signing URL than the probe
    signed (host/port/proto drift), the self-test catches it (403 → not ok)."""
    listener = WebhookListener(public_base_url="https://wrong-host.example")
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    try:
        # Probe through the live origin, but the listener thinks its public
        # base is wrong-host.example → reconstructed URL != probe-signed URL.
        result = await run_ingress_self_test(
            listener,
            public_base_url=f"http://{c.host}:{c.port}",
        )
        assert not result.ok
        assert result.status == 403
    finally:
        await c.close()


async def test_self_test_leaves_no_demux_residue() -> None:
    listener = WebhookListener(public_base_url="https://x.example")
    server = TestServer(listener.app)
    c = TestClient(server)
    await c.start_server()
    try:
        await run_ingress_self_test(listener, public_base_url=f"http://{c.host}:{c.port}")
        # The probe number must not linger in the demux map.
        assert listener.lookup("+18005550100") is None
    finally:
        await c.close()


# ── connector wiring (fail-fast on self-test failure) ─────────────────


async def test_connector_self_test_fail_fast_raises(monkeypatch) -> None:
    """With a misconfigured public base URL pointing at an unreachable
    origin, setup() fails the container start (fail-closed default)."""
    settings = Settings(
        host="127.0.0.1",
        port=0,
        public_base_url="http://127.0.0.1:1/twilio",  # unreachable
        self_test_fail_fast=True,
        self_test_timeout_seconds=2.0,
    )
    connector = SmsConnector(settings=settings, base_url="http://test", token="aios_runtime_x")

    raised = False
    try:
        async with __import__("asyncio").TaskGroup() as tg:
            await connector.setup(tg)
            tg._abort()  # type: ignore[attr-defined]
    except* RuntimeError:
        raised = True
    finally:
        await connector.teardown()
    assert raised


async def test_connector_self_test_skipped_without_base_url() -> None:
    """No public base URL ⇒ self-test skipped, setup() does not raise."""
    settings = Settings(host="127.0.0.1", port=0, public_base_url=None)
    connector = SmsConnector(settings=settings, base_url="http://test", token="aios_runtime_x")
    import asyncio

    async with asyncio.TaskGroup() as tg:
        await connector.setup(tg)
        # Listener is up and the self-test was skipped (no exception).
        tg._abort()  # type: ignore[attr-defined]
    await connector.teardown()
