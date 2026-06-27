"""Startup ingress self-test (design §6).

The public HTTPS ingress is a **first-class deliverable, not an ops
footnote** (design §6): the only inbound-reachable surface in the fleet,
and ``X-Twilio-Signature`` correctness depends on the TLS-termination
point, the configured public base URL, the trusted-proxy set, the
``allowedHosts`` list, and the **port-keeping rule**. A drift in any of
those (a proxy that rewrites the Host, a TLS cert that points at the wrong
SAN, a port that gets stripped) does not fail loudly — it *silently eats
traffic*, because every real Twilio webhook then fails the HMAC and 403s.

The self-test catches that drift at container start: it builds a synthetic
Twilio inbound for a throwaway number with a throwaway ``auth_token``,
signs it against the **configured public base URL** exactly as Twilio
would, registers a temporary demux entry for that number, then **POSTs it
through the sidecar's own public URL** and asserts the listener verifies
it (200 + empty TwiML). If the request comes back 403 the signing URL the
listener reconstructed != the URL the probe signed ⇒ host/port/proto/cert
drift ⇒ fail the start (fail-closed) before traffic is silently eaten.

This is deliberately an **end-to-end probe through the real public URL**
(not an in-process unit check): the whole point is to exercise the proxy +
TLS + port path that a pure unit test cannot see.
"""

from __future__ import annotations

import secrets as _secrets
from dataclasses import dataclass

import httpx
import structlog

from .verify import compute_signature
from .webhook import TWILIO_SIGNATURE_HEADER, DemuxEntry, InboundQueue, WebhookListener

__all__ = ["SelfTestError", "SelfTestResult", "run_ingress_self_test"]

log = structlog.get_logger(__name__)

# A reserved-for-documentation E.164 (NANP 555-01xx) used only as the
# throwaway ``To`` for the probe — never a real connection's number.
_PROBE_NUMBER = "+18005550100"
_PROBE_PEER = "+18005550199"
_PROBE_PATH = "/twilio/inbound"


class SelfTestError(RuntimeError):
    """Raised when the ingress self-test fails (fail-closed start)."""


@dataclass(frozen=True, slots=True)
class SelfTestResult:
    ok: bool
    status: int
    detail: str


def _probe_params() -> dict[str, str]:
    # Unique MessageSid so a probe never collides with a real inbound's
    # dedup key (the entry is torn down immediately regardless).
    return {
        "To": _PROBE_NUMBER,
        "From": _PROBE_PEER,
        "Body": "aios-sms ingress self-test",
        "MessageSid": "ST" + _secrets.token_hex(16),
    }


async def run_ingress_self_test(
    listener: WebhookListener,
    *,
    public_base_url: str,
    timeout_seconds: float = 10.0,
) -> SelfTestResult:
    """POST a synthetic signed request through the public URL and assert it
    verifies.

    Registers a temporary demux entry for ``_PROBE_NUMBER`` with a fresh
    random ``auth_token``, signs the probe against ``public_base_url`` (the
    URL Twilio would sign against), POSTs to ``<public_base_url>/twilio/inbound``,
    and asserts a ``200``. Always unregisters the probe entry afterward, so
    a self-test leaves no residue in the demux map.

    Raises nothing — returns a :class:`SelfTestResult`. The caller decides
    whether to fail-fast on ``ok is False`` (design §6: fail-closed start).
    """
    probe_token = _secrets.token_urlsafe(32)
    params = _probe_params()
    base = public_base_url.rstrip("/")
    signed_url = base + _PROBE_PATH
    signature = compute_signature(probe_token, signed_url, params)

    # A drained throwaway queue: the probe should *verify* and enqueue; we
    # don't care whether anything drains it, only that it cleared verify.
    probe_queue: InboundQueue = InboundQueue(maxsize=1)
    listener.register(
        _PROBE_NUMBER,
        DemuxEntry(connection_id="__selftest__", auth_token=probe_token, queue=probe_queue),
    )
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(
                signed_url,
                data=params,
                headers={TWILIO_SIGNATURE_HEADER: signature},
            )
        ok = resp.status_code == 200
        detail = (
            "verified through the public URL"
            if ok
            else (
                f"expected 200 but got {resp.status_code} — the listener's "
                "reconstructed signing URL does not match the probe's "
                f"(host/port/proto/cert drift vs {signed_url!r})"
            )
        )
        return SelfTestResult(ok=ok, status=resp.status_code, detail=detail)
    except httpx.HTTPError as exc:
        return SelfTestResult(
            ok=False,
            status=0,
            detail=f"could not reach the public URL {signed_url!r}: {exc!r}",
        )
    finally:
        listener.unregister(_PROBE_NUMBER)
