"""Per-connection runtime state + the shared inbound-demux registration.

Scoped to the inbound/transport MVP slice (#1253): the compliance-,
spend-, and outbound-attempt stores described in the design (§3.1, §4,
§5) land in later slices and are intentionally absent here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import INBOUND_QUEUE_MAXSIZE
from .webhook import InboundQueue

__all__ = ["SmsConnectionState"]


@dataclass
class SmsConnectionState:
    """One row per active ``sms`` connection.

    ``from_number`` is the AIOS-owned Twilio number in normalized E.164
    (== ``external_account_id``); it is the key the shared webhook
    listener routes inbound ``To`` by and the key the status route routes
    outbound ``From`` by (a later slice).

    ``auth_token`` is the Twilio master token used **only** to verify
    ``X-Twilio-Signature`` (design §5.9: send creds are a separate API
    Key, not this token).

    The per-connection ``inbound_queue`` decouples the Twilio ack from
    the ``emit_inbound`` round-trip (design §3.2): the webhook handler
    enqueues + returns 200 immediately, and ``serve_connection`` drains.
    It is **bounded** (design §5.3): a full queue sheds rather than
    growing unbounded, since a dropped inbound is recoverable via
    Twilio's retry but an OOM is not.
    """

    connection_id: str
    from_number: str
    auth_token: str
    inbound_queue: InboundQueue = field(
        default_factory=lambda: InboundQueue(maxsize=INBOUND_QUEUE_MAXSIZE)
    )
