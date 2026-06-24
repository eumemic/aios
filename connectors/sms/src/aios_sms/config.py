"""Container-wide configuration for the SMS connector listener.

The webhook listener is the **only inbound-reachable surface in the
connector fleet** (design §1, §6), so its knobs are first-class config,
not magic numbers buried in the handler.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "EMPTY_TWIML",
    "INBOUND_DEBOUNCE_SECONDS",
    "INBOUND_QUEUE_MAXSIZE",
    "MAX_BODY_BYTES",
    "TWIML_CONTENT_TYPE",
    "Settings",
]

# Pre-parse request-body cap (design §5.3, security sev 52). A Twilio
# inbound webhook is a small urlencoded form; well under 64 KiB. Reading
# more than this before verifying is a DoS lever, so we cap *before*
# parse and reject oversize bodies outright.
MAX_BODY_BYTES = 64 * 1024

# Bounded per-connection inbound queue (design §5.3). A full queue sheds
# (ack 200, drop) rather than growing unbounded — a dropped inbound is
# recoverable via Twilio retry; an OOM is not.
INBOUND_QUEUE_MAXSIZE = 1000

# SMS connections want server-side ``inbound_debounce_seconds > 0``
# (#799, design §3.2) so a carrier that splits a concatenated MO message
# into separate webhooks (distinct MessageSids → distinct event_ids → no
# dedup collision) coalesces into one model wake instead of fragmenting
# the turn. The debounce knob itself lives in aios-api server config
# (``Settings.inbound_debounce_seconds``); this constant is the value the
# operator runbook must set for an SMS deployment, surfaced here so the
# requirement is co-located with the connector that depends on it.
INBOUND_DEBOUNCE_SECONDS = 3.0

# Empty TwiML returned synchronously on every verified inbound (design
# §3.2): the agent's reply is always async via the ``sms_send`` tool, so
# the webhook response carries no message.
EMPTY_TWIML = "<Response></Response>"
TWIML_CONTENT_TYPE = "text/xml"


class Settings(BaseSettings):
    """Listener configuration, read from env at container start.

    ``public_base_url`` is the operator-configured public origin Twilio
    posts to (e.g. ``https://sms.example.com``). It is the **preferred**
    canonical signing URL (design §5.4) — reconstructing from forwarded
    headers is the host-header-injection surface, so a configured base is
    strongly recommended in production.
    """

    model_config = SettingsConfigDict(env_prefix="AIOS_SMS_", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    public_base_url: str | None = Field(default=None)
