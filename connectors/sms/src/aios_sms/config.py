"""Container-wide configuration for the SMS connector listener.

The webhook listener is the **only inbound-reachable surface in the
connector fleet** (design §1, §6), so its knobs are first-class config,
not magic numbers buried in the handler.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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

    # The public port Twilio signs over (design §3.2: "Pin the expected
    # port from config"). For SMS over HTTPS the port is kept in the
    # signed URL — Twilio drops it only for Voice. When ``public_base_url``
    # already carries a port that wins; this is the pin used by the
    # forwarded-header fallback and the startup self-test so a config/proxy
    # drift surfaces as a *closed* failure, not a silent one.
    public_port: int = Field(default=443)

    # Forwarded-header fallback gate (design §3.2 step 3, §5.4). Only
    # consulted when ``public_base_url`` is unset. BOTH must be non-empty
    # for the fallback to be enabled at all — an empty allowlist or an
    # empty trusted-proxy set fails the fallback closed.
    #
    # ``allowed_hosts``: hostnames Twilio may legitimately post to (e.g.
    #   ``sms.example.com``); a forwarded ``X-Forwarded-Host`` not in this
    #   set is refused.
    # ``trusted_proxies``: bare IPs or CIDRs of the ingress proxies whose
    #   forwarded headers we trust — matched against the *socket-peer* IP,
    #   never a header-derived value.
    allowed_hosts: Annotated[frozenset[str], NoDecode] = Field(default_factory=frozenset)
    trusted_proxies: Annotated[frozenset[str], NoDecode] = Field(default_factory=frozenset)

    # Startup self-test (design §6): POST a synthetic signed request
    # through the sidecar's *own public URL* and assert it verifies,
    # catching host/port/proto/cert drift before it silently eats traffic.
    # Only runs when a ``public_base_url`` is configured (nothing to probe
    # otherwise).
    self_test_enabled: bool = Field(default=True)
    # Fail container start on a self-test failure (fail-closed). When
    # ``False`` the failure is logged loudly but the container still serves
    # (useful for a first bring-up before the route is wired).
    self_test_fail_fast: bool = Field(default=True)
    self_test_timeout_seconds: float = Field(default=10.0)

    @field_validator("allowed_hosts", "trusted_proxies", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept a comma/space-separated env string for the set fields.

        ``AIOS_SMS_ALLOWED_HOSTS="sms.example.com, sms2.example.com"`` and
        ``AIOS_SMS_TRUSTED_PROXIES="10.0.0.0/8 192.0.2.7"`` both parse.
        """
        if isinstance(value, str):
            return frozenset(p for p in value.replace(",", " ").split() if p)
        return value
