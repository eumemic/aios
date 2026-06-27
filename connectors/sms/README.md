# aios-sms

Provider-neutral SMS connector for aios (`connector = "sms"`), Twilio-first.
See `docs/design/sms-connector.md` for the full design.

This package implements the **inbound / transport layer** slice
(#1253, design §3.1–§3.3) plus the **public HTTPS ingress + ingress
config** slice (#1265, design §5.3, §6):

- One container-wide `aiohttp` webhook listener stood up in `setup(tg)`
  (the signal shared-daemon fan-out shape), serving:
  - `POST /twilio/inbound` — MO messages.
  - `POST /twilio/status` — delivery status callbacks (verified + acked;
    session correlation lands in a later slice).
- **Verify-before-parse** (`verify.py`): read the raw body, route by the
  signed `To` number to the connection's cached `auth_token`, reconstruct
  the signed URL preferring the operator-configured public base URL and
  **keeping the port**, then `HMAC-SHA1` + base64 + constant-time compare.
  Missing/invalid → **403**; cold-start (number not yet discovered) →
  **transient 5xx** so Twilio retries. *Cold-start inbound is briefly
  unavailable per number, never briefly unauthenticated.*
- `serve_connection(connection_id, secrets)`: fetch creds, normalize
  `from_number` (single E.164 normalizer, symmetric at store + lookup),
  register `from_number → (connection_id, auth_token, queue)` in the
  shared demux map, then drain its per-connection queue →
  `emit_inbound(chat_id=From, sender={"display_name": From}, content=Body,
  event_id=MessageSid)`.
- `event_id = MessageSid` is a single-source invariant (the deprecated
  `SmsSid` / `SmsMessageSid` aliases are never used).

## Security posture (design §5.3)

- Pre-parse body cap (`MAX_BODY_BYTES`) before reading the form.
- Off-loop HMAC via `asyncio.to_thread` so a verify flood can't starve
  the single event loop.
- Bounded per-connection inbound queue that **sheds** on overflow (a
  dropped inbound is recoverable via Twilio retry; an OOM is not).
- Uniform 403 on any unverified/unroutable request (no enumeration
  oracle).
- `From` is a **routing key, not a trust anchor**: sender provenance is
  stamped `sender_verified=False` toward the model.

## Configuration

Listener settings are read from `AIOS_SMS_*` env (see `config.py`):

| env | default | meaning |
| --- | --- | --- |
| `AIOS_SMS_HOST` | `0.0.0.0` | listener bind host |
| `AIOS_SMS_PORT` | `8080` | listener bind port |
| `AIOS_SMS_PUBLIC_BASE_URL` | _(none)_ | operator-configured public origin Twilio posts to — the **preferred** canonical signing URL |
| `AIOS_SMS_PUBLIC_PORT` | `443` | the public port Twilio signs over (SMS over HTTPS **keeps the port**); pins the expected port from config |
| `AIOS_SMS_ALLOWED_HOSTS` | _(empty)_ | comma/space-separated hostnames Twilio may post to — gates the forwarded-header fallback (see below) |
| `AIOS_SMS_TRUSTED_PROXIES` | _(empty)_ | comma/space-separated IPs/CIDRs of ingress proxies whose forwarded headers we trust, matched against the **socket-peer** IP |
| `AIOS_SMS_SELF_TEST_ENABLED` | `true` | run the startup ingress self-test (only when a public base URL is set) |
| `AIOS_SMS_SELF_TEST_FAIL_FAST` | `true` | fail the container start on a self-test failure (fail-closed) |
| `AIOS_SMS_SELF_TEST_TIMEOUT_SECONDS` | `10.0` | self-test HTTP timeout |

### Signing-URL reconstruction & the forwarded-header fallback (design §5.4)

`X-Twilio-Signature` is computed over the **exact public URL** Twilio
posted to. We reconstruct it in priority order:

1. **Prefer `AIOS_SMS_PUBLIC_BASE_URL`** — the operator-registered webhook
   origin, which is by construction the URL Twilio signed. **Strongly
   recommended in production.** With it set, forwarded headers are
   ignored for signing.
2. **Fallback to `X-Forwarded-Proto` / `X-Forwarded-Host`** only when no
   base URL is configured — and only behind a fail-closed gate
   (`ingress.py`): the **socket-peer IP must be in `AIOS_SMS_TRUSTED_PROXIES`**
   *and* the `X-Forwarded-Host` must be a valid RFC-1123 host (no `@`
   userinfo) present in a **non-empty `AIOS_SMS_ALLOWED_HOSTS`**. Any gate
   miss → uniform **403** (fail closed). Empty allowlist or empty
   trusted-proxy set disables the fallback entirely.

For SMS over HTTPS the **port is kept** in the signed URL (Twilio drops it
only for Voice). Because the HMAC key is the per-connection `auth_token`
selected by the *signed* `To`, controlling the URL component alone cannot
forge a signature — a misconfig fails **closed** (availability), never open.

### Startup ingress self-test (design §6)

The public HTTPS ingress is a **first-class deliverable**: the only
inbound-reachable surface in the fleet, and signature correctness depends
on the TLS-termination point, the configured base URL, the trusted-proxy
set, the `allowedHosts` list, and the port-keeping rule. On start (when a
public base URL is configured) the sidecar **POSTs a synthetic signed
request through its own public URL** and asserts it verifies (200). A
mismatch (host/port/proto/cert drift) → loud `sms.selftest.failed` log and,
by default, a **fail-closed container start** — catching the drift *before
it silently eats traffic* (every real Twilio webhook would otherwise 403).

## Paired eumemic-ops deliverable (lockstep — design §6)

Per the durable-sandboxes promote-gate precedent (*never ship the code
half without the ops half*), this connector requires a paired
**eumemic-ops** change, tracked alongside this slice:

- a dedicated Traefik/Coolify route + **TLS** for the SMS sidecar;
- a **Twilio source-IP allowlist + rate-limiting** at the edge;
- **strip client-supplied `X-Forwarded-Host`** and set it authoritatively
  (so the connector's forwarded-header fallback can only ever see a value
  the proxy vouched for);
- **monitoring**: cert expiry, a `403`-signature-failure rate `> 0`, and a
  sustained inbound-rate drop.

The connector-side config above is written *against* those ops artifacts:
`AIOS_SMS_PUBLIC_BASE_URL` must equal the Traefik route's public origin,
`AIOS_SMS_TRUSTED_PROXIES` must list the proxy's egress IP(s), and
`AIOS_SMS_PUBLIC_PORT` must match the TLS listener port.

SMS deployments should set the aios-api server config
`inbound_debounce_seconds > 0` (#799) so a carrier that splits a
concatenated MO message into separate webhooks coalesces into one model
wake (`config.INBOUND_DEBOUNCE_SECONDS` documents the recommended value).

Per-connection secrets (`from_number`, `auth_token`) live on the
connection record, encrypted at rest, fetched via `/runtime/secrets`.

## Deferred (later #1252 children)

`sms_send`, the consent ledger, spend/registration gates, MMS, and the
status-callback → originating-session delivery-failure surfacing.
