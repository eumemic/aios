# aios-sms

Provider-neutral SMS connector for aios (`connector = "sms"`), Twilio-first.
See `docs/design/sms-connector.md` for the full design.

This package currently implements the **inbound / transport layer** slice
(#1253, design §3.1–§3.3):

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

SMS deployments should set the aios-api server config
`inbound_debounce_seconds > 0` (#799) so a carrier that splits a
concatenated MO message into separate webhooks coalesces into one model
wake (`config.INBOUND_DEBOUNCE_SECONDS` documents the recommended value).

Per-connection secrets (`from_number`, `auth_token`) live on the
connection record, encrypted at rest, fetched via `/runtime/secrets`.

## Deferred (later #1252 children)

`sms_send`, the consent ledger, spend/registration gates, MMS, and the
status-callback → originating-session delivery-failure surfacing.
