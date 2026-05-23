---
name: aios-whatsapp-pair
description: This skill should be used when the user asks to "smoke the whatsapp connector", "pair whatsapp", "link whatsapp", "scan the QR for whatsapp", "set up whatsapp smoke", "bring up the whatsapp bot", "rebind the whatsapp connection", "redo the whatsapp pairing", or otherwise wants to attach a real WhatsApp account to the aios runtime running in the current worktree.  Bundles the daemon cross-compile (Docker `golang:1.25` when local Go < 1.25), runtime-token mint, daemon spawn, start-pairing/confirm-pairing API calls, and — critically — the **ASCII-in-terminal QR rendering** that makes the pair flow actually work.  Hand off to `aios-live-monitor` for narration after `status=success`.
---

# aios WhatsApp pairing

Attach a real WhatsApp account to the aios runtime so the connector can receive inbounds and the model can call `whatsapp_send` / `whatsapp_react` / `whatsapp_edit_message` / `whatsapp_delete_message` / `whatsapp_list_groups` / `whatsapp_create_group` / `whatsapp_rename_group`.

## When to use

- Smoke-testing the WhatsApp connector PR against a real account
- Re-pairing after an operator-initiated `unpair`
- Recovering from a paired device that got unlinked server-side (e.g. user clicked "Log out from this device" in the WhatsApp mobile app)

Skip this skill for Telegram / Signal smokes — those are covered by `aios-smoke-setup`.  This is WhatsApp-only.

## Pre-flight rules (every pair attempt)

1. **Local Go ≥ 1.25 OR Docker available.**  whatsmeow's `go.mod` requires Go 1.25; macOS Homebrew ships 1.18 by default.  The script falls back to a `golang:1.25` Docker image when local Go is older; the build artifact lands at `/tmp/whatsapp-build/whatsapp-daemon`.

2. **Render the QR as ASCII in the terminal, not PNG.**  THIS IS LOAD-BEARING.  Preview-rendered PNGs degrade enough to fail WhatsApp's parser; the resulting "Can't link new devices right now" dialog is the *same* message WhatsApp shows for rate-limiting, so the failure mode looks like throttling when it isn't.  See memory `reference_whatsapp_pairing_ratelimit.md` for the full backstory.

3. **Pair ONCE; avoid retry storms.**  Each failed scan-attempt counts against an opaque per-device anti-abuse counter that DOES enforce real rate-limits if tripped enough.  If a pair attempt fails (and the QR rendering is verified ASCII), STOP — don't repeatedly call `start-pairing`.  Wait ~24h or switch to a different scanning device.

4. **Connection-secret has `phone=+<E.164>`.**  WhatsApp connections store the phone in the connection's secrets dict, not the bot-token field that Telegram uses.  Create with `aios connections create --connector=whatsapp --external-account-id="+<E.164>" --secret "phone=+<E.164>"`.

## The fast path

```bash
.claude/skills/aios-whatsapp-pair/scripts/pair.sh "+16575274288"
```

The script:
1. Verifies the daemon binary exists at `/tmp/whatsapp-build/whatsapp-daemon` (or cross-compiles via Docker if missing).
2. Verifies a `whatsapp` connector runtime is up and the connection is bound to a session.
3. Calls `POST /v1/connectors/whatsapp/start-pairing` with the phone.
4. Renders the returned code as ASCII in the terminal — the operator scans directly from the terminal window.
5. Calls `POST /v1/connectors/whatsapp/confirm-pairing` which blocks until terminal outcome.
6. Prints the resulting JID + status.

For an unattended setup, the operator just needs the terminal visible to their phone's camera during the ~20-second QR window.

## The four-step manual sequence (when the script doesn't apply)

```bash
# 1. Cross-compile the daemon (only if local Go < 1.25).
docker run --rm \
  -v "$PWD/connectors/whatsapp/daemon:/src" \
  -v aios-whatsapp-gocache:/go \
  -v /tmp/whatsapp-build:/out \
  -w /src -e GOPATH=/go -e GOCACHE=/go/build-cache \
  -e GOOS=darwin -e GOARCH=arm64 -e CGO_ENABLED=0 \
  golang:1.25 go build -o /out/whatsapp-daemon ./cmd/whatsapp-daemon

# 2. Mint a runtime token for the whatsapp connector.
TOKEN=$(curl -sS -X POST "http://127.0.0.1:${AIOS_API_PORT}/v1/runtime-tokens" \
  -H "Authorization: Bearer ${AIOS_API_KEY}" -H "Content-Type: application/json" \
  -d '{"connector": "whatsapp"}' | python3 -c "import json,sys; print(json.load(sys.stdin)['plaintext'])")

# 3. Spawn the connector (it auto-spawns one daemon subprocess per attached connection).
AIOS_URL="http://127.0.0.1:${AIOS_API_PORT}" AIOS_RUNTIME_TOKEN="$TOKEN" \
  nohup uv run python -m aios_whatsapp >> .logs/connector.log 2>&1 &

# 4. Pair: start-pairing → ASCII QR → confirm-pairing.
RESP=$(curl -sS -X POST "http://127.0.0.1:${AIOS_API_PORT}/v1/connectors/whatsapp/start-pairing" \
  -H "Authorization: Bearer ${AIOS_API_KEY}" -H "Content-Type: application/json" \
  -d "{\"external_account_id\": \"$PHONE\"}")
CODE=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('code',''))")
echo "$CODE" | uvx --from qrcode python -c \
  "import sys, qrcode; q=qrcode.QRCode(border=2); q.add_data(sys.stdin.read().strip()); q.print_ascii(invert=True)"

# Operator scans from terminal, then:
curl -sS -X POST "http://127.0.0.1:${AIOS_API_PORT}/v1/connectors/whatsapp/confirm-pairing" \
  -H "Authorization: Bearer ${AIOS_API_KEY}" -H "Content-Type: application/json" \
  -d "{\"external_account_id\": \"$PHONE\"}" | python3 -m json.tool
```

`confirm-pairing` blocks until whatsmeow reports `success`, `timeout`, or `error`.  On `success` the response carries `jid` and (eventually) `push_name`.

## Diagnosing a failing pair

When the phone shows "Can't link new devices right now":

1. **First**, verify the QR rendering: was it ASCII in the terminal?  If it was a PNG opened in Preview, regenerate as ASCII.  This is the failure mode 90% of the time.
2. **Second**, scan `https://web.whatsapp.com`'s QR from the same scanning device.  If that links fine, the issue is our QR (rendering or daemon-side); if it ALSO fails, the scanning device is genuinely rate-limited.
3. **Third**, check `aios connections get <conn_id>` shows the right `external_account_id` and the connection is `attached`.  An unbound connection won't have a daemon and `start-pairing` will route nowhere useful.

## Rebind to a different phone

The daemon is per-phone (per-connection); the connector spawns one subprocess per attached whatsapp connection.  To switch which account a session listens to:

```bash
uv run aios connections detach <old_conn_id>
uv run aios connections create --connector=whatsapp \
  --external-account-id="+<new_phone>" --secret "phone=+<new_phone>"
uv run aios connections attach <new_conn_id> --session-id=<session_id>
```

The connector picks up the new connection via SSE discovery and spawns a fresh daemon.

## After pair: hand off to aios-live-monitor

Per the locked smoke flow, once `confirm-pairing` returns `status=success`:

1. Read the current `last_event_seq` from `aios sessions get <session_id>`.
2. Arm the chat monitor via `.claude/skills/aios-live-monitor/scripts/preflight.sh <session_id>`.
3. Tell the operator to DM the bot — and narrate the round-trip per the live-monitor skill's etiquette.

## Additional resources

- **`scripts/pair.sh`** — all-in-one wrapper for the four-step sequence
- **`references/diagnostics.md`** — failure-mode → root-cause → fix table (web-QR-test, lifetime ctx, PairPhone alternative)
- Memory `reference_whatsapp_pairing_ratelimit.md` — why the ASCII rendering is load-bearing
