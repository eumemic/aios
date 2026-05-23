#!/usr/bin/env bash
# pair.sh — render a fresh WhatsApp pairing QR as ASCII in the terminal
# and block on confirm-pairing.  Operator scans from the terminal window
# during the ~20 s QR window.
#
# Usage: pair.sh "+<E.164>"
#
# Pre-conditions:
#   * aios api running on $AIOS_API_PORT (loaded from .env)
#   * aios_whatsapp connector running (auto-spawns the daemon for the bound connection)
#   * The phone has an active WhatsApp connection bound to a session
#
# Why ASCII not PNG: Preview-rendered PNGs degrade enough that WhatsApp's
# parser rejects them, but the error dialog ("Can't link new devices
# right now") is the same one shown for rate-limiting — see memory
# reference_whatsapp_pairing_ratelimit.md.

set -euo pipefail

PHONE="${1:-}"
if [[ -z "$PHONE" ]]; then
  echo "usage: pair.sh +<E.164>" >&2
  exit 2
fi

# Source the worktree .env so AIOS_API_PORT + AIOS_API_KEY are populated.
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
: "${AIOS_API_PORT:?AIOS_API_PORT not set; source the worktree .env}"
: "${AIOS_API_KEY:?AIOS_API_KEY not set; source the worktree .env}"

API="http://127.0.0.1:${AIOS_API_PORT}"

# Verify the connector is up.  If start-pairing 502s, the daemon isn't running.
echo "→ POST $API/v1/connectors/whatsapp/start-pairing  external_account_id=$PHONE"
RESP=$(curl -sS -X POST "$API/v1/connectors/whatsapp/start-pairing" \
  -H "Authorization: Bearer $AIOS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"external_account_id\": \"$PHONE\"}")

CODE=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('code',''))" 2>/dev/null || true)

if [[ -z "$CODE" ]]; then
  echo "no code in response:" >&2
  echo "$RESP" | python3 -m json.tool >&2 || echo "$RESP" >&2
  exit 1
fi

echo
echo "scan NOW from the terminal below (~20 s window before whatsmeow rotates internally):"
echo

echo "$CODE" | uvx --from qrcode python -c \
  "import sys, qrcode; q=qrcode.QRCode(border=2); q.add_data(sys.stdin.read().strip()); q.print_ascii(invert=True)"

echo
echo "→ POST $API/v1/connectors/whatsapp/confirm-pairing  (blocks until terminal)"
curl -sS -X POST "$API/v1/connectors/whatsapp/confirm-pairing" \
  -H "Authorization: Bearer $AIOS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"external_account_id\": \"$PHONE\"}" \
  | python3 -m json.tool
