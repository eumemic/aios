# WhatsApp pairing failure-mode diagnostics

Symptom → root cause → fix table for the failure modes observed during the
capstone smoke (PR #625, 2026-05-22).

## "Can't link new devices right now" on the phone

By far the most common failure.  WhatsApp uses this same dialog for SEVERAL
distinct conditions; differential diagnosis below.

### 1. PNG-rendered QR (~90% of cases)

**Symptom:** Phone shows the dialog immediately on scan.  Daemon log shows
the QR was generated and `drainQRChannel` is cycling refreshes.  Repeated
attempts all fail the same way regardless of which account.

**Root cause:** The QR was rendered to PNG and opened in macOS Preview /
similar viewer.  Preview's anti-aliasing + color profile + scaling degrade
the QR enough that WhatsApp's iOS parser rejects it — but the error dialog
is the same as the rate-limit dialog.

**Fix:** Render as ASCII in the terminal via the `pair.sh` script (or the
manual one-liner in SKILL.md).  Scan directly from the terminal window.

**Verification path** (when uncertain whether it's PNG or rate-limit):

* Open `https://web.whatsapp.com` in a browser.  Scan THAT QR from the same
  scanning device.
* If the web QR links fine → our QR rendering is the issue (PNG).
* If the web QR also fails → the scanning device IS rate-limited (rare).

### 2. Scanning-device rate-limit (rare)

**Symptom:** Even WhatsApp's own `web.whatsapp.com` QR is rejected on this
device.

**Root cause:** WhatsApp's per-device anti-abuse counter was tripped by too
many failed link attempts in a short window.  Retries reset the timer.

**Fix:** STOP attempting.  Wait ~24h with no link attempts on this scanning
device.  Confirm by retesting `web.whatsapp.com` first; only re-run our pair
flow once web works.

### 3. Account suspended (rarest)

**Symptom:** Even normal messaging from the WhatsApp account fails (not just
linking).

**Fix:** Out of scope — needs WhatsApp's support or account replacement.

## "device not paired" on `unpair`

**Symptom:** `POST /v1/connectors/whatsapp/unpair` returns 502 with body
`{"error": "device not paired"}` even though the daemon log shows the
pairing earlier succeeded.

**Root cause:** WhatsApp's server invalidated the device link (user pressed
"Log out from this device" in the WhatsApp mobile app, or 14-day inactivity,
or server-side cleanup).  whatsmeow's `events.LoggedOut` cleared the local
store; subsequent ops return `not paired`.

**Fix:** A new `start-pairing` works against the same connection — the
daemon's `replaceWhatsmeowClient` machinery (PR #622 originally added this
for operator-initiated `Unpair`; the PR #625 smoke caught that the
peer-side `events.LoggedOut` handler hadn't been wired through, and the
fix-up commit on PR #625 wired it so peer-logout now also swaps in a
fresh Client behind `atomic.Pointer` and the in-process re-pair works
without restart).

## "pairing already in progress" on a fresh `start-pairing`

**Symptom:** `start-pairing` returns
`{"error": "pairing already in progress"}` even when no QR is currently
displayed.

**Root cause:** A prior `start-pairing` call's QR cycle is still active
(whatsmeow's QR channel hasn't reached the `timeout` event yet — total
window is ~100 s across QR refreshes).

**Fix:** Either:
* Wait for the existing cycle to time out naturally (~2 min).
* Restart the daemon (kill + connector respawns it) to reset the pair state.

## "Got 515 code, reconnecting..." in daemon logs

**Symptom:** Daemon's whatsmeow logger emits
`{"level":"INFO","msg":"Got 515 code, reconnecting..."}` shortly after a
successful `confirm-pairing`.

**Root cause:** Normal — WhatsApp's pair handshake completes with a
re-connect signal so the client establishes its post-pair session.  Followed
within a second by `"Successfully authenticated"`.

**Fix:** None needed.  If `Successfully authenticated` doesn't follow within
~3 s, escalate.

## "Failed to store app state sync key … database is locked (5)" warnings

**Symptom:** Daemon spams `Failed to store app state sync key … database is
locked (5) (SQLITE_BUSY)` for ~30 s post-pair.

**Root cause:** whatsmeow + modernc.org/sqlite have a known concurrency
issue where the post-pair app-state-sync write contends with other writes.
Mostly self-corrects.

**Fix:** Cosmetic for v1.  Future PR can move to a write-coalescing
sqlstore wrapper.  Doesn't block pair functionality.

## Inbound messages flow but bot doesn't respond

**Symptom:** `aios sessions events --kind message` shows the inbound user
message, but no assistant turn follows; session sits `idle`.

**Differential:**

* Worker log shows `step.litellm_failed` → model provider error.  Check the
  agent's `model` field; WhatsApp's deferred smoke surfaced an OpenRouter
  → Bedrock 400 (cf. aios issue #623).
* Worker log shows no `wake_session` jobs → the inbound never reached the
  worker.  Check `aios connections get <id>` for `session_id`; an unbound
  inbound goes to the resolver's per-chat ephemeral path which may not be
  visible.
* No errors anywhere → the model wake fired but produced no tool call.
  Check the assistant message at the latest seq; if it's only an internal
  monologue without `tool_calls`, the model decided not to respond.

## Connector log spams `mark_read_failed`

**Symptom:** Every outbound send is followed by a
`wameow.mark_read_failed` warning.

**Root cause:** `flushReadReceipts` tried to `MarkRead` a message id that
doesn't correspond to a peer text/media — usually a reaction envelope ID
that slipped into the unread queue.

**Fix:** PR #625's round-2 fix-up (`8e05ddc`) filters
`isReactionOnly` envelopes out of the unread queue.  If the warning
persists on a daemon built from a commit before that, rebuild.
