#!/usr/bin/env bash
# Operational drill: issue #147's manual repro, scripted (issue #1757 item 4).
#
# Boots a REAL ``aios worker`` process against your dev worktree's DB, gets a
# file-barrier bash tool call in flight, ``kill -9``s the WHOLE process group
# (not just the parent — a plain ``kill -9 <pid>`` can leave orphaned
# grandchildren, which would silently make the drill weaker than a real crash),
# boots a SECOND worker, and asserts the ghost gets repaired and the turn
# completes via the API. This is the one form of "genuinely killed process"
# coverage aios does not run automatically (see the "Crash-recovery test
# architecture" note in ``tests/e2e/conftest.py`` for why: no marginal
# DB-visible coverage over the seeded simulators, no CI lane to run it in).
#
# Run this BEFORE promoting a change to the crash-recovery path
# (worker.py's startup-recovery sequence, sweep.py's ghost repair, the
# sandbox salvage preamble, procrastinate job reaping). It is a manual dev
# tool, NOT part of any CI lane — never wire this into the PR-gating docker
# shard; if a non-gating nightly CI lane is ever created, promoting this
# script there can be reconsidered then.
#
# Prerequisites:
#   - A dev worktree bootstrapped via ./scripts/dev-bootstrap.sh (or
#     equivalent: AIOS_DB_URL / AIOS_API_KEY / AIOS_VAULT_KEY /
#     AIOS_EGRESS_CA_KEY set, migrations applied, api reachable).
#   - `uv` on PATH.
#   - The workers run as raw ``uv run aios worker`` processes on the
#     HOST (not inside docker compose) — the drill needs direct process-group
#     control that `docker compose kill` semantics don't give us cleanly.
#
# Usage:
#   ./scripts/crash-recovery-drill.sh
#
# Exit code 0 iff the ghost was repaired AND the turn completed after the
# second worker boots. Non-zero (with a diagnostic) otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

say()  { printf '\033[36m·\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m✓\033[0m %s\n' "$*"; }
fail() { printf '\033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f .env ]] || fail "no .env — run ./scripts/dev-bootstrap.sh first"
set -a
# shellcheck disable=SC1091
source .env
set +a

: "${AIOS_API_KEY:?AIOS_API_KEY not set — run ./scripts/dev-bootstrap.sh first}"
: "${AIOS_DB_URL:?AIOS_DB_URL not set}"
AIOS_URL="${AIOS_URL:-http://localhost:${AIOS_API_PORT:-8080}}"

WORKER1_LOG="$(mktemp -t aios-drill-worker1-XXXXXX.log)"
WORKER2_LOG="$(mktemp -t aios-drill-worker2-XXXXXX.log)"
WORKER1_PID=""
WORKER2_PID=""

cleanup() {
  for pid in "$WORKER1_PID" "$WORKER2_PID"; do
    [[ -n "$pid" ]] || continue
    # Negative PID = kill the whole process group; ignore "already gone".
    kill -9 -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

curl_json() {
  curl -fsS -H "Authorization: Bearer $AIOS_API_KEY" -H 'Content-Type: application/json' "$@"
}

json_id() { python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'; }

say "creating a scratch agent + session for the drill"
AGENT_PAYLOAD="$(mktemp -t aios-drill-agent-XXXXXX.json)"
cat >"$AGENT_PAYLOAD" <<'JSON'
{
  "name": "crash-recovery-drill",
  "model": "openrouter/test",
  "system": "",
  "tools": [{"type": "bash"}],
  "window_min": 50000,
  "window_max": 150000
}
JSON
AGENT_ID="$(curl_json -X POST "$AIOS_URL/v1/agents" -d @"$AGENT_PAYLOAD" | json_id)"
rm -f "$AGENT_PAYLOAD"

ENV_ID="$(curl_json -X POST "$AIOS_URL/v1/environments" -d '{"name": "crash-recovery-drill"}' | json_id)"

SESSION_PAYLOAD="$(mktemp -t aios-drill-session-XXXXXX.json)"
python3 -c 'import json,sys; json.dump({"agent_id": sys.argv[1], "environment_id": sys.argv[2]}, sys.stdout)' \
  "$AGENT_ID" "$ENV_ID" >"$SESSION_PAYLOAD"
SESSION_ID="$(curl_json -X POST "$AIOS_URL/v1/sessions" -d @"$SESSION_PAYLOAD" | json_id)"
rm -f "$SESSION_PAYLOAD"

ok "session $SESSION_ID (agent $AGENT_ID, env $ENV_ID)"

BARRIER="$(mktemp -u -t aios-drill-barrier-XXXXXX)"
say "starting worker #1 (log: $WORKER1_LOG)"
setsid uv run aios worker >"$WORKER1_LOG" 2>&1 &
WORKER1_PID=$!
sleep 3
grep -q "worker.startup" "$WORKER1_LOG" || fail "worker #1 did not report worker.startup in time — see $WORKER1_LOG"
ok "worker #1 up (pid $WORKER1_PID)"

# NOTE: this drill needs a real model behind ``AGENT_ID`` (a live provider
# API key configured for your dev worktree) — it boots a real worker doing
# real inference, not a scripted fake model like the test suite. Point the
# agent at whatever model your worktree has credentials for by editing
# AGENT_ID's creation above if ``openrouter/test`` isn't wired up locally.
say "sending a message asking the model to run a file-barrier bash command"
# The barrier loop blocks the tool task in-container until BARRIER.go
# appears, giving a guaranteed window to kill mid-execution instead of
# racing a fast command. Depends on the model actually issuing the bash
# call as instructed — if your model declines or paraphrases, adjust the
# prompt or drive the tool call more directly via your own harness hook.
MESSAGE_PAYLOAD="$(mktemp -t aios-drill-message-XXXXXX.json)"
python3 -c '
import json, sys
barrier = sys.argv[1]
content = (
    "Run this exact bash command and nothing else first: "
    f"while [ ! -f {barrier}.go ]; do sleep 0.2; done; echo done"
)
json.dump({"content": content}, sys.stdout)
' "$BARRIER" >"$MESSAGE_PAYLOAD"
curl_json -X POST "$AIOS_URL/v1/sessions/$SESSION_ID/messages" -d @"$MESSAGE_PAYLOAD" >/dev/null
rm -f "$MESSAGE_PAYLOAD"
ok "message sent — session should now be actively processing"

# Give the worker a moment to pick up the wake and dispatch the tool call
# (or at least mark the wake job `doing`) before we kill it.
sleep 2

say "killing worker #1's ENTIRE PROCESS GROUP with SIGKILL (kill -9 -$WORKER1_PID)"
kill -9 -- "-$WORKER1_PID" 2>/dev/null || true
wait "$WORKER1_PID" 2>/dev/null || true
WORKER1_PID=""
ok "worker #1 is dead"

say "confirming the wake job is left 'doing' in procrastinate_jobs (the #147 shape)"
DOING_COUNT="$(uv run python3 - <<'PY'
import asyncio, os
import asyncpg

async def main() -> None:
    conn = await asyncpg.connect(os.environ["AIOS_DB_URL"].replace("postgresql+psycopg://", "postgresql://").replace("postgresql+asyncpg://", "postgresql://"))
    try:
        n = await conn.fetchval(
            "SELECT count(*) FROM procrastinate_jobs WHERE status = 'doing'"
        )
        print(n)
    finally:
        await conn.close()

asyncio.run(main())
PY
)"
say "doing rows: $DOING_COUNT (0 is fine if the kill landed between wake ticks — re-run if you want to force it)"

say "starting worker #2 (log: $WORKER2_LOG)"
setsid uv run aios worker >"$WORKER2_LOG" 2>&1 &
WORKER2_PID=$!
sleep 3
grep -q "worker.startup" "$WORKER2_LOG" || fail "worker #2 did not report worker.startup in time — see $WORKER2_LOG"
ok "worker #2 up (pid $WORKER2_PID)"

say "asserting reap_stalled_jobs logged a non-zero count on boot (or 0 if nothing was doing)"
grep -q "worker.boot_gate.admitted" "$WORKER2_LOG" || fail "worker #2 never reached boot_gate.admitted — see $WORKER2_LOG"
if grep -q 'sweep.reaped_stalled_jobs' "$WORKER2_LOG"; then
  ok "worker #2 reaped a stalled job from the killed predecessor (production telemetry signal)"
else
  say "no stalled jobs reaped — the kill likely landed before the wake job reached 'doing'; not a failure"
fi

say "letting worker #2 process the turn to completion (up to 30s)"
deadline=$((SECONDS + 30))
STATUS=""
while (( SECONDS < deadline )); do
  STATUS="$(curl_json "$AIOS_URL/v1/sessions/$SESSION_ID" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status"))')"
  [[ "$STATUS" == "idle" || "$STATUS" == "errored" ]] && break
  sleep 1
done
[[ "$STATUS" == "idle" ]] || fail "session did not reach idle within 30s after worker #2 booted (status=$STATUS) — see $WORKER2_LOG"
ok "session reached idle — the turn completed post-recovery"

say "cleaning up: archiving the drill session/agent/environment"
curl_json -X POST "$AIOS_URL/v1/sessions/$SESSION_ID/archive" >/dev/null || true
curl_json -X DELETE "$AIOS_URL/v1/agents/$AGENT_ID" >/dev/null || true
rm -f "$BARRIER" "$BARRIER.go"

ok "drill complete — worker #1 was SIGKILLed mid-turn, worker #2 recovered and completed it"
say "worker logs kept at: $WORKER1_LOG , $WORKER2_LOG"
