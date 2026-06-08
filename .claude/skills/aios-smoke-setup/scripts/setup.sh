#!/usr/bin/env bash
# Bring up an isolated aios runtime in this worktree.
#
# Two shapes:
#   * connector mode (default) — pointed at a real Telegram/Signal bot,
#     ready to receive DMs.  Needs --bot-token for telegram.
#   * headless mode (--no-connector) — api+worker only, no connector,
#     driven via the CLI.  For /verify of harness changes (sweep, loop,
#     context, tool_dispatch) where a connector is irrelevant.
#
# Idempotent.  Re-running refreshes the DB + fixtures.  --restart reloads
# api/worker on the new HEAD without rebuilding the DB.
#
# Usage:
#   setup.sh --bot-token <token> [--connector telegram] \
#            [--agent-name smoke] [--system-prompt-file <path>] \
#            [--source-env ~/code/aios/.env] [--port <num>] \
#            [--db <name>] [--restart]
#   setup.sh --no-connector [--agent-name verify] [--port <num>] [--db <name>]
#
# Output (last line):
#   smoke_session_id=sess_01...
#
# NOTE for Claude-driven sessions: a plain `bash setup.sh` started via the
# Bash tool spawns api/worker that are reaped when the tool call returns.
# If you need the runtime to outlive the call, start api+worker yourself via
# two `run_in_background` Bash calls and use `--phase fixtures` here.  See
# SKILL.md "Headless bring-up when Claude is driving".

set -euo pipefail

# ── arg parsing ───────────────────────────────────────────────────────
BOT_TOKEN=""
CONNECTOR="telegram"
NO_CONNECTOR=false
AGENT_NAME=""
SYSTEM_PROMPT_FILE=""
SOURCE_ENV="${HOME}/code/aios/.env"
PORT=""
DB=""
RESTART_ONLY=false
PHASE="all"   # all | prep | fixtures

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bot-token) BOT_TOKEN="$2"; shift 2;;
    --connector) CONNECTOR="$2"; shift 2;;
    --no-connector) NO_CONNECTOR=true; CONNECTOR=""; shift;;
    --agent-name) AGENT_NAME="$2"; shift 2;;
    --system-prompt-file) SYSTEM_PROMPT_FILE="$2"; shift 2;;
    --source-env) SOURCE_ENV="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --db) DB="$2"; shift 2;;
    --restart) RESTART_ONLY=true; shift;;
    --phase) PHASE="$2"; shift 2;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

[[ -z "$AGENT_NAME" ]] && { $NO_CONNECTOR && AGENT_NAME="verify" || AGENT_NAME="smoke"; }

WORKTREE="$(pwd)"
BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo nobranch)"
BRANCH_SHORT="$(echo "$BRANCH" | sed 's|wt/||; s|/|-|g; s/[^a-z0-9_-]/_/g' | head -c 30)"

[[ -z "$DB" ]] && DB="aios_smoke_${BRANCH_SHORT//-/_}"

say()   { printf "\033[36m▸\033[0m %s\n" "$*"; }
warn()  { printf "\033[33m⚠\033[0m %s\n" "$*"; }
fail()  { printf "\033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }
ok()    { printf "\033[32m✓\033[0m %s\n" "$*"; }

export DOCKER_HOST="${DOCKER_HOST:-unix:///Users/tom/.docker/run/docker.sock}"

# Resolve the aios Postgres container by the port it publishes (5433),
# falling back to the conventional compose name.  The hardcoded `aios-pg`
# this replaced silently broke once compose started naming it
# `aios-postgres-1`.
pg_container() {
  local c
  c="$(docker ps --filter "publish=5433" --format '{{.Names}}' 2>/dev/null | head -1)"
  echo "${c:-aios-postgres-1}"
}

# ── 1. preflight ──────────────────────────────────────────────────────
preflight() {
  say "preflight"

  # 1a. Bot token uncontested (telegram connector only)
  if [[ "$CONNECTOR" == "telegram" && -n "$BOT_TOKEN" ]]; then
    local resp
    resp="$(curl -sS "https://api.telegram.org/bot${BOT_TOKEN}/getUpdates?timeout=2&limit=1" 2>&1 || echo '{"ok":false}')"
    if echo "$resp" | grep -q '"error_code":409'; then
      fail "telegram bot getUpdates returns 409 Conflict — token is being polled elsewhere; rotate the token or stop the rogue poller"
    fi
    if ! echo "$resp" | grep -q '"ok":true'; then
      fail "telegram getMe/getUpdates rejected: $resp"
    fi
    ok "telegram bot uncontested"
  fi

  # 1b. Sibling worktree warning (don't kill, just warn)
  local siblings
  siblings="$(ps -ef | grep -E "aios (api|worker)" | grep -v grep | grep -v "$WORKTREE" || true)"
  if [[ -n "$siblings" ]]; then
    warn "sibling aios runtime detected (different worktree) — leaving it alone:"
    echo "$siblings" | head -3 | sed 's/^/    /'
  fi

  # 1c. Pick a free port if not provided
  if [[ -z "$PORT" ]]; then
    for candidate in 8091 8092 8093 8094 8095 8096 8097; do
      if ! lsof -iTCP:"$candidate" -sTCP:LISTEN >/dev/null 2>&1; then
        PORT="$candidate"
        break
      fi
    done
    [[ -z "$PORT" ]] && fail "no free port in 8091..8097"
  elif lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "port $PORT already in use"
  fi
  ok "port $PORT free"
}

# ── 2. write .env with overrides ──────────────────────────────────────
write_env() {
  say "writing .env (db=$DB port=$PORT connector=${CONNECTOR:-none})"
  [[ -e "$SOURCE_ENV" ]] || fail "source env $SOURCE_ENV missing"

  # Start from source env, then override the critical lines.  Any existing
  # .env is overwritten — this script owns it on smoke runtimes.
  cp "$SOURCE_ENV" "$WORKTREE/.env"

  python3 - "$WORKTREE/.env" "$DB" "$PORT" "$CONNECTOR" "$BOT_TOKEN" "$WORKTREE" <<'PY'
import sys, pathlib
path, db, port, connector, token, worktree = sys.argv[1:7]
overrides = {
    "AIOS_DB_URL": f"postgresql://aios:aios@localhost:5433/{db}",
    "AIOS_API_PORT": port,
    # The CLI targets AIOS_URL; without pinning it to THIS runtime's port a
    # port-overridden smoke runtime would have its fixtures created against
    # whatever the source .env points at (often :8090 — a sibling worktree).
    "AIOS_URL": f"http://127.0.0.1:{port}",
    "AIOS_CONNECTORS_ENABLED": connector,
    "AIOS_DEFAULT_MCP_PERMISSION_POLICY": "always_allow",
    # Pydantic rejects a relative workspace root ("must be an absolute
    # path"); the source .env ships `./workspaces`.  Resolve it under the
    # worktree so api + worker agree on the same absolute path.
    "AIOS_WORKSPACE_ROOT": f"{worktree}/workspaces",
    # Neutralise the one JSON-valued var: every phase below does `set -a;
    # source .env`, and bash strips the inner double-quotes of a JSON value
    # (`[{"match":…}]` -> `[{match:…}]`), so the exported var becomes invalid
    # JSON and shadows the correct .env-file value pydantic would have read.
    # Smoke runtimes don't exercise operator OAuth apps, so `[]` is correct and
    # bash-source-safe. (The app itself never sources .env — pydantic reads the
    # file directly — so this only bites the shell-sourcing the script does.)
    "AIOS_OAUTH_PROVIDER_APPS": "[]",
}
if connector == "telegram" and token:
    overrides["AIOS_TELEGRAM_BOT_TOKEN"] = token

lines = pathlib.Path(path).read_text().splitlines()
seen = set()
out = []
for line in lines:
    key = line.split("=", 1)[0] if "=" in line and not line.startswith("#") else None
    if key and key in overrides:
        out.append(f"{key}={overrides[key]}")
        seen.add(key)
    else:
        out.append(line)
for key, val in overrides.items():
    if key not in seen:
        out.append(f"{key}={val}")
pathlib.Path(path).write_text("\n".join(out) + "\n")
PY
  mkdir -p "$WORKTREE/workspaces"
  ok ".env written"
}

# ── 3. fresh DB + migrate ─────────────────────────────────────────────
ensure_db() {
  local pg
  pg="$(pg_container)"
  say "creating database $DB (container=$pg)"
  docker exec "$pg" psql -U aios -d postgres -c "DROP DATABASE IF EXISTS \"$DB\"" >/dev/null
  docker exec "$pg" psql -U aios -d postgres -c "CREATE DATABASE \"$DB\" OWNER aios" >/dev/null

  say "applying schemas (alembic + procrastinate via 'aios migrate')"
  ( set -a; source "$WORKTREE/.env"; set +a; uv run aios migrate >/dev/null )
  ok "schemas applied"
}

# ── 3b. bootstrap root account ────────────────────────────────────────
# A fresh DB has no account_keys rows, so every CLI/API call 401s with
# "invalid api key" — the .env's AIOS_API_KEY is NOT what the server
# validates (auth is DB-backed via account_keys).  Mint the root account
# and write its plaintext key back into .env so the CLI authenticates.
bootstrap_account() {
  say "bootstrapping root account"
  local key
  key="$( set -a; source "$WORKTREE/.env"; set +a
    uv run python -c "
import asyncio, os
from aios.db.pool import create_pool
from aios.services.accounts import bootstrap_root
async def main():
    pool = await create_pool(os.environ['AIOS_DB_URL'], max_size=2)
    try:
        r = await bootstrap_root(pool, display_name='smoke-root')
        print(r.plaintext_key)
    finally:
        await pool.close()
asyncio.run(main())
" 2>/dev/null )"
  [[ -z "$key" ]] && fail "bootstrap_root produced no key — see migrate/env"
  python3 - "$WORKTREE/.env" "$key" <<'PY'
import sys, pathlib
path, key = sys.argv[1:3]
lines = pathlib.Path(path).read_text().splitlines()
out, seen = [], False
for line in lines:
    if line.startswith("AIOS_API_KEY="):
        out.append(f"AIOS_API_KEY={key}"); seen = True
    else:
        out.append(line)
if not seen:
    out.append(f"AIOS_API_KEY={key}")
pathlib.Path(path).write_text("\n".join(out) + "\n")
PY
  ok "root account bootstrapped; AIOS_API_KEY written to .env"
}

# ── 4. start api + worker ─────────────────────────────────────────────
start_runtime() {
  mkdir -p "$WORKTREE/.logs"
  # Don't double-start: kill any previous OUR-worktree processes only.
  pkill -f "${WORKTREE}.*-m aios api" 2>/dev/null || true
  pkill -f "${WORKTREE}.*-m aios worker" 2>/dev/null || true
  sleep 0.5

  say "starting api + worker"
  # Unlink rather than truncate: the previous worker's still-open stdout
  # fd keeps appending to the inode it opened, which is now anonymous,
  # so its shutdown traceback can't bleed into the new log file.  See #298.
  rm -f "$WORKTREE/.logs/api.log" "$WORKTREE/.logs/worker.log"
  ( set -a; source "$WORKTREE/.env"; set +a
    nohup uv run python -m aios api > "$WORKTREE/.logs/api.log" 2>&1 &
    nohup uv run python -m aios worker > "$WORKTREE/.logs/worker.log" 2>&1 & )

  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    # Fatal-error short-circuit (covers both modes).  ValidationError
    # catches the relative-workspace-root class early.
    if grep -qE "RuntimeError|Conflict|Errno 48|duplicate_instance|ValidationError" \
         "$WORKTREE/.logs/worker.log" "$WORKTREE/.logs/api.log" 2>/dev/null; then
      tail -20 "$WORKTREE/.logs/worker.log" "$WORKTREE/.logs/api.log"
      fail "runtime startup failed — see .logs/"
    fi
    if $NO_CONNECTOR; then
      # Headless readiness: api port listening + worker booted.  There's no
      # connector to identify, so the connector-log grep would never fire.
      if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 \
         && grep -q "worker.startup" "$WORKTREE/.logs/worker.log" 2>/dev/null; then
        ok "runtime up (headless, no connector)"
        return 0
      fi
    else
      if grep -q "telegram\.bot\.identified\|signal\.account\.ready" "$WORKTREE/.logs/worker.log" 2>/dev/null; then
        ok "connector running"
        return 0
      fi
    fi
    sleep 0.5
  done
  fail "runtime didn't become ready within 60s — see .logs/"
}

# ── 5. fixtures ───────────────────────────────────────────────────────
build_fixtures() {
  ( set -a; source "$WORKTREE/.env"; set +a

    say "creating env"
    local env_id
    env_id="$(uv run aios -f json envs create --data '{"name":"smoke"}' 2>/dev/null \
              | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")"
    [[ -z "$env_id" ]] && fail "env create failed (auth? run bootstrap_account; is the runtime up on \$AIOS_URL?)"

    say "creating agent"
    local agent_json
    agent_json="$(mktemp)"
    local sysprompt
    if [[ -n "$SYSTEM_PROMPT_FILE" && -e "$SYSTEM_PROMPT_FILE" ]]; then
      sysprompt="$(cat "$SYSTEM_PROMPT_FILE")"
    elif $NO_CONNECTOR; then
      sysprompt="You are a verification assistant. Run tools as asked; keep replies brief."
    else
      sysprompt="You are a smoke-test assistant on Telegram. Speak in your own voice; never claim to be Claude Code. Keep replies brief and conversational."
    fi
    python3 - "$agent_json" "$AGENT_NAME" "$sysprompt" <<'PY'
import json, sys
path, name, system = sys.argv[1:4]
json.dump({
    "name": name,
    "model": "anthropic/claude-sonnet-4-6",
    "system": system,
    "tools": [
        {"type": "bash", "enabled": True},
        {"type": "read", "enabled": True},
        {"type": "write", "enabled": True},
    ],
}, open(path, "w"))
PY
    local agent_id
    agent_id="$(uv run aios -f json agents create --file "$agent_json" 2>/dev/null \
                | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")"
    rm -f "$agent_json"
    [[ -z "$agent_id" ]] && fail "agent create failed"

    say "creating session"
    local sess_id
    sess_id="$(uv run aios -f json sessions create --agent "$agent_id" --environment-id "$env_id" 2>/dev/null \
               | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")"
    [[ -z "$sess_id" ]] && fail "session create failed"

    if $NO_CONNECTOR; then
      cat <<EOF

✓ headless runtime ready (no connector)
  session:   $sess_id
  agent:     $agent_id
  api:       http://127.0.0.1:$PORT
  db:        $DB

Drive it via the CLI (AIOS_URL/AIOS_API_KEY are in .env):
  uv run aios sessions send $sess_id "run sleep 30"
  uv run aios sessions events $sess_id --kind message

smoke_session_id=$sess_id
EOF
      return 0
    fi

    say "fetching bot account id"
    local bot_id
    bot_id="$(uv run aios -f json connectors list 2>/dev/null \
              | python3 -c "
import json,sys
data=json.load(sys.stdin)
for c in data['connectors']:
    if c['connector'] == '$CONNECTOR' and c['accounts']:
        print(c['accounts'][0]['id']); break")"
    [[ -z "$bot_id" ]] && fail "no account in $CONNECTOR snapshot"

    say "creating connection"
    local conn_id
    conn_id="$(uv run aios -f json connections create --connector="$CONNECTOR" --account="$bot_id" 2>/dev/null \
               | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")"

    say "attaching connection → session"
    uv run aios connections attach "$conn_id" --session-id="$sess_id" >/dev/null

    cat <<EOF

✓ smoke runtime ready
  session:   $sess_id
  agent:     $agent_id
  connector: $CONNECTOR/$bot_id
  api:       http://127.0.0.1:$PORT
  db:        $DB

Next: arm monitors via .claude/skills/aios-live-monitor/scripts/preflight.sh $sess_id

smoke_session_id=$sess_id
EOF
  )
}

# ── main ──────────────────────────────────────────────────────────────
if $RESTART_ONLY; then
  start_runtime
  exit 0
fi

if [[ "$PHASE" == "fixtures" ]]; then
  # Processes already started elsewhere (e.g. Claude's run_in_background).
  build_fixtures
  exit 0
fi

if [[ -z "$BOT_TOKEN" && "$CONNECTOR" == "telegram" ]]; then
  fail "--bot-token is required for connector=telegram (or pass --no-connector for headless)"
fi

preflight
write_env
ensure_db
bootstrap_account

if [[ "$PHASE" == "prep" ]]; then
  cat <<EOF

✓ prep complete (db + .env + root account)
  Start the runtime yourself (persists across tool calls), then build fixtures:
    # two run_in_background Bash calls:
    ( set -a; source .env; set +a; uv run python -m aios api )
    ( set -a; source .env; set +a; uv run python -m aios worker )
    # then:
    $0 ${NO_CONNECTOR:+--no-connector }--phase fixtures --port $PORT --db $DB
EOF
  exit 0
fi

start_runtime
build_fixtures
