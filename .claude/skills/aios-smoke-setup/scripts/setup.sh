#!/usr/bin/env bash
# Bring up an isolated aios smoke runtime in this worktree, ready to
# receive real Telegram messages and respond.
#
# Idempotent.  Re-running with the same flags refreshes the DB + fixtures.
# Use --restart to reload api/worker on the new HEAD without rebuilding the
# DB (the smoke-branch commit-cycle pattern).
#
# Usage:
#   setup.sh --bot-token <token> [--connector telegram] \
#            [--agent-name smoke] [--system-prompt-file <path>] \
#            [--source-env ~/code/aios/.env] [--port <num>] \
#            [--db <name>] [--restart]
#
# Output (last line):
#   smoke_session_id=sess_01...

set -euo pipefail

# ── arg parsing ───────────────────────────────────────────────────────
BOT_TOKEN=""
CONNECTOR="telegram"
AGENT_NAME="smoke"
SYSTEM_PROMPT_FILE=""
SOURCE_ENV="${HOME}/code/aios/.env"
PORT=""
DB=""
RESTART_ONLY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bot-token) BOT_TOKEN="$2"; shift 2;;
    --connector) CONNECTOR="$2"; shift 2;;
    --agent-name) AGENT_NAME="$2"; shift 2;;
    --system-prompt-file) SYSTEM_PROMPT_FILE="$2"; shift 2;;
    --source-env) SOURCE_ENV="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --db) DB="$2"; shift 2;;
    --restart) RESTART_ONLY=true; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

WORKTREE="$(pwd)"
BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo nobranch)"
BRANCH_SHORT="$(echo "$BRANCH" | sed 's|wt/||; s|/|-|g; s/[^a-z0-9_-]/_/g' | head -c 30)"

[[ -z "$DB" ]] && DB="aios_smoke_${BRANCH_SHORT//-/_}"

say()   { printf "\033[36m▸\033[0m %s\n" "$*"; }
warn()  { printf "\033[33m⚠\033[0m %s\n" "$*"; }
fail()  { printf "\033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }
ok()    { printf "\033[32m✓\033[0m %s\n" "$*"; }

# ── 1. preflight ──────────────────────────────────────────────────────
preflight() {
  say "preflight"

  # 1a. Bot token uncontested
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
    for candidate in 8091 8092 8093 8094 8095; do
      if ! lsof -iTCP:"$candidate" -sTCP:LISTEN >/dev/null 2>&1; then
        PORT="$candidate"
        break
      fi
    done
    [[ -z "$PORT" ]] && fail "no free port in 8091..8095"
  elif lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    fail "port $PORT already in use"
  fi
  ok "port $PORT free"
}

# ── 2. write .env with overrides ──────────────────────────────────────
write_env() {
  say "writing .env (db=$DB port=$PORT)"
  [[ -e "$SOURCE_ENV" ]] || fail "source env $SOURCE_ENV missing"

  # Start from source env, then override the four critical lines.  Any
  # existing .env is overwritten — this script owns it on smoke runtimes.
  cp "$SOURCE_ENV" "$WORKTREE/.env"

  python3 - "$WORKTREE/.env" "$DB" "$PORT" "$CONNECTOR" "$BOT_TOKEN" <<'PY'
import sys, pathlib
path, db, port, connector, token = sys.argv[1:6]
overrides = {
    "AIOS_DB_URL": f"postgresql://aios:aios@localhost:5433/{db}",
    "AIOS_API_PORT": port,
    "AIOS_CONNECTORS_ENABLED": connector,
    "AIOS_DEFAULT_MCP_PERMISSION_POLICY": "always_allow",
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
  ok ".env written"
}

# ── 3. fresh DB + migrate ─────────────────────────────────────────────
ensure_db() {
  say "creating database $DB"
  DOCKER_HOST="${DOCKER_HOST:-unix:///Users/tom/.docker/run/docker.sock}" \
    docker exec aios-pg psql -U aios -d postgres -c "DROP DATABASE IF EXISTS \"$DB\"" >/dev/null
  DOCKER_HOST="${DOCKER_HOST:-unix:///Users/tom/.docker/run/docker.sock}" \
    docker exec aios-pg psql -U aios -d postgres -c "CREATE DATABASE \"$DB\" OWNER aios" >/dev/null

  say "applying schemas (alembic + procrastinate via 'aios migrate')"
  ( set -a; source "$WORKTREE/.env"; set +a; uv run aios migrate >/dev/null )
  ok "schemas applied"
}

# ── 4. start api + worker ─────────────────────────────────────────────
start_runtime() {
  mkdir -p "$WORKTREE/.logs"
  # Kill the supervisor first (sentinel is in its bash command line) so
  # it doesn't restart the worker we're about to kill.
  pkill -f "AIOS_WORKER_SUPERVISOR_FOR=${WORKTREE}" 2>/dev/null || true
  pkill -f "${WORKTREE}.*-m aios api" 2>/dev/null || true
  pkill -f "${WORKTREE}.*-m aios worker" 2>/dev/null || true
  sleep 0.5

  say "starting api + worker"
  : > "$WORKTREE/.logs/api.log"
  : > "$WORKTREE/.logs/worker.log"
  ( set -a; source "$WORKTREE/.env"; set +a
    nohup uv run python -m aios api > "$WORKTREE/.logs/api.log" 2>&1 &
    # Worker runs under a supervisor loop with capped backoff.  Backoff
    # protects against startup-failure storms (e.g. lock contention on
    # an aborted prior run); a process that lives >30s is considered
    # healthy and resets the backoff.
    nohup bash -c '# AIOS_WORKER_SUPERVISOR_FOR='"$WORKTREE"'
      delay=1
      while true; do
        started=$SECONDS
        uv run python -m aios worker
        ec=$?
        ran=$((SECONDS - started))
        if (( ran > 30 )); then delay=1; fi
        echo "[supervisor] worker exited ec=$ec ran=${ran}s, restarting in ${delay}s" >&2
        sleep "$delay"
        if (( delay < 15 )); then delay=$((delay * 2)); fi
        if (( delay > 15 )); then delay=15; fi
      done
    ' > "$WORKTREE/.logs/worker.log" 2>&1 & )

  # Wait for bot identification or fatal
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if grep -q "telegram\.bot\.identified\|signal\.account\.ready" "$WORKTREE/.logs/worker.log" 2>/dev/null; then
      ok "connector running"
      return 0
    fi
    if grep -qE "RuntimeError|Conflict|Errno 48|duplicate_instance" "$WORKTREE/.logs/worker.log" "$WORKTREE/.logs/api.log" 2>/dev/null; then
      tail -20 "$WORKTREE/.logs/worker.log"
      fail "runtime startup failed — see .logs/"
    fi
    sleep 0.5
  done
  fail "runtime didn't identify within 60s — see .logs/"
}

# ── 5. fixtures: env + agent + connection + session + attach ──────────
build_fixtures() {
  ( set -a; source "$WORKTREE/.env"; set +a

    say "creating env"
    local env_id
    env_id="$(uv run aios -f json envs create --data '{"name":"smoke"}' 2>/dev/null \
              | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")"
    [[ -z "$env_id" ]] && fail "env create failed"

    say "creating agent"
    local agent_json
    agent_json="$(mktemp)"
    local sysprompt
    if [[ -n "$SYSTEM_PROMPT_FILE" && -e "$SYSTEM_PROMPT_FILE" ]]; then
      sysprompt="$(cat "$SYSTEM_PROMPT_FILE")"
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

    say "creating session"
    local sess_id
    sess_id="$(uv run aios -f json sessions create --agent "$agent_id" --environment-id "$env_id" 2>/dev/null \
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

if [[ -z "$BOT_TOKEN" && "$CONNECTOR" == "telegram" ]]; then
  fail "--bot-token is required for connector=telegram"
fi

preflight
write_env
ensure_db
start_runtime
build_fixtures
