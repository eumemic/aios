#!/usr/bin/env bash
# Pre-flight check before arming the live chat monitor on an aios session.
#
# Verifies api/worker reachability, prints the session's current
# last_event_seq (single round-trip via `sessions get`), and emits the exact
# Monitor commands (chat stream + worker error filter) to copy-paste.
#
# Usage:
#   preflight.sh <session_id> [worktree_dir]
#
# worktree_dir defaults to the current directory; must contain a .env (or
# symlink to one) with AIOS_URL / AIOS_API_KEY.
set -euo pipefail

SESSION_ID="${1:-}"
WORKTREE="${2:-$(pwd)}"

if [[ -z "$SESSION_ID" ]]; then
  echo "usage: preflight.sh <session_id> [worktree_dir]" >&2
  exit 2
fi

cd "$WORKTREE"
if [[ ! -e .env ]]; then
  echo "ERROR: no .env in $WORKTREE — chat monitor needs AIOS_URL/AIOS_API_KEY" >&2
  exit 2
fi
set -a; source .env; set +a

# 1. Reachability
if ! uv run aios status >/dev/null 2>&1; then
  echo "ERROR: aios api unreachable at ${AIOS_URL:-default} — start with 'uv run python -m aios api'" >&2
  exit 1
fi

# 2. Tail seq via the session-info endpoint (single RTT — `last_event_seq` is
#    returned directly).  Walking the events page would touch every row in
#    the log, defeating the cost rationale this skill exists to avoid.
TAIL_SEQ=$(uv run aios --format json sessions get "$SESSION_ID" 2>/dev/null \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['last_event_seq'])")

if [[ -z "$TAIL_SEQ" || "$TAIL_SEQ" == "None" ]]; then
  echo "ERROR: could not read last_event_seq for $SESSION_ID — does the session exist?" >&2
  exit 1
fi

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$WORKTREE/.logs/worker.log"

echo "session : $SESSION_ID"
echo "tail seq: $TAIL_SEQ"
echo
echo "── chat monitor (Monitor tool, persistent: true) ──"
echo "cd $WORKTREE && set -a && source .env && set +a && \\"
echo "  uv run aios sessions stream $SESSION_ID --after-seq $TAIL_SEQ --raw 2>&1 | \\"
echo "  python3 $SKILL_DIR/chat_filter.py"
echo
echo "── error filter (Monitor tool, persistent: true) ──"
if [[ -f "$LOG" ]]; then
  echo "tail -f $LOG | grep --line-buffered -E 'error|ERROR|exception|Exception|traceback|Traceback|RuntimeError|KeyError|FAILED'"
else
  echo "(worker.log not found at $LOG — check the worktree's logs dir)"
fi
