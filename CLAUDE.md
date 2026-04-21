# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
uv sync --dev

# Run checks (do all three before every commit)
uv run mypy src
uv run ruff check src tests && uv run ruff format --check src tests
uv run pytest tests/unit -q                    # ~170 tests, <1s, no Docker needed

# E2E tests (need Docker for testcontainer Postgres + sandbox)
DOCKER_HOST=unix:///Users/tom/.docker/run/docker.sock uv run pytest tests/e2e -q

# Run a single test
uv run pytest tests/unit/test_context.py::TestBuildMessages::test_basic_user_assistant -xvs

# Migrations
set -a && source .env && set +a
uv run alembic upgrade head

# Start the system (two processes)
uv run python -m aios api      # API server on :8090
uv run python -m aios worker   # procrastinate worker
```

### Client CLI

`aios` is a typer-based client CLI that talks to a running API. Config is read from env (`AIOS_URL`, `AIOS_API_KEY`) or `.env`; every command accepts `--url` / `--api-key` overrides and a global `--format {table,json}`.

```bash
# Reachability + auth check
uv run aios status

# Resource inspection
uv run aios agents list
uv run aios agents get <agent_id>
uv run aios sessions list --status running
uv run aios sessions events <session_id> --kind message

# Interactive chat (creates a session, streams the reply)
uv run aios chat --agent <agent_id> --environment-id <env_id>

# One-shot: send a message and stream until the turn ends
uv run aios chat --agent <agent_id> --environment-id <env_id> -m "list /workspace"

# Tail a running session from another terminal
uv run aios sessions stream <session_id>

# Post a user message without entering the REPL
uv run aios sessions send <session_id> "hello"

# Create resources from JSON (thin wire — server validates)
uv run aios agents create --file agent.json
uv run aios skills create --dir path/to/my-skill --title "My Skill"
```

Every resource has CRUD subcommands. See `aios <resource> --help` for specifics. Operator commands (`aios api`, `aios worker`, `aios migrate`) are also typer-backed but keep their original behavior.

## Architecture

aios is an event-driven agent runtime. The headline property: **every tool is implicitly async** — the model stays responsive to user messages even while tools are running.

### The step model

There is no loop. A procrastinate `wake_session` job runs `run_session_step()` which calls the model **once**, kicks off tool calls as fire-and-forget asyncio tasks, and returns. Tool completion appends a result event and defers a new wake. The "loop" is the job queue re-entering the step function.

Key flow: `wake_session` → `run_session_step` (loop.py) → `stream_litellm` (completion.py) → `launch_tool_calls` (tool_dispatch.py) → tool tasks append results → `defer_wake` → next step.

### Process split

- **API process** (`aios api`): FastAPI server. Handles HTTP requests, appends user messages, defers wake jobs, serves SSE streams. Does NOT run tools or call models.
- **Worker process** (`aios worker`): Runs procrastinate jobs. Calls the model, dispatches tools, manages sandbox containers. Holds worker-scoped globals on `harness/runtime.py`.

Both share Postgres: same database, same `LISTEN/NOTIFY`, same procrastinate job queue.

### Context builder

`harness/context.py` builds the chat-completions message list from the event log. It's the most critical file in the harness:
- **Monotonicity invariant**: appending events only appends to the context, never rewrites earlier messages (prompt cache stability).
- **Pending-result synthesis**: in-flight tool calls get synthetic `"pending"` results so the message structure is valid.
- **Blind-spot injection**: tool results that arrived during inference are injected as user messages at the tail.
- **`reacting_to` watermark**: each assistant message records the max seq of events it saw, so `should_call_model` knows what's "new."

### Custom tools (client-executed)

When the model calls a custom tool (type="custom"), the harness does NOT execute it. Instead:
1. Session idles with `stop_reason: {"type": "requires_action", "event_ids": [...]}`
2. Client executes the tool externally
3. Client POSTs result to `POST /sessions/:id/tool-results`
4. Result appended → wake deferred → next step fires

Built-in and custom tools can be called in the same response. `should_call_model`'s batch gating waits for ALL tool_call_ids to have results.

### Two pools in one process

The worker runs **two** Postgres connection pools:
- **asyncpg** (ours): event log, session state, all application queries
- **psycopg3** (procrastinate's): job queue. Separate connector, separate connections.

They don't share connections; they share Postgres state.

## Conventions

- **Python 3.13+** with PEP 695 type parameter syntax: `class ListResponse[T](BaseModel)`, `def select_window[T](...)`. Do NOT use `TypeVar` + `Generic[T]`.
- **`from __future__ import annotations`** at the top of every source file.
- **Raw SQL + asyncpg**, no ORM. Every query lives in `db/queries.py`.
- **`pg_notify($1, $2)` function form**, never literal `NOTIFY`. Postgres case-folds unquoted identifiers; our ULIDs have uppercase.
- **structlog** with `structlog.stdlib.LoggerFactory()`. NOT `PrintLoggerFactory`.
- **"Mind" not "brain"** — load-bearing terminology for the model/inference concept.
- **Extreme simplicity.** No defensive guards for model mistakes, no fuzzy matching, no model-specific shims. The model sees raw errors and retries through the session log. Ask: "can the model handle this failure itself?" If yes, don't add complexity.
- **Events are OpenAI chat-completions format**, not Anthropic Messages format. LiteLLM translates at the provider boundary.
- **Conventional commits**: `feat:`, `fix:`, `refactor:`, etc. Substantial commit bodies.
- **Co-Authored-By trailer** on AI-authored commits.
- **Ask non-trivial design decisions** before writing code.

## How to approach changes

- **Investigate before fixing** — when something behaves oddly (unexpected errors, stuck jobs, state inconsistencies), find the root cause first. Don't kill processes, delete rows, or reset state as a shortcut.
- **Fail hard, no fallbacks** — no silent skipping, no dummy values, no error suppression. The model sees raw errors and retries through the session log; that IS the design.
- **Correct-by-construction** — the Key invariants below (gapless seq, monotonic context, tool-always-appends-result) are examples of this stance. Prefer designs that produce valid state in one pass over multi-stage corrective pipelines.
- **Push back on bad requests** — if a request conflicts with existing architecture, or has an obvious root-cause fix the user missed, flag it before implementing. You have context the user may not in the moment.
- **Don't deprecate, delete** — remove old code paths rather than leaving shims. Git history preserves them.
- **After refactors, grep exhaustively** — for any rename or move, search the whole tree (including tests) for the old name before declaring done.

## Key invariants

1. **Gapless seq per session** — every `append_event` locks the session row, increments `last_event_seq`, inserts. No gaps.
2. **Monotonic context** — the context is a monotonic function of the log. Prompt cache stays stable within a windowing chunk.
3. **`reacting_to` watermark** — every assistant message carries the seq of the latest user/tool event in its context.
4. **Tool-always-appends-result** — every async tool task MUST append exactly one tool_result event and defer a wake (try/except/finally).
5. **Procrastinate lock** — `lock="{session_id}"` for mutual exclusion, `queueing_lock="{session_id}"` for wake deduplication.
6. **NOTIFY after commit** — `pg_notify` fires outside the transaction so subscribers don't see uncommitted rows.

## Environment variables

All aios settings use the `AIOS_` prefix (Pydantic settings with `env_prefix="AIOS_"`):
- `AIOS_API_KEY` — bearer auth key
- `AIOS_VAULT_KEY` — base64-encoded 32-byte libsodium key (do NOT regenerate if Postgres has encrypted data)
- `AIOS_DB_URL` — Postgres connection string
- `AIOS_API_PORT` — default 8080
- `AIOS_TAVILY_API_KEY` — for web_fetch/web_search tools

Model provider keys use standard LiteLLM env vars (no `AIOS_` prefix): `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
