---
name: aios-smoke-setup
description: This skill should be used when the user asks to "smoke test the connector", "DM the bot in this worktree", "set up a fresh aios runtime", "spin up aios", "prep a smoke session", "bring up aios on this branch", "bring up aios headless", "spin up aios without a bot", "no-connector runtime", or needs a running aios runtime to "verify a harness change" / repro a sweep/loop/context/tool_dispatch issue — whether pointed at a real Telegram/Signal/WhatsApp bot OR headless (api+worker only, driven via the CLI).  Bundles the pre-flight checks (bot getUpdates conflict, sibling worktree non-interference, free port), the isolated `.env` overrides (separate DB + port + ``AIOS_URL`` + absolute ``AIOS_WORKSPACE_ROOT`` + ``AIOS_DEFAULT_MCP_PERMISSION_POLICY=always_allow``), the right migration command (``aios migrate``, not ``alembic upgrade head``), the fresh-DB ``bootstrap_root`` step (a new DB 401s every call until an account key is minted), and the env/agent/(connection)/session resource chain in one ``setup.sh`` so the first DM (or first CLI round-trip) works in seconds instead of forty-five minutes.  Use ``--no-connector`` for harness verification; for WhatsApp the pair flow differs (QR scan, no bot-token) — hand off to ``aios-whatsapp-pair`` then ``aios-live-monitor``.
---

# aios smoke session setup

Bring up an isolated aios runtime in a worktree, ready to receive real Telegram (or Signal) messages and respond.  The "isolated" part is load-bearing: parallel smoke sessions in sibling worktrees share the same Postgres instance and would otherwise fight over the supervisor lock + API port.

## When to use

- Smoke-testing a connector / harness PR against a live bot before merging
- Reproducing a user-reported issue on a real platform account
- Demoing a feature end-to-end on a fresh branch

Skip this skill if you're just running unit / e2e tests — `uv run pytest` is the right tool, no runtime setup needed.

## The fast path

```bash
.claude/skills/aios-smoke-setup/scripts/setup.sh \
  --bot-token <telegram_token> \
  [--agent-name smoke] \
  [--system-prompt-file <path>]
```

Reads source `.env` from `~/code/aios/.env`, writes a worktree-scoped `.env` with overrides, creates a fresh `aios_smoke_<branch_short>` Postgres database, runs migrations, bootstraps the root account, starts api + worker, builds the env/agent/connection/session resource chain, and prints `session_id` + the focal-channel address on stdout.

## Headless bring-up (no connector — for `/verify` of harness changes)

When verifying a change to the harness core (sweep / loop / context / tool_dispatch / completion), a connector is dead weight — you just need a running runtime you can drive via the CLI. Use `--no-connector`: it skips the bot-token requirement and the connector-readiness wait, and builds env+agent+session **without** the connection/attach steps.

```bash
.claude/skills/aios-smoke-setup/scripts/setup.sh --no-connector [--port <num>]
# → prints a session_id; drive it with:
uv run aios sessions send <sess_id> "run sleep 30"
uv run aios sessions events <sess_id> --kind message
```

### When Claude is driving (not a human terminal)

`setup.sh`'s api/worker are spawned with `nohup … &`. That survives a **human** running the script in a terminal, but **not** Claude running it via the Bash tool — the harness reaps the call's process group on completion (`nohup` only ignores SIGHUP). The runtime dies before you can drive it. Use the three-step `--phase` split so the processes are owned by `run_in_background` calls that persist across tool calls:

1. `setup.sh --no-connector --phase prep` — writes `.env`, creates+migrates the DB, bootstraps the root account. No long-lived processes.
2. Start **api and worker as two separate `run_in_background` Bash calls**:
   `( set -a; source .env; set +a; uv run python -m aios api )` and `… -m aios worker`.
3. `setup.sh --no-connector --phase fixtures` — builds env+agent+session once the runtime is up.

Then SIGKILL/restart the worker (`kill -9 <pid>` → restart via `run_in_background`) to exercise crash-recovery paths, and read the event log with `aios sessions events` or `docker exec <pg> psql`. See `references/gotchas.md` #9–13 for the cold-start failure modes this sequence avoids.

## Pre-flight rules (every smoke run)

1. **Telegram getUpdates conflict.** Telegram's bot polling is exclusive — if the token is already being polled by a remote deployment / coworker / stale process, the connector spawns and dies in a `Conflict 409` loop.  The script tests this with `curl …/getUpdates?timeout=2` before starting the worker; if it returns `409`, ask the user to either rotate the token or stop the rogue poller.

2. **Sibling worktree non-interference.** Run `ps -ef | grep -E "aios (api|worker)" | grep -v grep`.  If a process is running from a *different* worktree path, **do not kill it** — it might be a managed monitor session another Claude Code instance is driving.  Choose a port that doesn't conflict (the script auto-finds the first free port from 8091).

3. **Bot identity.** The connector calls `Bot.get_me()` at startup; the returned `bot_id` becomes the `account` field of every connection and the `<account>` segment of channel addresses.  The script captures this from `aios connectors list` and uses it for `aios connections create --account=<bot_id>`.

## The four overrides that always belong in the worktree's .env

```dotenv
AIOS_DB_URL=postgresql://aios:aios@localhost:5433/aios_smoke_<branch_short>
AIOS_API_PORT=8091   # or next free port from 8091
AIOS_CONNECTORS_ENABLED=telegram   # narrow to what's actually installed in this venv
AIOS_DEFAULT_MCP_PERMISSION_POLICY=always_allow
```

Without the **DB** and **port** overrides, two parallel smoke sessions will deadlock each other on the procrastinate supervisor lock and the uvicorn bind.  Without **`always_allow`**, every model tool call parks in `requires_action` until you POST a `tool-confirmation` — fine for production, terrible for unattended smoke.  Without **narrowing `AIOS_CONNECTORS_ENABLED`**, a missing connector entry-point in this worktree's venv (e.g. signal not installed) crashes the worker at startup.

## The migration step is `aios migrate`, not `alembic upgrade head`

CLAUDE.md mentions `alembic upgrade head` for migrations — that creates aios tables but **not** the procrastinate schema.  The worker then crashes on first job with `procrastinate.exceptions.ConnectorException`.  Run `uv run aios migrate` instead; it does both in one shot.

## The resource chain

A fresh DB has zero envs, agents, connections, sessions.  In order:

```bash
uv run aios envs create --data '{"name":"smoke"}'                               # → env_<id>
uv run aios agents create --file agent.json                                     # → agent_<id>
uv run aios connections create --connector=telegram --account=<bot_id>          # → conn_<id>
uv run aios sessions create --agent <agent_id> --environment-id <env_id>        # → sess_<id>
uv run aios connections attach <conn_id> --session-id=<sess_id>                 # binds connection
```

CLI quirks: `envs` is plural; `--session-id` not `--session`; `sessions create` requires both `--agent` AND `--environment-id`; the `agent.json` `tools[]` items must use bare type names (`bash`, `read`, `write`, …) and **`switch_channel` is auto-included — do not list it** (the API rejects it).

## WhatsApp variant

Telegram and Signal pair via a bot token / signal-cli registration that
the setup script can resolve up-front.  WhatsApp's pair is interactive
(QR scan) and so happens *after* `setup.sh` brings up the runtime, not
during it.  Variant flow:

1. Run `setup.sh --connector whatsapp` (no `--bot-token` needed — that
   flag is Telegram-only).  This brings up api+worker+connector+empty
   resource chain.
2. Hand off to `aios-whatsapp-pair` to scan the QR and bind the
   account.  That skill owns the daemon spawn, ASCII QR rendering, and
   `confirm-pairing` block.
3. Hand off to `aios-live-monitor` for chat narration.

The connection-create step also differs: WhatsApp connections take a
`--secret phone=+<E.164>` instead of Telegram's `--account=<bot_id>`.
The setup script handles that branch when `--connector whatsapp` is
passed.

## After setup: hand off to aios-live-monitor

`scripts/setup.sh` ends by printing the `session_id` and a hint:

```
✓ smoke runtime ready
  session: sess_01...
  focal:   telegram/<bot_id>/<chat_id-pending>
  api:     http://127.0.0.1:8091
  port-free: yes  bot-uncontested: yes  always_allow: yes

Next: arm monitors via .claude/skills/aios-live-monitor/scripts/preflight.sh sess_01...
```

Then DM the bot from your phone.  The first inbound message triggers the supervisor's auto-route to the attached session.

## Restart cadence

Per saved feedback (`feedback_restart_after_commit.md`), every commit on the smoke branch auto-triggers a stop+restart on the new HEAD.  The script's `--restart` flag short-circuits the DB+fixtures phases (they're already there) and just re-runs the api/worker:

```bash
.claude/skills/aios-smoke-setup/scripts/setup.sh --restart
```

## Common gotchas

See **`references/gotchas.md`** for the eight that cost ~5–15 min each during the parity-PR smoke session, with the symptom → root cause → one-liner fix table.

## Known CLI/API affordance gaps

See **`references/missing-affordances.md`** — wishlist of admin-side ergonomics that would have shaved an hour off the parity-PR smoke.  Worth filing as enhancement issues; high-impact ones include `aios sessions confirm <id> <tool_call_id>`, `aios connectors check <name>` (bot pre-flight), supervisor auto-creating connections on `accounts_updated` (not just on first inbound), and consistent CLI flag naming (`envs`/`environments`, `--session`/`--session-id`).

## Additional resources

- **`scripts/setup.sh`** — the all-in-one bring-up; idempotent on re-run, supports `--restart` for commit-cycle re-deploys
- **`references/gotchas.md`** — symptom-to-fix table for the eight common failure modes
- **`references/missing-affordances.md`** — admin CLI/API gaps surfaced by smoke testing; candidate enhancement issues
