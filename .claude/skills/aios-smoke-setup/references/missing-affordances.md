# Admin CLI/API affordance gaps surfaced by smoke testing

Each item below cost real time during the openclaw-parity smoke session.  Listed as candidate enhancement issues with a one-line proposal and rough impact estimate.  Highest-value items at the top.

## High value

### 1. `aios connectors check <name>` — pre-flight health probe

**Today:** the only signal that a connector is healthy is `aios connectors list` showing `status: running`.  But a connector can be "running" while its inner poll loop is failing (e.g., Telegram getUpdates 409 conflict in a tight respawn loop).  No way to detect this short of reading worker.log.

**Proposal:** `aios connectors check <connector>[:<instance>]` performs a synthetic round-trip against the platform (Telegram: `getMe` + a one-shot `getUpdates?timeout=2`; Signal: a `getDevices` etc.) and reports `ok` / `bot_uncontested: true|false` / `last_poll_error: <str|null>` / `auth_ok: true|false`.

**Impact:** ~10 min of bot-token-conflict diagnosis avoided per smoke session.

### 2. `aios sessions confirm <id> <tool_call_id> --allow|--deny` 

**Today:** unblocking a `requires_action` tool call requires a manual `curl POST /v1/sessions/<id>/tool-confirmations` with hand-crafted JSON (and the schema field is `result` not `decision`, which trips up first-time users).

**Proposal:** thin CLI wrapper.  Default `--allow`; `--deny` takes optional `--message`.

**Impact:** ~5 min the first time, and removes the curl-vs-CLI dissonance.

### 3. Auto-create connection on `accounts_updated` (not just on first inbound)

**Today:** when the connector reports a new account via `notifications/aios/accounts`, the supervisor records it in the snapshot but doesn't create the `connections` row.  The operator must `aios connections create --connector=X --account=<bot_id>` manually before they can attach.  The auto-create-on-first-inbound path mentioned in `connections.py` only fires after a real user message arrives — chicken-and-egg for an attach-before-DM flow.

**Proposal:** when `accounts_updated` lands, idempotent-create a detached connection per (connector, account) tuple.  The operator only ever needs to `attach` it.

**Impact:** ~5 min per fresh-DB smoke; removes one of five steps in the resource chain.

### 4. CLI flag and noun consistency

- `aios envs` ↔ `aios environments` — only `envs` works; `environments` returns "No such command".  Add an alias.
- `aios connections attach --session-id` ↔ `--session` — only `--session-id` works.  Add `--session` alias for symmetry with `aios sessions stream <id>`.
- `aios connections list --account=<id>` — currently rejected with "No such option"; only `--connector` filter exists.  Add `--account`.
- `aios -f json sessions get <id>` works; `aios sessions -f json get <id>` doesn't.  This is a typer convention but worth a one-line note in `--help`.

**Impact:** ~3 min trial-and-error per command, repeated.

## Medium value

### 5. Connector status surfaces `last_poll_error`

`aios connectors list` JSON includes `last_error`, but it's null in the cases I observed (the worker silently respawned the connector subprocess on `Conflict 409` rather than holding the error).  Surface the most recent poll-loop error string here so `connectors check` can read it without reading log files.

### 6. `aios envs create --name <str>` shorthand

Today requires `--data '{"name":"smoke"}'` or a file.  A one-flag shorthand for the trivial case would make the resource-chain script half a line shorter.

### 7. `aios agents create --name <str> --model <str> --system <str> --tool <type>...` shorthand

Same idea.  For the smoke scenario the JSON is always trivial — env vars or repeated `--tool` flags would let the script skip the temp-file dance.

### 8. `--instructions` parameter on `agents create` to override system prompt at session level

When connecting an existing well-loved agent (e.g., `ultron`) to a new platform for smoke testing, the agent's system prompt may be platform-specific (Signal phone number, etc.).  Letting `aios sessions create` pass an `additional_system` would avoid forking the agent for each smoke run.

## Low value (style)

### 9. `aios envs list` returning `(none)` literally

Could be `0 environments` or an empty-table-with-headers shape consistent with other list commands.  Cosmetic.

### 10. Document `aios migrate` in CLAUDE.md as the canonical migration command

`alembic upgrade head` is currently in CLAUDE.md but it's insufficient on its own (misses procrastinate schema).  Replace with `aios migrate`.

---

## Filing strategy

If filing each as a separate issue, items 1–4 are the worth-doing-now bucket.  5–10 are nice-to-haves; bundle them as a single "smoke-setup CLI ergonomics" tracking issue if filing at all.
