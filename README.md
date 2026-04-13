# aios

An open-source agent runtime inspired by [Anthropic's Managed Agents](https://www.anthropic.com/engineering/managed-agents): Postgres-backed sessions, Docker sandbox, any LiteLLM-compatible model.

**The model behind an agent is just a URL.** The agent's `model` field is a LiteLLM-compatible string — `anthropic/claude-opus-4-6`, `ollama/llama3.3`, `openrouter/moonshotai/kimi-k2.5`, anything LiteLLM speaks. The harness POSTs chat-completions requests; what's behind that URL is somebody else's problem.

## Architecture

```
                    ┌─────────────────┐
                    │   API server    │  FastAPI, stateless
                    │  (aios api)     │  HTTP + SSE
                    └───────┬─────────┘
                            │  Postgres
                            │  LISTEN/NOTIFY
                            │  procrastinate jobs
                    ┌───────┴─────────┐
                    │    Worker(s)    │  procrastinate, stateless
                    │ (aios worker)   │  model calls, tool dispatch
                    └───────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────┴─────┐ ┌────┴────┐ ┌──────┴──────┐
        │  LiteLLM  │ │ Docker  │ │   Tavily    │
        │  (model)  │ │ sandbox │ │  (web tools) │
        └───────────┘ └─────────┘ └─────────────┘
```

Three primitives, each independently replaceable:

- **Session** — Postgres-backed append-only event log. The durable truth. Survives crashes, supports resume.
- **Harness** — the step function. Reads events, builds context, calls the model once, dispatches tool calls as fire-and-forget async tasks, returns. The "loop" is the job queue re-entering the step function.
- **Sandbox** — a Docker container per session with a bind-mounted workspace volume. Lazily provisioned on first tool call. Cattle, not pets.

The harness is split into a stateless **API server** (`aios api`) and one or more **workers** (`aios worker`). They communicate through Postgres — same database, same `LISTEN/NOTIFY`, same job queue (procrastinate).

## Features

### Tools (8 built-in + custom + MCP)

| Tool | Description |
|---|---|
| `bash` | Shell commands via `docker exec` |
| `read` | Read files with line numbers |
| `write` | Write files (base64 safe for arbitrary content) |
| `edit` | Find-and-replace with unified diff |
| `glob` | File pattern matching (ripgrep) |
| `grep` | Content search with output modes, context, multiline regex (ripgrep) |
| `web_fetch` | Fetch URLs and return markdown (Tavily) |
| `web_search` | Search the web (Tavily) |
| **Custom tools** | Client-executed tools with `requires_action` flow |
| **MCP tools** | Connect remote MCP servers, auto-discover tools |

### Agent versioning

Every update creates an immutable version. Full version history. Sessions can pin to a specific version or float on `latest` (auto-updating).

### Tool permissions

Per-tool `always_allow` / `always_ask` policies. The `always_ask` flow idles the session with `requires_action` until the client confirms or denies.

### Environments

Container configuration with pre-installed packages (pip, npm, apt, cargo, gem, go) and network policies (`unrestricted` or `limited` with domain allowlist via iptables).

### Skills

Reusable, versioned knowledge resources with progressive disclosure. Loaded into the model's context on demand when relevant to the task.

### Vaults

Encrypted credential storage for MCP server auth. Sessions bind vaults at creation time. OAuth and static bearer token types.

### Streaming

Real-time SSE streaming with per-token deltas via `pg_notify`. Span events with per-request token usage.

### Session lifecycle

Create, update, archive, delete, interrupt. Mutable sessions. Rescheduling with auto-retry (3 attempts, 5s delay) for transient errors.

## Quickstart

```bash
# Install
uv sync --dev

# Set up Postgres (or use your own)
docker run -d --name aios-pg -p 5433:5432 \
  -e POSTGRES_USER=aios -e POSTGRES_PASSWORD=aios -e POSTGRES_DB=aios \
  postgres:16-alpine

# Build the sandbox image
docker build -t aios-sandbox:latest -f docker/Dockerfile.sandbox docker/

# Configure
cat > .env << 'EOF'
AIOS_API_KEY=your-secret-key
AIOS_VAULT_KEY=$(python3 -c "import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())")
AIOS_DB_URL=postgresql://aios:aios@localhost:5433/aios
AIOS_API_PORT=8090
EOF

# Migrate
set -a && source .env && set +a
uv run python -m aios migrate

# Start (two processes)
uv run python -m aios api      # API server
uv run python -m aios worker   # Worker process

# Create an environment + agent + session
export H="Authorization: Bearer your-secret-key"
export CT="Content-Type: application/json"

curl -H "$H" -H "$CT" -X POST http://localhost:8090/v1/environments \
  -d '{"name": "default"}'

curl -H "$H" -H "$CT" -X POST http://localhost:8090/v1/agents \
  -d '{
    "name": "assistant",
    "model": "openrouter/anthropic/claude-sonnet-4-6",
    "system": "You are a helpful assistant.",
    "tools": [{"type": "bash"}, {"type": "read"}, {"type": "write"}, {"type": "edit"},
              {"type": "glob"}, {"type": "grep"}, {"type": "web_fetch"}, {"type": "web_search"}]
  }'

curl -H "$H" -H "$CT" -X POST http://localhost:8090/v1/sessions \
  -d '{"agent_id": "agent_01...", "environment_id": "env_01...",
       "initial_message": "What is the top story on Hacker News right now?"}'

# Stream the response
curl -N -H "$H" http://localhost:8090/v1/sessions/sess_01.../stream
```

Model API keys are configured via standard LiteLLM environment variables: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. For web tools: `AIOS_TAVILY_API_KEY`.

## API

All endpoints require `Authorization: Bearer <AIOS_API_KEY>`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/agents` | Create agent |
| `GET` | `/v1/agents` | List agents |
| `GET` | `/v1/agents/:id` | Get agent |
| `PUT` | `/v1/agents/:id` | Update agent (creates new version) |
| `DELETE` | `/v1/agents/:id` | Archive agent |
| `GET` | `/v1/agents/:id/versions` | List version history |
| `GET` | `/v1/agents/:id/versions/:n` | Get specific version |
| `POST` | `/v1/environments` | Create environment |
| `GET` | `/v1/environments` | List environments |
| `GET` | `/v1/environments/:id` | Get environment |
| `PUT` | `/v1/environments/:id` | Update environment |
| `DELETE` | `/v1/environments/:id` | Archive environment |
| `POST` | `/v1/sessions` | Create session |
| `GET` | `/v1/sessions` | List sessions |
| `GET` | `/v1/sessions/:id` | Get session (includes usage + vault_ids) |
| `PUT` | `/v1/sessions/:id` | Update session |
| `POST` | `/v1/sessions/:id/messages` | Send user message |
| `POST` | `/v1/sessions/:id/interrupt` | Interrupt session |
| `POST` | `/v1/sessions/:id/archive` | Archive session |
| `DELETE` | `/v1/sessions/:id` | Delete session |
| `POST` | `/v1/sessions/:id/tool-results` | Submit custom tool result |
| `POST` | `/v1/sessions/:id/tool-confirmations` | Confirm/deny always_ask tool |
| `GET` | `/v1/sessions/:id/events` | List events |
| `GET` | `/v1/sessions/:id/stream` | SSE event stream |
| `POST` | `/v1/vaults` | Create vault |
| `GET` | `/v1/vaults` | List vaults |
| `GET` | `/v1/vaults/:id` | Get vault |
| `PUT` | `/v1/vaults/:id` | Update vault |
| `DELETE` | `/v1/vaults/:id` | Archive vault |
| `POST` | `/v1/vaults/:id/credentials` | Create credential |
| `GET` | `/v1/vaults/:id/credentials` | List credentials |
| `PUT` | `/v1/vaults/:id/credentials/:id` | Update credential |
| `DELETE` | `/v1/vaults/:id/credentials/:id` | Delete credential |
| `POST` | `/v1/skills` | Create skill |
| `GET` | `/v1/skills` | List skills |
| `GET` | `/v1/skills/:id` | Get skill |
| `PUT` | `/v1/skills/:id` | Update skill (creates new version) |
| `DELETE` | `/v1/skills/:id` | Archive skill |
| `GET` | `/v1/skills/:id/versions` | List skill versions |
| `GET` | `/v1/skills/:id/versions/:n` | Get skill version |

## Divergences from Anthropic Managed Agents

aios shares the core architecture from [the Managed Agents blog post](https://www.anthropic.com/engineering/managed-agents) (session = append-only log, harness = stateless loop, sandbox = cattle-not-pets containers) but diverges in several ways to support **long-lived assistant entities** rather than task-scoped sessions:

- **Mutable sessions.** Sessions can be updated after creation (`PUT /v1/sessions/:id`) to change the agent binding, version, title, metadata, or vault bindings. Anthropic sessions are immutable after creation.
- **Auto-updating sessions.** By default (`agent_version: null`), sessions always use the latest agent config — updating an agent immediately affects all unpinned sessions on the next step. Anthropic always pins at creation time.
- **Model-agnostic.** The `model` field is any LiteLLM URL, not limited to Claude. Tested with Ollama (local), OpenRouter, Moonshot, and Anthropic models.
- **OpenAI wire format.** Events in the session log and streaming use OpenAI chat-completions message format (roles: system/user/assistant/tool, `tool_calls` array), not Anthropic's Messages API format. LiteLLM translates at the provider boundary.

## License

MIT
