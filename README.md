# aios

An open-source agent runtime: Postgres-backed sessions, Docker sandbox, any LiteLLM-compatible model.

aios is inspired by [Anthropic's Managed Agents](https://www.anthropic.com/engineering/managed-agents) but with a clean-room API and one significant difference: **the model behind an agent is just a URL.** The agent's `model` field is a LiteLLM-compatible string (`anthropic/claude-opus-4-6`, `ollama_chat/llama3.3`, `openai/gpt-5`, `openrouter/...`, anything LiteLLM speaks). The harness POSTs chat-completions requests; what's behind that URL is somebody else's problem.

## Architecture

Three primitives, each independently replaceable:

- **Session** — Postgres-backed append-only event log; the durable truth
- **Harness** — the loop. Owns all I/O. Reads events, prepares the chat-completions context, calls the model, runs tool calls in the sandbox, persists results
- **Sandbox** — a Docker container per session, with a bind-mounted workspace volume

The harness is split into a stateless API server (`aios serve api`) and one or more workers (`aios serve worker`). They communicate through Postgres — same database, same `LISTEN/NOTIFY`, same job queue substrate (procrastinate).

## v1 status

Pre-alpha. The directory structure exists but most of it is empty. See [the implementation plan](../.claude/plans/curious-painting-starlight.md) for the full design and phase breakdown.

## Quickstart (when v1 phase 3 lands)

```bash
# 1. Spin up Postgres + the aios server
docker compose up -d

# 2. Migrate
uv run python -m aios migrate

# 3. Create a credential, environment, agent, and session
export AIOS_API_KEY=...
curl -H "Authorization: Bearer $AIOS_API_KEY" -H "Content-Type: application/json" \
     -X POST http://localhost:8080/v1/credentials \
     -d '{"name":"local-ollama","provider":"ollama","value":"unused"}'

curl -H "Authorization: Bearer $AIOS_API_KEY" -H "Content-Type: application/json" \
     -X POST http://localhost:8080/v1/environments \
     -d '{"name":"default"}'

curl -H "Authorization: Bearer $AIOS_API_KEY" -H "Content-Type: application/json" \
     -X POST http://localhost:8080/v1/agents \
     -d '{
       "name":"hn-fetcher",
       "model":"ollama_chat/llama3.3",
       "system":"You are a helpful assistant with shell and file tools.",
       "tools":[{"type":"bash"},{"type":"read"},{"type":"write"}]
     }'

curl -H "Authorization: Bearer $AIOS_API_KEY" -H "Content-Type: application/json" \
     -X POST http://localhost:8080/v1/sessions \
     -d '{
       "agent_id":"agent_01...",
       "environment_id":"env_01...",
       "initial_message":"Fetch https://news.ycombinator.com and save the top story titles to /workspace/news.json"
     }'

# 4. Tail the SSE stream
curl -N -H "Authorization: Bearer $AIOS_API_KEY" \
     http://localhost:8080/v1/sessions/sess_01.../stream
```

## License

MIT
