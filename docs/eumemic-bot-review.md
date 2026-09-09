# eumemic-bot PR reviews

On each non-draft pull request, [the workflow](../.github/workflows/eumemic-bot-review.yml) checks out the PR head and runs a coding agent directly on the GitHub Actions runner. It does not create an aios `dev-review` session.

The run is split into two launcher invocations with the token mint between them:

1. `eumemic_bot_review.py agent` runs the harness and writes the agent's final `### Code review` artifact to `REVIEW_ARTIFACT_PATH`. No installation token exists yet.
2. `actions/create-github-app-token` mints one, but only if the agent step succeeded.
3. `eumemic_bot_review.py publish` — a different process, which never launches a harness — POSTs that artifact and verifies GitHub returned the run-specific `<!-- eumemic-bot-review:<sha> -->` marker.

The default model is `gpt-5.6-sol`. Set the repository variable `EUMEMIC_BOT_REVIEW_MODEL` to select a model; routing is by prefix:

| Model | Harness | Endpoint |
|---|---|---|
| `gpt-*` | Codex CLI (Responses API) | `https://oai-proxy.eumemic.ai/v1` |
| `claude-*` | Claude Code | `https://ant-proxy.eumemic.ai` |
| `grok-*` | Pi | `https://xai-proxy.eumemic.ai/v1` |

For example, set the variable to an available `claude-*` model to use Claude Code, or `grok-4.6` to use Pi. The workflow installs only the harness for the routed prefix, so switching model changes what `npm install --global` fetches; only the secret for the selected family must contain a usable key.

## Required repository configuration

| Kind | Name | Purpose |
|---|---|---|
| Variable | `EUMEMIC_BOT_APP_ID` | GitHub App ID (`4752589`) |
| Variable | `EUMEMIC_BOT_REVIEW_MODEL` | Optional; defaults to `gpt-5.6-sol` |
| Secret | `EUMEMIC_BOT_PRIVATE_KEY` | PEM for the GitHub App |
| Secret | `OAI_PROXY_API_KEY` | oai-proxy client key for `gpt-*` reviews |
| Secret | `ANT_PROXY_API_KEY` | ant-proxy client key for `claude-*` reviews |
| Secret | `XAI_PROXY_API_KEY` | xai-proxy client key for `grok-*` reviews |

The launcher also accepts the conventional `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `XAI_API_KEY` names when run manually.

## How each harness reaches its proxy

- **Codex** is pointed at oai-proxy with an explicit provider (`-c model_provider=…` plus a `model_providers.…` table with `wire_api="responses"`), *not* `OPENAI_BASE_URL`. Codex's built-in `openai` provider pins `api.openai.com` and its own auth and ignores that variable, so an env-var-only setup silently 401s against the real OpenAI instead of using the proxy. It runs with `--sandbox danger-full-access`: GitHub-hosted runners reject the bubblewrap loopback setup used by Codex's `read-only` sandbox, while this trusted publisher job runs on an ephemeral runner that holds no installation token while the agent runs (see below).
- **Claude Code** honours `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` directly.
- **Pi** gets a generated `models.json` in a throwaway `PI_CODING_AGENT_DIR` declaring an `xai-proxy` provider, selected with `--provider xai-proxy`.

## Why the token is minted after the agent, not before

The agent reads PR-authored files (including `AGENTS.md` / `CLAUDE.md`), runs shell commands, and on the `gpt-*` route runs unsandboxed. It must not be able to obtain a credential that can write.

Stripping the child environment (`_STRIPPED_ENV`) does not achieve that by itself. `unsetenv` does not rewrite `/proc/<pid>/environ`, so an agent running as the same OS user can read every variable the launcher was started with off `/proc/$PPID/environ` no matter how the child env is scrubbed. A secret is only withheld from the agent if the *step* never receives it. Hence:

- The installation token is minted **after** the agent step has completed, so `GH_TOKEN` is never in the agent phase's process tree. The agent phase additionally refuses to start if `GH_TOKEN` is set, so the ordering cannot silently regress.
- The agent step is given **only the routed family's** proxy secret, selected by the `family` output of the harness-install step; the other two arrive as empty strings.
- Credentials in files are out of reach of both: `actions/checkout` persists the workflow token as an `http.*.extraheader` in `.git/config`, so the launcher unsets that header once it has finished pinning the checkout and before it starts the agent.

Residual, accepted: `--sandbox danger-full-access` (needed because GitHub-hosted runners reject Codex's bubblewrap loopback setup) leaves an agent that chose to be hostile able to tamper with the runner it shares — including the checkout that the publish step later executes. Closing that requires running the agent under a separate identity or container, not a different token order. The job is ephemeral, `pull_request` from forks receives no secrets, and `permissions: contents: read` bounds the workflow token.

## Scope and failure behaviour

Checkout uses `pull_request.head.sha` with full history. The launcher refuses to run if local `HEAD` does not match `HEAD_SHA`, and requires `BASE_SHA` (`pull_request.base.sha`) to be present locally — fetching it once if it is not — because the prompt hands the agent an explicit `git diff <base>...<head>` range rather than letting it guess the base branch.

Reviews instruct the agent not to modify the checkout and ask for focused verification only, not repository-wide suites. Review execution has a 15-minute budget inside a 20-minute job; the launcher's own timeout fires first so its `FATAL` (not a runner kill) ends the step. Failures remain `continue-on-error`; when the harness install, the agent, the token mint, or the publish step fails, the job summary records that no comment was posted.

The existing `infra/agents/dev-review.json` remains available to other callers but is not part of this Action path. Jarbot still uses the old session path until it is ported separately.
