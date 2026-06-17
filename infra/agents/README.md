# `infra/agents/` — versioned agent manifests (IaC-sync v1 for agents)

Live aios agents are otherwise 100% imperative and unversioned: a change to one
is a hand-edit no one can diff against intent. This directory brings the
dev-pipeline's opus agents under git as **thin, secret-free manifests** that are
the declarative **source of truth**; the live agent objects are a reconciled
**projection**. `scripts/reconcile_agents.py` reconciles each manifest **by name**
against live `/v1/agents`, and `.github/workflows/reconcile-agents.yml` runs it
on a `master` push touching `infra/agents/**` (deploy-on-merge).

This mirrors the workflow reconciler (`scripts/reregister_workflows.py` /
`.github/workflows/reregister-workflows.yml`, IaC-sync v1 for workflow objects,
issues #1179 / #1226). The shared `--check` read-only diff mode on both scripts is
the seed of the v2 in-aios ops-agent drift loop.

## Manifest shape

One JSON file per managed agent (filename = agent name, for legibility). The body
is **exactly the shape `AgentCreate` accepts** (`src/aios/models/agents.py`,
`extra="forbid"`) **minus all secrets**. Available fields:

```
name, model, system, tools, skills, mcp_servers, http_servers,
description, metadata, litellm_extra, window_min, window_max
```

Example (`dev-implement.json`):

```json
{
  "name": "dev-implement",
  "model": "anthropic/claude-opus-4-8",
  "system": "<full system prompt text>",
  "tools": [{"type": "bash"}, {"type": "read"}, {"type": "edit"}],
  "http_servers": [],
  "litellm_extra": {},
  "window_min": 50000,
  "window_max": 150000
}
```

### Secrets are by-name reference ONLY — never committed

Manifests carry **config**, never key material. Credentials are referenced by
their vault / connection **name** (the existing aios indirection) inside
`http_servers` / `mcp_servers` config — the literal key lives in the vault and is
resolved at runtime by the worker. `tests/unit/test_reconcile_agents.py` asserts
no committed manifest contains a key-shaped literal (`sk-…`, `ghp_…`, bearer
tokens) and no top-level `api_key` / `token` / `secret` key. A manifest that leaks
a secret fails CI, not prod.

## CRITICAL: reconcile-by-name vs reference-by-id (read before editing)

The dev-pipeline **workflow** references each agent by its server-minted **id**
(`agent_<ULID>`, from `make_id(AGENT)` — NOT caller-settable, NOT the name). This
reconciler reconciles **by name** (the human-stable handle in the manifest). The
two compose correctly **only on the UPDATE path**:

- **UPDATE (the 99% case):** the manifest name matches an existing live agent →
  read its `id` + `version` → `PUT /v1/agents/{id}` in place. The id is unchanged,
  so the workflow header `IMPLEMENT_AGENT_ID=agent_…` stays valid. Idempotent,
  version-bumped. This is what deploy-on-merge needs and it works.
- **CREATE (bootstrap / disaster-recovery only):** no live agent by that name →
  `POST /v1/agents` mints a **NEW** `agent_<ULID>`. The dev-pipeline workflow
  header still points at the **OLD** id, so the freshly-created agent is live but
  the pipeline **cannot reach it** until the workflow is ALSO re-registered with
  the new id. The reconciler emits a loud warning on any create:

  > `WARNING: created agent <name> id=<new-id>; the referencing workflow header still points at the prior id and must be re-registered to rewire`

  **Rewiring after a create is a manual follow-on** (re-register the dev-pipeline
  workflow with the new agent id). Because the 5 managed agents already exist in
  prod, you should never hit the create path in normal operation. A future
  enhancement may let a manifest carry the referencing workflow target so a create
  triggers a paired re-register — **not built here.**

## Seeding the manifests (one-time bootstrap by the implementer)

Manifests are **seeded from the live config** of the managed agents — a one-time
bootstrap done by hand, **not committed as code**:

1. List the live agent and grab its id:
   `GET /v1/agents?name=<name>` → take the single `id`.
2. Read its full config: `GET /v1/agents/{id}`.
3. Strip the server-only fields — `id`, `version`, `created_at`, `updated_at`,
   `archived_at` — keep the rest, and **remove any secret material** (reduce
   credentialed `http_servers` / `mcp_servers` to by-name references only).
4. Save as `infra/agents/<name>.json`.

`scripts/reconcile_agents.py --check` verifies a seeded manifest is byte-faithful
to live (`in-sync`) before you ever merge it — **always run `--check` against live
first** so the first reconcile is a proven no-op, never a surprise overwrite.

> **Managed agents:** `dev-implement`, `dev-review`, `dev-fix`, `dev-ci-watch`,
> `dev-risk` (the dev-pipeline's 5 opus judgment nodes; names are unique in prod).
> `dev-implement.json` is committed; the remaining four are seeded by an operator
> with live API access using the recipe above (this PR was authored in an
> environment without egress to prod). Run `reconcile_agents.py --check` to confir
> each seeded manifest is `in-sync` before the first non-`--check` reconcile.

## Out of scope (explicit follow-on issues)

- Trigger manifests (per-session — needs the attach-trigger primitive #1216/#1280).
- Vault / environment manifests.
- The continuous in-aios reconcile *loop* (meta-Ralph, v2): wraps
  `reregister_workflows.py --check` + `reconcile_agents.py --check` on a cron
  trigger that files an issue on drift. Gated on the trigger-action primitives.
