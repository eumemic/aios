# DEPLOY.md — aios on Coolify + Hetzner HIL

Day-one deploy plan for **aios** as the second service in the Coolify
constellation, alongside `ant-proxy`. Coolify control plane already lives
on Server A (`coolify.eumemic.ai`); aios goes on its own VM, registered
as Server B and managed remotely by the existing Coolify dashboard.

## Architecture

**One role per VM.** Server A keeps Coolify + ant-proxy alone; aios
gets its own host so a runaway sandbox can't OOM the management plane
or the proxy.

```
┌──────────────────────────────────────────────────────────────┐
│ Server A — CCX13 @ Hetzner HIL  (already deployed)           │
│ Role: Coolify control plane + ant-proxy                       │
│ Hostname: coolify.eumemic.ai                                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ Coolify SSH agent
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Server B — CCX13 @ Hetzner HIL                                │
│ Role: aios (api + worker + Postgres + sandbox host)           │
│ Hostname: aios.eumemic.ai                                     │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐   │
│  │ aios-api        │  │ aios-worker     │  │ Postgres     │   │
│  │ (Coolify App)   │  │ (Coolify App)   │  │ (Coolify DB) │   │
│  │ :8080           │  │ /var/run/docker │  │ :5432 inner  │   │
│  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘   │
│           │                     │                  │           │
│           └────────── eumemic-aios network ────────┘           │
│                                 │                              │
│           ┌─────────────────────┴────────────────┐             │
│           │  spawned siblings on host docker     │             │
│           │  (one per active session, label=     │             │
│           │   aios.managed=true)                 │             │
│           └──────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

**Coolify shape:**

- **Postgres** lives in a Coolify **Database** resource (managed: backups,
  restart, env injection — no `docker-compose.yml` to maintain).
- **api** and **worker** are two separate Coolify **Applications** built
  from the same git repo and the same `Dockerfile`, with different
  `--target` values (`api` vs `worker`). Same image cache, same deploy
  trigger, independent log streams + scale.
- All three live in one Coolify **Project** (`eumemic-aios`) on Server B,
  so they auto-share an internal Docker network.

**Why two Applications instead of one Compose stack:**

- Coolify-managed Postgres includes a backup scheduler and managed-DB
  ergonomics; rolling our own in compose would re-implement that.
- API and worker have different deploy semantics: api is stateless and
  deploys with zero downtime via healthcheck swap; worker holds in-flight
  sandbox state and takes a brief hit on rollover. Independent
  Applications make this explicit instead of bundled.
- The repo doesn't yet have a `docker-compose.yml`; introducing one
  would couple deploy topology to the repo, which is the wrong direction
  if api and worker eventually split into separate repos (the plan).

**Sandbox model:** the worker mounts `/var/run/docker.sock` from the host
and spawns sibling sandbox containers via the host's docker daemon
("Docker-outside-of-Docker"). This is a single-tenant, root-equivalent
pattern — fine for our setup, not safe for multi-tenant SaaS. The
sandbox base image (`aios-sandbox:latest`) lives on the host's docker
image store and is built once at deploy time.

**Workspace persistence:** sessions write to
`/srv/aios/workspaces/<session_id>/`. The worker container bind-mounts
this same path at the same location (`/srv/aios/workspaces:/srv/aios/workspaces`)
so the path the worker passes to `docker run -v` matches the host's
view of the filesystem — no path translation logic needed inside the
worker. Workspaces survive sandbox lifetimes; cleanup of stale dirs is
deferred.

Your laptop → `https://aios.eumemic.ai` → Server B → aios-api container
→ Postgres (Coolify-managed) + aios-worker (which spawns sandboxes on
the host docker daemon → workspaces in `/srv/aios/workspaces/`).

**What this deploy covers:** Server B provisioned + hardened + Docker
installed; Coolify registers it as a remote Server; Postgres + api +
worker stand up; sandbox base image is built and present; alembic
migrations run on every deploy; smoke-tested via `aios chat` from the
laptop. HTTPS via Coolify's Traefik + Let's Encrypt. Auto-deploy on
push to `main`. Nightly Postgres dumps + workspace tarballs to off-box
storage.

**Deferred:** Tailscale to home-5090 for inference (Phase 2 LiteLLM
work), Grafana Cloud observability (separate doc), Jarvis 2.0 migration
(post-AIOS-stability).

**Rough time to execute:** ~90 min the first time, given Server A
patterns are already proven.

---

## Prerequisites

- **Hetzner Cloud account** with a payment method.
- **Cloudflare account** with `eumemic.ai` already configured (used by
  ant-proxy).
- **GitHub access** to `github.com/eumemic/aios`.
- **Coolify control plane already deployed** on Server A
  (`coolify.eumemic.ai`) — see `~/code/ant-proxy/DEPLOY.md` if not.
- **Local tooling**: `ssh`, `scp`, `~/code/aios/.venv/` with the `aios`
  CLI on PATH (`uv sync --dev` from the repo).
- **The `Dockerfile` at the aios repo root** (added in this PR — the api/
  worker images don't exist on `master` before it).
- **1Password (or equivalent)** for storing secrets.

---

## Step 1 — Provision the Hetzner VM

### 1a. Create the server

Hetzner Cloud Console → **Servers → Add Server**:

- **Location**: `Hillsboro, OR (hil)` — same as Server A.
- **Image**: `Ubuntu 24.04`.
- **Type**: Shared vCPU → `CCX13` (2 dedicated vCPU, 8 GB RAM, 80 GB
  disk, 20 TB traffic, ~€14/mo). Tight for ~5 concurrent sandbox
  containers but cheap; plan to upgrade to **CCX23** (4 vCPU / 16 GB,
  ~€28/mo) when you see the worker host pressure on `free -h`, OOM
  events in `dmesg`, or postgres `shared_buffers` thrashing. Hetzner
  upgrades are in-place (single reboot, ~30 s downtime).
- **Networking**: IPv4 + IPv6 (default).
- **SSH Keys**: select `tom-laptop` (already uploaded for Server A).
- **Backups**: enable (+20% ≈ €2.80/mo, 7 rolling daily whole-disk
  backups). Whole-disk is the last-resort restore primitive; primary DR
  is the postgres + workspace backups in step 13.
- **Name**: `aios-hil-01`.

Click **Create & Buy now**. Record the IPv4 address.

### 1b. Update DNS

Cloudflare → `eumemic.ai` → **DNS → Records**:

| Type | Name | Content | Proxy |
|---|---|---|---|
| A | `aios` | `<server B IPv4>` | DNS only (grey) |

Grey-cloud for the same Let's Encrypt HTTP-01 reason as ant-proxy — see
the comparable note in `ant-proxy/DEPLOY.md`. Verify:

```bash
dig +short aios.eumemic.ai
# → Server B's IP
```

---

## Step 2 — Initial server hardening

SSH in as root (DNS may take a moment; you can also use the IP):

```bash
ssh root@aios.eumemic.ai
```

This is the same hardening sequence as Server A — copy/pasteable:

```bash
# 1. Update packages
apt-get update && apt-get upgrade -y

# 2. Create a non-root user; keep sudo password-protected.
adduser --disabled-password --gecos "" tom
usermod -aG sudo tom
mkdir -p /home/tom/.ssh
cp /root/.ssh/authorized_keys /home/tom/.ssh/
chown -R tom:tom /home/tom/.ssh
chmod 700 /home/tom/.ssh && chmod 600 /home/tom/.ssh/authorized_keys
passwd tom                       # strong password → 1Password

# 3. Disable root SSH and password auth.
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh

# 4. UFW — public ports only. Coolify reaches us over SSH for deploys,
#    not over an open port.
apt-get install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# 5. fail2ban for SSH brute-force.
apt-get install -y fail2ban
systemctl enable --now fail2ban

# 6. Docker (Coolify needs it on remote Servers; install up front).
curl -fsSL https://get.docker.com | sh
usermod -aG docker tom
```

Log out, back in as `tom`:

```bash
exit
ssh tom@aios.eumemic.ai
docker ps    # works without sudo
```

---

## Step 3 — Prepare host directories

aios needs two host paths that pre-exist before the first deploy:

```bash
# Workspace root — bind-mounted into the worker at the same path so
# the worker doesn't have to translate sandbox `-v` paths.
sudo mkdir -p /srv/aios/workspaces
sudo chmod 700 /srv/aios/workspaces

# Postgres data dir is managed by Coolify's Database resource; nothing
# to prep here.

# Off-box backup staging.
sudo mkdir -p /var/backups/aios
sudo chown tom:tom /var/backups/aios
```

The workspace dir's ownership doesn't need an explicit `chown`: the
worker runs as root inside its container, and the sandbox containers
it spawns also run as root (the `aios-sandbox:latest` image inherits
the default `python:3.13-slim` root user; tightening that is tracked
separately). Both writers are uid 0, so workspace contents end up
root-owned regardless of how the directory starts. The `chmod 700`
keeps the dir from being world-readable while still letting root
descend into it. Note the override of `AIOS_WORKSPACE_ROOT` from the
in-tree default (`/var/lib/aios/workspaces` per `src/aios/config.py`)
to `/srv/aios/workspaces` is deliberate — `/srv/` is the conventional
location for service data on a deployment host.

---

## Step 4 — Build the sandbox base image on the host

The worker assumes `aios-sandbox:latest` is present in the host's
docker image cache; without it, the first session crashes on `docker
run`. Build it now.

```bash
# On Server B, as `tom`:
mkdir -p ~/build && cd ~/build
git clone --depth=1 https://github.com/eumemic/aios.git
cd aios/docker
docker build -t aios-sandbox:latest -f Dockerfile.sandbox .

# Verify
docker images | grep aios-sandbox
# aios-sandbox   latest   <hash>   ~180 MB
```

You can delete `~/build/aios` after — the image stays in the docker
cache. Rebuilds happen whenever `docker/Dockerfile.sandbox` changes
(realistically: ~monthly on this codebase, e.g. when iptables or
package-manager support gets added). The "What's next" section
mentions an eventual move to CI + GHCR so this stops being a
manual-on-host step; until then, when the upstream Dockerfile.sandbox
changes, re-run this step on Server B before the next deploy.

---

## Step 5 — Register Server B in Coolify

In your browser at `https://coolify.eumemic.ai` (Server A's control
plane), navigate to **Servers → Add Server**:

| Field | Value |
|---|---|
| Name | `aios-hil-01` |
| IP | `<Server B IPv4>` |
| User | `tom` |
| Port | `22` |
| Private Key | Coolify's existing host key (used for Server A) |

Coolify validates SSH connectivity, then installs its remote agent on
Server B (~30 s). When the dashboard shows the new Server as **Validated
+ Reachable**, move on.

Note: this registers Server B without re-installing Coolify on it —
Server A's Coolify is the only control plane. If Coolify ever asks to
"install Coolify on this server," the answer is **no** — that would
clone the management plane and break the one-control-plane model.

---

## Step 6 — Generate production secrets

On your laptop:

```bash
cd ~/code/aios && source .venv/bin/activate

API_KEY=$(openssl rand -hex 32)
VAULT_KEY=$(openssl rand -base64 32)

echo "AIOS_API_KEY=$API_KEY"
echo "AIOS_VAULT_KEY=$VAULT_KEY"
```

**Store both in 1Password under `aios.eumemic.ai`** before you paste
them into Coolify. The vault key encrypts credentials at rest; if it
rotates after data is encrypted, the data is unrecoverable. Treat it
like a database root password.

You'll also want at least one client API key (for the laptop):

```bash
LAPTOP_KEY=$(openssl rand -hex 32)
echo "LAPTOP_KEY=$LAPTOP_KEY"
# Store in 1Password as `aios.eumemic.ai laptop client key`.
```

The `AIOS_API_KEY` is the *server's* shared secret — clients send it as
`Authorization: Bearer <key>`. It's a single-tenant model in v1 (one
key, one client per deploy is the current shape; multi-key client-table
support is on the roadmap, parallel to ant-proxy's `clients` table).
For now, use the same value for `AIOS_API_KEY` and the client key, or
share `LAPTOP_KEY` as both — they're identical in v1.

---

## Step 7 — Create the Coolify project + Postgres database

Coolify → **Projects → New Project** → name it `eumemic-aios`. Set
**Default Server** to `aios-hil-01`.

Inside the project: **New Resource → Database → PostgreSQL 16**.

| Field | Value |
|---|---|
| Name | `aios-postgres` |
| Image version | `postgres:16` |
| Database name | `aios` |
| Username | `aios` |
| Password | `<generated by Coolify; copy to 1Password>` |
| Public | **off** (internal only — workers reach it on the project network) |
| Backup schedule | `0 3 * * *` (daily 3am UTC) |
| Backup retention | 14 days |
| Backup destination | `s3` (configure in step 13) or local for now |

Click **Save**, then **Start**. Coolify provisions a Postgres container
on Server B and exposes a `DATABASE_URL` env var to other resources in
the same project. Verify via the Coolify "Logs" tab that Postgres is
ready.

Coolify's DSN env var name is `DATABASE_URL`; aios reads
`AIOS_DB_URL`. Both Applications (api + worker) will set
`AIOS_DB_URL=${DATABASE_URL}` in step 8/9 — Coolify substitutes the
value at runtime.

---

## Step 8 — Connect GitHub source

If the Coolify GitHub App is already installed on the `eumemic` org
(from ant-proxy setup), it sees `aios`. Otherwise: Coolify → **Sources →
GitHub Apps → Install/Update** → grant `aios` repo access.

---

## Step 9 — Create the api Application

Inside the `eumemic-aios` project: **New Resource → Application →
Private Repository (GitHub App)** → pick `eumemic/aios`.

| Field | Value |
|---|---|
| Name | `aios-api` |
| Branch | `main` |
| Build Pack | `Dockerfile` |
| Dockerfile Location | `/Dockerfile` |
| Base Directory | `/` |
| Build Target Stage | `api` |
| Ports Exposed | `8080` |
| Domains | `https://aios.eumemic.ai` |

**Environment variables** (app → **Environment Variables**):

```
AIOS_API_KEY=<from step 6>
AIOS_VAULT_KEY=<from step 6>
AIOS_DB_URL=${DATABASE_URL}
AIOS_API_HOST=0.0.0.0
AIOS_API_PORT=8080
AIOS_LOG_LEVEL=INFO
AIOS_WORKSPACE_ROOT=/srv/aios/workspaces
AIOS_DOCKER_IMAGE=aios-sandbox:latest
AIOS_SANDBOX_BACKEND=docker
```

Mark `AIOS_API_KEY` and `AIOS_VAULT_KEY` as **Is Secret**.

**Pre-Deploy Command** (app → **Settings → Pre-Deploy Command**):

```
aios migrate
```

Runs inside the new image before traffic swaps. `aios migrate` does
three things in `src/aios/cli/commands/ops.py`: alembic
`upgrade head` (idempotent via the version table), one-shot install of
the procrastinate job-queue schema if missing (gated by a
`to_regclass(procrastinate_jobs)` check), and a `DROP TRIGGER IF
EXISTS … CREATE TRIGGER …` for a procrastinate lock-release helper.
The trigger DDL re-runs every deploy but is safe (drop + recreate is
atomic in Postgres). Net behavior on a healthy second deploy:
fast no-op, no failures. The api is the only Application that runs
this — the worker doesn't need to migrate since it shares the DB.

**No persistent storage on the api Application.** It's stateless.

Click **Save**. **Don't deploy yet** — the worker needs to be configured
first so a startup race doesn't matter (the api's first health probe
won't succeed until the migration applies, which is fine, but better to
have both Applications ready before either deploys).

---

## Step 10 — Create the worker Application

Same project → **New Resource → Application → Private Repository
(GitHub App)** → pick `eumemic/aios`.

| Field | Value |
|---|---|
| Name | `aios-worker` |
| Branch | `main` |
| Build Pack | `Dockerfile` |
| Dockerfile Location | `/Dockerfile` |
| Base Directory | `/` |
| Build Target Stage | `worker` |
| Ports Exposed | *(none — worker is internal)* |
| Domains | *(none)* |

**Environment variables** — same set as api **except `AIOS_API_HOST`
and `AIOS_API_PORT` are unused** (worker has no HTTP listener):

```
AIOS_API_KEY=<from step 6>
AIOS_VAULT_KEY=<from step 6>
AIOS_DB_URL=${DATABASE_URL}
AIOS_LOG_LEVEL=INFO
AIOS_WORKSPACE_ROOT=/srv/aios/workspaces
AIOS_DOCKER_IMAGE=aios-sandbox:latest
AIOS_SANDBOX_BACKEND=docker
AIOS_WORKER_CONCURRENCY=4
```

Mark `AIOS_API_KEY` and `AIOS_VAULT_KEY` as **Is Secret**. Use the same
values as api — they share the vault.

**Persistent storage / mounts** (app → **Storages**):

| Type | Source (host) | Destination (container) |
|---|---|---|
| Bind | `/srv/aios/workspaces` | `/srv/aios/workspaces` |
| Bind | `/var/run/docker.sock` | `/var/run/docker.sock` |

Both must be **Bind Mounts**, not named volumes. Coolify lets you set
"Type: Bind Mount" in the Storages UI; double-check this — a named
volume here would silently break sandbox spawning because the worker
would tell the host daemon a path the host doesn't recognize.

**No Pre-Deploy Command** on the worker (api owns migrations).

**Deploy Order**: in the Project's **Settings → Deploy Order**, set
api before worker. Coolify deploys both in parallel by default; api-
first ensures the migration runs before the worker starts opening
sessions.

---

## Step 11 — First deploy

Coolify → `aios-api` → **Deploy**. Watch the Deployments tab. The
build streams `uv sync` then `aios migrate` then the health probe goes
green on `/health`.

While that runs, go to `aios-worker` → **Deploy**. Worker boots,
acquires the procrastinate advisory lock, tails the job queue, idles.

Verify on the laptop:

```bash
curl -s https://aios.eumemic.ai/health | jq
# {"ok": true, "version": "...", ...}
```

If `/health` doesn't return 200: Coolify → app → **Logs** for the api
container. Most first-deploy failures trace to:

- `AIOS_DB_URL` not resolving — the `${DATABASE_URL}` substitution
  needs the Postgres Database resource in the same project. Confirm
  it's healthy in the Coolify dashboard.
- Migration failure — usually a permission issue on the Coolify-
  managed Postgres user. Coolify creates the role with full DB perms
  by default; if you customized the role, check it has `CREATE` on the
  `public` schema.
- `AIOS_VAULT_KEY` malformed — must be base64 of exactly 32 bytes.

---

## Step 12 — Smoke test from the laptop

On the laptop:

```bash
export AIOS_URL=https://aios.eumemic.ai
export AIOS_API_KEY=<LAPTOP_KEY from step 6>

# Reachability
uv run aios status
# → server reachable, version=..., backend=docker

# Drive a real session through the bash tool. We need an environment
# (sandbox config) AND an agent; the chat command refuses to start a
# new session without --environment-id (chat.py:_resolve_session_id).

ENV_ID=$(uv run aios envs create --data '{"name": "smoke-env"}' \
    | jq -r '.id')

AGENT_ID=$(uv run aios agents create --file - <<'JSON' | jq -r '.id'
{
  "name": "smoke-bash",
  "description": "smoke test agent",
  "system": "You are a shell agent. Use the bash tool.",
  "tools": [{"type": "bash"}],
  "model": "anthropic/claude-haiku-4-5"
}
JSON
)

uv run aios chat --agent "$AGENT_ID" --environment-id "$ENV_ID" \
    -m "echo 'hello from server B' && hostname && pwd"
```

You should see a live SSE stream with the model's reply, a tool_call
to bash, and the bash tool's output (`hello from server B / aios-worker-...
/ /workspace`).

On Server B, verify a sandbox container was actually spawned. The
orphan-reaper filters on both `aios.managed=true` AND
`aios.instance_id=<settings.instance_id>` (default `default`); if you
ever override `AIOS_INSTANCE_ID` per-deployment, swap that label below
to match.

```bash
ssh tom@aios.eumemic.ai
docker ps \
    --filter label=aios.managed=true \
    --filter label=aios.instance_id=default
# → one running container, image aios-sandbox:latest, ~30-second age
ls /srv/aios/workspaces/
# → one session_id directory matching the chat session
```

After ~5 minutes of idle (the default
`AIOS_CONTAINER_IDLE_TIMEOUT_SECONDS=300`), the sandbox auto-releases:

```bash
docker ps --filter label=aios.managed=true \
          --filter label=aios.instance_id=default   # → empty
ls /srv/aios/workspaces/                            # → still there
```

Workspace files persist across container lifetimes. The next message in
the same session re-provisions a fresh sandbox bind-mounting the same
workspace dir.

---

## Step 13 — Backups

### 13a. Postgres dumps

Coolify's Database resource has a built-in backup scheduler. Verify
it's running:

Coolify → `aios-postgres` → **Backups**. Schedule from step 7 should be
`0 3 * * *` daily, 14 days retention. First backup completes the next
night; you can trigger one manually now to confirm the path works.

### 13b. Workspace tarballs

Coolify doesn't back up bind-mounted host paths. Roll our own:

```bash
# On Server B, as tom:
crontab -e
# Add:
30 3 * * * tar czf /var/backups/aios/workspaces-$(date +\%Y\%m\%d).tar.gz -C /srv/aios workspaces 2>>/home/tom/aios-backup.log
0 4 * * 0 find /var/backups/aios/ -name 'workspaces-*.tar.gz' -mtime +14 -delete 2>/dev/null
```

Daily tarballs at 3:30 (after the postgres dump finishes), prune to 14
days. For a typical aios workload (≤10 active sessions, each with a
workspace under ~100 MB), tarballs are <1 GB and the prune keeps disk
predictable.

### 13c. Off-box sync (optional but strongly recommended)

Whole-disk Hetzner backups (step 1a) handle catastrophic loss; for
per-data DR you want both postgres dumps and workspace tarballs
mirrored off-box. Same `rclone` pattern as ant-proxy:

```bash
sudo apt-get install -y rclone
rclone config              # configure b2 / hetzner storage box / etc.

crontab -e
# Add:
0 5 * * 0 rclone sync /var/backups/aios b2:aios-backups/weekly 2>>/home/tom/rclone.log
```

Coolify's postgres backup destination supports S3-compatible directly —
configure that in `aios-postgres` → **Backups → Destination** and skip
the rclone for Postgres.

---

## Step 14 — Verify auto-deploy on `main`

Make a trivial change in `aios` (edit README, bump a comment), commit,
`git push`. Coolify's webhook fires on both `aios-api` and `aios-worker`
within seconds. Watch both Deployments tabs.

Rolling swap on api (healthcheck-gated). Worker takes a brief stop-
start because it owns sandbox state — that's expected. Sandboxes
spawned by the old worker process get reaped on the new worker's
startup via the orphan reaper (see `sandbox/registry.py:reap_orphans`).

If a deploy doesn't fire: Coolify → app → **Settings → Auto Deploy** is
on, and the GitHub App's webhook deliveries page shows 200s from
Coolify.

---

## Rollback / disaster recovery

**Bad deploy**: Coolify → app → **Deployments** → previous green build
→ **Rollback**. Code reverts in <30 s; volume + DB unchanged so no
state loss. Roll back api and worker independently if only one is bad.

**DB corruption**: Coolify → `aios-postgres` → **Backups** → pick a
snapshot → **Restore**. ~30 s for a small DB. Sessions in flight at
the snapshot point are the most you'd lose.

**Workspace loss**: tarball from `/var/backups/aios/` (or the off-box
mirror), `tar xzf` into `/srv/aios/`, restart `aios-worker`.

**VM lost**: provision new CCX13 with same steps 1–4. Rebuild
`aios-sandbox:latest` (step 4 = ~3 min). In Coolify, edit the
`aios-hil-01` Server's IP → it re-validates → re-deploys all three
resources. Restore postgres from the latest backup, untar the latest
workspace archive, update Cloudflare A record.

**Coolify on Server A corrupts**: aios on Server B keeps running on
its own docker daemon — only the management plane is gone. Recover
Coolify per `ant-proxy/DEPLOY.md` rollback section.

---

## Capacity signals — when to upgrade Server B

Watch for any of these on Server B:

- **`free -h`** shows `available` < 1 GB during normal load. Sandbox
  containers hold ~200–500 MB each; postgres `shared_buffers` is at
  Coolify's default ~128 MB; the rest is api + worker + page cache.
  Hitting <1 GB available means you're swapping or about to.
- **`dmesg | grep -i kill`** shows the OOM killer firing on a
  postgres or sandbox PID.
- **`docker stats`** shows the worker steadily climbing past ~1 GB RSS.
- **Average `iostat 5` `%util`** above 60% — postgres is checkpoint-
  bound or workspace I/O is saturating the local NVMe.

Hetzner upgrades are in-place — single reboot, ~30 s downtime.
**CCX13 → CCX23** ($14 → $28/mo) doubles vCPU + RAM, gets you to ~10
concurrent sandboxes comfortably. **CCX23 → CCX33** ($28 → $56/mo)
when you start landing real workloads or run Jarvis 2.0 alongside
(though the plan is for Jarvis on Server C).

---

## What's next

Once aios is running cleanly for a few days:

- **Tailscale to home-5090**: install Tailscale on Server B and the home
  workstation. Configure aios's LiteLLM router to target the home box
  for a fast/cheap local model. The mesh IP just becomes another
  upstream URL — no Coolify or DNS changes.
- **Grafana Cloud**: ship Server B logs (Coolify's container logs are
  JSON if `AIOS_LOG_LEVEL=INFO` and structlog is configured) + host
  metrics via Grafana Agent. Same pattern you'll deploy on Server A
  next.
- **Connectors**: Signal/Telegram subprocesses are off in this deploy
  (no env vars set). When you turn them on, mount their config dirs
  (e.g. `/srv/aios/signal-cli-config`) the same way the workspace dir
  is mounted.
- **Jarvis 2.0**: comes after this is stable. Plan is its own Server C
  (CCX23+), with aios as a dependency it talks to over the eumemic.ai
  LAN.

---

## v2 hardening (not required day one)

- **Cloudflare proxy + DNS-01 cert issuance** — same upgrade as
  `ant-proxy`'s v2 section. Switch `aios.eumemic.ai` to orange-cloud
  after Let's Encrypt has issued via HTTP-01.
- **Cloudflare Access** in front of `aios.eumemic.ai` if you ever
  expose anything beyond the API (e.g. a future admin dashboard). The
  raw API stays bearer-token-only.
- **User-namespace remapping** for the worker container (`--userns
  remap`) so root-inside-the-container is not root-on-host. Real win
  if aios ever runs untrusted code (multi-tenant). Single-tenant: not
  worth the friction.
- **Per-service firewall**: tighten 443 to Cloudflare ranges only after
  enabling the proxy.
- **Sandbox image build in CI + GHCR**: when the sandbox Dockerfile
  starts changing more than monthly, move the build to a GitHub Action
  that pushes to GHCR. Worker pulls on startup. Until then, the
  on-host one-shot from step 4 is fine.
- **Postgres tuning**: the Coolify default is conservative
  (`shared_buffers=128MB`, `max_connections=100`). When workload grows,
  edit via Coolify's Postgres advanced config: `shared_buffers=2GB`,
  `effective_cache_size=6GB` on a CCX23.
