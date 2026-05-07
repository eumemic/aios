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
push to `master`. Nightly Postgres dumps + workspace tarballs to off-box
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

Inside the project: **New Resource → Database → PostgreSQL 18** (Coolify
v4 currently defaults to PostgreSQL 18; older versions are still listed
under "Image" if you have a reason to pin lower).

| Field | Value |
|---|---|
| Name | `aios-postgres` |
| Image version | `postgres:18-alpine` |
| Database name | `postgres` (Coolify's default; aios doesn't care about the name) |
| Username | `postgres` |
| Password | `<generated by Coolify; copy to 1Password>` |
| Public | **off** (internal only — workers reach it on the project network) |

Backups are configured in **step 13** rather than at create time —
cleaner to verify the DB itself is healthy first, then layer on the
schedule.

Click **Save**, then **Start**. Coolify provisions a Postgres container
on Server B and exposes a `DATABASE_URL` env var to other resources in
the same project. Verify via the Coolify "Logs" tab that Postgres is
ready.

Coolify's DSN env var name is `DATABASE_URL`; aios reads
`AIOS_DB_URL`. Both Applications (api + worker) will set
`AIOS_DB_URL` in steps 9/10 — but **don't** use the
`${DATABASE_URL}` shortcut here. Coolify's substituted value uses the
`postgres://` scheme, which SQLAlchemy 2.x (used by alembic in
`aios migrate`) rejects with `NoSuchModuleError: Can't load plugin:
sqlalchemy.dialects:postgres`. Substitute the value yourself and
write the literal `postgresql://...` into the env var. The `aios`
runtime accepts both schemes; only alembic's import path is strict.

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
| Branch | `master` |
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
AIOS_DB_URL=postgresql://postgres:<password>@<postgres-uuid>:5432/postgres
AIOS_API_HOST=0.0.0.0
AIOS_API_PORT=8080
AIOS_LOG_LEVEL=INFO
AIOS_WORKSPACE_ROOT=/srv/aios/workspaces
AIOS_DOCKER_IMAGE=aios-sandbox:latest
AIOS_SANDBOX_BACKEND=docker
```

Copy the `postgresql://` DSN from `aios-postgres` → **Internal URL**
in the Coolify dashboard, but **swap the scheme to `postgresql://`**
if Coolify gave you `postgres://` (see step 7's note on the SQLAlchemy
2.x scheme requirement). Mark `AIOS_API_KEY` and `AIOS_VAULT_KEY` as
**Is Secret**.

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
| Branch | `master` |
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
AIOS_DB_URL=postgresql://postgres:<password>@<postgres-uuid>:5432/postgres
AIOS_LOG_LEVEL=INFO
AIOS_WORKSPACE_ROOT=/srv/aios/workspaces
AIOS_DOCKER_IMAGE=aios-sandbox:latest
AIOS_SANDBOX_BACKEND=docker
AIOS_WORKER_CONCURRENCY=4
ANTHROPIC_API_KEY=<from 1Password — required for the LLM call>
ANTHROPIC_BASE_URL=https://ant-proxy.eumemic.ai
```

The `postgresql://` scheme requirement is the same as api (see step 9).
Worker also needs the `ANTHROPIC_*` pair — api never calls the LLM,
worker does — and routes through ant-proxy so usage tracking + cost
attribution stay centralized.

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

**Stop-then-start deploy strategy** (app → **Settings → Consistent
Container Name**, toggle on): the worker holds a Postgres advisory
lock at startup (`pg_try_advisory_lock` in
`harness/worker.py:_acquire_worker_lock`) to enforce single-instance
ownership of connector subprocesses. Coolify's default rolling deploy
starts the new container before stopping the old one, so the new
worker tries to acquire the lock the old one still holds and exits
with `worker.duplicate_instance_refused`. With **Consistent Container
Name** on, both containers share the same name (`<uuid>` instead of
`<uuid>-<deployment_id>`), and Docker refuses to start the second
until the first is removed — forcing a stop-then-start with ~10 s
downtime per deploy. Acceptable: procrastinate jobs queue in Postgres,
so no work is lost.

If Coolify's UI doesn't expose this toggle, set it via DB-direct
because the API allow-list rejects the field:

```
docker exec coolify-db psql -U coolify -c "
UPDATE application_settings s SET is_consistent_container_name_enabled = true
FROM applications a
WHERE s.application_id = a.id AND a.name = 'aios-worker';"
```

The first deploy after flipping this flag is still vulnerable —
the *currently-running* container has the old `<uuid>-<deployment_id>`
name, so Docker doesn't see a name conflict during the transition.
Manual workaround for the transitional deploy:
`docker rm -f <uuid>-<old-deployment-id>` once the new container is
crashlooping. From the second post-flag deploy onward, the consistent
name does the work automatically.

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

- `AIOS_DB_URL` scheme is `postgres://` — alembic's SQLAlchemy import
  rejects this. Use `postgresql://` (see step 7).
- Migration failure — usually a permission issue on the Coolify-
  managed Postgres user. Coolify creates the role with full DB perms
  by default; if you customized the role, check it has `CREATE` on the
  `public` schema.
- `AIOS_VAULT_KEY` malformed — must be base64 of exactly 32 bytes.

**Verify the schema actually applied**: `/health` doesn't touch the
DB, so the api can come up green even if `aios migrate` failed.
Check explicitly:

```bash
ssh tom@aios.eumemic.ai
docker exec <postgres-container> psql -U postgres -c "\dt" | head -5
# Expect ~26 tables: agents, sessions, events, procrastinate_*, ...
```

If the table list is empty or just shows alembic's version table,
the Pre-Deploy didn't run. The most common cause is Coolify's
Pre-Deploy mechanism: it `docker exec`s into the **already-running**
container, which has the OLD env vars. So if you change `AIOS_DB_URL`
between deploys, the Pre-Deploy uses the previous value. Force a
recreate (`docker compose up --force-recreate -d` on the host, or
the equivalent Coolify "Restart" button) before redeploying so the
new env reaches the migration command.

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

Coolify → `aios-postgres` → **Backups → New Scheduled Backup**:

| Field | Value |
|---|---|
| Frequency | `0 3 * * *` (daily 03:00 UTC) |
| Retention (days, locally) | `7` |
| Retention (count, locally) | `10` |
| Save to S3 | off (configure in 13c if you want off-box) |

Click **Trigger backup now** to verify the pipeline works. Backups
land at
`/data/coolify/backups/databases/<team>/<db-uuid>/pg-dump-<db>-<ts>.dmp`
on the host as pg_dump custom-format archives — restore with
`pg_restore`. The retention pair is "delete when older than N days OR
when there are more than M files" (whichever fires first).

Equivalent via API if the UI is fighting you:

```bash
curl -X POST "https://coolify.eumemic.ai/api/v1/databases/<postgres-uuid>/backups" \
  -H "Authorization: Bearer $COOLIFY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"frequency":"0 3 * * *","enabled":true,"save_s3":false,
       "database_backup_retention_days_locally":7,
       "database_backup_retention_amount_locally":10,
       "dump_all":false,"backup_now":true}'
```

### 13b. Workspace tarballs

Coolify doesn't back up bind-mounted host paths. Roll our own as a
`/etc/cron.daily/` script — Ubuntu fires `cron.daily` at 06:25 UTC
(when anacron is absent, per `/etc/crontab`), comfortably after the
03:00 Postgres dump:

```bash
ssh root@aios.eumemic.ai
cat > /etc/cron.daily/aios-workspace-backup <<'EOF'
#!/bin/sh
# Nightly tarball of session workspaces; 7-day local retention.
# Postgres handles its own backups via Coolify.
set -eu
dest=/var/backups/aios
date=$(date -u +%Y-%m-%d)
file=$dest/workspaces-$date.tar.gz
mkdir -p $dest
[ -f "$file" ] && exit 0
tar -czf "$file.tmp" -C /srv/aios workspaces 2>/dev/null
mv "$file.tmp" "$file"
find $dest -maxdepth 1 -name 'workspaces-*.tar.gz' -mtime +7 -delete
EOF
chmod 0755 /etc/cron.daily/aios-workspace-backup
# Run once to seed today's tarball + verify the script:
/etc/cron.daily/aios-workspace-backup
ls -la /var/backups/aios/
```

For a typical aios workload (≤10 active sessions, each with a
workspace under ~100 MB), tarballs are <1 GB and the prune keeps disk
predictable.

### 13c. Off-box sync (known gap)

Steps 13a/b both land on Server B's `/`, so a VM loss takes them with
it. Whole-disk Hetzner backups (step 1a) handle the catastrophic case,
but per-data DR ideally has both Postgres dumps and workspace
tarballs mirrored off-box. Two routes:

- **Coolify-side**: configure an S3-compatible destination
  (`Settings → S3 Storages`) and set `save_s3: true` on the schedule
  from 13a. Postgres dumps then land both locally and on S3.
- **Host-side**: `apt-get install rclone`, `rclone config` against B2
  / Hetzner Storage Box / etc., then a weekly cron `rclone sync
  /var/backups/aios b2:aios-backups`. Picks up the workspace
  tarballs too.

Currently neither is configured. Tracked as a follow-up.

---

## Step 14 — Verify auto-deploy on `master`

Set **Auto Deploy** on for both Applications. The toggle is at app
→ **Settings → General → Auto Deploy**, or via API:

```bash
for uuid in <api-uuid> <worker-uuid>; do
  curl -X PATCH "https://coolify.eumemic.ai/api/v1/applications/$uuid" \
    -H "Authorization: Bearer $COOLIFY_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"is_auto_deploy_enabled":true,"instant_deploy":false}'
done
```

The GET response *won't* include `is_auto_deploy_enabled` (Coolify's
serializer omits it) — verify against the DB if you doubt the PATCH:

```bash
docker exec coolify-db psql -U coolify -c "
SELECT a.name, s.is_auto_deploy_enabled
FROM applications a JOIN application_settings s ON s.application_id = a.id
WHERE a.deleted_at IS NULL;"
```

Make a trivial change (edit a comment, bump a doc line), commit,
`git push origin master`. Coolify's webhook fires on both
Applications within seconds. Watch both Deployments tabs.

api: rolling swap — new container boots while old keeps serving;
healthcheck on `/health` gates the cutover. ~10–30 s, zero downtime.

worker: stop-then-start (per the consistent-container-name flag set
in step 10) — old container shuts down, new one starts. ~10 s
sandbox-spawning blackout while procrastinate jobs queue in
Postgres. Sandboxes the old worker had spawned get orphan-reaped on
the new worker's startup (`sandbox/registry.py:reap_orphans`).

If a deploy doesn't fire: the GitHub App's installation-level webhook
should show a 200 response from Coolify per push event. The webhook
is at the org/installation level (not per-repo), so an empty
`gh api /repos/eumemic/aios/hooks` is expected and not the problem.

**Watch out for**: a Coolify deployment with status `finished` does
NOT mean the new container is steadily up — Coolify's deploy job
ends after starting the new container; post-handoff health is
invisible to it. The worker's `HEALTHCHECK CMD true` passes
unconditionally, so a crashlooping worker (e.g. lock collision)
looks "healthy" to docker too. Verify with `docker ps` + log spot-
check for `worker.startup` after deploys.

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
