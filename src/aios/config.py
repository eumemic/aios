"""Server configuration loaded from environment variables.

Single source of truth for every env-driven knob in aios. Imported once at
process start; downstream modules read attributes off the ``settings`` instance
rather than touching ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aios.models.vaults import OAuthProviderApp

# Hardcoded mirror of ``aios.harness.loop._JOB_TIMEOUT_S`` (300.0). We
# don't import it because ``aios.harness.loop`` itself imports config
# at module load — a top-level import here would risk a cycle, and a
# lazy one inside the validator is more noise than the constant is
# worth. If ``_JOB_TIMEOUT_S`` ever moves, update this too; the failure
# mode is a config validator that no longer matches reality, not a
# crash. ``test_github_clone_session_timeout_below_step_budget``
# cross-checks the two values stay aligned.
_HARNESS_STEP_TIMEOUT_S: float = 960.0


class Settings(BaseSettings):
    """All env-driven aios configuration.

    Reads from process env first, then a ``.env`` file if present, then defaults.
    Required values without defaults will raise on instantiation if missing.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIOS_",
        # Layered: user-level shared secrets (seeded once) then per-worktree
        # overrides. Later file wins; process env still beats both. `~` is not
        # expanded by pydantic-settings, so we resolve Path.home() explicitly.
        # Missing files are silently skipped.
        env_file=(
            str(Path.home() / ".aios" / "secrets.env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── instance identity ──────────────────────────────────────────────────
    instance_id: str = Field(
        default="default",
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Distinguishes concurrent aios deployments on a shared host. "
        "Flows into Docker container labels so each worker's orphan-reaper only "
        "touches its own containers, and into dev-bootstrap DB naming. The "
        "`default` value preserves pre-existing single-instance behavior; `aios "
        "dev bootstrap` writes a unique value per git worktree. The pattern "
        "restricts the value to chars safe for Postgres identifiers and Docker "
        "labels.",
    )

    # ── crypto (required) ──────────────────────────────────────────────────
    vault_key: SecretStr = Field(
        ...,
        description="Base64-encoded 32-byte master key for libsodium secretbox.",
    )
    vault_key_previous: SecretStr | None = Field(
        default=None,
        description="Optional previous AIOS_VAULT_KEY used only by `aios rekey` to decrypt during rotation.",
    )
    egress_ca_key: SecretStr = Field(
        ...,
        description="Base64-encoded 32-byte key for the sandbox egress CA.",
    )
    bootstrap_token: SecretStr | None = Field(
        default=None,
        description="Unlocks ``POST /v1/accounts/bootstrap`` while the ``accounts`` table "
        "has no root row; the endpoint is 404 once a root exists. Generate with "
        "``openssl rand -base64 32``.",
    )

    default_spend_limit_usd: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Default lifetime USD spend limit for accounts that do not set or "
            "inherit account config spend_limit_usd. Unset means ungated."
        ),
    )

    # ── interactive OAuth (vault credential "Connect") ─────────────────────
    oauth_provider_apps: list[OAuthProviderApp] = Field(
        default_factory=list,
        description="Operator-registered OAuth client apps for MCP providers that don't "
        "support Dynamic Client Registration (Google, Microsoft, Slack, …). When a "
        "server's discovered OAuth issuer/host matches an app's ``match``, the interactive "
        "Connect flow uses that app so end users sign in without supplying any credentials. "
        "Set ``AIOS_OAUTH_PROVIDER_APPS`` to a JSON list of "
        "{match, client_id, client_secret?, token_endpoint_auth_method?, "
        "token_endpoint_hosts?, scope?, authorize_params?}. Register "
        "each app with the provider using the console callback "
        "``<console_url>/api/auth/mcp-oauth/callback`` as the redirect URI.",
    )
    oauth_allow_insecure_hosts: str = Field(
        default="",
        description="DEV-ONLY comma-separated host[:port] allowlist that bypasses the "
        "interactive-Connect SSRF guard (https requirement + private-range block) for "
        "the listed hosts. Lets a self-hosted MCP fleet reachable only over plain http "
        "on an internal host (e.g. ``workspace-mcp:8000``) be connected from dev. Leave "
        "empty in production (prod MCP servers are https), where any value would re-open "
        "SSRF to internal hosts. Plain comma string (not JSON) so the natural ``.env`` "
        "format ``host-a,host-b:8000`` parses without brackets.",
    )

    # ── database (required) ────────────────────────────────────────────────
    db_url: str = Field(
        ...,
        description="Postgres connection URL (postgresql://user:pass@host:port/db).",
    )

    # ── sandbox ────────────────────────────────────────────────────────────
    docker_image: str = Field(
        default="ghcr.io/eumemic/aios-sandbox:latest",
        description="Container image used for all v1 sessions. Published from "
        "`docker/Dockerfile.sandbox` to GHCR by the build-sandbox workflow. "
        "Override to a local tag for development "
        "(`docker build -t aios-sandbox:latest -f docker/Dockerfile.sandbox .`).",
    )
    workspace_root: Path = Field(
        default=Path("/var/lib/aios/workspaces"),
        description="Host directory containing per-session workspace subdirectories. "
        "Each session gets <workspace_root>/<session_id> bind-mounted to /workspace "
        "inside its sandbox container.",
    )
    workspaces_owner_uid: int = Field(
        default=1000,
        ge=0,
        description="uid that should own every directory under workspace_root. The "
        "worker (root) chowns newly-created shared-tree components to this uid so the "
        "api container (running as this uid) can write into them. Set via "
        "AIOS_WORKSPACES_OWNER_UID.",
    )
    workspaces_owner_gid: int = Field(
        default=1000,
        ge=0,
        description="gid counterpart to workspaces_owner_uid. Set via AIOS_WORKSPACES_OWNER_GID.",
    )
    sandbox_cpu_quota: float | None = Field(
        default=None,
        ge=0.01,
        description="Maximum cumulative CPU cores a sandbox can use, in "
        "decimal core units (e.g. 2.0 = two cores at 100%, 0.5 = half "
        "of one core). Translates to ``docker run --cpus``. The lower "
        "bound is Docker's effective floor — below ~0.01 the runtime "
        "rounds the CFS quota down and the value becomes meaningless. "
        "``None`` leaves the host's default in place (unlimited).",
    )
    sandbox_memory_bytes: int | None = Field(
        default=None,
        ge=4 * 1024 * 1024,
        description="Maximum resident memory a sandbox can use, in bytes. "
        "Translates to ``docker run --memory`` and ``--memory-swap`` "
        "(swap pinned to the same value so the sandbox can't lean on "
        "swap to exceed the cap). The kernel OOM-kills the offending "
        "process when this is breached. ``None`` leaves the host's "
        "default (unlimited) in place. Minimum 4 MiB to satisfy Docker's "
        "own floor.",
    )
    sandbox_pids_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of PIDs the sandbox cgroup can hold. "
        "Translates to ``docker run --pids-limit``. Mitigates fork-bomb "
        "denial-of-service; ``None`` leaves the host's default in place.",
    )
    sandbox_snapshot_budget_bytes: int | None = Field(
        default=4 * 1024 * 1024 * 1024,
        ge=10 * 1024 * 1024,
        description="Per-session snapshot budget in **unique** bytes (durable "
        "session sandboxes, §5.7). When a session's unique snapshot bytes "
        "would exceed this at idle/teardown, the snapshot verb FLATTENS "
        "(collapse + whiteout, the definitive secret scrub) instead of "
        "committing another layer; commit-and-flag, never refuse (refusal "
        "would destroy the agent's work as punishment for a state it wasn't "
        "awake to prevent). Replaces the dead ``sandbox_disk_bytes`` "
        "writable-layer cap, which required overlay2-on-xfs+pquota and never "
        "worked on prod ext4. ``None`` ⇒ unbounded (the depth backstop still "
        "applies). Per-environment overridable via "
        "``EnvironmentConfig.snapshot_budget_bytes``.",
    )
    sandbox_snapshot_ttl_seconds: int = Field(
        default=2_592_000,  # 30 days
        ge=60,
        description="Dormancy TTL for a persisted session snapshot. The GC "
        "reconciler removes a snapshot whose session's last activity is older "
        "than this (appending a model-visible ``sandbox_fs_expired`` notice), "
        "keyed on the session's ``(session_id, last_event_seq)`` event row. "
        "Bounds supply-chain accumulation (agent-installed software re-executes "
        "at every resume) and unbounded snapshot retention.",
    )
    sandbox_snapshot_pool_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Per-host snapshot disk budget in bytes (durable session "
        "sandboxes, §5.7) — the load-bearing pool bound. When this host's "
        "total snapshot bytes exceed it, the GC evicts the MOST-DORMANT "
        "sessions first (``sandbox_fs_expired {disk_pressure}``). Required in "
        "production once the eumemic-ops prune-cron exemption lands (that "
        "exemption removes the only existing disk control). ``None`` ⇒ "
        "unbounded retention (dev default); operators MUST set real host "
        "headroom in production. v1 enforces per-host and reports global.",
    )
    sandbox_snapshot_empty_floor_bytes: int = Field(
        default=8192,
        ge=0,
        description="Writable-layer byte floor at/below which a snapshot is "
        "treated as a no-write identity (``skipped_empty`` — no new image, no "
        "chain growth). NOT zero: on the production **containerd image store** "
        "a no-write container reports ``SizeRw == 4096``, not 0, so an "
        "``== 0`` test would never fire in prod and chat-only / read-only "
        "sessions would grow a chain every idle. The floor (default 8 KiB) is "
        "what keeps them from ever snapshotting.",
    )
    sandbox_seccomp_profile: str = Field(
        default=str(Path(__file__).resolve().parents[2] / "docker" / "seccomp-sandbox.json"),
        description="Path to the authored seccomp profile applied to every sandbox "
        "container via `docker run --security-opt seccomp=<path>`. The docker CLI "
        "reads this file from the WORKER container's filesystem and ships the JSON "
        "to the daemon. Defaults to the baked-in profile shipped in the worker image. "
        "Emergency rollback ONLY: set AIOS_SANDBOX_SECCOMP_PROFILE=unconfined to "
        "disable seccomp filtering. Never defaults to unconfined; the flag is always emitted.",
    )
    sandbox_backend: str = Field(
        default="docker",
        description="Sandbox backend implementation. 'docker' (default) runs "
        "DockerBackend. Override only where an alternate SandboxBackend impl is installed.",
    )
    sandbox_runtime: str | None = Field(
        default=None,
        description="Optional Docker container runtime for sandboxes and their "
        "network-lockdown sidecars. Unset (or Docker's configured default, normally "
        "runc) preserves local/CI behavior; set AIOS_SANDBOX_RUNTIME=runsc to run "
        "containers under gVisor where the host has runsc installed.",
    )
    bash_default_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Default ceiling for a single bash tool call, in seconds. "
        "The agent can override per-call up to this maximum. A session bound "
        "to an environment with ``EnvironmentConfig.bash_timeout_seconds`` set "
        "uses that environment's ceiling instead — letting heavy dev workloads "
        "run >120s commands without raising the global default for every "
        "session on the worker (issue #725).",
    )
    bash_max_output_bytes: int = Field(
        default=100_000,
        ge=1_000,
        description="Maximum bytes of stdout+stderr returned from a bash tool call. "
        "Output beyond this is truncated with a [truncated] marker.",
    )
    tool_result_max_chars: int = Field(
        default=200_000,
        ge=1_000,
        description="Maximum characters of a tool result stored inline in the "
        "event log. A larger result is spilled to a file under the session's "
        "attachments mount and replaced inline with a stub pointing the model "
        "to read it, so a single oversized result can't exceed the context window.",
    )
    upload_max_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
        description="Maximum size of a single file uploaded via "
        "``POST /v1/sessions/<id>/files``. Overflow is rejected with 413 "
        "after the multipart body is drained so the client sees a clean "
        "response rather than a transport reset.",
    )
    model_call_deadline_s: float = Field(
        default=900.0,
        ge=1.0,
        description="Total-duration deadline for a single model call, in seconds. "
        "Must stay strictly below the harness step timeout so job-level "
        "cancellation remains a final safety net instead of the normal model "
        "budget.",
    )
    github_clone_session_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Wall-clock bound on each per-session "
        "``git clone --reference --dissociate`` (and the short admin ops "
        "around it: ``remote set-url``, ``config user.*``). Must stay "
        "well below the 960s harness step timeout — otherwise a hung "
        "clone burns a whole user turn before the step-level cap fires. "
        "See issue #697.",
    )
    github_clone_cache_timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Wall-clock bound on the bare-cache clone/fetch "
        "(``<workspace_root>/_github_repos/<sha256(url)>``). Cold-case "
        "clones of large repos can legitimately take minutes; failures "
        "here only delay the cache, they don't block the per-session "
        "working tree past the per-session budget. See issue #697.",
    )

    # ── worker ─────────────────────────────────────────────────────────────
    worker_concurrency: int = Field(
        default=4,
        ge=1,
        description="Concurrent session steps per worker process.",
    )

    # ── container lifecycle ────────────────────────────────────────────────
    container_idle_timeout_seconds: int = Field(
        default=1800,
        ge=30,
        description="Seconds of inactivity before a session's sandbox container "
        "is released. The idle reaper checks every 60s. Raised from 300 to 1800 "
        "(30 min) for durable session sandboxes (§5.10): teardown now costs a "
        "commit + a layer + eventual flatten, so keeping an idle ``tail -f`` "
        "container alive (~1 MB RSS) is the cheap option. At 300s a daily-driver "
        "session would commit 20-80x/day and a 15-min cron-trigger session "
        "~96x/day, hitting the flatten wall in days; at 1800s a 15-min cron "
        "session never idles out (zero commits until it stops) and a "
        "conversational session commits a handful of times a day.",
    )
    mcp_pool_idle_timeout_seconds: int = Field(
        default=900,
        ge=30,
        description="Seconds of inactivity before a pooled MCP session is "
        "closed. An OAuth token refresh rotates the bearer, orphaning the "
        "old keyed entry; the idle reaper (checks every 60s) reclaims it.",
    )

    # ── host scratch-dir reaper (#1192) ────────────────────────────────────
    host_dir_reaper_enabled: bool = Field(
        default=True,
        description="Kill-switch for the idle host scratch-dir reaper "
        "(``_session_repos`` working-tree clones and ``_runs`` per-run "
        "scratch). When False the reaper is never started and deletes "
        "nothing — so the worker that just had a P1 disk-fill can disable "
        "irreversible host deletion without a redeploy (set "
        "``AIOS_HOST_DIR_REAPER_ENABLED=false``). Default-on: the missing GC "
        "is itself the floodgates incident (#1192).",
    )
    host_dir_reaper_min_age_seconds: int = Field(
        default=3600,
        ge=0,
        description="Minimum age (by directory mtime) before an idle "
        "``_session_repos``/``_runs`` host dir is eligible for reaping. A "
        "race floor: a dir freshly created by an in-flight provision whose "
        "DB row has not yet committed (or a run whose terminal status is "
        "about to flip back) is left alone until it ages past this floor. "
        "The dominant safety check is DB liveness; this is the belt-and-"
        "suspenders for the provision/commit gap.",
    )
    host_dir_reaper_interval_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Interval for the periodic host scratch-dir reaper sweep "
        "(an immediate first sweep runs at worker startup, then repeats at "
        "this cadence — mirrors the snapshot GC reconciler).",
    )

    # ── archived-session workspace reaper (#40, the 45G hole) ──────────────────
    workspace_reaper_enabled: bool = Field(
        default=False,
        description="Kill-switch for the archived-session workspace reaper "
        "(reclaims the per-session ``/workspace`` host dir of "
        "``archived_at IS NOT NULL`` sessions — the 45G hole #1192 did not "
        "cover). Ships DARK (default OFF) and is enabled after review (set "
        "``AIOS_WORKSPACE_REAPER_ENABLED=true``): unlike the #1192 reaper, this "
        "deletes a session's actual working files, not reconstructible clones, "
        "so it does not ship enabled. When False the reaper is never started "
        "and deletes nothing.",
    )
    workspace_reaper_dry_run: bool = Field(
        default=False,
        description="When True the archived-session workspace reaper logs what "
        "it WOULD reap (paths + bytes reclaimable) and the structured per-sweep "
        "count, but deletes nothing. Lets a deploy verify the keep-set/predicate "
        "on real data before the destructive sweep is armed "
        "(``AIOS_WORKSPACE_REAPER_DRY_RUN=true``).",
    )
    workspace_reaper_min_archived_age_seconds: int = Field(
        default=86_400,
        ge=0,
        description="Minimum age, by ``sessions.archived_at`` (DB archive time, "
        "NOT file mtime), before an archived session's ``/workspace`` dir is "
        "eligible for reaping. Conservative default 24h: guards a just-archived "
        "session whose workspace something might still be reading in the window "
        "right after archive.",
    )
    workspace_reaper_min_mtime_age_seconds: int = Field(
        default=3600,
        ge=0,
        description="Belt-and-suspenders floor on the workspace dir's own mtime "
        "for the archived-session workspace reaper: a dir touched more recently "
        "than this (e.g. a step that was mid-write at archive time) is left "
        "alone regardless of archive age. The dominant gate is the DB "
        "archived-and-aged check; this is the provision/write race floor.",
    )
    workspace_reaper_interval_seconds: float = Field(
        default=3600.0,
        gt=0.0,
        description="Interval for the periodic archived-session workspace "
        "reaper sweep (an immediate first sweep runs at worker startup, then "
        "repeats at this cadence).",
    )
    tool_broker_socket_path: Path | None = Field(
        default=None,
        description="Host path for the tool broker's Unix-domain socket. "
        "When set, the broker dual-listens on TCP (ephemeral port, used by "
        "any out-of-container tooling) AND on this UDS path; sandbox "
        "containers receive ``TOOL_BROKER_URL=unix:///var/run/aios/tool-broker.sock`` "
        "and a bind mount of the socket file. Use this in deployments where "
        "the worker's TCP port isn't reachable from the sandbox network — "
        "e.g. Coolify production where TCP port publishing isn't done. When "
        "unset (the default), sandboxes reach the broker over TCP via "
        "``http://aios-worker:<port>``.",
    )

    # ── database pool ──────────────────────────────────────────────────────
    db_pool_max_size: int = Field(
        default=8,
        ge=1,
        description="Maximum asyncpg pool size. Should comfortably exceed "
        "worker_concurrency * ~3 connections per turn.",
    )

    # ── api server ─────────────────────────────────────────────────────────
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8080, ge=1, le=65535)

    # ── web tools ──────────────────────────────────────────────────────────
    tavily_api_key: str | None = Field(
        default=None,
        description="Tavily API key for web_fetch and web_search tools.",
    )

    default_mcp_permission_policy: Literal["always_allow", "always_ask"] | None = Field(
        default=None,
        description="Fallback permission policy for MCP tools whose server "
        "has no matching ``mcp_toolset`` entry on the calling agent. "
        "When unset (the default), unmounted MCP toolsets gate on "
        "``always_ask`` confirmation — the safe default that protects "
        "against connector-mounted tools the agent never opted into. "
        "Operators in trusted environments can set this to "
        "``always_allow`` so connector tools dispatch immediately on any "
        "session that has the connection attached, without per-agent "
        "``mcp_toolset`` declarations.  Explicit per-toolset policies on "
        "``agent.tools`` always win — this only changes the fallback "
        "for unmounted servers.",
    )

    trusted_inference_api_bases: list[str] = Field(
        default_factory=list,
        description="Operator allowlist of trusted inference endpoints for the "
        "model-identity clamp at the workflow ``agent()`` spawn edge (#823). A "
        "named-agent child's effective ``api_base`` (from its ``litellm_extra``) "
        "must equal the launcher's (a workflow run has none — i.e. the default "
        "operator endpoint, so ``api_base`` must be unset) OR appear in this "
        "allowlist; otherwise the spawn fails closed with an "
        "``untrusted_api_base`` rejection. Empty (the default) means *no* "
        "redirected endpoint is trusted: a child may only run on the default "
        "endpoint. This is sound today (``create_agent`` is operator-only, so no "
        "runtime principal can mint a hostile ``api_base``); the clamp makes the "
        "boundary real and frozen ahead of native self-management, where the "
        "trusted-catalog precondition lapses. Values are matched verbatim against "
        "the agent's ``litellm_extra['api_base']`` string.",
    )

    # ── triggers ───────────────────────────────────────────────────────────
    triggers_per_account_max: int = Field(
        default=100,
        ge=1,
        description="Per-account ceiling on enabled trigger rows "
        "across all sessions. Enforced at ``add_trigger`` time; on exceed, "
        "the call raises ``RateLimitedError``. Bounds the worst-case work "
        "the event-driven scheduler has to enumerate on each tick, and "
        "the fire-storm a misbehaving agent could deflect at the broker. "
        "Includes one-shot ``schedule_wake`` rows; bump if your workload "
        "legitimately needs more standing timers per tenant.",
    )
    trigger_runs_retention_days: int = Field(
        default=30,
        ge=1,
        description="Days of per-fire trigger audit rows (``trigger_runs``) to "
        "retain; older rows are pruned by the worker's periodic sweep. "
        "Time-based by design — see ``prune_trigger_runs`` for why a "
        "count-cap would be unsafe.",
    )
    schedule_wake_max_delay_seconds: int = Field(
        default=60 * 60 * 24 * 30,  # 30 days
        ge=60,
        description="Upper bound on the agent-requested wake delay (via "
        "``schedule_wake``'s ``delay_seconds`` or the resolved offset of "
        "its ``at`` argument). Bounds (a) unbounded scheduler MIN-query "
        "enumeration over agent-set-and-forget rows that may never fire "
        "within a worker's lifetime, and (b) catches obvious mistakes "
        "(``delay_seconds=86400000`` instead of ``86400``) at the tool "
        "boundary with a clear typed error instead of letting them sit "
        "in the DB for years. Set high for legitimate long-horizon "
        "reminders.",
    )

    # ── workflows ──────────────────────────────────────────────────────────
    workflow_max_agent_calls: int = Field(
        default=1000,
        ge=1,
        description="Per-run lifetime ceiling on the number of ``agent()`` "
        "children a workflow may spawn. Counted from the run's ``call_started`` "
        "journal (monotonic), checked before each step's spawns: a step that "
        "would push the total past the cap errors the run (``too_many_agents``) "
        "before spawning any of that step's new children. Bounds a runaway "
        "script that spawns children in an unbounded loop. The per-call "
        "single-``parallel()`` fan-out width is bounded separately, in the "
        "host (``wf_script_host.MAX_PARALLEL_FANOUT``).",
    )
    workflow_default_child_model: str | None = Field(
        default=None,
        description="Default model for generic workflow agent() children launched via operator/HTTP runs when no per-run or per-call model is supplied.",
    )
    workflow_max_inflight_children_per_run: int = Field(
        default=8,
        ge=1,
        description="Per-run cap on concurrently in-flight ``agent()`` children. A "
        "single workflow wake opens its emitted frontier in waves: at most this many "
        "new agent() children are spawned per step while that many are already "
        "in-flight; the rest are journaled as ``frontier_deferred`` markers (not "
        "spawned) and admitted on later wakes as earlier children resolve and free "
        "slots. Orthogonal to ``workflow_max_agent_calls`` (the lifetime ceiling) and "
        "to ``worker_concurrency`` (cross-run worker parallelism): this bounds a "
        "single run's standing agent() fan-out. The re-drive uses the existing "
        "child-completion re-wake — no new machinery.",
    )
    workflow_agent_deadline_seconds: float = Field(
        default=60 * 60,  # 1 hour
        gt=0,
        description="Wall-clock budget for a single ``agent()`` call, measured "
        "from its ``call_started``. A child that never goes idle (e.g. loops on "
        "tools forever) never trips the quiescence nudge, so without this its "
        "caller would suspend forever; past the deadline the harvest writes a "
        "``timeout`` error response (exactly-once), resolving the call as a "
        "catchable ``AgentError`` and keeping the run total. Generous by default "
        "— a child doing real model+tool work can legitimately run for minutes; "
        "this is the never-resolves backstop, not a tight SLA.",
    )
    trace_max_nodes: int = Field(
        default=2000,
        ge=1,
        description="Node-count ceiling for a single ``/trace`` walk (#1149). "
        "#1124's depth counter bounds path length (≤10 by construction) but NOT "
        "the node count — ``workflow_max_agent_calls`` defaults to 1000 lifetime "
        "``agent()`` children per run, and fan-out caps are concurrency, not "
        "totals. So the trace walk carries this explicit ceiling: once this many "
        "nodes have been emitted, the frontier is cut and the response carries a "
        "typed ``truncated: {at_nodes}`` marker rather than silently dropping "
        "tail subtrees or running unbounded. Tunable upward for deliberately "
        "deep-audit traces.",
    )
    workflow_runs_per_launcher_max: int = Field(
        default=20,
        ge=1,
        description="Per-launcher-session ceiling on OUTSTANDING (non-terminal) "
        "runs — the horizontal fan-out bound on the agent's ``create_run`` "
        "builtin (the operator/HTTP path has no launcher and is exempt). A "
        "concurrency cap, not a rate or lifetime budget: slots free as runs "
        "reach a terminal status, so a sequential launch loop is unbounded by "
        "design. Enforced with the per-account cap under one advisory lock; "
        "on exceed, ``create_run`` raises ``RateLimitedError``.",
    )
    workflow_runs_per_account_max: int = Field(
        default=100,
        ge=1,
        description="Per-account ceiling on OUTSTANDING (non-terminal) runs "
        "across all launchers, operator launches included — the backstop the "
        "per-launcher cap can't provide (many sessions, or a run's agent() "
        "children, are each fresh launchers). Same concurrency-not-budget "
        "semantics; worst-case standing exposure is this many runs each "
        "entitled to ``workflow_max_agent_calls`` children.",
    )
    open_goals_per_session_max: int = Field(
        default=10,
        ge=1,
        description="Per-session ceiling on OPEN self-issued goals (#1414). A goal "
        "is a self-edge (``set_goal``): an awaited ``request_opened`` whose caller "
        "is the session itself. ``set_goal`` rejects (``RateLimitedError``) when the "
        "session already owns this many open self-goals — slots free as goals are "
        "answered (``return``/``error``) or retracted (``cancel_goal``). An admission "
        "cap, not a rate/lifetime budget. Pairs with the obligations-block rendered "
        "cap (#1413): without it, allow-N ``set_goal`` makes the owned-obligation "
        "count unbounded so B's reserved tail budget could crash the step. Counts "
        "self-goals ONLY — peer-invoke / workflow-child obligations don't consume it.",
    )
    workflow_wake_batch_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Coalescing window for run wakes from child completions and "
        "tool results: completions landing within the window collapse into one "
        "re-drive (the scheduled wake occupies the dedup lock, so further defers "
        "are absorbed — including, for the window's duration, otherwise-immediate "
        "wakes like a gate resume; their signals are still harvested by the "
        "delayed step). 0 disables (every completion wakes immediately — today's "
        "behavior). Note the effective floor: a not-yet-due scheduled job is "
        "picked up by worker polling (~5s on an otherwise-idle worker), so "
        "sub-second windows buy nothing.",
    )

    # ── connectors ─────────────────────────────────────────────────────────
    inbound_debounce_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Debounce window (seconds) before the FIRST wake of an "
        "idle session on a connector inbound. aios already coalesces "
        "mid-turn arrivals, but the very first wake of an idle session "
        "fires immediately, so a bursty sender that jitters several "
        "messages a few seconds apart (e.g. Jarvis, 2-5s between sends) "
        "spawns one turn per message instead of one turn for the burst. "
        "When > 0, the inbound path schedules the wake this many seconds "
        "out via ``defer_wake``'s ``delay_seconds``; the existing "
        "``queueing_lock=session_id`` dedup then collapses any follow-on "
        "inbounds that land inside the window into the same single wake. "
        "0.0 (the default) disables debounce — the wake fires immediately, "
        "preserving pre-feature behavior. Fixed delay, not random jitter "
        "(v1). Only the connector-inbound first wake is affected; the "
        "harness retry-backoff path (cause='reschedule') and the async "
        "tool-completion path (cause='connector_tool_result') are "
        "untouched.",
    )
    connector_backfill_max_age_seconds: int = Field(
        default=60 * 60,  # 1 hour
        ge=60,
        description="Age ceiling, in seconds, on connector sends surfaced by the "
        "SSE subscribe-time backfill (``list_pending_calls_for_connector``). A "
        "pending connector send whose parent assistant turn is older than this is "
        "SKIPPED — excluded from the backfill, not expired (the event log is left "
        "untouched). A legitimately in-flight send executes within seconds of the "
        "connector reconnecting, so the default 1h never touches a real one; the "
        "bound only suppresses dormant, weeks-stale sends from being re-transmitted "
        "on a worker restart (#744). Does NOT affect ``Session.awaiting`` / the "
        "unresolved-tool-call read-model, which surfaces all unresolved calls "
        "regardless of age (#741).",
    )
    confirmed_dispatch_max_age_seconds: int = Field(
        default=60 * 60,  # 1 hour
        ge=60,
        description="Age ceiling, in seconds, on the confirmation of an "
        "operator-confirmed-but-unresolved tool call surfaced for auto-dispatch "
        "on resume (``list_confirmed_unresolved_tool_calls`` / sweep "
        "``CONFIRMED_ROWS_SQL``). A confirmed call whose ``tool_confirmed`` event "
        "is older than this is SKIPPED — excluded from re-dispatch, not expired "
        "(the event log is left untouched, no synthetic result). The bound is on "
        "the CONFIRM event's ``created_at``, NOT the assistant turn: an operator "
        "can confirm an OLD proposal, which is a FRESH intent to dispatch, so the "
        "dispatchable-since timestamp is when confirmation was granted. A "
        "legitimately confirmed call dispatches within seconds, so the default 1h "
        "never touches a real one; the bound only suppresses weeks-stale "
        "confirmations from re-running a side-effecting tool on a worker restart "
        "(#746). Parallel to ``connector_backfill_max_age_seconds`` (#744); both "
        "detection (sweep) and dispatch (resolver) apply this same bound in sync "
        "so they resolve the identical condition (no wake-with-no-progress loop, "
        "the #155 symptom).",
    )

    client_tool_call_max_age_seconds: int = Field(
        default=24 * 60 * 60,  # 24 hours
        ge=60,
        description="Age ceiling, in seconds, after which an unresolved "
        "CLIENT-result-pending tool call (a tool the harness never dispatches "
        "because the CLIENT executes it and returns the result — the non-MCP, "
        "not-in-registry branch of ``sweep._was_dispatched``) is treated as "
        "ABANDONED by ghost repair: a synthetic timeout/abandoned error result is "
        "appended, resolving the call. Unlike "
        "``connector_backfill_max_age_seconds`` / "
        "``confirmed_dispatch_max_age_seconds`` (which only SKIP, leaving the log "
        "untouched), this bound EXPIRES the call — without it the call's "
        "``open_tool_call_count`` contribution keeps the session a permanent wake "
        "candidate (``CANDIDATE_ROWS_SQL`` / ``_SESSION_ACTIVE_EXPR``) with no "
        "progress to make (the #155 wake-no-progress loop; regression from the "
        "open-tool-call-count predicate added in #750). The bound is on the "
        "ASSISTANT turn's ``created_at`` (when the call was emitted), consistent "
        "with the dispatched-ghost age semantics. The default 24h is deliberately "
        "generous — a legitimate slow client (a human-driven connector, a "
        "long-running custom tool) returns its result well within a day, so the "
        "bound only fires for a client that disconnected and will never return. "
        "Confirmation-pending calls (``always_ask`` tools awaiting a "
        "``tool_confirmed`` event) are EXCLUDED: those wait on the USER, not a "
        "client, and erroring them would kill a slow human-in-the-loop "
        "confirmation.",
    )

    # ── observability ──────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    @property
    def oauth_allow_insecure_host_set(self) -> frozenset[str]:
        """Parsed ``oauth_allow_insecure_hosts`` as a set of host[:port] entries."""
        return frozenset(h.strip() for h in self.oauth_allow_insecure_hosts.split(",") if h.strip())

    @model_validator(mode="after")
    def _require_absolute_workspace_root(self) -> Settings:
        if not self.workspace_root.is_absolute():
            raise ValueError(
                f"AIOS_WORKSPACE_ROOT must be an absolute path; got '{self.workspace_root}'. "
                "Note: pathlib does not expand '~'; write the full path. "
                "API and worker processes resolve relative paths against their own CWD, "
                "producing diverging session workspace_path values and ForbiddenError "
                "on tool calls."
            )
        return self

    @model_validator(mode="after")
    def _model_call_deadline_under_step_budget(self) -> Settings:
        if self.model_call_deadline_s >= _HARNESS_STEP_TIMEOUT_S:
            raise ValueError(
                f"AIOS_MODEL_CALL_DEADLINE_S={self.model_call_deadline_s} must be "
                f"strictly less than the harness step budget "
                f"({_HARNESS_STEP_TIMEOUT_S}s, aios.harness.loop._JOB_TIMEOUT_S)."
            )
        return self

    @model_validator(mode="after")
    def _github_clone_session_timeout_under_step_budget(self) -> Settings:
        # The whole point of #697: per-session clone budget must fit
        # strictly inside the harness step budget so a hung clone can't
        # burn a whole user turn before the step-level cap fires. Reject
        # misconfiguration loudly at startup rather than letting an
        # operator silently defeat the fix.
        if self.github_clone_session_timeout_seconds >= _HARNESS_STEP_TIMEOUT_S:
            raise ValueError(
                f"AIOS_GITHUB_CLONE_SESSION_TIMEOUT_SECONDS="
                f"{self.github_clone_session_timeout_seconds} must be strictly less than "
                f"the harness step budget ({_HARNESS_STEP_TIMEOUT_S}s, "
                f"aios.harness.loop._JOB_TIMEOUT_S); otherwise a hung clone "
                f"burns a whole user turn before the step-level cap fires. "
                f"See issue #697."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` singleton.

    Cached so every call site sees the same instance and env reads happen once.
    Tests that need different settings should use ``Settings(...)`` directly
    inside a fixture.
    """
    return Settings()
