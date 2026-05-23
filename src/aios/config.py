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
    bootstrap_token: SecretStr | None = Field(
        default=None,
        description="Unlocks ``POST /v1/accounts/bootstrap`` while the ``accounts`` table "
        "has no root row; the endpoint is 404 once a root exists. Generate with "
        "``openssl rand -base64 32``.",
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
    bash_default_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Default timeout for a single bash tool call. The agent can "
        "override per-call up to this maximum.",
    )
    bash_max_output_bytes: int = Field(
        default=100_000,
        ge=1_000,
        description="Maximum bytes of stdout+stderr returned from a bash tool call. "
        "Output beyond this is truncated with a [truncated] marker.",
    )
    upload_max_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
        description="Maximum size of a single file uploaded via "
        "``POST /v1/sessions/<id>/files``. Overflow is rejected with 413 "
        "after the multipart body is drained so the client sees a clean "
        "response rather than a transport reset.",
    )

    # ── worker ─────────────────────────────────────────────────────────────
    worker_concurrency: int = Field(
        default=4,
        ge=1,
        description="Concurrent session steps per worker process.",
    )

    # ── container lifecycle ────────────────────────────────────────────────
    container_idle_timeout_seconds: int = Field(
        default=300,
        ge=30,
        description="Seconds of inactivity before a session's sandbox container "
        "is released. The idle reaper checks every 60s.",
    )
    mcp_pool_idle_timeout_seconds: int = Field(
        default=900,
        ge=30,
        description="Seconds of inactivity before a pooled MCP session is "
        "closed. An OAuth token refresh rotates the bearer, orphaning the "
        "old keyed entry; the idle reaper (checks every 60s) reclaims it.",
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

    # ── observability ──────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` singleton.

    Cached so every call site sees the same instance and env reads happen once.
    Tests that need different settings should use ``Settings(...)`` directly
    inside a fixture.
    """
    return Settings()
