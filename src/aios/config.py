"""Server configuration loaded from environment variables.

Single source of truth for every env-driven knob in aios. Imported once at
process start; downstream modules read attributes off the ``settings`` instance
rather than touching ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
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

    # ── auth + crypto (required) ───────────────────────────────────────────
    api_key: SecretStr = Field(
        ...,
        description="Shared API key clients send as `Authorization: Bearer <key>`.",
    )
    vault_key: SecretStr = Field(
        ...,
        description="Base64-encoded 32-byte master key for libsodium secretbox.",
    )
    bootstrap_token: SecretStr | None = Field(
        default=None,
        description="Optional one-shot token that unlocks ``POST /v1/accounts/bootstrap`` "
        "when the ``accounts`` table has no root row. Once a root exists, the bootstrap "
        "endpoint is 404 regardless of this value. Operator generates a random token "
        "(`openssl rand -base64 32`), sets it on a fresh deployment, hits the endpoint, "
        "captures the returned account_id + plaintext key, then unsets the env. Leaving "
        "the env set after bootstrap is harmless — the endpoint is already closed.",
    )

    # ── database (required) ────────────────────────────────────────────────
    db_url: str = Field(
        ...,
        description="Postgres connection URL (postgresql://user:pass@host:port/db).",
    )

    # ── sandbox ────────────────────────────────────────────────────────────
    sandbox_backend: Literal["docker"] = Field(
        default="docker",
        description="Which sandbox backend the worker uses to provision and run "
        "session sandboxes. Today only 'docker' is implemented; future backends "
        "(e.g. host-subprocess for environments without a Docker daemon) will "
        "extend this enum without changing call sites.",
    )
    docker_image: str = Field(
        default="ghcr.io/eumemic/aios-sandbox:latest",
        description="Container image used for all v1 sessions. Published from "
        "`docker/Dockerfile.sandbox` to GHCR by the build-sandbox workflow. "
        "Override to a local tag for development "
        "(`docker build -t aios-sandbox:latest -f docker/Dockerfile.sandbox docker/`).",
    )
    workspace_root: Path = Field(
        default=Path("/var/lib/aios/workspaces"),
        description="Host directory containing per-session workspace subdirectories. "
        "Each session gets <workspace_root>/<session_id> bind-mounted to /workspace "
        "inside its sandbox container.",
    )
    sandbox_network_mode: str = Field(
        default="bridge",
        description="Docker network mode for sandbox containers. 'bridge' (default) "
        "gives full outbound internet access, which the HN demo needs. 'none' "
        "disables networking entirely; 'host' shares the host network namespace.",
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

    # ── database pool ──────────────────────────────────────────────────────
    db_pool_max_size: int = Field(
        default=16,
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

    connectors_auto_create: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-connector ``auto_create_connections`` knob (plan §13). "
        "Inbound for an unknown ``(connector, account)`` pair auto-creates a "
        "detached connection row by default; setting ``{name: false}`` disables "
        "that for ``name``, dropping such inbounds with a ``no_connection`` "
        "counter increment instead.  Names not present in the dict default to "
        "``True``.\n\n"
        "Shape note: the plan documented this as nested "
        "``connectors.<name>.auto_create_connections``, but pydantic-settings "
        "doesn't compose well with that shape under the ``AIOS_`` env prefix; "
        "a flat ``AIOS_CONNECTORS_AUTO_CREATE='{\"signal\":false}'`` is "
        "operationally equivalent and easier to override from systemd / env.",
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

    # ── windowing defaults (per-agent overrides live on the agent record) ──
    default_window_min: int = Field(default=50_000, ge=1)
    default_window_max: int = Field(default=150_000, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide ``Settings`` singleton.

    Cached so every call site sees the same instance and env reads happen once.
    Tests that need different settings should use ``Settings(...)`` directly
    inside a fixture.
    """
    return Settings()
