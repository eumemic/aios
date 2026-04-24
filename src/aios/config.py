"""Server configuration loaded from environment variables.

Single source of truth for every env-driven knob in aios. Imported once at
process start; downstream modules read attributes off the ``settings`` instance
rather than touching ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # ── database (required) ────────────────────────────────────────────────
    db_url: str = Field(
        ...,
        description="Postgres connection URL (postgresql://user:pass@host:port/db).",
    )

    # ── sandbox ────────────────────────────────────────────────────────────
    docker_image: str = Field(
        default="aios-sandbox:latest",
        description="Container image used for all v1 sessions. Build via "
        "`docker build -t aios-sandbox:latest -f docker/Dockerfile.sandbox docker/`.",
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
