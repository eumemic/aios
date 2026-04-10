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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
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
        description="Container image used for all v1 sessions.",
    )
    workspace_root: Path = Field(
        default=Path("/var/lib/aios/workspaces"),
        description="Host directory containing per-session workspace subdirectories.",
    )
    pool_size_max: int = Field(
        default=16,
        ge=1,
        description="Maximum concurrent Docker containers in the sandbox pool.",
    )

    # ── worker ─────────────────────────────────────────────────────────────
    worker_concurrency: int = Field(
        default=4,
        ge=1,
        description="Concurrent session loops per worker process.",
    )

    # ── lease (Phase 2) ────────────────────────────────────────────────────
    lease_duration_seconds: int = Field(
        default=30,
        ge=5,
        description="How long a worker's lease on a session is valid.",
    )
    lease_refresh_seconds: int = Field(
        default=10,
        ge=1,
        description="How often the lease refresh task extends the lease. "
        "Should be comfortably less than lease_duration_seconds.",
    )
    lease_reschedule_delay_seconds: int = Field(
        default=5,
        ge=1,
        description="When acquire_lease fails because another worker holds it, "
        "delay before retrying via a deferred wake job.",
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
