"""Server configuration loaded from environment variables.

Single source of truth for every env-driven knob in aios. Imported once at
process start; downstream modules read attributes off the ``settings`` instance
rather than touching ``os.environ`` directly.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Instance names: stricter than ``Settings.instance_id`` (which allows a
# leading underscore for Postgres identifier compatibility) because the
# supervisor's AIOS_<CONNECTOR_UPPER>_<INSTANCE_UPPER>_* env-var
# re-export would produce double-underscore POSIX env names like
# ``AIOS_SIGNAL__BOT_TOKEN`` if leading underscores were allowed.
_INSTANCE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Sentinel ``instance`` value the CLI / API passes when the operator
# omits the instance segment.  The worker resolves it to the sole
# enabled instance of that connector type, or returns an
# ``ambiguous_instance`` error envelope.  Distinct from any valid
# instance name (which must match :data:`_INSTANCE_NAME_RE`).  Lives
# here rather than in ``aios.harness.connector_tasks`` so the CLI can
# import it without triggering the harness's procrastinate-app load
# (which requires ``db_url`` and other worker-only settings).
DEFAULT_INSTANCE_SENTINEL = "_"


class ConnectorInstance(NamedTuple):
    """Parsed ``connectors_enabled`` entry.

    ``connector`` is the entry-point name (e.g. ``"signal"``).
    ``instance`` is operator-supplied (e.g. ``"main"``); defaults to the
    connector name when omitted in the env value.  The supervisor uses
    ``(connector, instance)`` as the registry key so multiple instances
    of the same connector type can run side-by-side as separate
    subprocesses.
    """

    connector: str
    instance: str


def parse_connector_entry(raw: str) -> ConnectorInstance:
    """Parse one ``<connector>[:<instance>]`` ``connectors_enabled`` entry.

    Default instance = connector name when ``:`` is omitted (so a
    single-instance setup like ``connectors_enabled=signal`` continues to
    work without per-instance env scoping).  Both halves must match
    :data:`_INSTANCE_NAME_RE` so the env-var re-export stays POSIX-valid.
    """
    if ":" in raw:
        connector, _, instance = raw.partition(":")
    else:
        connector = raw
        instance = raw
    if not _INSTANCE_NAME_RE.match(connector):
        raise ValueError(f"connector name {connector!r} must match ^[a-z][a-z0-9_]*$")
    if not _INSTANCE_NAME_RE.match(instance):
        raise ValueError(
            f"instance name {instance!r} must match ^[a-z][a-z0-9_]*$ "
            f"(letters/digits/underscore only; lowercase; must start with a letter)"
        )
    return ConnectorInstance(connector=connector, instance=instance)


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

    # ── connectors ─────────────────────────────────────────────────────────
    connectors_enabled: list[str] = Field(
        default_factory=list,
        description="Connector instances the worker should spawn at startup, "
        "as a CSV of ``<connector>[:<instance>]`` entries (e.g. "
        "``signal:main,telegram:support,telegram:alerts``).  When ``:`` is "
        "omitted, instance defaults to the connector name (so a single-instance "
        "deployment can write ``connectors_enabled=signal,telegram``).  Each "
        "connector name is resolved against the ``aios.connectors`` Python "
        "entry-point group; the resolved spec describes how to launch the "
        "subprocess.  Multiple instances of the same connector type are "
        "permitted — each gets its own subprocess and per-instance env scope.  "
        "Empty (the default) means the worker boots no connector children.",
    )
    connectors_dir: Path = Field(
        default=Path.home() / ".aios" / "connectors",
        description="Per-connector working-directory root. The supervisor cd's "
        "into ``<connectors_dir>/<connector>/`` for default-instance setups "
        "(``instance == connector``) and ``<connectors_dir>/<connector>/<instance>/`` "
        "for non-default instances, so spool databases and other state files "
        "live next to each connector subprocess.",
    )

    @field_validator("connectors_enabled")
    @classmethod
    def _validate_connectors_enabled(cls, value: list[str]) -> list[str]:
        """Reject malformed entries and duplicate ``(connector, instance)`` pairs.

        Validation runs ``parse_connector_entry`` for each value to surface
        bad characters early; the parsed list is recomputed lazily in
        :meth:`connector_instances` rather than stored, so the public
        attribute stays a plain ``list[str]`` for env-CSV ergonomics.
        """
        seen: set[tuple[str, str]] = set()
        for raw in value:
            parsed = parse_connector_entry(raw)
            key = (parsed.connector, parsed.instance)
            if key in seen:
                raise ValueError(
                    f"duplicate connector instance {parsed.connector}:{parsed.instance} "
                    f"in connectors_enabled"
                )
            seen.add(key)
        return value

    def connector_instances(self) -> list[ConnectorInstance]:
        """Return ``connectors_enabled`` parsed into ``(connector, instance)`` pairs.

        Cheap to recompute (one regex per entry); not cached because
        ``Settings`` instances are themselves cached at the module level.
        """
        return [parse_connector_entry(raw) for raw in self.connectors_enabled]

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
