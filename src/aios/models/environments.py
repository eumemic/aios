"""Environment resource: a sandbox configuration template.

Environments configure the container each session runs in: pre-installed
packages, network access rules, and (later) custom base images. The
``config`` field stores this as JSONB.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

# ── networking config ─────────────────────────────────────────────────────────

# Hostname: RFC 952 / RFC 1123 labels joined by dots.  Only characters that
# are safe to embed in a shell script (no metacharacters, no slashes).
# Public so vault env-var credentials (``models/vaults.py``) validate their
# host allowlists against the same grammar the sandbox networking layer uses.
HOSTNAME_RE = re.compile(
    r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*$"
)


class UnrestrictedNetworking(BaseModel):
    """Full outbound network access (default)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["unrestricted"] = "unrestricted"


class LimitedNetworking(BaseModel):
    """Deny-all with domain allowlist.

    Outbound HTTP/HTTPS is restricted to ``allowed_hosts`` plus any hosts
    implied by the boolean flags.  DNS (port 53) remains open so tools
    like ``curl`` can resolve names.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["limited"]
    allowed_hosts: list[str] = Field(default_factory=list)
    allow_package_managers: bool = False

    @field_validator("allowed_hosts", mode="after")
    @classmethod
    def _validate_hosts(cls, v: list[str]) -> list[str]:
        for host in v:
            if not host:
                raise ValueError("allowed_hosts entries must not be empty")
            if len(host) > 253:
                raise ValueError(f"hostname too long ({len(host)} > 253): {host!r}")
            # fullmatch, not match: a trailing ``$`` lets ``re.match`` accept a
            # single trailing newline ("host\n"), which would then be embedded
            # verbatim into the iptables lockdown script (sandbox/setup.py).
            if not HOSTNAME_RE.fullmatch(host):
                raise ValueError(
                    f"invalid hostname {host!r}: only alphanumerics, hyphens, and dots allowed"
                )
        return v


def _networking_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("type", "unrestricted"))
    return str(getattr(v, "type", "unrestricted"))


NetworkingConfig = Annotated[
    Annotated[UnrestrictedNetworking, Tag("unrestricted")]
    | Annotated[LimitedNetworking, Tag("limited")],
    Discriminator(_networking_discriminator),
]


# ── environment config ────────────────────────────────────────────────────────

# Reserved prefix for per-session snapshot images (durable session sandboxes,
# §5.8). Snapshot tags are ``aios-sbx-{instance}-{session}:latest`` — single
# path component, so they resolve locally with no registry pull. Because
# persistence makes another session's snapshot mountable via a free-form
# ``image`` string, a tenant-supplied image starting with this prefix would
# mount a victim session's entire root FS (and its baked secrets); ULIDs are
# not secrets (they appear in URLs/logs). The cross-tenant gate rejects it at
# two layers covering different writer sets: this pydantic validator (422 at
# the API boundary, UX + window-shrinking) and the spec-build choke point all
# writers traverse including direct SQL (``build_spec_from_session``).
RESERVED_SANDBOX_IMAGE_PREFIX = "aios-sbx-"


class EnvironmentConfig(BaseModel):
    """Container configuration for an environment."""

    model_config = ConfigDict(extra="forbid")

    image: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description=(
            "Container image for sessions bound to this environment. When "
            "unset, sessions provision from the worker's global "
            "``settings.docker_image``. Lets a purpose-built environment "
            "(e.g. an autodev dev image with toolchains baked in) pin its "
            "own image without changing the image every other session on "
            "the shared worker uses (issue #724). Accepts any reference "
            "the worker's Docker daemon can resolve: a registry image "
            "(``ghcr.io/eumemic/aios-sandbox:pinned``) or a bare local "
            "tag for development."
        ),
    )
    packages: dict[str, list[str]] | None = Field(
        default=None,
        description='Package manager → package list, e.g. {"pip": ["pandas"], "npm": ["express"]}.',
    )
    networking: NetworkingConfig | None = Field(
        default=None,
        description=(
            'Network access rules.  None or {"type": "unrestricted"} for full '
            'access; {"type": "limited", "allowed_hosts": [...]} to restrict.'
        ),
    )
    env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Environment variables injected into every session container "
            "using this environment.  Per-session env overrides these. A "
            "vaulted environment_variable credential whose secret_name "
            "matches a key — here or in the per-session env — outranks "
            "both: that key resolves to the credential's opaque "
            "placeholder, not the value set here."
        ),
    )
    snapshot_budget_bytes: int | None = Field(
        default=None,
        ge=10 * 1024 * 1024,
        description=(
            "Per-session snapshot budget in **unique** bytes for sessions "
            "bound to this environment (durable session sandboxes). When "
            "unset, falls back to the worker's global "
            "``settings.sandbox_snapshot_budget_bytes`` (4 GiB). When a "
            "session's unique snapshot bytes would exceed this at teardown, "
            "the snapshot flattens (collapse + whiteout) instead of growing "
            "another layer — commit-and-flag, never a refusal. Replaces the "
            "former ``disk_bytes`` writable-layer cap, which required "
            "overlay2+pquota and never worked on prod ext4. Minimum 10 MiB."
        ),
    )
    bash_timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Ceiling, in seconds, for a single bash tool call in sessions "
            "bound to this environment. When unset, falls back to the "
            "worker's global ``settings.bash_default_timeout_seconds`` "
            "(120s). Lets heavy dev workloads run >120s commands without "
            "raising the global default for every session on the worker. "
            "The agent can still request a shorter per-call timeout; this "
            "is the maximum it is capped to (issue #725)."
        ),
    )

    @field_validator("image", mode="after")
    @classmethod
    def _reject_reserved_image_prefix(cls, v: str | None) -> str | None:
        """Reject the reserved per-session-snapshot image prefix (422).

        Persistence makes another session's snapshot mountable via this
        free-form string; the prefix is reserved so a tenant can't point an
        environment at ``aios-sbx-<victim>`` and mount its root FS + baked
        secrets. Case-insensitive (Docker lowercases image refs). This is the
        UX/window-shrinking layer; ``build_spec_from_session`` is the boundary
        gate that also catches direct-SQL writers.
        """
        if v is not None and v.lower().startswith(RESERVED_SANDBOX_IMAGE_PREFIX):
            raise ValueError(
                f"image may not use the reserved {RESERVED_SANDBOX_IMAGE_PREFIX!r} prefix "
                "(reserved for per-session snapshot images)"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def _normalize_networking(cls, data: Any) -> Any:
        """Treat ``networking: {}`` from legacy DB rows as None (unrestricted)."""
        if isinstance(data, dict) and data.get("networking") == {}:
            data = {**data, "networking": None}
        return data


class EnvironmentCreate(BaseModel):
    """Request body for `POST /v1/environments`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


class EnvironmentUpdate(BaseModel):
    """Request body for ``PUT /v1/environments/{id}``.

    All fields are optional; omitted fields are preserved.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)
    config: EnvironmentConfig | None = None


class Environment(BaseModel):
    """Read view of an environment."""

    id: str
    name: str
    config: EnvironmentConfig
    created_at: datetime
    archived_at: datetime | None = None
