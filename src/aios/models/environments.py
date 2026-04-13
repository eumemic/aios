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
_HOSTNAME_RE = re.compile(
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
    # TODO: resolve MCP server hosts when MCP config is available
    allow_mcp_servers: bool = False

    @field_validator("allowed_hosts", mode="after")
    @classmethod
    def _validate_hosts(cls, v: list[str]) -> list[str]:
        for host in v:
            if not host:
                raise ValueError("allowed_hosts entries must not be empty")
            if len(host) > 253:
                raise ValueError(f"hostname too long ({len(host)} > 253): {host!r}")
            if not _HOSTNAME_RE.match(host):
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


class EnvironmentConfig(BaseModel):
    """Container configuration for an environment."""

    model_config = ConfigDict(extra="forbid")

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
