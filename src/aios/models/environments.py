"""Environment resource: a sandbox configuration template.

Environments configure the container each session runs in: pre-installed
packages, network access rules, and (later) custom base images. The
``config`` field stores this as JSONB.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentConfig(BaseModel):
    """Container configuration for an environment."""

    model_config = ConfigDict(extra="forbid")

    packages: dict[str, list[str]] | None = Field(
        default=None,
        description='Package manager → package list, e.g. {"pip": ["pandas"], "npm": ["express"]}.',
    )
    networking: dict[str, Any] | None = Field(
        default=None,
        description='Network access rules, e.g. {"type": "unrestricted"}.',
    )


class EnvironmentCreate(BaseModel):
    """Request body for `POST /v1/environments`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    config: EnvironmentConfig = Field(default_factory=EnvironmentConfig)


class Environment(BaseModel):
    """Read view of an environment."""

    id: str
    name: str
    config: EnvironmentConfig
    created_at: datetime
    archived_at: datetime | None = None
