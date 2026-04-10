"""Environment resource: a sandbox configuration template.

In v1 the environment is essentially just a name; aios uses one fixed Docker
image. The resource exists so the schema is forward-compatible with later
phases that add per-environment image, package, and networking config.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentCreate(BaseModel):
    """Request body for `POST /v1/environments`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)


class Environment(BaseModel):
    """Read view of an environment."""

    id: str
    name: str
    created_at: datetime
    archived_at: datetime | None = None
