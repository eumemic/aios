"""Routing rule resource.

A rule maps a *prefix* of channel addresses onto a *target*: either an
agent (``agent:<id>[@<version>]`` — auto-creates a session at resolve
time) or an existing session (``session:<id>``).

When an inbound message arrives and no explicit binding exists, the
resolver picks the longest matching prefix.  The match is segment-aware:
``signal`` matches ``signal/abc`` but not ``signalfoo``.

``session_params`` carries the args the resolver passes to
``queries.insert_session`` for ``agent:`` targets.  It is meaningless and
must be empty for ``session:`` targets — service-level validation enforces
this at create/update.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionParams(BaseModel):
    """Args used to spin up a fresh session for an ``agent:`` target.

    ``title`` may contain ``{address}`` which the resolver substitutes
    with the matched channel address.
    """

    model_config = ConfigDict(extra="forbid")

    environment_id: str | None = None
    vault_ids: list[str] = Field(default_factory=list)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingRuleCreate(BaseModel):
    """Request body for ``POST /v1/routing-rules``."""

    model_config = ConfigDict(extra="forbid")

    prefix: str = Field(min_length=1)
    target: str = Field(min_length=1)
    session_params: SessionParams = Field(default_factory=SessionParams)


class RoutingRuleUpdate(BaseModel):
    """Request body for ``PUT /v1/routing-rules/{id}``.

    ``prefix`` is immutable after creation.
    """

    model_config = ConfigDict(extra="forbid")

    target: str | None = Field(default=None, min_length=1)
    session_params: SessionParams | None = None


class RoutingRule(BaseModel):
    """Read view of a routing rule."""

    id: str
    prefix: str
    target: str
    session_params: SessionParams
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
