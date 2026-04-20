"""Routing rule resource.

Rules are owned by a :class:`Connection` — the ``{connector}/{account}``
prefix of the full channel address is implicit from the owning connection.
A rule's ``prefix`` is the *path portion* (the segments after
``{connector}/{account}/``) that an inbound message's path must start with
for the rule to match; ``""`` is the per-connection catch-all.

When an inbound message arrives and no explicit binding exists, the
resolver evaluates rules for the owning connection and picks the longest
matching prefix.  The match is segment-aware: a rule of ``group`` matches
``group/thread-1`` but not ``groupchat``.

``session_params`` carries the args the resolver passes to
``queries.insert_session`` for ``agent:`` targets.  It is meaningless and
must be empty for ``session:`` targets — service-level validation enforces
this at create/update.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aios.models._paths import validate_path_segments


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
    """Request body for ``POST /v1/connections/{id}/routing-rules``.

    ``prefix`` is a path within the owning connection — everything after
    ``{connector}/{account}/`` in the full channel address.  ``""`` is the
    per-connection catch-all; any other value must be one or more
    ``/``-joined non-empty segments with no ``..``.
    """

    model_config = ConfigDict(extra="forbid")

    prefix: str
    target: str = Field(min_length=1)
    session_params: SessionParams = Field(default_factory=SessionParams)

    @field_validator("prefix")
    @classmethod
    def _valid_prefix(cls, v: str) -> str:
        validate_path_segments(v, allow_empty=True)
        return v


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
    connection_id: str
    prefix: str
    target: str
    session_params: SessionParams
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
