"""Connection resource — the unified routing primitive.

A *connection* is a registered ``(connector, account)`` pair plus an
optional routing-mode binding.  The schema enforces three valid shapes
via :sql:`connections_one_mode_ck`:

* **detached** — ``session_id`` and ``session_template_id`` are both
  NULL.  Inbound messages drop with a counter increment.
* **single_session** — ``session_id`` populated.  Every inbound for this
  account appends to that one session.
* **per_chat** — ``session_template_id`` populated.  Each new chat
  partner spawns a fresh session via the template; the ``chat_id`` →
  ``session_id`` map lives in ``connection_chat_sessions``.

The active-row uniqueness on ``(connector, account)`` enforces
"one session per account" by schema — operators can't accidentally
double-bind a phone number to two sessions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ConnectionMode = Literal["detached", "single_session", "per_chat"]


class ConnectionCreate(BaseModel):
    """Request body for ``POST /v1/connections``.

    Created in detached mode — neither ``session_id`` nor
    ``session_template_id`` is set.  Use ``POST .../attach`` or
    ``POST .../configure-per-chat`` afterward to bind a routing mode.

    ``connector`` and ``account`` may not contain ``/`` — they're used
    in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
    and a ``/`` would create ambiguous segment boundaries.
    """

    model_config = ConfigDict(extra="forbid")

    connector: str = Field(min_length=1, max_length=64)
    account: str = Field(min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("connector", "account")
    @classmethod
    def _no_slash(cls, v: str) -> str:
        if "/" in v:
            raise ValueError("must not contain '/'")
        return v


class ConnectionAttach(BaseModel):
    """Request body for ``POST /v1/connections/{id}/attach``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str


class ConnectionConfigurePerChat(BaseModel):
    """Request body for ``POST /v1/connections/{id}/configure-per-chat``."""

    model_config = ConfigDict(extra="forbid")

    session_template_id: str


class Connection(BaseModel):
    """Read view of a connection.

    Mode is implicit in the populated field:

    * ``session_id`` set → single_session
    * ``session_template_id`` set → per_chat
    * neither → detached
    """

    id: str
    connector: str
    account: str
    session_id: str | None = None
    session_template_id: str | None = None
    metadata: dict[str, Any]
    created_at: datetime
    attached_at: datetime | None = None
    updated_at: datetime
    archived_at: datetime | None = None
