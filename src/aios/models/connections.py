"""Connection resource — the unified routing primitive.

A *connection* is a registered ``(connector, account)`` pair, optionally
attached to a routing target via the ``bindings`` table.  Three valid
shapes derived from the active binding:

* **detached** — no active binding row.  Inbound messages drop with a
  counter increment.
* **single_session** — active binding with ``mode='single_session'`` and
  ``session_id`` populated.  Every inbound for this account appends to
  that one session.
* **per_chat** — active binding with ``mode='per_chat'`` and
  ``session_template_id`` populated.  Each new chat partner spawns a
  fresh session via the template; the ``chat_id`` → ``session_id`` map
  lives in ``chat_sessions``.

The active-row uniqueness on ``(connector, account)`` enforces
"one session per account" by schema — operators can't accidentally
double-bind a phone number to two sessions.  The
``bindings_connection_active_uniq`` partial-unique index gives the same
guarantee at the binding level (at most one active binding per
connection).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aios.models.agents import ToolSpec

ConnectionMode = Literal["detached", "single_session", "per_chat"]

# A connection's *binding* exists only when curated — ``detached`` is "no
# binding row," not a binding mode.  This narrower alias is what the
# ``bindings`` table's ``mode`` column carries (enforced by the
# ``bindings_mode_ck`` CHECK constraint).
BindingMode = Literal["single_session", "per_chat"]


def _validate_connection_tools(tools: list[ToolSpec]) -> list[ToolSpec]:
    """Connection-declared tools must be ``type="custom"`` (#301).

    Connections expose model-facing tools that the connector executes
    externally via the ``requires_action`` flow.  Built-in tools live on
    the agent; ``mcp_toolset`` is for HTTP MCP servers, not connectors.
    """
    for t in tools:
        if t.type != "custom":
            raise ValueError(
                f"connection tools must be type='custom', got type={t.type!r}",
            )
    return tools


class ConnectionCreate(BaseModel):
    """Request body for ``POST /v1/connections``.

    Created in detached mode — neither ``session_id`` nor
    ``session_template_id`` is set.  Use ``POST .../attach`` or
    ``POST .../configure-per-chat`` afterward to bind a routing mode.

    ``connector`` and ``account`` may not contain ``/`` — they're used
    in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
    and a ``/`` would create ambiguous segment boundaries.

    ``tools`` declares the model-facing custom tools this connection
    contributes to any session it's attached to (see #301).
    """

    model_config = ConfigDict(extra="forbid")

    connector: str = Field(min_length=1, max_length=64)
    account: str = Field(min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tools: list[ToolSpec] = Field(default_factory=list)
    secrets: dict[str, str] | None = Field(
        default=None,
        description="Platform credentials (e.g. ``bot_token``).  Encrypted "
        "at rest via the server's ``AIOS_VAULT_KEY``; only ever read back "
        "via the connector-scoped ``GET /v1/connectors/secrets``.  "
        "Operator-facing reads return ``secrets_set: bool`` instead of values.",
    )

    @field_validator("connector", "account")
    @classmethod
    def _no_slash(cls, v: str) -> str:
        if "/" in v:
            raise ValueError("must not contain '/'")
        return v

    @field_validator("tools")
    @classmethod
    def _custom_only(cls, v: list[ToolSpec]) -> list[ToolSpec]:
        return _validate_connection_tools(v)


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

    ``session_id`` / ``session_template_id`` / ``attached_at`` are
    projected from the connection's active binding row at read time.
    ``attached_at`` is "when did the active binding land," not "when
    was this connection first attached," so detach+re-attach moves
    it forward — operator dashboards keying off the timestamp see
    that motion.

    Secrets are *write-only* on the operator surface — the model carries
    ``secrets_set: bool`` rather than the values themselves.  The only
    decryption path is the runtime-scoped
    ``GET /v1/connectors/runtime/secrets``, which returns the dict
    for a connection of the caller's connector type.
    """

    id: str
    connector: str
    account: str
    session_id: str | None = None
    session_template_id: str | None = None
    metadata: dict[str, Any]
    tools: list[ToolSpec] = Field(default_factory=list)
    secrets_set: bool = False
    created_at: datetime
    attached_at: datetime | None = None
    updated_at: datetime
    archived_at: datetime | None = None


class ConnectionSetSecrets(BaseModel):
    """Request body for ``PUT /v1/connections/{id}/secrets``.

    Replaces the connection's secrets dict wholesale.  Encrypted at
    rest server-side via ``AIOS_VAULT_KEY``; the operator never reads
    them back.

    Pass an empty dict to clear secrets.
    """

    model_config = ConfigDict(extra="forbid")

    secrets: dict[str, str] = Field(default_factory=dict)


class ConnectorSecrets(BaseModel):
    """Response shape for ``GET /v1/connectors/secrets``.

    Only the connector container's bearer token (which scopes to one
    ``connection_id``) can hit this route.  Returns the decrypted dict
    the operator stored at create / set-secrets time.  Empty dict when
    the connection has no secrets configured.
    """

    secrets: dict[str, str] = Field(default_factory=dict)


class BindChatRequest(BaseModel):
    """Request body for ``POST /v1/connections/{id}/bind-chat``.

    Pre-populates a ``chat_sessions`` row so inbound on ``chat_id``
    routes to ``session_id`` regardless of the connection's mode-default
    fallback (#215).  Operators use this to point different chats on a
    single account at different operator-curated existing sessions.
    """

    model_config = ConfigDict(extra="forbid")

    chat_id: str = Field(min_length=1, max_length=512)
    session_id: str


class BoundChat(BaseModel):
    """Read view of one ``chat_sessions`` row.

    Returned by ``GET /v1/connections/{id}/bound-chats``.  Operator-bound
    rows and per-chat-spawned rows are returned together — the table
    doesn't tag the writer.
    """

    chat_id: str
    session_id: str
    created_at: datetime


class RecentChat(BaseModel):
    """Distinct chat_id observed on a connection's account, with the
    most-recent inbound timestamp.

    Returned by ``GET /v1/connections/{id}/recent-chats`` so operators
    can find the chat_id for a specific peer without digging through
    event logs before calling ``bind-chat``.
    """

    chat_id: str
    last_seen_at: datetime
