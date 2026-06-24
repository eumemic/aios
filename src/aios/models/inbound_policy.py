"""Per-connection inbound admission policy — the connector *human* inbound plane.

A connection carries an ``inbound_policy``: a discriminated union over ``kind``
that answers the single question "is this sender allowed to talk to this agent."
The connector inbound path was historically **fail-open** — every message a
connector daemon posted for a bound connection was appended to a session and
woke the model, with no predicate gating the sender. This module supplies the
predicate's data model; the gate lives in :func:`aios.services.inbound.handle_inbound`.

Growth rule (mirrors triggers' ``source`` / ``action`` unions): a new admission
behavior is always a new ``kind``, never a flag on an existing one. The illegal
state "``AllowAll`` carrying a chat-id list" is unrepresentable by construction.

Members
-------
* ``AllowAll`` — explicit "anyone may talk" acknowledgement.
* ``AllowList`` — only the listed ``chat_ids`` are admitted; an empty list is
  rejected at validation time (422), never a silent deny-all.
* ``DenyAll`` — explicit fail-closed; the **server default** when the column is
  NULL (resolved in :mod:`aios.db.queries.inbound_policy`).

The ``InboundPolicyReplace`` union is the required-on-update variant: a partial
update that omits ``AllowList.chat_ids`` must 422 rather than silently widening.
``AllowList.chat_ids`` is already required (no default), so the Replace union is
structurally identical here, but it is named separately so the operator-surface
PR can pin update semantics without re-deriving it. (Mirrors
``TriggerSource`` / ``TriggerSourceReplace`` in :mod:`aios.models.triggers`.)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class AllowAll(BaseModel):
    """Explicit "anyone may talk to this agent" acknowledgement."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["allow_all"] = "allow_all"


class AllowList(BaseModel):
    """Admit only the enumerated ``chat_ids``.

    ``chat_ids`` is required and ``min_length=1`` — an empty list is a
    validation error (422), never a silent deny-all. (Use ``DenyAll`` to
    deny everyone explicitly.)
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["allow_list"] = "allow_list"
    chat_ids: list[str] = Field(min_length=1)


class DenyAll(BaseModel):
    """Explicit fail-closed — admit no one. The server default."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["deny_all"] = "deny_all"


InboundPolicy = Annotated[
    AllowAll | AllowList | DenyAll,
    Field(discriminator="kind"),
]
"""Discriminated union over ``kind`` — the stored / read-model shape."""

InboundPolicyReplace = Annotated[
    AllowAll | AllowList | DenyAll,
    Field(discriminator="kind"),
]
"""Required-on-update variant. Structurally identical to ``InboundPolicy``
today (``AllowList.chat_ids`` is already required, so a partial update that
omits it 422s), named separately so the operator-surface PR can pin update
semantics without re-deriving it. Defined here but not yet wired to any
endpoint — see the issue's Out-of-scope note."""
