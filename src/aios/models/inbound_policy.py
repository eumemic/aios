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

from pydantic import BaseModel, ConfigDict, Field, RootModel


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


class InboundPolicyReplace(RootModel[InboundPolicy]):
    """Wire wrapper for ``PUT /v1/connections/{id}/inbound-policy``.

    A ``RootModel`` over the :data:`InboundPolicy` discriminated union so the
    request body is the bare ``{"kind": ..., "chat_ids"?: [...]}`` shape (no
    envelope key) and the validated member is reachable as ``body.root``.

    **Replace, not Patch.** This is the *required-on-update* variant:
    ``AllowList.chat_ids`` is required with ``min_length=1``, so a partial
    body ``{"kind": "allow_list"}`` (no ``chat_ids``) 422s rather than
    silently widening to an unbounded allow-everyone or re-defaulting, and an
    empty ``{"kind": "allow_list", "chat_ids": []}`` 422s at the write edge.
    ``DenyAll`` / ``AllowAll`` bodies carry no ``chat_ids`` and are accepted.
    An unknown or missing ``kind`` 422s via the discriminated-union
    validation plus each member's ``extra="forbid"``. (Mirrors
    ``TriggerSourceReplace`` in :mod:`aios.models.triggers`.)

    Revocation (§9) is expressed as a Replace with the smaller list — there
    is no separate patch shape.
    """
