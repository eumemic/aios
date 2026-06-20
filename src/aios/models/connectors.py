"""Typed connector-capability descriptor — a ``tools_schema`` sibling.

The connector subsystem renders the SAME externally-executed dispatch (an
``always_ask`` tool-confirmation, an in-progress assistant delta) differently
per platform: Slack can show inline buttons and edit a draft message; a plain
text channel can only post a prompt and wait for the committed message.  This
module is the *declared, typed channel* through which a connector advertises
which richer renderings it supports, so shared rendering code can branch on a
declared **KIND** (``capabilities.draft_streaming is not None``) instead of a
``connector == 'slack'`` identity shim.

**Variation is a KIND, not a flag.**  Each capability is a present/absent
typed sub-object — never a boolean.  ``supports_draft_streaming: bool`` is
rejected in favour of ``draft_streaming: DraftStreaming | None``: the
present/absent object subsumes the boolean and binds its parameters into the
same KIND, so an illegal combo (capability "on" but its parameters missing) is
unrepresentable rather than runtime-guarded.

This descriptor sits as a sibling to the connector-type catalog's existing
``tools_schema`` (``connectors.capabilities jsonb`` — migration 0109), is
published on the same root-only path, and is read alongside the same
per-session query.  It declares NO authority: capabilities constrain RENDERING
(how a surface displays), never what any principal may invoke, so a capability
descriptor is never consulted in an authorization decision.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NativeButtons(BaseModel):
    """Present == connector renders externally-executed tool-confirmations as
    inbound platform buttons; a button-tap maps to the existing
    ``POST /sessions/:id/tool-confirmations``.  Absent == text-prompt fallback.
    """

    model_config = ConfigDict(extra="forbid")

    max_buttons: int  # platform cap (Slack 5, Telegram rows, …)


class DraftStreaming(BaseModel):
    """Present == connector can render in-progress assistant deltas as an
    editable draft.  Absent == connector waits for the committed message.
    """

    model_config = ConfigDict(extra="forbid")

    overflow_limit: int | None = None  # chars before truncate/finalize; None == unbounded


class ConnectorCapabilities(BaseModel):
    """Typed richness descriptor — a ``tools_schema`` sibling on the catalog
    row.  Each field is a present/absent typed sub-object (a declared KIND),
    never a bool flag.  An absent field == capability not declared == the
    conservative rendering floor.
    """

    model_config = ConfigDict(extra="forbid")

    draft_streaming: DraftStreaming | None = None
    native_buttons: NativeButtons | None = None
