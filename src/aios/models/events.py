"""Event resource: append-only entries on the session log.

Events come in four kinds, distinguished by `kind`:

* ``message`` — a chat-completions message dict (whatever LiteLLM returns,
  stored opaquely so reasoning_content / thinking_blocks come along for free)
* ``lifecycle`` — session state transitions (turn started/ended, status
  changes, stop_reason)
* ``span`` — observability markers around model calls and tool calls
* ``interrupt`` — user-issued cancel signal

The `data` field is intentionally opaque (`dict[str, Any]`) so we don't
over-validate at the boundary. Per-kind shapes are documented but not
enforced via pydantic discriminated unions, because the message kind in
particular has to round-trip arbitrary LiteLLM extensions without rejecting
them.

Schema note (#1140): a (child-)*session* event is ``{kind, data}`` (this
model). A *run* event is the DIFFERENT shape ``{type, payload, seq}`` (see
``aios.models.workflows.WfRunEvent``). Consumers watching both surfaces must
not assume a single schema; ``docs/reference/run-observability.md`` documents
the split (and, for ``kind == "message"``, the ``data.role`` ∈
user/assistant/tool axis).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal["message", "lifecycle", "span", "interrupt"]

# Durable-session-sandbox FS-loss notices (§5.9): the only non-``message``
# events the context builder renders, and so the only non-``message`` events
# the windowing read must carry through to it. Defined here, the lowest layer,
# so both the harness renderer (``harness/context.py``) and the db read
# (``db/queries/events.py``: ``read_windowed_context_events``) can name the same
# allowlist without the db layer importing the harness.
# ``connector_delivery_failed`` (#1261): a connector-generic, NON-SMS-special
# kind a connector appends (via the session-targeted lifecycle route) when an
# outbound the model *consciously sent* did not arrive (carrier block / delivery
# failure). Unlike the FS-loss notices it is paired at the producer with a wake
# (the session-targeted lifecycle route's ``wake=True``), so the failure reaches
# the originating session in a form the model acts on; the connector carries the
# carrier-specific detail in ``data`` so core stays transport-agnostic.
MODEL_VISIBLE_LIFECYCLE_EVENTS: frozenset[str] = frozenset(
    {
        "sandbox_fs_reset",
        "sandbox_fs_expired",
        "sandbox_fs_over_limit",
        "connector_delivery_failed",
        # ``connector_message_delivered`` / ``connector_message_edited`` (#1341):
        # the success-path complement to ``connector_delivery_failed``. A
        # connector appends one of these (via the session-/chat-lifecycle route)
        # when the platform confirmed an outbound the model consciously sent was
        # delivered, or that an edit landed. Informational (non-stimulus-bearing,
        # ``wake=False``): the model reads the ack on its next turn. Connector
        # specifics (``platform_message_id``/``tool_call_id``) ride in ``data``
        # so core stays transport-agnostic.
        "connector_message_delivered",
        "connector_message_edited",
    }
)


# ── Error-latch lifecycle vocabulary (#1084) ──────────────────────────────
# The error latch (``harness/loop.py``) writes a ``turn_ended`` lifecycle event
# with this exact ``stop_reason`` string; ``append_event``
# (``db/queries/events.py``) reads the SAME constant to recognize the event and
# bump ``last_error_seq`` (which drives ``_SESSION_ERRORED_EXPR`` and so the
# sweep's errored-session park). The value round-trips writer → ``json.dumps``
# → Postgres JSONB → asyncpg → ``dict[str, Any]``, so the read is type ``Any``
# and the checker CANNOT bind the write literal to the read literal. Two free
# strings on either side of that ``Any`` boundary type-check and pass CI even
# when they DIVERGE — flip one (e.g. the tempting ``"errored"`` transposition
# to match ``ERRORED_LIFECYCLE_STATUS``) and the park silently breaks: the
# session busy-wakes forever with no progress (#155 class). Routing BOTH sides
# through this single ``Literal`` constant is the binding; the coupling is
# pinned by ``tests/unit/test_errored_lifecycle_coupling.py`` (the floor), which
# the ``Any`` read can otherwise evaporate.
ERRORED_LIFECYCLE_STOP_REASON: Literal["error"] = "error"
ERRORED_LIFECYCLE_STATUS: Literal["errored"] = "errored"


def is_errored_lifecycle_event(kind: str, data: dict[str, Any]) -> bool:
    """The errored-park predicate: does this lifecycle event latch the session
    into ``errored``?

    ``data`` is the JSONB-round-tripped event payload (``dict[str, Any]``), so
    the ``stop_reason`` it carries is type ``Any`` — the checker cannot bind it
    to the writer's literal. This helper is the SINGLE read site
    (``db/queries/events.py:append_event`` calls it to bump ``last_error_seq``)
    and reads the SAME ``ERRORED_LIFECYCLE_STOP_REASON`` the latch writes, so the
    write↔read coupling is one constant rather than two free strings (#1084).
    """
    return kind == "lifecycle" and data.get("stop_reason") == ERRORED_LIFECYCLE_STOP_REASON


class Event(BaseModel):
    """Read view of a single event from the session log.

    Schema (#1140): a session event is ``{kind, data, seq}`` — a DIFFERENT
    shape from a *run* event (``{type, payload, seq}``, see
    ``aios.models.workflows.WfRunEvent``). See module docstring and
    ``docs/reference/run-observability.md`` for the split.
    """

    id: str
    session_id: str
    seq: int
    kind: EventKind
    data: dict[str, Any]
    cumulative_tokens: int | None = Field(default=None, exclude=True)
    created_at: datetime
    # Un-hidden (#1613) so LIST consumers can distinguish "where the inbound
    # came from" from the resolved turn ``channel`` — parity with the SSE
    # serializer, which already emits ``orig_channel``.
    orig_channel: str | None = None
    focal_channel_at_arrival: str | None = Field(default=None, exclude=True)
    # Derived "which channel does this event belong to?" — stamped at
    # append time. For user events, == orig_channel; for assistant
    # events, == focal_channel_at_arrival; for tool events, == the
    # parent assistant's focal_channel_at_arrival (so a tool call
    # started in A and completing after a switch to B still belongs to
    # A). NULL for non-message events and for events that belong to no
    # channel (e.g. assistant emitted while focal was cleared).
    #
    # Un-hidden (#1613): the LIST API now emits ``channel`` so historical
    # consumers (relay/cockpit/audit) can read the authoritative per-event
    # channel instead of reconstructing it by heuristic focal-tracking.
    channel: str | None = None
