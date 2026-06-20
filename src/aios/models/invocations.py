"""The API caller's request-*writer* + the one completion *awaiter* (#1128).

``POST /v1/invocations`` is the one kind-agnostic request-writer for an
external/operator caller: it materializes the trusted request edge (#1123) and
constructs-or-resolves a servicer (a session or a run), returning a **structured
handle**. The ephemeral caller then awaits that handle via the one unified
awaiter ``GET /v1/invocations/{task_id}/await`` — both servicer kinds, one
:class:`AwaitResponse` envelope.

The handle is explicitly **not an auth boundary** — ``await`` re-authorizes by
``account_id`` — so it ships as plain JSON fields, never an opaque/encoded
string.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The caller-kind written onto the request edge for the HTTP/operator path.
# Generalizes the run-only ``{kind:"run", id:<run_id>}`` provenance to the
# external client (#1123 / #1128).
API_CALLER_KIND = "api"


class InvocationRequest(BaseModel):
    """Request body for ``POST /v1/invocations`` — the API caller's request-writer.

    ``target`` is an ``agent_id | workflow_id | session_id`` and ``target_kind``
    discriminates it:

    * ``agent``     — create a **session** servicer and inject a channel-less
      request into it (the API analog of ``invoke_agent``).
    * ``workflow``  — create a **run** servicer of the workflow.
    * ``session``   — invoke an **existing** same-account session by id (the API
      analog of #1127's ``invoke(session_id)``). No ``environment_id`` applies —
      the session already exists.

    ``output_schema`` is the per-request JSON Schema the response ``value`` must
    satisfy; it rides the edge (``metadata.request.output_schema``), coexisting
    with any definition-level schema. ``environment_id`` is ownership-checked
    against the caller's account on the ``agent`` / ``workflow`` create-paths
    (the per-field containment clamp is #1130's deliverable).
    """

    model_config = ConfigDict(extra="forbid")

    target_kind: Literal["agent", "workflow", "session"]
    target: str = Field(
        description="An agent_id / workflow_id / session_id, discriminated by ``target_kind``.",
    )
    input: Any = Field(
        default=None,
        description="The request payload delivered to the servicer (arbitrary JSON or a string).",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional JSON Schema the response ``value`` must satisfy. Rides the "
            "request edge, per-request — coexists with any definition-level schema."
        ),
    )
    environment_id: str | None = Field(
        default=None,
        description=(
            "Environment to bind a created servicer to (agent → session, "
            "workflow → run). Ownership-checked against the caller's account. "
            "Inapplicable for ``target_kind=session`` (the session already exists)."
        ),
    )


class InvocationHandle(BaseModel):
    """Structured handle returned by ``POST /v1/invocations``.

    Plain JSON fields, no opaque encoding: the handle is **not** an auth boundary
    (``await`` re-authorizes by ``account_id``). Await the invocation at the one
    unified awaiter ``GET /v1/invocations/{servicer_id}/await`` — the ``task_id``
    path segment is the ``servicer_id`` and its kind is read off the id prefix; a
    ``session`` servicer additionally needs ``?request_id=`` to correlate the
    response, a ``run`` servicer resolves off its terminal row.
    """

    servicer_kind: Literal["session", "run"]
    servicer_id: str
    request_id: str


class AwaitResponse(BaseModel):
    """The one completion envelope — ``GET /v1/invocations/{task_id}/await``.

    Unifies the session and run completion long-polls. ``outcome`` is the
    terminal state minus liveness (the trace's ``TerminalState`` with
    ``suspended``/``running`` folded into pending): ``None`` means **still
    pending** — the long-poll timed out before the invocation reached a terminal
    state, so re-poll. ``result`` carries the servicer's return value on ``ok``;
    ``error`` carries the ``{kind, message, …}`` detail on ``errored`` /
    ``cancelled``.
    """

    outcome: Literal["ok", "errored", "cancelled"] | None = Field(
        default=None,
        description=(
            "The invocation's terminal outcome, or null while it is still pending "
            "(the long-poll timed out — call again to keep blocking)."
        ),
    )
    result: Any = Field(
        default=None,
        description="The servicer's return value when outcome=='ok'; null otherwise.",
    )
    error: dict[str, Any] | None = Field(
        default=None,
        description="On outcome 'errored'/'cancelled', the {kind, message, …} detail; null otherwise.",
    )


# ─── cancel supervision side-table rows (cancel-design §0/§9) ─────────────────


class CancelIntent(BaseModel):
    """The durable cancel **tombstone** — one row per ``cancel_invocation`` call, keyed by
    the cancelled edge handle ``(servicer_kind, servicer_id, request_id)``.

    Written first, independent of cascade progress. ``outstanding`` is the §9 monotone
    quiescence counter (seed 1; each node adjusts it in its own terminal/withdraw txn);
    ``quiesced_at`` is set the instant it reaches 0 — the cascade is then complete.
    """

    servicer_kind: Literal["session", "run"]
    servicer_id: str
    request_id: str
    account_id: str
    outstanding: int
    quiesced_at: datetime | None = None
    created_at: datetime


class SessionCancelMarker(BaseModel):
    """A session-side **exit-marker** — a durable cancel signal the target session's own
    step harvests under its lock (the run side reuses ``wf_run_signals kind='cancel'``).

    Keyed by the target edge ``(session_id, request_id)``; ``harvested_at`` flips once the
    session's step has applied it, so the sweep wakes only still-unharvested markers (C2).
    """

    session_id: str
    request_id: str
    account_id: str
    harvested_at: datetime | None = None
    created_at: datetime
