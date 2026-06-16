"""The trace normalizer (#1149 / #1140) — pure functions, no I/O.

``terminal_state ∈ {ok, errored, cancelled, suspended, running}`` plus
``error_kind: str | None`` carrying the **raw** discriminant verbatim. The rule
unifies both servicer kinds (session, run) by **reusing #1126's resolver**
(``derive_response`` / ``derive_run_response``): the walk fetches the resolved
response dict and hands it here, so terminal-ness is never re-derived
independently.

These functions are deliberately I/O-free so the (DB-mocked) unit tier can pin
the full truth table without a live Postgres.
"""

from __future__ import annotations

from typing import Any

from aios.models.trace import TerminalState

# Run statuses are persisted on ``wf_runs.status``; sessions have no status
# column (their lifecycle is derived from ``stop_reason``).
_TERMINAL_RUN_STATUSES = {"completed", "errored", "cancelled"}


def normalize_response(response: dict[str, Any] | None) -> tuple[TerminalState, str | None]:
    """Normalize a **servicer** node's caller's-eye outcome from a resolved response.

    ``response`` is the dict ``derive_response`` / ``derive_run_response``
    returns (or ``None`` when still pending):

    * ``None`` → still running (alive and unanswered). Note: an *absent* response
      means running — it is **never** ``no_return``. ``no_return`` is a
      *present* ``request_response`` with ``error.kind == 'no_return'`` (written
      by the quiescence guard), so it lands in the ``is_error`` branch below.
    * ``is_error`` → ``errored`` + the raw ``error.kind`` (``no_return``,
      ``child_gone``, …) passed through verbatim.
    * otherwise → ``ok``.
    """
    if response is None:
        return "running", None
    if response.get("is_error"):
        error = response.get("error") or {}
        kind = error.get("kind") if isinstance(error, dict) else None
        return "errored", kind
    return "ok", None


def normalize_run_root(
    *,
    status: str,
    run_completed_error: dict[str, Any] | None,
    run_completed_is_error: bool,
) -> tuple[TerminalState, str | None]:
    """Normalize a **root run** (no inbound request) from its own status.

    * ``completed`` → ``errored`` if the ``run_completed`` payload ``is_error``
      (``error_kind`` from ``run_completed.payload.error.kind`` — surfaces
      ``output_schema_violation``, ``author_exception``, ``script_host_timeout``,
      ``budget_exceeded``, …), else ``ok``.
    * ``errored`` → ``errored`` (+ raw ``error.kind`` when present).
    * ``cancelled`` → ``cancelled``.
    * ``suspended`` → ``suspended``.
    * ``pending`` / ``running`` (or anything else) → ``running``.
    """
    if status == "completed":
        if run_completed_is_error:
            return "errored", _kind_of(run_completed_error)
        return "ok", None
    if status == "errored":
        return "errored", _kind_of(run_completed_error)
    if status == "cancelled":
        return "cancelled", None
    if status == "suspended":
        return "suspended", None
    return "running", None


def normalize_session_root(
    stop_reason: dict[str, Any] | None,
    *,
    owes_open_request: bool,
    owed_request_response: dict[str, Any] | None = None,
    is_archived: bool = False,
) -> tuple[TerminalState, str | None]:
    """Normalize a **root session** (no inbound request) from ``stop_reason.type``.

    A session has no status column and no hard terminal flag, so the state is
    derived from the most-recent step's ``stop_reason``:

    * ``error`` → ``errored`` (``error_kind`` from the reason's
      ``error.kind`` / ``finish_reason`` when present).
    * ``interrupt`` → ``cancelled``.
    * ``rescheduling`` → ``running`` (a model-error backoff is still live).
    * ``end_turn`` → ``ok`` if it owes no open request, else ``running``.
    * an archived session with no stop_reason resolves its owed request
      (``child_gone`` → ``errored + child_gone``) else ``ok``.
    """
    reason_type = (stop_reason or {}).get("type")
    if reason_type == "error":
        return "errored", _session_error_kind(stop_reason)
    if reason_type == "interrupt":
        return "cancelled", None
    if reason_type == "rescheduling":
        return "running", None
    if reason_type == "end_turn":
        return ("running", None) if owes_open_request else ("ok", None)
    # No (or unknown) stop_reason. An archived session can never run again, so
    # resolve any owed request; a live one is simply still running.
    if is_archived:
        if owed_request_response is not None and owed_request_response.get("is_error"):
            return "errored", _kind_of(owed_request_response.get("error"))
        return "ok", None
    return "running", None


def _kind_of(error: dict[str, Any] | None) -> str | None:
    if isinstance(error, dict):
        kind = error.get("kind")
        return kind if isinstance(kind, str) else None
    return None


def _session_error_kind(stop_reason: dict[str, Any] | None) -> str | None:
    """Best-effort raw discriminant for an errored session's stop_reason.

    The harness stores ``{type:'error', message?, finish_reason?}`` and, for a
    child whose turn errored, fails its open requests with a structured
    ``error.kind`` separately. Prefer an explicit ``error.kind`` if the reason
    carries one, then ``finish_reason`` (e.g. ``content_filter``), so the raw
    discriminant is preserved verbatim rather than flattened to ``errored``.
    """
    if not isinstance(stop_reason, dict):
        return None
    error = stop_reason.get("error")
    if isinstance(error, dict):
        kind = error.get("kind")
        if isinstance(kind, str):
            return kind
    finish = stop_reason.get("finish_reason")
    return finish if isinstance(finish, str) else None
