"""Per-counterparty inbound rate/cost budget (Fast-follow B, #1504).

The inbound-admission gate (#1500) decides *whether* a counterparty may
talk to the agent; this module bounds *how much*. Once admitted, every
inbound message a sender posts still costs exactly one model inference
(``handle_inbound`` appends a ``role=user`` event and ``defer_wake``s a
``wake_session`` job → one ``run_session_step``). An admitted-but-abusive
counterparty can therefore drive unbounded inference spend.

:func:`check_inbound_budget` is a coarse per-``(connection_id, chat_id)``
rolling-count budget enforced at the shared connector-write boundary —
the ``RuntimeAuthDep`` routes that append an event and defer a wake.
It is the *cap-before-side-effect* shape already proven in
:func:`aios.services.wake.deliver_cross_session_wake` and
:func:`aios.services.wake.count_recent_wakes_from`: validate the
counterparty against the budget, then — and only then — append and wake.

The budget counts *already-persisted* admitted inbounds over a rolling
window. The natural per-counterparty key already on every admitted inbound
is ``orig_channel`` (``_append_with_dedup`` stamps
``orig_channel = f"{connector}/{external_account_id}/{chat_id}"`` on each
``role=user`` event), so the allow path adds **no new write** — the next
message's count naturally includes the ones that landed.

**Disabled by default.** Both knobs default to ``0`` (disabled). When the
threshold is disabled the helper short-circuits to "admit" with no query,
so an inbound and a wake-bearing lifecycle call each take a path
byte-identical to pre-PR behavior.

**Cross-tenant isolation by construction.** The window query is
``account_id``-scoped (two tenants holding the same external ``chat_id``
get independent budgets), identical to how ``count_recent_wakes_from``
scopes by ``account_id``.

v1 enforces a **message-rate** cap only. The token/cost variant is a
strictly-additive extension of the same window query (a different
aggregate / predicate over the same rolling window) and may ship later;
the setting names leave room for it.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.config import get_settings

# The "inference-bearing inbound" shape, defined ONCE and shared by both
# ``_count_recent_*`` counters (single-source rather than duplicated-by-
# vigilance). Every counted row pairs an append with a ``defer_wake`` → one
# ``run_session_step`` → one model inference:
#   * an admitted inbound message (``kind='message'``, ``role='user'``), and
#   * a wake-bearing connector lifecycle (``kind='lifecycle'`` stamped with
#     ``wake=True`` by the two single-target wake routes, #1558).
# ``(data->>'wake')::boolean IS TRUE`` matches ONLY the ``wake=True`` lifecycle
# writes: a no-wake lifecycle and every internal harness lifecycle transition
# carry no ``wake`` key (``->>`` yields ``NULL`` → not true), so the free paths
# stay uncounted by construction.
_INFERENCE_BEARING_PREDICATE = """
              (kind = 'message'   AND data->>'role' = 'user')
           OR (kind = 'lifecycle' AND (data->>'wake')::boolean IS TRUE)
"""


def inbound_orig_channel(connector: str, external_account_id: str, chat_id: str) -> str:
    """Build the ``orig_channel`` key a ``role=user`` event is stamped with.

    Mirrors ``_append_with_dedup``'s
    ``f"{connector}/{external_account_id}/{chat_id}"`` so the budget window
    counts exactly the rows the admitted-inbound append writes.
    """
    return f"{connector}/{external_account_id}/{chat_id}"


async def _count_recent_inbounds(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    orig_channel: str,
    window_seconds: int,
) -> int:
    """Count admitted ``role=user`` inbound events on ``orig_channel`` in the
    last ``window_seconds`` — the rolling-window aggregate the budget reads.

    Reuses the ``count_recent_wakes_from`` shape: a single ``SELECT count(*)``
    over the ``events`` log, ``account_id``-scoped, no new table. Reads only
    already-persisted state (no write on the allow path).
    """
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            f"""
            SELECT count(*)
            FROM events
            WHERE account_id = $1
              AND orig_channel = $2
              AND ({_INFERENCE_BEARING_PREDICATE})
              AND created_at > now() - make_interval(secs => $3::bigint)
            """,
            account_id,
            orig_channel,
            window_seconds,
        )
    return int(count or 0)


async def _count_recent_session_inbounds(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str,
    window_seconds: int,
) -> int:
    """Count admitted ``role=user`` inbound events on one ``session_id`` in the
    last ``window_seconds``.

    The session-grain variant of :func:`_count_recent_inbounds`, used by the
    session-lifecycle route, which carries a ``session_id`` but no ``chat_id``.
    A wake-bearing session-lifecycle maps to exactly one session, so keying the
    window on ``(account_id, session_id)`` is sufficient (and avoids a reverse
    ``session_id → chat_id`` lookup). Same ``events``-log window shape,
    ``account_id``-scoped.
    """
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            f"""
            SELECT count(*)
            FROM events
            WHERE account_id = $1
              AND session_id = $2
              AND ({_INFERENCE_BEARING_PREDICATE})
              AND created_at > now() - make_interval(secs => $3::bigint)
            """,
            account_id,
            session_id,
            window_seconds,
        )
    return int(count or 0)


async def check_inbound_budget_session(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str,
) -> bool:
    """Session-grain budget check for the session-lifecycle wake path (#1504).

    The session-lifecycle route carries ``body.session_id`` but no ``chat_id``;
    a wake-bearing session-lifecycle maps to exactly one session, so the window
    is keyed on ``(account_id, session_id)`` (the event's ``session_id``
    column) rather than ``orig_channel``. Same disabled-by-default
    short-circuit and admit/throttle semantics as :func:`check_inbound_budget`.
    """
    settings = get_settings()
    threshold = settings.inbound_rate_max_per_window
    window_seconds = settings.inbound_rate_window_seconds
    if threshold <= 0 or window_seconds <= 0:
        return True

    recent = await _count_recent_session_inbounds(
        pool,
        account_id=account_id,
        session_id=session_id,
        window_seconds=window_seconds,
    )
    return recent < threshold


async def check_inbound_budget(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    connector: str,
    external_account_id: str,
    chat_id: str,
) -> bool:
    """Return whether one ``(connection_id, chat_id)`` counterparty is within
    its rolling per-window inbound budget.

    ``True`` = admit (within budget, or budget disabled); ``False`` = throttle
    (over budget). The caller drops an over-budget inbound BEFORE any append +
    wake — zero session spawn, zero event row, zero ``defer_wake``.

    **Disabled-by-default short-circuit.** When the per-window threshold is
    ``0`` (the default), this returns ``True`` immediately with no query, so the
    path is byte-identical to pre-PR behavior. The window-length knob being
    ``0`` likewise disables (an empty/zero window is meaningless).

    The window query is ``account_id``-scoped, so cross-tenant isolation holds
    by construction: two tenants holding the same external ``chat_id`` get
    independent budgets.
    """
    settings = get_settings()
    threshold = settings.inbound_rate_max_per_window
    window_seconds = settings.inbound_rate_window_seconds
    if threshold <= 0 or window_seconds <= 0:
        # Disabled: admit with no query (byte-identical to pre-PR behavior).
        return True

    orig_channel = inbound_orig_channel(connector, external_account_id, chat_id)
    recent = await _count_recent_inbounds(
        pool,
        account_id=account_id,
        orig_channel=orig_channel,
        window_seconds=window_seconds,
    )
    # ``recent`` counts the already-appended admitted inbounds in the window.
    # Admit while strictly under threshold; the message about to be appended is
    # the ``recent``-th+1, so ``recent >= threshold`` means this one tips over.
    return recent < threshold
